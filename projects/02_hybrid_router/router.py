"""
Hybrid Router — route agent calls between fast and frontier model tiers.

Classifies task complexity using the fast model, then routes:
  - Below threshold → Mistral Small 4 (119B, fast, cheap)
  - Above threshold → Mistral Large 3 (675B MoE, complex reasoning)

Both tiers served through the same NVIDIA API catalog
(integrate.api.nvidia.com) — one key, one endpoint pattern.

This is the same pattern NVIDIA's OpenShell Privacy Router implements
at the infrastructure level — but here it's application-layer routing
you control.

Usage:
    from router import HybridRouter

    router = HybridRouter(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=NVIDIA_API_KEY,
    )
    result = router.route("Classify this email as spam or not spam")
    print(result.backend)   # "fast" or "frontier"
    print(result.answer)
"""

import json
import os
import time
from dataclasses import dataclass, field

from openai import OpenAI


@dataclass
class RoutingDecision:
    """Result of a routing decision."""
    task: str
    complexity_score: float
    threshold: float
    backend: str                    # "fast" or "frontier"
    answer: str
    classification_ms: float        # time to classify
    completion_ms: float            # time to generate answer
    total_ms: float
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class RouterStats:
    """Aggregate stats across multiple routing decisions."""
    decisions: list[RoutingDecision] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.decisions)

    @property
    def fast_count(self) -> int:
        return sum(1 for d in self.decisions if d.backend == "fast")

    @property
    def frontier_count(self) -> int:
        return sum(1 for d in self.decisions if d.backend == "frontier")

    @property
    def fast_pct(self) -> float:
        return (self.fast_count / self.total * 100) if self.total else 0

    @property
    def avg_complexity(self) -> float:
        if not self.decisions:
            return 0
        return sum(d.complexity_score for d in self.decisions) / self.total

    @property
    def avg_latency_ms(self) -> float:
        if not self.decisions:
            return 0
        return sum(d.total_ms for d in self.decisions) / self.total

    def to_viz_data(self) -> list[dict]:
        """Export for visualization."""
        return [
            {
                "task": d.task[:80],
                "complexity": d.complexity_score,
                "backend": "local" if d.backend == "fast" else "frontier",
                "latency_ms": d.total_ms,
                "tokens": d.input_tokens + d.output_tokens,
            }
            for d in self.decisions
        ]


CLASSIFIER_PROMPT = """\
You are a task complexity classifier. Given a task description, score its \
complexity from 0.0 to 1.0 based on these criteria:

- 0.0-0.3: Simple lookup, classification, extraction, formatting
- 0.3-0.6: Moderate analysis, summarization, structured generation
- 0.6-0.8: Multi-step reasoning, synthesis across sources, nuanced judgment
- 0.8-1.0: Complex reasoning chains, ambiguous problems, creative/strategic work

Respond with ONLY a JSON object: {"score": <float>, "reason": "<one sentence>"}"""


class HybridRouter:
    """
    Routes tasks between fast and frontier model tiers.

    Both tiers use the same OpenAI-compatible API (NVIDIA catalog).
    The classifier runs on the fast model (cheap, fast). Only tasks
    above the complexity threshold get sent to the frontier model.
    """

    def __init__(
        self,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        api_key: str | None = None,
        fast_model: str = "mistralai/mistral-small-4-119b-2603",
        frontier_model: str = "mistralai/mistral-large-3-instruct-2512",
        threshold: float = 0.6,
    ):
        self.threshold = threshold
        self.fast_model = fast_model
        self.frontier_model = frontier_model
        self.stats = RouterStats()

        # Single client — both tiers go through the same API
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or os.environ.get("NVIDIA_API_KEY"),
        )

    def classify(self, task: str) -> tuple[float, str]:
        """Score task complexity using the fast model. Returns (score, reason)."""
        response = self.client.chat.completions.create(
            model=self.fast_model,
            messages=[
                {"role": "system", "content": CLASSIFIER_PROMPT},
                {"role": "user", "content": task},
            ],
            temperature=0.0,
            max_tokens=100,
        )
        text = response.choices[0].message.content or ""
        try:
            data = json.loads(text)
            return float(data.get("score", 0.5)), data.get("reason", "")
        except (json.JSONDecodeError, ValueError):
            return 0.5, f"Parse error, raw: {text[:100]}"

    def _complete_fast(self, task: str) -> tuple[str, int, int]:
        """Generate answer using fast model (Mistral Small 4)."""
        response = self.client.chat.completions.create(
            model=self.fast_model,
            messages=[{"role": "user", "content": task}],
        )
        usage = response.usage
        return (
            response.choices[0].message.content or "",
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )

    def _complete_frontier(self, task: str) -> tuple[str, int, int]:
        """Generate answer using frontier model (Mistral Large 3)."""
        response = self.client.chat.completions.create(
            model=self.frontier_model,
            messages=[{"role": "user", "content": task}],
        )
        usage = response.usage
        return (
            response.choices[0].message.content or "",
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )

    def route(self, task: str) -> RoutingDecision:
        """Classify, route, and complete a task."""
        # Step 1: classify complexity (always on fast model)
        t0 = time.perf_counter()
        score, reason = self.classify(task)
        classify_ms = (time.perf_counter() - t0) * 1000

        # Step 2: route based on threshold
        backend = "frontier" if score >= self.threshold else "fast"

        # Step 3: complete on chosen backend
        t1 = time.perf_counter()
        if backend == "fast":
            answer, in_tok, out_tok = self._complete_fast(task)
        else:
            answer, in_tok, out_tok = self._complete_frontier(task)
        complete_ms = (time.perf_counter() - t1) * 1000

        decision = RoutingDecision(
            task=task,
            complexity_score=score,
            threshold=self.threshold,
            backend=backend,
            answer=answer,
            classification_ms=classify_ms,
            completion_ms=complete_ms,
            total_ms=classify_ms + complete_ms,
            input_tokens=in_tok,
            output_tokens=out_tok,
        )
        self.stats.decisions.append(decision)
        return decision

    def route_batch(self, tasks: list[str]) -> list[RoutingDecision]:
        """Route multiple tasks sequentially, collecting stats."""
        return [self.route(task) for task in tasks]
