"""
Demo: Run the hybrid router on sample tasks and visualize routing decisions.

Requires MISTRAL_API_KEY and ANTHROPIC_API_KEY env vars.

Usage:
  export MISTRAL_API_KEY=your_key
  export ANTHROPIC_API_KEY=your_key

  # Run with default tasks
  python -m projects.02-hybrid-router.demo

  # Custom threshold
  python -m projects.02-hybrid-router.demo --threshold 0.5

  # Use local NIM/vLLM instead of Mistral API
  python -m projects.02-hybrid-router.demo --local-url http://localhost:8000/v1
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "projects" / "02_hybrid_router"))

from router import HybridRouter
from viz.routing import render as render_routing


SAMPLE_TASKS = [
    # Simple — should route local
    "Classify this email subject as spam or not spam: 'You won $1M click here'",
    "Extract the total amount from: 'Invoice #4521, Total: $2,340.00'",
    "What day of the week is July 4, 2026?",
    "Convert 72°F to Celsius",
    "Parse this JSON and list the keys: {\"name\": \"Ada\", \"role\": \"eng\"}",
    "Translate 'Good morning' to Spanish, French, and Japanese",
    "Summarize this in one sentence: 'The quarterly revenue was $4.2B, up 12% YoY'",

    # Complex — should route frontier
    "Analyze the competitive positioning of Anthropic vs OpenAI vs Google in enterprise AI, considering pricing, model quality, safety approach, and developer experience",
    "Write a persuasive appeal letter for a denied health insurance claim for physical therapy after knee surgery, citing relevant policy clauses",
    "Design a migration plan from a Django monolith to a microservices architecture, considering data consistency, team structure, and phased rollout",
    "Evaluate the trade-offs between PostgreSQL, DynamoDB, and CockroachDB for a multi-region financial application with strict consistency requirements",

    # Borderline — interesting routing decisions
    "Write a professional email declining a meeting invitation due to scheduling conflict",
    "Generate 5 unit test cases for a function that validates email addresses",
    "Rewrite this paragraph for a C-suite audience: 'We need to fix the data pipeline because it keeps breaking and losing customer records'",
]


def main():
    parser = argparse.ArgumentParser(description="Hybrid router demo")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--local-model", default="mistral-small-latest")
    parser.add_argument("--local-url", default="https://api.mistral.ai/v1")
    parser.add_argument("--local-key", default=None)
    parser.add_argument("--frontier-model", default="claude-sonnet-4-20250514")
    parser.add_argument("--out", default="output")
    args = parser.parse_args()

    local_key = args.local_key or os.environ.get("MISTRAL_API_KEY")
    frontier_key = os.environ.get("ANTHROPIC_API_KEY")

    if not local_key:
        print("Error: Set MISTRAL_API_KEY env var or pass --local-key")
        sys.exit(1)
    if not frontier_key:
        print("Error: Set ANTHROPIC_API_KEY env var")
        sys.exit(1)

    print(f"Local:     {args.local_model} @ {args.local_url}")
    print(f"Frontier:  {args.frontier_model}")
    print(f"Threshold: {args.threshold}")
    print(f"Tasks:     {len(SAMPLE_TASKS)}")
    print()

    router = HybridRouter(
        local_base_url=args.local_url,
        local_api_key=local_key,
        local_model=args.local_model,
        frontier_model=args.frontier_model,
        threshold=args.threshold,
    )

    for i, task in enumerate(SAMPLE_TASKS, 1):
        print(f"[{i:2d}/{len(SAMPLE_TASKS)}] ", end="", flush=True)
        decision = router.route(task)
        icon = "🟠" if decision.backend == "local" else "🔵"
        print(f"{icon} {decision.complexity_score:.2f} → {decision.backend:8s} "
              f"({decision.total_ms:,.0f}ms)  {task[:60]}")

    # Summary
    stats = router.stats
    print(f"\n{'='*60}")
    print(f"Total:    {stats.total} tasks")
    print(f"Local:    {stats.local_count} ({stats.local_pct:.0f}%)")
    print(f"Frontier: {stats.frontier_count} ({100-stats.local_pct:.0f}%)")
    print(f"Avg latency: {stats.avg_latency_ms:,.0f}ms")

    # Save and visualize
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = stats.to_viz_data()
    data_file = out_dir / "routing-decisions.json"
    with open(data_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nData saved → {data_file}")

    render_routing(out_dir, data, args.threshold)


if __name__ == "__main__":
    main()
