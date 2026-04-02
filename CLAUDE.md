# agentic-inference — Agent Instructions

## Purpose

This repository demonstrates cost-effective agentic AI inference using the NVIDIA NIM ecosystem. Every project runs through a single NVIDIA API endpoint with one key, two model tiers.

## Architecture

Three-layer stack:
1. **Governance** — OpenShell sandbox, policy engine, privacy router
2. **Serving** — Dynamo orchestration, NIM containers, TensorRT-LLM, vLLM, NIXL
3. **Models** — Hybrid routing between Mistral Small 4 (119B, fast tier) and Mistral Large 3 (675B MoE, frontier tier)

## Agent Behavior Rules

- **Route by complexity.** Use Mistral Small 4 for extraction, classification, formatting, and simple queries (complexity < 0.6). Escalate to Mistral Large 3 only for multi-step reasoning, synthesis, or ambiguous problems.
- **Minimize token waste.** Every tool call costs tokens. Batch parallel tool calls where possible (e.g., fetch all RSS feeds in a single turn rather than one per turn).
- **Expose telemetry.** Every agent run must capture: turns, tool calls per turn, tokens (input + output), latency per step, and total cost. Save trace as JSON.
- **Fail gracefully.** Tool errors should be caught and returned as JSON `{"error": "..."}` — never crash the agent loop.
- **Strict tool schemas.** Tools are defined with Python type hints. The registry auto-generates OpenAI-compatible JSON schemas. Keep tool interfaces narrow and well-typed.

## Project Structure

```
projects/
  01_tool_calling/    # Core while(tool_use) agent loop + tool registry
  02_hybrid_router/   # Complexity classifier → fast/frontier routing
  03_news_aggregator/ # AI news digest agent (9 RSS sources → markdown)
viz/
  theme.py            # Shared NVIDIA-green dark palette
  stack.py            # Three-layer architecture diagram
  cost.py             # API vs self-hosted cost crossover
  trace.py            # Agent loop execution trace
  routing.py          # Hybrid routing decisions scatter
  trace_analysis.py   # Per-turn latency breakdown with bottleneck annotation
output/               # Generated visualizations + digests
```

## Running

```bash
pip install -r requirements.txt
export NVIDIA_API_KEY=your_key  # from build.nvidia.com

# Visualizations (no API key needed)
python -m viz.stack
python -m viz.cost
python -m viz.trace
python -m viz.routing

# Live demos (requires API key)
python projects/01_tool_calling/demo.py
python projects/02_hybrid_router/demo.py --threshold 0.6
python projects/03_news_aggregator/demo.py --days 7
```

## Coding Standards

- Python 3.10+ with type hints on all function signatures
- Dataclasses for structured results (not dicts)
- OpenAI-compatible client interface (works with NIM, vLLM, Mistral API)
- Strip unsupported message fields for NVIDIA NIM compatibility (`audio`, `refusal`, `annotations`)
- Minimal dependencies — stdlib where possible (urllib, xml.etree, html)
