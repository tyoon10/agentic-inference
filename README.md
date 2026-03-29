# agentic-inference

Exploring the agentic AI inference stack — from open-weight models to self-hosted serving to hybrid routing.

Companion repo to [Agents Don't Need Better Models. They Need Better Infrastructure.](https://twyoon.com/post/agents-need-better-infrastructure/)

## Quickstart

```bash
pip install -r requirements.txt
```

### Visualizations

Four publication-quality figures generated from code:

```bash
python -m viz.stack                    # three-layer architecture diagram
python -m viz.cost                     # API vs self-hosted cost crossover
python -m viz.trace                    # agent loop execution trace
python -m viz.routing                  # hybrid routing decisions scatter
```

All accept `--out path/` for custom output directory.

| Figure | What it shows |
|--------|---------------|
| `stack` | Governance → Serving → Models with hybrid routing arrows |
| `cost` | Monthly cost crossover at ~5,500 agent calls/day |
| `trace` | Step-by-step agent loop: model → tool call → result → answer |
| `routing` | Tasks plotted by complexity, colored by local vs frontier backend |

### Run the demos (requires API keys)

```bash
export MISTRAL_API_KEY=your_key
export ANTHROPIC_API_KEY=your_key

# 01 — Agent loop on Mistral Small 4
python projects/01_tool_calling/demo.py

# 02 — Hybrid router: Mistral (local) + Claude (frontier)
python projects/02_hybrid_router/demo.py --threshold 0.6

# 03 — AI news aggregator (weekly digest)
python projects/03_news_aggregator/demo.py --days 7
```

Both demos run the real agent/router, save trace data as JSON, and auto-generate visualizations.

## Projects

Mini projects showcasing [Mistral Small 4](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603) (119B, agentic tool-calling, open weights) on the NVIDIA inference stack.

### 01 — Tool-Calling Agent Loop ✅

> **The core pattern.** A `while(tool_use)` agent loop running entirely on Mistral Small 4.

The model decides which tools to call and when to stop — no hardcoded step sequence. Demonstrates that open-weight models can drive autonomous tool-calling loops, not just answer questions.

```
projects/01_tool_calling/
  agent.py    — Agent class: while(tool_use) loop with trace capture
  tools.py    — ToolRegistry + 5 built-in tools (calculator, file read, etc.)
  demo.py     — Run agent → save trace JSON → render viz
```

- Works with any OpenAI-compatible endpoint: NIM, vLLM, Mistral API
- Auto-generates tool schemas from Python type hints
- Full trace capture: tokens, latency, tool calls per turn
- Visualization: `viz/trace.py` renders the execution as a vertical flow diagram

### 02 — Hybrid Router ✅

> **Route by complexity.** Easy calls → self-hosted Mistral. Hard calls → frontier API (Claude).

A lightweight routing layer that classifies incoming requests and dispatches them to the right backend. The same architecture described in the blog article — and the same pattern NVIDIA's OpenShell Privacy Router implements at the infrastructure level.

```
projects/02_hybrid_router/
  router.py   — HybridRouter: classify → route → complete, with stats
  demo.py     — Run 15 sample tasks → save decisions JSON → render viz
```

- Classifier: Mistral Small 4 scores task complexity (0–1) in a single call
- Threshold routing: below 0.6 → local Mistral, above → Claude API
- Aggregate stats: local %, avg latency, tokens per backend
- Visualization: `viz/routing.py` plots decisions as a scatter with threshold line

### 03 — AI News Aggregator ✅

> **Smart weekly digest.** An agent that fetches, filters, and summarizes the week's most significant AI news.

The agent autonomously crawls RSS feeds from 9 major AI sources (OpenAI, Google, Anthropic, NVIDIA, Mistral, DeepMind, HuggingFace, MIT Tech Review, arXiv), identifies the most impactful stories, fetches full article text for top picks, and compiles a ranked markdown digest.

```
projects/03_news_aggregator/
  news_tools.py — RSS fetching, article extraction, date filtering, digest saver
  demo.py       — Run agent → fetch all feeds → produce weekly digest
```

- 9 built-in AI news sources (RSS/Atom)
- Full article text extraction for deep summaries
- Configurable lookback window (`--days 3`, `--days 14`)
- Works with NIM, vLLM, or Mistral API — same OpenAI-compatible interface
- Outputs ranked markdown digest + full agent trace

### 04 — Inference Pattern Benchmarks

> **Measure what the article claims.** Fan-out, chaining, and iterative loops — benchmarked.

| Pattern | Workload | What to measure |
|---------|----------|-----------------|
| Parallel fan-out | 23 concurrent classification calls | Throughput ceiling, rate limit impact |
| Sequential chain | 4-step transcript → summary → actions → push | End-to-end latency |
| Iterative loop | 3-pass draft → evaluate → revise | Token cost scaling, context growth |

### 05 — Devstral Code Agent

> **Agentic coding on open weights.** [Devstral](https://huggingface.co/mistralai/Devstral-Small-2507) (24B active / 123B total) as a local Copilot alternative.

Build a minimal code agent that reads a file, identifies issues, proposes fixes, and applies them — the same loop Claude Code and Codex run, but on a self-hosted model.

## Structure

```
agentic-inference/
  viz/                          # Visualization modules (matplotlib, dark theme)
    theme.py                    # Shared NVIDIA-green palette
    stack.py                    # Three-layer architecture diagram
    cost.py                     # API vs self-hosted cost crossover
    trace.py                    # Agent loop execution trace
    routing.py                  # Hybrid routing decisions scatter
  projects/
    01_tool_calling/            # ✅ Agent loop + tool registry
    02_hybrid_router/           # ✅ Complexity-based routing
    03_news_aggregator/         # ✅ AI news digest agent
    04_inference_benchmarks/    # Benchmark fan-out, chain, loop
    05_devstral_code_agent/     # Code agent on Devstral
```

## Stack Reference

| Layer | Tool | Role |
|-------|------|------|
| Orchestration | Dynamo 1.0 | Multi-node inference coordination |
| Governance | OpenShell | Sandbox + policy engine + privacy router |
| Serving | NIM | One-command model containers |
| Optimization | TensorRT-LLM | GPU compiler optimization |
| Inference | vLLM | Community inference engine |
| Context | CMX (BlueField-4) | Hardware context memory offload |

## License

MIT
