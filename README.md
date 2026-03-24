# agentic-inference

Exploring the agentic AI inference stack — from open-weight models to self-hosted serving to hybrid routing.

Companion repo to [Agents Don't Need Better Models. They Need Better Infrastructure.](https://twyoon.com/post/agents-need-better-infrastructure/)

## Visualizations

Publication-quality figures of the post-GTC 2026 NVIDIA AI stack.

```bash
pip install -r requirements.txt

python -m viz.stack                    # three-layer architecture diagram
python -m viz.cost                     # API vs self-hosted cost crossover
python -m viz.stack --out figures/     # custom output directory
```

| Figure | What it shows |
|--------|---------------|
| `stack` | Governance → Serving → Models with hybrid routing arrows |
| `cost` | Monthly cost crossover at ~5,500 agent calls/day |

## Projects

Mini projects showcasing [Mistral Small 4](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603) (119B, agentic tool-calling, open weights) on the NVIDIA inference stack.

### 01 — Tool-Calling Agent Loop

> **The core pattern.** A `while(tool_use)` agent loop running entirely on Mistral Small 4.

The model decides which tools to call and when to stop — no hardcoded step sequence. Demonstrates that open-weight models can drive autonomous tool-calling loops, not just answer questions.

- Run locally via [vLLM](https://github.com/vllm-project/vllm) or [NIM](https://build.nvidia.com/)
- Tools: file reader, web fetcher, calculator
- Compare: latency and tool-selection accuracy vs Claude Sonnet on the same tasks

### 02 — Hybrid Router

> **Route by complexity.** Easy calls → self-hosted Mistral. Hard calls → frontier API.

A lightweight routing layer that classifies incoming requests and dispatches them to the right backend. The same architecture described in the blog article — and the same pattern NVIDIA's OpenShell Privacy Router implements at the infrastructure level.

- Classifier: Mistral Small 4 scores task complexity (0–1) in a single call
- Threshold routing: below 0.6 → local Mistral, above → Claude API
- Measure: cost savings vs quality tradeoff at different thresholds
- Log every decision for analysis

### 03 — Job Scanner on Open Stack

> **Port a real workflow from API to self-hosted.** Same logic, different infrastructure.

The [career monitor](https://twyoon.com/post/agents-need-better-infrastructure/) from the blog article currently runs on Claude's API — 345 calls/day across 23 companies. Port the two-stage screening pipeline (title filter → JD keyword classification) to Mistral Small 4 on NIM.

- Stage 1 (title classification): should match or exceed API accuracy — it's a simple classification task
- Stage 2 (JD analysis): test whether 119B parameters handle nuanced keyword matching
- Benchmark: accuracy parity, latency difference, cost difference over 30 days

### 04 — Inference Pattern Benchmarks

> **Measure what the article claims.** Fan-out, chaining, and iterative loops — benchmarked.

Run the three agent inference patterns from the blog against both API and self-hosted backends. Quantify latency, throughput, and cost for each.

| Pattern | Workload | What to measure |
|---------|----------|-----------------|
| Parallel fan-out | 23 concurrent classification calls | Throughput ceiling, rate limit impact |
| Sequential chain | 4-step transcript → summary → actions → push | End-to-end latency |
| Iterative loop | 3-pass draft → evaluate → revise | Token cost scaling, context growth |

- Self-hosted: vLLM + Mistral Small 4 on a single A100
- API: Claude Sonnet via Anthropic API
- Output: comparison table + visualization (extend `viz/`)

### 05 — Devstral Code Agent

> **Agentic coding on open weights.** [Devstral](https://huggingface.co/mistralai/Devstral-Small-2507) (24B active / 123B total) as a local Copilot alternative.

Build a minimal code agent that reads a file, identifies issues, proposes fixes, and applies them — the same loop Claude Code and Codex run, but on a self-hosted model.

- Use Devstral's native tool-calling for file read/write/search
- Test on real tasks: bug fixes, refactors, test generation
- Compare: fix quality and autonomy vs Claude Code on the same tasks

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
