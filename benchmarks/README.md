# Serving benchmark — what prefix caching actually buys

A reproducible measurement of one thing the [companion post](https://twyoon.com/writings/agents-need-better-infrastructure/)
argues from theory: on agentic workloads that reuse a system prompt,
serving-layer caching is a real lever, not a marketing line.

It serves **Qwen2.5-3B-Instruct** on **vLLM** and **SGLang**, each with its
prefix cache **on vs off**, over a **shared-prefix workload**, and reports the
delta in time-to-first-token and throughput.

**This is a per-engine A/B (cache on vs off), not a vLLM-vs-SGLang race.**
Kernel-backend differences on consumer GPUs make a cross-engine winner claim
unsafe; "caching cut TTFT by X% on this engine, on this workload" is a number
that survives review.

## Run it

Any CUDA GPU works. Two paths:

**RunPod (recommended, ~$0.25, plain shell):**

| Setting | Value |
|---|---|
| GPU | RTX 4090 24GB (SM89) or A40 48GB — Qwen-3B fits with room to spare |
| Template | a PyTorch image on **CUDA 12.x** — avoid CUDA 13.2 images (a known mid-2026 bug garbles Qwen output) |
| Disk | 30GB+ |

```bash
bash benchmarks/run_bench.sh        # both engines, ~15-20 min
python benchmarks/summarize.py      # -> figure, RESULTS.md, registry entries
```

**Google Colab Pro** (if you already pay for it): see **[COLAB.md](COLAB.md)** —
the notebook cell sequence, plus how to isolate the two engines (they pin
conflicting deps) and pull artifacts off the ephemeral disk.

The runner takes an optional engine selector so the two can run separately
(`run_bench.sh vllm` / `run_bench.sh sglang`); `summarize.py` composes whatever
result files exist.

`run_bench.sh` installs both engines, serves the model four ways
(vLLM cache-on/off, SGLang radix-on/off), and benches each. `summarize.py`
turns the raw output into:

- `output/serving-prefix-cache-benchmark.png` — the figure
- `benchmarks/RESULTS.md` — the table + the measured deltas (fill in your GPU)
- `benchmarks/registry-entries.yaml` — `first-party-benchmark` claims records,
  ready to paste into the [cuda-ecosystem-radar](https://github.com/tyoon10/cuda-ecosystem-radar)
  claims registry

Tunables are env vars (`NUM_PROMPTS`, `INPUT_LEN`, `OUTPUT_LEN`, `PREFIX_LEN`,
`CONC`). The shared prefix (`PREFIX_LEN`, default 512 tokens) is what the cache
reuses — set it to zero and the effect vanishes, which is itself the point.

## Known gotcha

SGLang's FlashInfer kernels hit issues on some SM86/SM89 consumer cards. If
SGLang fails to start, re-run with the Triton attention backend:

```bash
SGLANG_EXTRA="--attention-backend triton" bash benchmarks/run_bench.sh
```

(Slightly slower, but it unblocks the run. Note the fallback in RESULTS.md if
you used it — it affects the SGLang absolute numbers, not the on-vs-off delta.)

## Honesty notes

- Every number in the outputs comes from the raw result files; nothing is
  hand-entered. A config that fails to run is reported as missing, not guessed.
- Absolute numbers are small-GPU, small-model, single-node — they illustrate the
  *mechanism*, not datacenter performance. The claim is the on-vs-off delta.
- Fill the real GPU name into RESULTS.md and the registry `attribution` before
  citing anything externally.
