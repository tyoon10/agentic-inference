#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Serving benchmark: what does prefix caching actually buy?
#
# Runs Qwen2.5-3B-Instruct on vLLM and SGLang, each with prefix/radix caching
# ON vs OFF, over a SHARED-PREFIX workload (the case caching is built for, and
# the shape of a real agentic workload that reuses a system prompt). Reports
# TTFT and throughput for each config so the delta is a measured number, not a
# marketing claim.
#
# This is a per-engine measurement (cache on vs off), NOT a vLLM-vs-SGLang race
# — kernel-backend differences on consumer GPUs make a cross-engine winner claim
# unsafe. The honest, defensible result is "caching cut TTFT by X% on this engine
# for this workload."
#
# WHERE TO RUN: a CUDA GPU. Recommended RunPod pod:
#   GPU     : RTX 4090 24GB (SM89) or A40 48GB — Qwen-3B fits with room to spare
#   Template: a PyTorch image on CUDA 12.x  (AVOID CUDA 13.2 images — a known
#             mid-2026 bug produces gibberish on Qwen)
#   Disk    : 30GB+ for the model + envs
#
# USAGE (from the repo root, on the GPU):
#   bash benchmarks/run_bench.sh            # both engines (RunPod / one env)
#   bash benchmarks/run_bench.sh vllm       # vLLM only   (Colab: run, then Restart runtime)
#   bash benchmarks/run_bench.sh sglang     # SGLang only (Colab: fresh runtime)
#   python benchmarks/summarize.py          # -> figure + RESULTS.md + registry YAML
#
# vLLM and SGLang pin conflicting deps — installing BOTH in one env is the main
# failure mode. On RunPod the sequential install usually survives; on Colab, run
# each engine in a fresh runtime (see benchmarks/COLAB.md). summarize.py combines
# whatever result files exist, so per-engine runs compose cleanly.
#
# ~15-20 min wall / a few GPU-dollars. Everything writes to benchmarks/results/.
# ---------------------------------------------------------------------------
set -uo pipefail

ENGINE="${1:-all}"   # all | vllm | sglang

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
OUT="${BENCH_RESULTS:-benchmarks/results}"   # override to a Drive path for Colab's per-engine flow
LOGS="$OUT/logs"
mkdir -p "$LOGS"

# Workload — a shared prefix (system-prompt-shaped) plus a short unique tail.
NUM_PROMPTS="${NUM_PROMPTS:-200}"
INPUT_LEN="${INPUT_LEN:-256}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
PREFIX_LEN="${PREFIX_LEN:-512}"   # shared prefix tokens (what the cache reuses)
CONC="${CONC:-32}"

# SGLang on some consumer cards (SM86/SM89) hits FlashInfer kernel issues.
# If SGLang errors on startup, re-run with:  SGLANG_EXTRA="--attention-backend triton"
SGLANG_EXTRA="${SGLANG_EXTRA:-}"

log()  { printf '\n\033[1;32m>>> %s\033[0m\n' "$*"; }
warn() { printf '\n\033[1;33m!!! %s\033[0m\n' "$*"; }

wait_ready() {  # $1 = url, $2 = label, times out after ~5 min
  local url="$1" label="$2" i=0
  until curl -sf "$url" >/dev/null 2>&1; do
    sleep 3; i=$((i+1))
    if [ $i -gt 100 ]; then warn "$label did not become ready — check $LOGS"; return 1; fi
  done
  log "$label ready"
}

# =========================== vLLM ===========================
if [ "$ENGINE" = "all" ] || [ "$ENGINE" = "vllm" ]; then
log "Installing vLLM (bench extras)"
pip install -q "vllm[bench]" || { warn "vllm install failed"; }

run_vllm () {  # $1 = tag, $2 = extra `vllm serve` flags
  local tag="$1" serve_flags="$2"
  log "vLLM serve ($tag): $serve_flags"
  vllm serve "$MODEL" --port 8000 --disable-log-requests $serve_flags \
    > "$LOGS/vllm-$tag.log" 2>&1 &
  local svr=$!
  if wait_ready "http://localhost:8000/v1/models" "vLLM($tag)"; then
    vllm bench serve --model "$MODEL" --base-url "http://localhost:8000" \
      --dataset-name random --num-prompts "$NUM_PROMPTS" \
      --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
      --random-prefix-len "$PREFIX_LEN" --max-concurrency "$CONC" \
      --save-result --result-filename "$OUT/vllm-$tag.json" \
      2>&1 | tee -a "$LOGS/vllm-$tag.log"
  fi
  kill "$svr" 2>/dev/null; wait "$svr" 2>/dev/null; sleep 5
}
run_vllm cache-on  ""
run_vllm cache-off "--no-enable-prefix-caching"
fi

# =========================== SGLang ===========================
if [ "$ENGINE" = "all" ] || [ "$ENGINE" = "sglang" ]; then
log "Installing SGLang"
pip install -q "sglang[all]" || { warn "sglang install failed"; }

run_sglang () {  # $1 = tag, $2 = extra launch flags
  local tag="$1" launch_flags="$2"
  log "SGLang launch ($tag): $launch_flags $SGLANG_EXTRA"
  python -m sglang.launch_server --model-path "$MODEL" --port 8001 \
    $launch_flags $SGLANG_EXTRA > "$LOGS/sglang-$tag.log" 2>&1 &
  local svr=$!
  if wait_ready "http://localhost:8001/health" "SGLang($tag)"; then
    # generated-shared-prefix exercises RadixAttention directly
    python -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 8001 \
      --model "$MODEL" --dataset-name generated-shared-prefix \
      --gsp-num-groups 8 --gsp-prompts-per-group 25 \
      --gsp-system-prompt-len "$PREFIX_LEN" --gsp-question-len "$INPUT_LEN" \
      --gsp-output-len "$OUTPUT_LEN" --max-concurrency "$CONC" \
      --output-file "$OUT/sglang-$tag.jsonl" \
      2>&1 | tee -a "$LOGS/sglang-$tag.log"
  fi
  kill "$svr" 2>/dev/null; wait "$svr" 2>/dev/null; sleep 5
}
run_sglang cache-on  ""
run_sglang cache-off "--disable-radix-cache"
fi

log "Done ($ENGINE). Raw results in $OUT/. When both engines are done, run:  python benchmarks/summarize.py"
ls -la "$OUT"
