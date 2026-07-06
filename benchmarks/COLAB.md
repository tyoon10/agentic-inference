# Running the benchmark on Google Colab Pro

Colab Pro works. The honest tradeoff vs. RunPod: Colab is a notebook with an
ephemeral disk, and vLLM + SGLang pin conflicting dependencies, so isolating
the two engines takes a little care. RunPod (a plain persistent shell, ~$0.25)
is less total hassle; use Colab if you already pay for it.

**Runtime → Change runtime type → GPU.** On Pro you'll usually get an **L4**
(24GB, plenty for Qwen-3B) or an **A100**. Colab's CUDA is 12.x, which is what
we want (avoids the CUDA-13.2 Qwen bug). L4 is Ada (SM89), so if SGLang's
FlashInfer kernels error, use the Triton fallback shown below.

---

## Path A — one runtime (try this first)

**Use `!bash` (streams live), never `%%bash` (Colab buffers it and shows nothing
until the whole cell finishes — you'll think it hung).** Run the two engines as
separate cells so you see vLLM finish before SGLang starts, and keep the vLLM
results if SGLang's install clashes.

```python
# Cell 1 — clone
!git clone https://github.com/tyoon10/agentic-inference.git
%cd agentic-inference
!nvidia-smi --query-gpu=name,memory.total --format=csv
```

```python
# Cell 2 — vLLM (streams live; ~8-12 min incl. install + a ~6GB model download).
# Watch for: pip install -> "vllm x.y.z" -> ">>> vLLM serve" -> bench numbers.
!bash benchmarks/run_bench.sh vllm
```

```python
# Cell 3 — SGLang (Triton backend pre-set; L4's FlashInfer kernels are flaky)
!SGLANG_EXTRA="--attention-backend triton" bash benchmarks/run_bench.sh sglang
```

```python
# Cell 4 — summarize + show
!pip -q install matplotlib
!python benchmarks/summarize.py
from IPython.display import Image
Image("output/serving-prefix-cache-benchmark.png")
```

```python
# Cell 5 — pull the artifacts out (Colab disk is ephemeral)
from google.colab import files
print(open("benchmarks/RESULTS.md").read())
print(open("benchmarks/registry-entries.yaml").read())
files.download("output/serving-prefix-cache-benchmark.png")
files.download("benchmarks/RESULTS.md")
files.download("benchmarks/registry-entries.yaml")
```

The harness fails fast and loudly: if an engine's install/import breaks, it says
so and skips that engine instead of hanging. If SGLang's install clashes with
vLLM's in this shared runtime (Cell 3 reports a skip), your vLLM results are
already saved — do SGLang via **Path B** (fresh runtime) and `summarize.py` will
compose both.

---

## Path B — separate runtimes via Drive (if Path A's SGLang install conflicts)

Persist raw results to Drive so a fresh runtime for each engine doesn't lose them.

```python
# Cell 1 — Drive + clone (run once)
from google.colab import drive; drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/agentic-bench
!git clone https://github.com/tyoon10/agentic-inference.git
%cd agentic-inference
```

```bash
%%bash
# Cell 2 — vLLM only, results to Drive
export BENCH_RESULTS=/content/drive/MyDrive/agentic-bench
bash benchmarks/run_bench.sh vllm
```

**Now: Runtime → Disconnect and delete runtime.** Reconnect with a fresh GPU,
re-mount Drive, `%cd agentic-inference` (re-clone if needed), then:

```bash
%%bash
# Cell 3 — SGLang only, same Drive folder (fresh runtime = clean deps)
export BENCH_RESULTS=/content/drive/MyDrive/agentic-bench
# SGLANG_EXTRA="--attention-backend triton" \
bash benchmarks/run_bench.sh sglang
```

```python
# Cell 4 — summarize from Drive, show, download
import os; os.environ["BENCH_RESULTS"] = "/content/drive/MyDrive/agentic-bench"
!pip -q install matplotlib
!python benchmarks/summarize.py
from IPython.display import Image
Image("output/serving-prefix-cache-benchmark.png")
```

---

## After the run

1. Fill the real GPU (e.g. "Colab L4 24GB") into `RESULTS.md` and each registry
   entry's `attribution` line.
2. Commit the figure + `RESULTS.md` to this repo, and paste the entries in
   `registry-entries.yaml` into `cuda-ecosystem-radar/claims/`.
3. Add one line to `benchmarks/RESULTS.md` if you used the Triton fallback — it
   affects SGLang's absolute numbers, not the on-vs-off delta.
