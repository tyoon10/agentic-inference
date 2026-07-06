"""
Summarize the serving benchmark: parse vLLM + SGLang results, render one figure,
write RESULTS.md, and emit ready-to-paste claims-registry entries.

Reads benchmarks/results/{vllm-cache-on,vllm-cache-off}.json (vLLM --save-result)
and {sglang-cache-on,sglang-cache-off}.jsonl (SGLang --output-file), all produced
by run_bench.sh. Every number here comes from those files — nothing is invented;
missing configs are reported as missing, not guessed.

Usage:
  python benchmarks/summarize.py                 # after run_bench.sh
  python benchmarks/summarize.py --out benchmarks
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Override with BENCH_RESULTS to read raw results from a Drive path (Colab per-engine flow).
RESULTS = Path(os.environ.get("BENCH_RESULTS", ROOT / "benchmarks" / "results"))

# Which metrics we lift out of each engine's result blob (defensive key lookup).
METRICS = {
    "median_ttft_ms": ["median_ttft_ms"],
    "p99_ttft_ms": ["p99_ttft_ms"],
    "output_throughput": ["output_throughput"],          # tokens/s
    "request_throughput": ["request_throughput"],         # req/s
    "median_itl_ms": ["median_itl_ms"],
    "num_prompts": ["num_prompts", "completed"],
}

CONFIGS = [
    ("vLLM", "cache-on", RESULTS / "vllm-cache-on.json", "json"),
    ("vLLM", "cache-off", RESULTS / "vllm-cache-off.json", "json"),
    ("SGLang", "cache-on", RESULTS / "sglang-cache-on.jsonl", "jsonl"),
    ("SGLang", "cache-off", RESULTS / "sglang-cache-off.jsonl", "jsonl"),
]


def _load(path: Path, kind: str) -> dict | None:
    if not path.exists():
        return None
    try:
        if kind == "json":
            return json.loads(path.read_text())
        # jsonl: take the last non-empty object (most recent run appended)
        objs = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
        return objs[-1] if objs else None
    except (json.JSONDecodeError, IndexError):
        return None


def _pick(blob: dict, keys: list[str]):
    for k in keys:
        if k in blob and blob[k] is not None:
            return blob[k]
    return None


def collect() -> dict:
    """Return {engine: {tag: {metric: value}}} for whatever ran."""
    out: dict = {}
    for engine, tag, path, kind in CONFIGS:
        blob = _load(path, kind)
        if blob is None:
            print(f"  [missing] {engine} {tag}  ({path.name}) — skipped")
            continue
        out.setdefault(engine, {})[tag] = {m: _pick(blob, ks) for m, ks in METRICS.items()}
    return out


def pct_change(new, base):
    if base in (None, 0) or new is None:
        return None
    return (new - base) / base * 100.0


def render_figure(data: dict, out_dir: Path):
    import matplotlib.pyplot as plt
    import numpy as np
    from viz.theme import apply_theme, CARD_BG, BORDER, TEXT, TEXT_DIM, GREEN, BLUE

    engines = [e for e in ("vLLM", "SGLang") if e in data]
    if not engines:
        print("  no data to plot"); return None
    apply_theme()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 6))
    x = np.arange(len(engines)); w = 0.36
    ON, OFF = GREEN, TEXT_DIM  # cache-on = NVIDIA green, cache-off = muted

    def bars(ax, metric, title, unit, lower_is_better):
        on = [(data[e].get("cache-on") or {}).get(metric) for e in engines]
        off = [(data[e].get("cache-off") or {}).get(metric) for e in engines]
        on_v = [v if v is not None else 0 for v in on]
        off_v = [v if v is not None else 0 for v in off]
        b1 = ax.bar(x - w/2, on_v, w, label="cache ON", color=ON, zorder=3)
        b2 = ax.bar(x + w/2, off_v, w, label="cache OFF", color=OFF, zorder=3)
        for bars_, vals in ((b1, on), (b2, off)):
            for rect, v in zip(bars_, vals):
                if v is None:
                    continue
                ax.text(rect.get_x() + rect.get_width()/2, rect.get_height(),
                        f"{v:,.0f}", ha="center", va="bottom",
                        fontsize=9, color=TEXT, weight="bold")
        ax.set_xticks(x); ax.set_xticklabels(engines, fontsize=11)
        arrow = "lower is better" if lower_is_better else "higher is better"
        ax.set_title(title, fontsize=13, weight="bold", color=TEXT, pad=12)
        ax.set_ylabel(f"{unit}  ({arrow})", fontsize=10.5)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=9.5, framealpha=0.9, facecolor=CARD_BG,
                  edgecolor=BORDER, labelcolor=TEXT)

    bars(axL, "median_ttft_ms", "Median Time-to-First-Token", "ms", True)
    bars(axR, "output_throughput", "Output Throughput", "tokens / sec", False)

    fig.suptitle("What Prefix Caching Buys — Qwen2.5-3B, shared-prefix workload",
                 fontsize=15, weight="bold", color=TEXT, y=0.99)
    fig.text(0.5, 0.005,
             "Per-engine A/B on one GPU. Shared-prefix workload (agentic system-prompt reuse). "
             "Measured with `vllm bench serve` / `sglang.bench_serving`. Not a cross-engine race.",
             ha="center", fontsize=7.5, color=TEXT_DIM)
    out = out_dir / "output" / "serving-prefix-cache-benchmark.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"  figure -> {out}")
    return out


def write_results_md(data: dict, out_dir: Path):
    lines = ["# Serving Benchmark — What Prefix Caching Buys", "",
             "Qwen2.5-3B-Instruct, shared-prefix workload, single GPU. Per-engine "
             "A/B (caching on vs off) — a measured delta, not a marketing claim. "
             "**Fill in the hardware line below from your pod.**", "",
             "**Hardware:** _<GPU, e.g. RTX 4090 24GB>_ · vLLM _<ver>_ · SGLang _<ver>_ · CUDA 12.x", "",
             "| Engine | Config | Median TTFT (ms) | p99 TTFT (ms) | Output tok/s | Req/s |",
             "|---|---|--:|--:|--:|--:|"]
    for engine, tags in data.items():
        for tag in ("cache-on", "cache-off"):
            m = tags.get(tag)
            if not m:
                continue
            def f(v, p=0): return f"{v:,.{p}f}" if isinstance(v, (int, float)) else "—"
            lines.append(f"| {engine} | {tag} | {f(m['median_ttft_ms'])} | "
                         f"{f(m['p99_ttft_ms'])} | {f(m['output_throughput'])} | "
                         f"{f(m['request_throughput'],1)} |")
    lines += ["", "## What caching bought (cache-off → cache-on)", ""]
    for engine, tags in data.items():
        on, off = tags.get("cache-on"), tags.get("cache-off")
        if not (on and off):
            continue
        ttft = pct_change(on["median_ttft_ms"], off["median_ttft_ms"])
        thr = pct_change(on["output_throughput"], off["output_throughput"])
        parts = []
        if ttft is not None:
            parts.append(f"median TTFT {abs(ttft):.0f}% {'lower' if ttft < 0 else 'higher'}")
        if thr is not None:
            parts.append(f"output throughput {abs(thr):.0f}% {'higher' if thr > 0 else 'lower'}")
        lines.append(f"- **{engine}:** " + "; ".join(parts) +
                     " with prefix caching on, on this workload.")
    (out_dir / "benchmarks" / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(f"  results -> benchmarks/RESULTS.md")


def emit_registry(data: dict, out_dir: Path):
    """Emit first-party-benchmark YAML for the cuda-ecosystem-radar claims registry."""
    from datetime import date
    today = str(date.today())
    blocks = []
    for engine, tags in data.items():
        on, off = tags.get("cache-on"), tags.get("cache-off")
        if not (on and off):
            continue
        ttft = pct_change(on["median_ttft_ms"], off["median_ttft_ms"])
        eng = engine.lower()
        cache = "PagedAttention prefix cache" if engine == "vLLM" else "RadixAttention"
        blocks.append(f"""id: {eng}-prefix-cache-ttft-firstparty
claim: >
  On a shared-prefix workload (Qwen2.5-3B, single GPU), {engine}'s {cache}
  reduced median TTFT by {abs(ttft):.0f}% vs caching disabled
  ({on['median_ttft_ms']:.0f}ms vs {off['median_ttft_ms']:.0f}ms).
value: "median TTFT -{abs(ttft):.0f}% ({on['median_ttft_ms']:.0f}ms vs {off['median_ttft_ms']:.0f}ms)"
scope: Qwen2.5-3B-Instruct, {on.get('num_prompts','?')} shared-prefix prompts, single GPU; my own measurement
source:
  type: first-party-benchmark
  title: "agentic-inference/benchmarks — {engine} prefix-cache A/B"
  url: https://github.com/tyoon10/agentic-inference/tree/main/benchmarks
  date: {today}
verified: {today}
status: verified
attribution: "My own measurement on <GPU>; reproducible via benchmarks/run_bench.sh. Fill in the exact GPU."
used_in: []""")
    if not blocks:
        print("  no complete engine A/Bs — no registry entries emitted"); return
    path = out_dir / "benchmarks" / "registry-entries.yaml"
    path.write_text("\n---\n".join(blocks) + "\n")
    print(f"  registry entries -> benchmarks/registry-entries.yaml  "
          f"(paste each into cuda-ecosystem-radar/claims/<id>.yaml after filling the GPU)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT))
    args = ap.parse_args()
    out_dir = Path(args.out)

    print("Collecting results…")
    data = collect()
    if not data:
        print(f"No results found in {RESULTS}. Run benchmarks/run_bench.sh first.")
        sys.exit(1)
    render_figure(data, out_dir)
    write_results_md(data, out_dir)
    emit_registry(data, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
