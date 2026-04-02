"""
News Aggregator Trace Analysis — Bottleneck Visualization

Renders a horizontal bar chart showing per-turn breakdown of the news
aggregator agent run: model latency vs tool latency, with annotations
highlighting the turn-4 synthesis bottleneck.

Works with trace data from output/news-trace.json or with any trace
JSON in the same format.

Usage:
  python -m viz.trace_analysis                          # default trace
  python -m viz.trace_analysis --json output/news-trace.json
  python -m viz.trace_analysis --out figures/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from .theme import apply_theme, BG, CARD_BG, BORDER, TEXT, TEXT_DIM
from .theme import GREEN, BLUE, ORANGE, RED_SOFT, PURPLE


def analyze_trace(trace: list[dict]) -> list[dict]:
    """Aggregate trace events into per-turn stats."""
    turns: dict[int, dict] = {}
    for event in trace:
        t = event.get("turn", 0)
        if t not in turns:
            turns[t] = {
                "turn": t,
                "model_ms": 0,
                "tool_ms": 0,
                "tool_calls": 0,
                "output_tokens": 0,
                "tools": [],
            }
        if event["type"] == "model":
            turns[t]["model_ms"] = event.get("latency_ms", 0)
            turns[t]["output_tokens"] = event.get("tokens", 0)
        elif event["type"] == "tool_call":
            turns[t]["tool_calls"] += 1
            turns[t]["tools"].append(event["tool"])
        elif event["type"] == "tool_result":
            turns[t]["tool_ms"] += event.get("duration_ms", 0)
    return [turns[t] for t in sorted(turns.keys())]


def draw(ax, turn_data: list[dict]):
    """Render per-turn latency breakdown as a horizontal stacked bar."""
    n = len(turn_data)
    y_pos = np.arange(n)

    model_ms = [d["model_ms"] for d in turn_data]
    tool_ms = [d["tool_ms"] for d in turn_data]
    tool_calls = [d["tool_calls"] for d in turn_data]
    tokens = [d["output_tokens"] for d in turn_data]

    # Stacked horizontal bars
    bars_model = ax.barh(y_pos, model_ms, height=0.55, color=BLUE,
                         alpha=0.85, label="Model inference", zorder=3)
    bars_tool = ax.barh(y_pos, tool_ms, height=0.55, left=model_ms,
                        color=ORANGE, alpha=0.85, label="Tool execution", zorder=3)

    # Turn labels on y-axis
    labels = []
    for d in turn_data:
        tc = d["tool_calls"]
        unique_tools = list(set(d["tools"]))
        if unique_tools:
            tool_name = unique_tools[0] if len(unique_tools) == 1 else f"{len(unique_tools)} tools"
            labels.append(f"Turn {d['turn']}\n{tc} call{'s' if tc != 1 else ''} → {tool_name}")
        else:
            labels.append(f"Turn {d['turn']}\nfinal answer")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()

    # Annotate each bar with timing
    for i, (m, t, tc, tok) in enumerate(zip(model_ms, tool_ms, tool_calls, tokens)):
        total = m + t
        if total > 0:
            # Time label at end of bar
            ax.text(total + 200, i, f"{total / 1000:.1f}s",
                    fontsize=9, color=TEXT, va="center", weight="bold")
            # Token count inside model bar if significant
            if tok > 0 and m > 2000:
                ax.text(m / 2, i, f"{tok:,} tok",
                        fontsize=8, color=BG, va="center", ha="center",
                        weight="bold", zorder=4)

    # Find and annotate bottleneck (longest turn)
    max_idx = max(range(n), key=lambda i: model_ms[i] + tool_ms[i])
    max_total = model_ms[max_idx] + tool_ms[max_idx]
    total_time = sum(model_ms) + sum(tool_ms)
    pct = max_total / total_time * 100 if total_time > 0 else 0

    ax.annotate(
        f"BOTTLENECK\n{pct:.0f}% of total time\n{turn_data[max_idx]['output_tokens']:,} output tokens",
        xy=(max_total, max_idx),
        xytext=(max_total + 3000, max_idx - 0.8),
        fontsize=9, color=RED_SOFT, weight="bold",
        arrowprops=dict(arrowstyle="-|>", color=RED_SOFT, lw=1.5),
        bbox=dict(fc=CARD_BG, ec=RED_SOFT, boxstyle="round,pad=0.4", alpha=0.95),
        zorder=5,
    )

    # Stats box
    total_tool_calls = sum(tool_calls)
    total_tokens = sum(tokens)
    stats = (
        f"Total: {total_time / 1000:.1f}s  ·  {n} turns  ·  {total_tool_calls} tool calls\n"
        f"Model: {sum(model_ms) / 1000:.1f}s ({sum(model_ms) / total_time * 100:.0f}%)  ·  "
        f"Tools: {sum(tool_ms) / 1000:.1f}s ({sum(tool_ms) / total_time * 100:.0f}%)"
    )
    ax.text(0.98, 0.02, stats, transform=ax.transAxes,
            fontsize=9, color=TEXT_DIM, va="bottom", ha="right",
            bbox=dict(fc=CARD_BG, ec=BORDER, boxstyle="round,pad=0.4", alpha=0.9))

    # Styling
    ax.set_xlabel("Latency (ms)", fontsize=11, labelpad=10)
    ax.set_title("News Aggregator: Per-Turn Latency Breakdown",
                 fontsize=14, weight="bold", color=TEXT, pad=15)
    ax.legend(fontsize=10, loc="lower right", framealpha=0.9,
              facecolor=CARD_BG, edgecolor=BORDER, labelcolor=TEXT)
    ax.grid(True, axis="x", alpha=0.2)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: f"{v / 1000:.0f}s" if v >= 1000 else f"{v:.0f}ms"))


def render(out_dir: Path, trace: list[dict] | None = None):
    apply_theme()
    if trace is None:
        # Try to load the default news trace
        default_trace = Path(__file__).resolve().parent.parent / "output" / "news-trace.json"
        if default_trace.exists():
            with open(default_trace) as f:
                trace = json.load(f)
        else:
            print("No trace data found. Run the news aggregator demo first, or pass --json.")
            return

    turn_data = analyze_trace(trace)
    fig, ax = plt.subplots(figsize=(14, 6))
    draw(ax, turn_data)
    out = out_dir / "news-trace-analysis.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="output", help="Output directory")
    parser.add_argument("--json", default=None, help="Trace JSON file")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    trace = None
    if args.json:
        with open(args.json) as f:
            trace = json.load(f)

    render(out_dir, trace)
