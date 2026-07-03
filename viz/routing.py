"""
Hybrid Router — Routing Decision Visualization

Scatter plot showing tasks by complexity score, colored by which backend
handled them (fast vs frontier tier — both NVIDIA API), with the threshold line.

Works with routing data from projects/02-hybrid-router/router.py or
with raw data dicts for demo/testing.

Usage:
  python -m viz.routing                     # renders demo data
  python -m viz.routing --out figures/      # custom output
  python -m viz.routing --json routing.json # render from saved decisions
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from .theme import apply_theme, BG, CARD_BG, BORDER, TEXT, TEXT_DIM
from .theme import GREEN, BLUE, ORANGE, RED_SOFT, PURPLE


# Demo routing data (matches RouterStats.to_viz_data() output)
DEMO_DATA = [
    {"task": "Classify this email as spam or not", "complexity": 0.15,
     "backend": "fast", "latency_ms": 180, "tokens": 95},
    {"task": "Extract the dollar amount from this invoice", "complexity": 0.20,
     "backend": "fast", "latency_ms": 210, "tokens": 82},
    {"task": "What day of the week is March 15, 2026?", "complexity": 0.10,
     "backend": "fast", "latency_ms": 150, "tokens": 45},
    {"task": "Summarize this 3-page contract", "complexity": 0.45,
     "backend": "fast", "latency_ms": 890, "tokens": 520},
    {"task": "Route this support ticket to the right team", "complexity": 0.25,
     "backend": "fast", "latency_ms": 195, "tokens": 78},
    {"task": "Parse JSON and validate schema", "complexity": 0.18,
     "backend": "fast", "latency_ms": 160, "tokens": 65},
    {"task": "Translate this paragraph to French", "complexity": 0.35,
     "backend": "fast", "latency_ms": 420, "tokens": 240},
    {"task": "Rewrite this paragraph for a different audience", "complexity": 0.50,
     "backend": "fast", "latency_ms": 680, "tokens": 380},
    {"task": "Analyze competitive positioning across 4 companies", "complexity": 0.75,
     "backend": "frontier", "latency_ms": 2800, "tokens": 1850},
    {"task": "Write a persuasive appeal for a denied insurance claim", "complexity": 0.82,
     "backend": "frontier", "latency_ms": 3200, "tokens": 2100},
    {"task": "Debug why this async pipeline deadlocks under load", "complexity": 0.88,
     "backend": "frontier", "latency_ms": 4100, "tokens": 2800},
    {"task": "Design a migration strategy from monolith to microservices", "complexity": 0.92,
     "backend": "frontier", "latency_ms": 5200, "tokens": 3500},
    {"task": "Draft a board memo on AI risk with regulatory citations", "complexity": 0.85,
     "backend": "frontier", "latency_ms": 3800, "tokens": 2400},
    {"task": "Evaluate trade-offs between 3 database architectures", "complexity": 0.70,
     "backend": "frontier", "latency_ms": 2500, "tokens": 1600},
    {"task": "Generate unit tests for this function", "complexity": 0.40,
     "backend": "fast", "latency_ms": 550, "tokens": 310},
    {"task": "Normalize these addresses to USPS format", "complexity": 0.22,
     "backend": "fast", "latency_ms": 175, "tokens": 90},
]

THRESHOLD = 0.6


def draw(ax, data: list[dict], threshold: float = THRESHOLD):
    """Render routing scatter plot."""

    fast = [d for d in data if d["backend"] == "fast"]
    frontier = [d for d in data if d["backend"] == "frontier"]

    # Scatter
    if fast:
        ax.scatter(
            [d["complexity"] for d in fast],
            [d["latency_ms"] for d in fast],
            c=ORANGE, s=90, alpha=0.85, edgecolors="white", linewidth=0.5,
            label=f"Fast tier — Small 4 via API  ({len(fast)} tasks)", zorder=3,
        )
    if frontier:
        ax.scatter(
            [d["complexity"] for d in frontier],
            [d["latency_ms"] for d in frontier],
            c=BLUE, s=90, alpha=0.85, edgecolors="white", linewidth=0.5,
            label=f"Frontier tier — Large 3 via API  ({len(frontier)} tasks)", zorder=3,
        )

    # Threshold line
    ax.axvline(threshold, color=RED_SOFT, linestyle="--", linewidth=1.5,
               alpha=0.8, zorder=2)
    ax.text(threshold + 0.02, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 4500,
            f"threshold = {threshold}",
            fontsize=9, color=RED_SOFT, weight="bold", va="top")

    # Zone labels (bottom of chart)
    ymax = max(d["latency_ms"] for d in data) * 1.15 if data else 5000
    ax.text(threshold / 2, ymax * 0.03,
            "FAST TIER\n(Mistral Small 4 119B)",
            fontsize=9, color=ORANGE, ha="center", weight="bold", alpha=0.6)
    ax.text((1 + threshold) / 2, ymax * 0.03,
            "FRONTIER TIER\n(Mistral Large 3 675B)",
            fontsize=9, color=BLUE, ha="center", weight="bold", alpha=0.6)

    # Fill zones
    ax.axvspan(0, threshold, alpha=0.04, color=ORANGE, zorder=0)
    ax.axvspan(threshold, 1, alpha=0.04, color=BLUE, zorder=0)

    # Task labels (annotate a few interesting ones)
    for d in data:
        if d["complexity"] in (0.10, 0.92, 0.50):  # extremes + near boundary
            short = d["task"][:35] + "..." if len(d["task"]) > 35 else d["task"]
            color = ORANGE if d["backend"] == "fast" else BLUE
            offset = (-150, -22) if d["complexity"] > 0.8 else (15, 10)
            ax.annotate(
                short,
                xy=(d["complexity"], d["latency_ms"]),
                xytext=offset, textcoords="offset points",
                fontsize=7, color=TEXT_DIM, alpha=0.8,
                arrowprops=dict(arrowstyle="-", color=TEXT_DIM, lw=0.5, alpha=0.4),
            )

    # Styling
    ax.set_xlabel("Complexity Score  (classified by Mistral Small 4)", fontsize=11,
                  labelpad=10)
    ax.set_ylabel("Response Latency (ms)", fontsize=11, labelpad=10)
    ax.set_title("Hybrid Routing: Small 4 vs Large 3 by Task Complexity",
                 fontsize=14, weight="bold", color=TEXT, pad=15)
    ax.set_xlim(-0.02, 1.02)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.9,
              facecolor=CARD_BG, edgecolor=BORDER, labelcolor=TEXT)
    ax.grid(True, alpha=0.2)

    # Stats box
    fast_pct = len(fast) / len(data) * 100 if data else 0
    avg_fast_ms = np.mean([d["latency_ms"] for d in fast]) if fast else 0
    avg_frontier_ms = np.mean([d["latency_ms"] for d in frontier]) if frontier else 0
    stats_text = (
        f"Fast tier: {fast_pct:.0f}% of tasks  ·  avg {avg_fast_ms:.0f}ms\n"
        f"Frontier: {100-fast_pct:.0f}% of tasks  ·  avg {avg_frontier_ms:.0f}ms"
    )
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            fontsize=8.5, color=TEXT_DIM, va="top", ha="right",
            bbox=dict(fc=CARD_BG, ec=BORDER, boxstyle="round,pad=0.4", alpha=0.9))


def render(out_dir: Path, data: list[dict] | None = None,
           threshold: float = THRESHOLD):
    apply_theme()
    is_demo = data is None
    data = data or DEMO_DATA
    fig, ax = plt.subplots(figsize=(12, 7))
    draw(ax, data, threshold)
    if is_demo:
        fig.text(0.5, 0.01,
                 "Illustrative demo data — synthetic tasks and latencies showing the routing pattern."
                 "  Both tiers are NVIDIA API calls; regenerate from real runs with --json.",
                 ha="center", fontsize=7.5, color=TEXT_DIM)
    out = out_dir / "hybrid-routing-decisions.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="output", help="Output directory")
    parser.add_argument("--json", default=None, help="Routing data JSON file")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = None
    if args.json:
        with open(args.json) as f:
            data = json.load(f)

    render(out_dir, data, args.threshold)
