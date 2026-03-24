"""
API vs Self-Hosted Cost Crossover for Agent Workloads

Shows where the economics flip from pay-per-token APIs to fixed-cost
self-hosted GPUs, annotated with real agent workflow data points.

Usage:
  python -m viz.cost                    # saves to output/
  python -m viz.cost --out path/to/dir  # custom output directory
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from .theme import apply_theme, BG, CARD_BG, BORDER, TEXT, TEXT_DIM
from .theme import GREEN, BLUE, ORANGE, RED_SOFT, PURPLE


def draw(ax):
    # Assumptions
    api_cost_per_call = 0.015   # $/call — blended mid-tier agent call
    gpu_daily_fixed = 83        # $/day  — A100 80GB lease ($2,500/mo)

    calls = np.arange(0, 8001, 50)
    api_monthly = calls * api_cost_per_call * 30
    gpu_monthly = np.full_like(calls, gpu_daily_fixed * 30, dtype=float)
    crossover = int(gpu_daily_fixed / api_cost_per_call)

    # Lines
    ax.plot(calls, api_monthly, color=BLUE, lw=2.5,
            label="Frontier API  (pay-per-token)", zorder=3)
    ax.plot(calls, gpu_monthly, color=ORANGE, lw=2.5, ls="--",
            label="Self-Hosted GPU  (fixed cost)", zorder=3)

    # Fills
    ax.fill_between(calls, api_monthly, gpu_monthly,
                     where=api_monthly < gpu_monthly, alpha=0.08,
                     color=BLUE, zorder=1)
    ax.fill_between(calls, api_monthly, gpu_monthly,
                     where=api_monthly >= gpu_monthly, alpha=0.08,
                     color=ORANGE, zorder=1)

    # Crossover marker
    if crossover <= 8000:
        cy = crossover * api_cost_per_call * 30
        ax.axvline(crossover, color=RED_SOFT, ls=":", lw=1.2, alpha=0.7, zorder=2)
        ax.plot(crossover, cy, "o", color=RED_SOFT, ms=8, zorder=4)
        ax.annotate(f"Crossover\n~{crossover:,} calls/day",
                    xy=(crossover, cy), xytext=(crossover - 1200, cy + 350),
                    fontsize=9, color=RED_SOFT, weight="bold",
                    arrowprops=dict(arrowstyle="-|>", color=RED_SOFT, lw=1.2),
                    zorder=5)

    # Annotations — real workloads
    for x, c, label_text in [
        (345,  GREEN,  "My job scanner\n345 calls/day"),
        (2000, PURPLE, "Small team\n~2,000 calls/day"),
    ]:
        y = x * api_cost_per_call * 30
        ax.plot(x, y, "D", color=c, ms=8, zorder=5)
        ax.annotate(label_text, xy=(x, y),
                    xytext=(x + 350, y + 140),
                    fontsize=8.5, color=c, weight="bold",
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1), zorder=5)

    # Enterprise zone
    ax.axvspan(5500, 8000, alpha=0.06, color=ORANGE, zorder=0)
    ax.text(6750, gpu_daily_fixed * 30 + 80, "Enterprise\nzone",
            fontsize=9, color=ORANGE, ha="center", va="bottom",
            weight="bold", alpha=0.8)

    # Styling
    ax.set_xlabel("Agent inference calls per day", fontsize=11, labelpad=10)
    ax.set_ylabel("Monthly cost ($)", fontsize=11, labelpad=10)
    ax.set_title("When Self-Hosted Beats API for Agent Workloads",
                 fontsize=15, weight="bold", color=TEXT, pad=15)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.9,
              facecolor=CARD_BG, edgecolor=BORDER, labelcolor=TEXT)
    ax.set_xlim(0, 8000)
    ax.set_ylim(0, max(api_monthly[-1], gpu_monthly[-1]) * 1.15)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))


def render(out_dir: Path):
    apply_theme()
    fig, ax = plt.subplots(figsize=(12, 6.5))
    draw(ax)
    fig.text(0.5, 0.01,
             "Assumptions: $0.015/call blended mid-tier agent call"
             "  ·  $2,500/mo A100 80 GB lease  ·  30-day month",
             ha="center", fontsize=7.5, color=TEXT_DIM)
    out = out_dir / "agent-cost-crossover.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="output", help="Output directory")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    render(out_dir)
