"""
Three-Layer Agentic AI Inference Stack — Architecture Diagram

Visualizes the post-GTC 2026 NVIDIA AI stack:
  Governance (OpenShell) → Serving (Dynamo, NIM, engines) → Models (hybrid)

Usage:
  python -m viz.stack                    # saves to output/
  python -m viz.stack --out path/to/dir  # custom output directory
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from .theme import apply_theme, BG, CARD_BG, BORDER, TEXT, TEXT_DIM
from .theme import GREEN, BLUE, ORANGE


def draw(ax):
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")

    def card(x, y, w, h, fc=CARD_BG, ec=BORDER, lw=1.2, zorder=2):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.15",
            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder,
        )
        ax.add_patch(box)

    def label(x, y, text, size=11, color=TEXT, weight="normal", ha="center",
              va="center", zorder=5, **kw):
        ax.text(x, y, text, fontsize=size, color=color, fontweight=weight,
                ha=ha, va=va, zorder=zorder, **kw)

    # Title
    label(7, 8.55, "Agentic AI Inference Stack", size=18, weight="bold",
          color=GREEN)
    label(7, 8.15, "Three layers every agent developer should understand",
          size=10, color=TEXT_DIM)

    # ── Layer 3: Governance ──
    card(0.5, 6.2, 13, 1.6, ec=GREEN, lw=1.8)
    label(1.3, 7.5, "GOVERNANCE", size=9, color=GREEN, weight="bold", ha="left")
    label(1.3, 7.15, "Who controls what", size=8, color=TEXT_DIM, ha="left")

    card(1.2, 6.45, 3.2, 0.55, fc="#1a2310", ec=GREEN, lw=0.8)
    label(2.8, 6.72, "OpenShell Sandbox", size=9, weight="bold")

    card(4.8, 6.45, 3.2, 0.55, fc="#1a2310", ec=GREEN, lw=0.8)
    label(6.4, 6.72, "Policy Engine", size=9, weight="bold")

    card(8.4, 6.45, 4.8, 0.55, fc="#1a2310", ec=GREEN, lw=0.8)
    label(10.8, 6.72, "Privacy Router", size=9, weight="bold")
    label(10.8, 6.5, "routes by data sensitivity policy", size=7, color=TEXT_DIM)

    # ── Layer 2: Serving ──
    card(0.5, 3.4, 13, 2.5, ec=BLUE, lw=1.8)
    label(1.3, 5.6, "SERVING", size=9, color=BLUE, weight="bold", ha="left")
    label(1.3, 5.25, "How it runs on hardware", size=8, color=TEXT_DIM, ha="left")

    card(1.2, 4.65, 11.8, 0.45, fc="#11192a", ec=BLUE, lw=0.8)
    label(7.1, 4.87,
          "Dynamo 1.0  —  multi-node orchestration  ·  KV-aware routing"
          "  ·  disaggregated serving  ·  SLA autoscaling",
          size=8, color=BLUE, weight="bold")

    engines = [
        ("NIM", "one-command\ncontainers"),
        ("TensorRT-LLM", "compiler\noptimization"),
        ("vLLM", "community\ninference"),
        ("Triton", "multi-model\nserving"),
        ("NIXL", "KV cache transfer\n(zero-copy RDMA)"),
    ]
    ew, gap = 2.2, 0.2
    total_w = len(engines) * ew + (len(engines) - 1) * gap
    sx = (14 - total_w) / 2
    for i, (name, desc) in enumerate(engines):
        x = sx + i * (ew + gap)
        ec_color = ORANGE if name == "NIXL" else BORDER
        card(x, 3.65, ew, 0.8, fc="#161b22", ec=ec_color, lw=0.8)
        label(x + ew / 2, 4.22, name, size=9.5, weight="bold")
        label(x + ew / 2, 3.9, desc, size=7, color=TEXT_DIM)

    # ── Layer 1: Models ──
    card(0.5, 0.6, 13, 2.5, ec=ORANGE, lw=1.8)
    label(1.3, 2.8, "MODELS", size=9, color=ORANGE, weight="bold", ha="left")
    label(1.3, 2.45, "What reasons", size=8, color=TEXT_DIM, ha="left")

    # Self-hosted
    card(1.2, 0.85, 5.5, 1.45, fc="#1f1a0f", ec=ORANGE, lw=0.8)
    label(3.95, 2.05, "Self-Hosted  (open weights)", size=9, weight="bold",
          color=ORANGE)
    for i, m in enumerate(["Mistral Small 4", "Nemotron", "Llama 4", "Devstral"]):
        label(2.2 + (i % 2) * 3.5, 1.6 - (i // 2) * 0.4, f"· {m}",
              size=8.5, ha="left")

    # Frontier
    card(7.3, 0.85, 5.5, 1.45, fc="#111827", ec=BLUE, lw=0.8)
    label(10.05, 2.05, "Frontier Tier  (NVIDIA API)", size=9, weight="bold",
          color=BLUE)
    for i, m in enumerate(["Mistral Large 3 (675B MoE)",
                           "DeepSeek R1", "Llama 4 Maverick"]):
        label(8.1, 1.6 - i * 0.35, f"· {m}", size=8.5, ha="left")

    # ── Routing arrows ──
    ax.annotate("", xy=(3.95, 2.35), xytext=(9.5, 6.45),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.4,
                                connectionstyle="arc3,rad=0.15"), zorder=4)
    label(3.5, 4.55, "high-volume\nclassification\nextraction\nrouting",
          size=7, color=ORANGE, ha="center",
          bbox=dict(fc=BG, ec=ORANGE, alpha=0.85, boxstyle="round,pad=0.3"))

    ax.annotate("", xy=(10.05, 2.35), xytext=(11.5, 6.45),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.4,
                                connectionstyle="arc3,rad=-0.15"), zorder=4)
    label(12.0, 4.55, "complex\nreasoning\nhigh-stakes\ndecisions",
          size=7, color=BLUE, ha="center",
          bbox=dict(fc=BG, ec=BLUE, alpha=0.85, boxstyle="round,pad=0.3"))

    label(7, 0.2, "Source: GTC 2026 announcements", size=7, color=TEXT_DIM)


def render(out_dir: Path):
    apply_theme()
    fig, ax = plt.subplots(figsize=(14, 9))
    draw(ax)
    out = out_dir / "agentic-stack-architecture.png"
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