"""
Agent Loop Trace — Visualization

Renders the step-by-step execution of an agent loop as a vertical
flow diagram: user prompt → model reasoning → tool calls → results → answer.

Works with trace data from projects/01-tool-calling/agent.py or
with raw trace dicts for demo/testing.

Usage:
  python -m viz.trace                     # renders demo trace
  python -m viz.trace --out figures/      # custom output
  python -m viz.trace --json trace.json   # render from saved trace
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from .theme import apply_theme, BG, CARD_BG, BORDER, TEXT, TEXT_DIM
from .theme import GREEN, BLUE, ORANGE, PURPLE


# Demo trace (matches the structure from agent.py AgentResult.trace)
DEMO_TRACE = [
    {"type": "model", "turn": 1, "content": "I'll check the current directory.",
     "tokens": 18, "latency_ms": 420},
    {"type": "tool_call", "turn": 1, "tool": "list_directory",
     "args": {"path": "."}},
    {"type": "tool_result", "turn": 1, "tool": "list_directory",
     "result": '{"path": ".", "entries": [{"name": "README.md"}, {"name": "viz/"}, {"name": "projects/"}]}',
     "duration_ms": 2},
    {"type": "model", "turn": 2, "content": "Let me read the README.",
     "tokens": 12, "latency_ms": 380},
    {"type": "tool_call", "turn": 2, "tool": "read_file",
     "args": {"path": "README.md"}},
    {"type": "tool_result", "turn": 2, "tool": "read_file",
     "result": '{"path": "README.md", "content": "# agentic-inference\\n..."}',
     "duration_ms": 1},
    {"type": "model", "turn": 3, "content": "Let me also count the words.",
     "tokens": 15, "latency_ms": 350},
    {"type": "tool_call", "turn": 3, "tool": "word_count",
     "args": {"text": "# agentic-inference..."}},
    {"type": "tool_result", "turn": 3, "tool": "word_count",
     "result": '{"words": 342, "sentences": 28, "characters": 2105}',
     "duration_ms": 0.1},
    {"type": "model", "turn": 4,
     "content": "The repo has 3 top-level items. The README is 342 words covering the viz module and 5 project stubs.",
     "tokens": 35, "latency_ms": 510},
]


def _wrap(text: str, max_chars: int = 55) -> str:
    """Word-wrap text to max_chars per line."""
    words = text.split()
    lines, line = [], ""
    for w in words:
        if line and len(line) + len(w) + 1 > max_chars:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        lines.append(line)
    return "\n".join(lines)


def draw(ax, trace: list[dict]):
    """Render trace as a vertical flow on the given axes."""
    ax.axis("off")

    # Layout constants
    col_x = 4.0        # center x of the flow
    card_w = 6.0
    row_h = 1.2        # vertical spacing
    y_top = len(trace) * row_h + 1.5

    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, y_top + 1.0)

    # Title
    ax.text(col_x + card_w / 2, y_top + 0.5,
            "Agent Loop Trace", fontsize=16, weight="bold",
            color=GREEN, ha="center", va="center")

    # Turn labels on the left
    turns_seen = set()

    for i, event in enumerate(trace):
        y = y_top - (i + 1) * row_h
        turn = event.get("turn", 0)

        # Turn marker (left gutter)
        if turn not in turns_seen:
            turns_seen.add(turn)
            ax.text(0.8, y + 0.15, f"Turn {turn}", fontsize=8, color=TEXT_DIM,
                    ha="center", va="center", weight="bold")

        etype = event["type"]

        if etype == "model":
            # Blue card — model reasoning
            color = BLUE
            content = _wrap(event["content"], 50)
            stats = f'{event["tokens"]} tok · {event["latency_ms"]:.0f}ms'
            icon = "LLM"
        elif etype == "tool_call":
            # Orange card — tool invocation
            color = ORANGE
            args_str = ", ".join(f"{k}={v!r}" for k, v in event["args"].items())
            if len(args_str) > 45:
                args_str = args_str[:42] + "..."
            content = f'{event["tool"]}({args_str})'
            stats = ""
            icon = "CALL"
        elif etype == "tool_result":
            # Green card — tool result
            color = GREEN
            result = event["result"]
            if len(result) > 60:
                result = result[:57] + "..."
            content = result
            stats = f'{event["duration_ms"]:.1f}ms'
            icon = "RES"
        else:
            continue

        # Card background
        card_h = 0.7
        box = FancyBboxPatch(
            (col_x - card_w / 2, y - card_h / 2), card_w, card_h,
            boxstyle="round,pad=0.1",
            facecolor=CARD_BG, edgecolor=color, linewidth=1.2, zorder=2,
        )
        ax.add_patch(box)

        # Icon badge (left gutter, outside card)
        badge = FancyBboxPatch(
            (col_x - card_w / 2 - 1.1, y - 0.12), 0.9, 0.25,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor=color, linewidth=0, zorder=3, alpha=0.9,
        )
        ax.add_patch(badge)
        ax.text(col_x - card_w / 2 - 0.65, y + 0.005,
                icon, fontsize=6.5, color=BG, weight="bold",
                ha="center", va="center", zorder=4)

        # Content text
        ax.text(col_x - card_w / 2 + 0.2, y,
                _wrap(content, 50), fontsize=7.5, color=TEXT,
                ha="left", va="center", zorder=3,
                fontfamily="monospace" if etype != "model" else "sans-serif")

        # Stats (right side)
        if stats:
            ax.text(col_x + card_w / 2 - 0.15, y - card_h / 2 + 0.12,
                    stats, fontsize=6.5, color=TEXT_DIM,
                    ha="right", va="center", zorder=3)

        # Arrow to next
        if i < len(trace) - 1:
            ax.annotate(
                "", xy=(col_x, y - card_h / 2 - 0.05),
                xytext=(col_x, y - card_h / 2 - row_h + card_h / 2 + 0.05),
                arrowprops=dict(arrowstyle="<|-", color=BORDER, lw=1.0),
                zorder=1,
            )

    # Legend
    legend_y = -0.2
    for lx, lc, lt in [(2.5, BLUE, "Model"), (5.0, ORANGE, "Tool Call"),
                        (7.5, GREEN, "Tool Result")]:
        ax.plot(lx - 0.3, legend_y, "s", color=lc, ms=8)
        ax.text(lx, legend_y, lt, fontsize=8, color=TEXT_DIM, va="center")


def render(out_dir: Path, trace: list[dict] | None = None):
    apply_theme()
    trace = trace or DEMO_TRACE
    h = max(8, len(trace) * 1.3 + 2.5)
    fig, ax = plt.subplots(figsize=(10, h))
    draw(ax, trace)
    out = out_dir / "agent-loop-trace.png"
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
