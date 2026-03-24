"""Shared dark theme for all visualizations."""

import matplotlib.pyplot as plt

# Palette — dark theme, NVIDIA green accent
BG       = "#0d1117"
CARD_BG  = "#161b22"
BORDER   = "#30363d"
TEXT     = "#e6edf3"
TEXT_DIM = "#8b949e"
GREEN    = "#76b900"   # NVIDIA green
BLUE     = "#58a6ff"   # frontier accent
ORANGE   = "#d29922"   # self-hosted accent
RED_SOFT = "#f85149"
PURPLE   = "#bc8cff"


def apply_theme():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Segoe UI", "Helvetica Neue", "Arial"],
        "text.color": TEXT,
        "axes.facecolor": BG,
        "figure.facecolor": BG,
        "axes.edgecolor": BORDER,
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT_DIM,
        "ytick.color": TEXT_DIM,
        "grid.color": BORDER,
        "grid.alpha": 0.4,
    })
