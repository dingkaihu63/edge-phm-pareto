"""Draw the configurable recurrent framework without version-specific results."""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "figures" / "fig_architecture"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def wrapped(text: str, width: int = 20) -> str:
    return "\n".join(textwrap.fill(line, width=width) for line in text.split("\n"))


fig, ax = plt.subplots(figsize=(7.2, 3.85))
ax.set_xlim(0, 14.4)
ax.set_ylim(0, 7.6)
ax.axis("off")


def box(
    x,
    y,
    width,
    height,
    text,
    face,
    edge,
    bold=False,
    fontsize=7,
    wrap_width=20,
):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.04,rounding_size=0.09",
        linewidth=1.0,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        wrapped(text, width=wrap_width),
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold" if bold else "normal",
        color="#20262E",
    )


def arrow(x1, y1, x2, y2, style="-"):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops={"arrowstyle": "-|>", "color": "#46515C", "lw": 0.95, "linestyle": style},
    )


def line(x1, y1, x2, y2, style="-"):
    ax.plot([x1, x2], [y1, y2], color="#46515C", lw=0.95, ls=style)


box(0.25, 5.65, 2.05, 1.05, "Industrial sensor\nsequence", "#E9F4EF", "#3D7C62", bold=True)
box(2.65, 5.65, 2.15, 1.05, "Unit-aware preprocessing\nand sliding windows", "#E9F4EF", "#3D7C62")
box(5.15, 5.65, 2.05, 1.05, "Two-layer LSTM\nsequence encoder", "#F7F1DE", "#9B7A22", bold=True)
box(7.65, 6.15, 2.25, 0.95, "Normalized Sigmoid\ntemporal attention", "#F8EAE6", "#B85C4A")
box(
    7.65,
    4.95,
    2.25,
    0.95,
    "Terminal hidden state\n(no attention)",
    "#EEF1F4",
    "#68737D",
    wrap_width=28,
)
box(10.35, 5.65, 1.75, 1.05, "Dense head and\noptional dropout", "#EEEAF6", "#735C93")
box(12.5, 5.65, 1.65, 1.05, "Fault-state\nprobability", "#F8EEDF", "#A66A28", bold=True)

box(
    3.0,
    2.55,
    3.0,
    1.05,
    "Validation-only mode and\noperating-threshold selection",
    "#E8F0F6",
    "#477A9B",
    bold=True,
    wrap_width=32,
)
box(6.45, 2.55, 2.25, 1.05, "Optional MC inference\nmean and dispersion", "#E8F0F6", "#477A9B")
box(9.15, 2.55, 2.25, 1.05, "Validation-selected\nfault-state decision", "#F8EEDF", "#A66A28")
box(11.85, 2.55, 2.25, 1.05, "Optional temporal\nattribution", "#F8EAE6", "#B85C4A")

box(
    1.2,
    0.25,
    12.0,
    1.05,
    "Task-aligned evidence: gradual degradation and cross-bearing stress testing\nPhysical-unit inference, auxiliary event recognition, and deployment reference",
    "#F4F5F6",
    "#8A9299",
    fontsize=6.4,
    wrap_width=95,
)

arrow(2.30, 6.18, 2.65, 6.18)
arrow(4.80, 6.18, 5.15, 6.18)
arrow(7.20, 6.18, 7.65, 6.63)
arrow(7.20, 6.18, 7.65, 5.43)
arrow(9.90, 6.63, 10.35, 6.18)
arrow(9.90, 5.43, 10.35, 6.18)
arrow(12.10, 6.18, 12.50, 6.18)

line(4.50, 5.65, 4.50, 3.85)
arrow(4.50, 3.85, 4.50, 3.60)
line(13.32, 5.65, 13.32, 4.15)
line(13.32, 4.15, 7.58, 4.15)
arrow(7.58, 4.15, 7.58, 3.60)
line(13.32, 4.15, 10.28, 4.15)
arrow(10.28, 4.15, 10.28, 3.60)
line(9.90, 6.63, 10.08, 6.63, style="--")
line(10.08, 6.63, 10.08, 3.90, style="--")
line(10.08, 3.90, 12.98, 3.90, style="--")
arrow(12.98, 3.90, 12.98, 3.60, style="--")
arrow(6.00, 3.08, 6.45, 3.08)
arrow(8.70, 3.08, 9.15, 3.08)

ax.text(7.72, 7.28, "optional", fontsize=6, color="#B85C4A")
ax.text(7.72, 4.72, "fallback", fontsize=6, color="#68737D")
ax.text(11.35, 4.02, "attention enabled", fontsize=5.8, color="#B85C4A")

fig.tight_layout(pad=0.2)
fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(OUTPUT.with_suffix(".svg"), bbox_inches="tight")
fig.savefig(OUTPUT.with_suffix(".png"), dpi=600, bbox_inches="tight")
fig.savefig(OUTPUT.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
plt.close(fig)
print("architecture redrawn")
