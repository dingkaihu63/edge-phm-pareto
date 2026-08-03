"""Clean architecture diagram with auto-wrapped labels."""

import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def wrap_text(text, width=18):
    return "\n".join(textwrap.fill(part, width=width) for part in text.split("\n"))


fig, ax = plt.subplots(figsize=(14, 7.4))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7.8)
ax.axis("off")


def box(x, y, w, h, text, fc="#eef3fb", ec="#1f4e79", fs=11, bold=False):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.14",
                       linewidth=1.6, edgecolor=ec, facecolor=fc)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, wrap_text(text), ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal", color="#17202a")


def line(x1, y1, x2, y2, color="#34495e", lw=1.8):
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, zorder=2)


def arrow(x1, y1, x2, y2, color="#34495e", lw=1.8):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw), zorder=3)


# Main pipeline row
box(0.15, 5.9, 2.15, 1.30, "Industrial sensor streams\n1 Hz telemetry", fc="#eafaf1", ec="#1e8449", fs=11, bold=True)
box(2.65, 5.9, 2.15, 1.30, "Preprocessing\nimpute, diff, windows", fc="#eafaf1", ec="#1e8449", fs=11)
box(5.15, 5.9, 1.85, 1.30, "LSTM 1\n64 units", fc="#fef9e7", ec="#b7950b", fs=11)
box(7.35, 5.9, 1.85, 1.30, "LSTM 2\n32 units", fc="#fef9e7", ec="#b7950b", fs=11)
box(9.55, 5.9, 2.15, 1.30, "Sigmoid attention\ntau = 0.5, normalized", fc="#fdedec", ec="#cb4335", fs=11, bold=True)
box(12.05, 5.9, 1.80, 1.30, "Context + MC-Dropout", fc="#f4ecf7", ec="#7d3c98", fs=11)

# Second row
box(2.20, 3.45, 2.15, 1.30, "Dense 16\nReLU + dropout", fc="#f4ecf7", ec="#7d3c98", fs=11)
box(4.70, 3.45, 2.15, 1.30, "Fault probability\nSigmoid output", fc="#fdebd0", ec="#ca6f1e", fs=11, bold=True)
box(7.25, 3.45, 2.75, 1.30, "MC uncertainty\nmean +/- std, K = 50", fc="#ebf5fb", ec="#2874a6", fs=11)
box(10.35, 3.45, 3.20, 1.30, "Attention + SHAP\nwhen and why", fc="#fdf2e9", ec="#d35400", fs=11)

# Bottom row
box(0.30, 0.85, 5.60, 1.55, "Edge profile\n56k params, 0.022 ms CPU,\n1.1 ms MC50", fc="#e8f8f5", ec="#148f77", fs=11, bold=True)
box(6.30, 0.85, 7.20, 1.55, "Evaluation\n5 seeds, AUROC / AUPRC / F2 / Brier / ECE\nUR3, C-MAPSS FD001/FD003, XJTU-SY", fc="#fef5e7", ec="#af601a", fs=11)

# Pipeline arrows
arrow(2.30, 6.55, 2.65, 6.55)
arrow(4.80, 6.55, 5.15, 6.55)
arrow(7.00, 6.55, 7.35, 6.55)
arrow(9.20, 6.55, 9.55, 6.55)
arrow(11.70, 6.55, 12.05, 6.55)

# Context down to Dense (elbow)
line(12.95, 5.90, 12.95, 4.90)
line(12.95, 4.90, 3.25, 4.90)
arrow(3.25, 4.90, 3.25, 4.75)

# Dense to Output
arrow(4.35, 4.10, 4.70, 4.10)

# Output to uncertainty
arrow(6.85, 4.10, 7.25, 4.10)

# Attention to explainability
line(10.65, 5.90, 10.65, 5.15)
line(10.65, 5.15, 11.95, 5.15)
arrow(11.95, 5.15, 11.95, 4.75)

# Output and uncertainty down to evaluation
line(5.75, 3.45, 5.75, 2.45)
line(5.75, 2.45, 9.90, 2.45)
arrow(9.90, 2.45, 9.90, 2.40)
line(8.65, 3.45, 8.65, 2.75)
arrow(8.65, 2.75, 8.65, 2.40)

# Evaluation to edge profile
arrow(6.30, 1.60, 5.90, 1.60)

fig.tight_layout()
fig.savefig("../figures/fig_architecture.pdf", bbox_inches="tight")
fig.savefig("../figures/fig_architecture.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("architecture redrawn")
