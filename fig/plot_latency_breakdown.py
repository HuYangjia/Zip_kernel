#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Latency breakdown figure: Naive vs Opt (W4A4 sparse-augmented GEMM).

Vertical stacked bars, Nunchaku-style reference layout:
  - full box frame, horizontal gridlines only
  - white bold in-bar numbers (moved outside when segment is too small)
  - dashed red drop-line from naive top to opt top, centred arrow + NNNx label
  - serif legend on the right, italic (a)-prefixed caption below

Palette matches docs/figures/pipeline_spacetime/naive_pipeline.tex.

Usage:
    python3 plot_latency_breakdown.py
Outputs latency_breakdown.pdf (vector, for LaTeX) and latency_breakdown.png.
"""

from __future__ import annotations

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

# ----------------------------------------------------------------------------
# Palette (verbatim from naive_pipeline.tex)
# ----------------------------------------------------------------------------
qC   = "#7F7F7F"      # quant
tcC  = "#D62728"      # dense GEMM  (INT4 Tensor Core)
spC  = "#9467BD"      # sparse branch
redC = "#17A2B8"      # reduce-add  (teal)
fpC  = "#2CA02C"      # fused (dense+sparse)
rtC  = "#C44E52"      # speed-up annotation
axisC = "#1F2933"

# ----------------------------------------------------------------------------
# Raw data  (microseconds)  -- from fig/data.md
#
# Sub-component values are rescaled (preserving ratios) so that the stacked
# totals match the measured end-to-end wall-clock of a single projection
# (Qwen3-8B O-proj, decode bs=4): NAIVE_e2e = 66.34 us, OPT_e2e = 35.00 us.
# Original micro-bench raw values were
#     NAIVE_raw = (19.96, 41.75, 23.12, 8.18)  sum = 93.01
#     OPT_raw   = (22.79, 21.73)                sum = 44.52
# scaling factors: NAIVE x 66.34/93.01 ~ 0.71325,  OPT x 35.00/44.52 ~ 0.78616.
# ----------------------------------------------------------------------------
NAIVE = [
    ("Quant",         14.24, qC),
    ("Dense GEMM",    29.78, tcC),
    ("Sparse GEMM",   16.49, spC),
    ("Reduce Add",     5.83, redC),
]
OPT = [
    ("Quant",                17.92, qC),
    ("Fused (dense+sparse)", 17.08, fpC),
]
NAIVE_TOTAL = sum(v for _, v, _ in NAIVE)   # 66.34
OPT_TOTAL   = sum(v for _, v, _ in OPT)     # 35.00
SPEEDUP     = NAIVE_TOTAL / OPT_TOTAL       # ~1.90x

# ----------------------------------------------------------------------------
# Matplotlib global style  (paper-grade serif)
# ----------------------------------------------------------------------------
rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Palatino", "Palatino Linotype", "Times New Roman",
                           "DejaVu Serif"],
    "mathtext.fontset":   "cm",
    "axes.edgecolor":     axisC,
    "axes.labelcolor":    axisC,
    "xtick.color":        axisC,
    "ytick.color":        axisC,
    "axes.linewidth":     0.9,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})

# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.2, 3.0))

BAR_W   = 0.52                                # narrow bars, wide gap
X_NAIVE = 0.0
X_OPT   = 1.0
# threshold (fraction of naive total) under which an in-bar number is
# promoted to an outside label to keep readability.
OUTSIDE_FRAC = 0.12


def _draw_stack(x: float, segments: list[tuple[str, float, str]]) -> float:
    """Stack `segments` vertically at column `x`; return running height."""
    y = 0.0
    for _, val, color in segments:
        ax.bar(x, val, bottom=y, width=BAR_W,
               color=color, edgecolor="white", linewidth=1.1, zorder=3)
        # ---- numeric annotation ------------------------------------------
        # small segments: place to the right of the bar in black;
        # large segments: place at segment centre in bold white.
        if val / NAIVE_TOTAL < OUTSIDE_FRAC:
            ax.text(x + BAR_W / 2 + 0.04, y + val / 2,
                    f"{val:.1f}", ha="left", va="center",
                    fontsize=8.2, color=axisC, zorder=5)
        else:
            ax.text(x, y + val / 2, f"{val:.1f}",
                    ha="center", va="center",
                    fontsize=9.0, color="white", fontweight="bold",
                    zorder=5)
        y += val
    return y


naive_top = _draw_stack(X_NAIVE, NAIVE)
opt_top   = _draw_stack(X_OPT,   OPT)

# total on top of each column (black, like Nunchaku's "17" on the very top)
ax.text(X_NAIVE, naive_top + 1.3, f"{NAIVE_TOTAL:.2f}",
        ha="center", va="bottom", fontsize=9.0, color=axisC,
        fontweight="bold")
ax.text(X_OPT,   opt_top   + 1.3, f"{OPT_TOTAL:.2f}",
        ha="center", va="bottom", fontsize=9.0, color=axisC,
        fontweight="bold")

# ----------------------------------------------------------------------------
# Speed-up annotation : dashed red line from naive-top right edge to opt-top,
# then a small centred arrow pointing down + "N.NNx" label.
# ----------------------------------------------------------------------------
drop_line_y = naive_top                                  # same height as naive
line_x0     = X_NAIVE + BAR_W / 2                        # right edge of naive
line_x1     = X_OPT                                      # centre of opt column
ax.plot([line_x0, line_x1], [drop_line_y, drop_line_y],
        color=rtC, lw=1.15, linestyle=(0, (5, 2.5)), zorder=4)

# centred drop arrow on top of opt bar
ax.annotate(
    "", xy=(X_OPT, opt_top + 1.0),
    xytext=(X_OPT, drop_line_y - 1.0),
    arrowprops=dict(arrowstyle="-|>", color=rtC, lw=1.25,
                    mutation_scale=12),
    zorder=5,
)
ax.text(X_OPT + 0.02, (drop_line_y + opt_top) / 2,
        f"${SPEEDUP:.2f}\\times$",
        ha="left", va="center", fontsize=10.5, color=rtC,
        fontweight="bold", zorder=6)

# ----------------------------------------------------------------------------
# Axes cosmetics
# ----------------------------------------------------------------------------
ax.set_xticks([X_NAIVE, X_OPT])
ax.set_xticklabels(["Na\u00efve", "Ours"], fontsize=10.5)
ax.set_ylabel(r"Latency ($\mu$s)", fontsize=10.5)

# --- tightened Y range: just cover the top annotation ------------------------
y_top = naive_top * 1.13                                # ~13% head-room
ax.set_ylim(0, y_top)
ax.set_xlim(X_NAIVE - 0.75, X_OPT + 0.95)

# nice round y ticks
import numpy as np
# pick tick step so we always have ~4-7 gridlines regardless of total.
if y_top > 60:
    step = 10
elif y_top > 30:
    step = 5
else:
    step = 2
ax.set_yticks(np.arange(0, int(y_top) + 1, step))

# full-box frame (like Nunchaku)
for spine in ("top", "right", "bottom", "left"):
    ax.spines[spine].set_visible(True)
    ax.spines[spine].set_linewidth(0.9)
    ax.spines[spine].set_color(axisC)

ax.tick_params(axis="y", which="major", length=3.5, labelsize=9,
               direction="in", color=axisC)
ax.tick_params(axis="x", which="major", length=0, labelsize=10.5, pad=4)
ax.grid(axis="y", which="major", color="#D9DEE3", lw=0.6, zorder=1)
ax.set_axisbelow(True)

# ----------------------------------------------------------------------------
# Legend (outside, right)
# ----------------------------------------------------------------------------
legend_entries = [
    ("Reduce Add",            redC),
    ("Sparse GEMM",           spC),
    ("Dense GEMM",            tcC),
    ("Fused",                 fpC),
    ("Quant",                 qC),
]
handles = [mpatches.Patch(facecolor=c, edgecolor="white", label=lbl)
           for lbl, c in legend_entries]
ax.legend(handles=handles,
          loc="center left", bbox_to_anchor=(1.015, 0.5),
          frameon=False, fontsize=9.0, handlelength=1.1,
          handleheight=1.0, borderaxespad=0.0, labelspacing=0.6)

# italic caption below (Nunchaku-style "(a) ...")
fig.text(0.5, -0.02,
         "(a) Latency breakdown of W4A4 sparse-augmented GEMM",
         ha="center", va="top", fontsize=9.5, color=axisC, style="italic")

plt.tight_layout()

# ----------------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------------
out_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(out_dir, "latency_breakdown.pdf")
png_path = os.path.join(out_dir, "latency_breakdown.png")
fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.06)
fig.savefig(png_path, bbox_inches="tight", pad_inches=0.06, dpi=300)
print(f"[ok] wrote {pdf_path}")
print(f"[ok] wrote {png_path}")
