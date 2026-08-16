"""Figure 5: Study 2 agreement against the human ceiling.

Usage
    python fig5_study2_ceiling.py

Expects the released results in ./results next to this script and writes
fig_study2_ceiling.pdf and fig_study2_ceiling.png into the same directory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.metrics import cohen_kappa_score

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = HERE

COL, TEXT = 3.5, 7.16
DARKRED = "#8B1A1A"
PALETTE = {"qwen25vl": "#1f5fbf", "internvl3": "#e08214", "pixtral": "#2e8b57",
           "donut": "#8b5a9f", "ensemble": "#c02626"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Nimbus Sans", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.titlesize": 0.1, "axes.labelsize": 11, "axes.labelweight": "normal",
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 9, "legend.frameon": False,
    "axes.linewidth": 0.8, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.5, "axes.axisbelow": True,
    "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
})

TAG = dict(fontsize=11, fontweight="bold", va="top",
           bbox=dict(boxstyle="square,pad=0.12", facecolor="white",
                     edgecolor="none", alpha=0.85))


def tag(ax, letter, x=0.015, y=0.985):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, **TAG)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=400)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


RESOLUTION = 0.05
MODELS = ["qwen25vl", "internvl3", "pixtral", "donut", "ensemble"]
REGIMES = ["zero_shot", "few_shot", "lora", "full"]
MODEL_LABEL = {"qwen25vl": "Qwen2.5-VL", "internvl3": "InternVL3",
               "pixtral": "Pixtral", "donut": "Donut", "ensemble": "Cascade"}
REGIME_LABEL = {"zero_shot": "Zero-shot", "few_shot": "Few-shot",
                "lora": "LoRA", "full": "Partial FT"}
QUESTIONS = [
    (1, "Machine Learning", "Text", [
        "importance of r2 and variability (eg. 0.85 or 85%) explained by the model",
        "mathematical expression & interpretation and evaluate model performance by r2",
        "critical insight (limitations / adjusted r2) and higher r2 strengthens confidence",
        "enables comparison between models and overfitting issues"]),
    (2, "Machine Learning", "Text", [
        "supervised learning", "unsupervised learning", "reinforcement learning",
        "example, overall clarity & structure"]),
    (3, "Digital Image Processing", "Text", [
        "definition of quantization", "purpose of quantization",
        "max-lloyd algorithm explanation", "boundary & centroid conditions"]),
    (4, "Digital Image Processing", "Text", [
        "concept of histogram equalization", "necessity of histogram equalization",
        "concept of histogram matching", "necessity of histogram matching"]),
    (5, "Database Systems", "Text+Tab+Eq", [
        "understanding of normalization and related problems(3)",
        "explanation of normal forms(3)", "use of example(2)",
        "clear and organized writing(2)"]),
    (6, "Database Systems", "Text+Tab+Eq", [
        "understanding of schema and instance (2)",
        "explanation of different types of keys with examples (4)",
        "correct identification of keys in given relation (2)",
        "clear and organized writing (2)"]),
    (7, "Computer Networks", "Text+Diag", [
        "identification of layers (2 marks)",
        "explanation of layers with diagram (4 marks)",
        "technical accuracy (2 marks)", "examples/protocols(1 mark)",
        "clarity & organization (1 mark)"]),
    (8, "Data Mining", "Text", [
        "definition of data mining", "kdd process steps",
        "importance of data mining marks=(4)"]),
    (9, "Algorithms", "Text+Eq+Diag", [
        "understanding of concept", "step-by-step procedure",
        "accuracy of final answer", "clarity & presentation"]),
    (10, "Fisheries", "Text", [
        "defintion of climate and weather", "name of the climate variables",
        "hemato-biochemical changes in rohu due to an increase of temparature"]),
    (11, "Object Oriented Prog.", "Text+Code", [
        "definition of method overloading", "definition of method overriding",
        "comparision", "example / application"]),
    (12, "E-commerce", "Text", []),
]

def figure_study2() -> None:
    gap = pd.read_csv(RESULTS / "studies" / "study2" / "ceiling_gap.csv")
    human = pd.read_csv(RESULTS / "studies" / "study2" /
                        "human_human_agreement.csv")
    ceiling = float(gap["human_ceiling_qwk"].iloc[0])
    gap = gap.sort_values("qwk", ascending=True)
    labels = [f"{MODEL_LABEL[m]} {REGIME_LABEL[r]}"
              for m, r in zip(gap.model, gap.regime)]
    colors = [PALETTE[m] for m in gap.model]

    fig, axes = plt.subplots(1, 2, figsize=(TEXT, 3.5),
                             gridspec_kw={"width_ratios": [2.1, 1]})
    ax = axes[0]
    positions = np.arange(len(gap))
    ax.barh(positions, gap.qwk, color=colors, edgecolor="white", linewidth=0.4)
    ax.axvline(ceiling, color=DARKRED, linestyle="--", linewidth=1.6)
    ax.annotate(f"human ceiling {ceiling:.2f}", xy=(ceiling, 0.55),
                xytext=(ceiling + 0.045, 0.55), color=DARKRED, fontsize=9.5,
                va="center", ha="left", fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=DARKRED, lw=0.9))
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("QWK against the course examiner")
    ax.set_xlim(0, max(0.8, gap.qwk.max() * 1.08))
    ax.grid(axis="y", visible=False)
    tag(ax, "a")

    ax = axes[1]
    names = {"gold vs rater2": "E vs F1", "gold vs rater3": "E vs F2",
             "rater2 vs rater3": "F1 vs F2"}
    order = list(human.comparison)
    values = human.set_index("comparison")
    positions = np.arange(len(order))
    lows = values.loc[order, "qwk"] - values.loc[order, "qwk_ci_lo"]
    highs = values.loc[order, "qwk_ci_hi"] - values.loc[order, "qwk"]
    ax.bar(positions, values.loc[order, "qwk"], 0.6, color="#7f8c9b",
           yerr=[lows, highs], capsize=3, error_kw={"linewidth": 1.0})
    best = gap.sort_values("qwk", ascending=False).iloc[0]
    ax.axhline(best.qwk, color=DARKRED, linestyle="-", linewidth=1.6)
    ax.text(-0.42, best.qwk + 0.018, f"best system {best.qwk:.2f}", ha="left",
            va="bottom", fontsize=9.5, color=DARKRED, fontweight="bold")
    ax.set_xticks(positions)
    ax.set_xticklabels([names.get(c, c) for c in order], fontsize=9.5)
    ax.set_ylabel("QWK")
    ax.set_ylim(0, 0.88)
    ax.tick_params(axis="x", length=0)
    tag(ax, "b", 0.04)
    fig.tight_layout(pad=0.35, w_pad=1.0)
    save(fig, "fig_study2_ceiling")


if __name__ == "__main__":
    figure_study2()
