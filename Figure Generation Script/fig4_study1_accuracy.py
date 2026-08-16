"""Figure 4: Study 1 accuracy panels.

Usage
    python fig4_study1_accuracy.py

Expects the released results in ./results next to this script and writes
fig_study1_accuracy.pdf and fig_study1_accuracy.png into the same directory.
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

def grouped_bars(ax, frame, metric, ylabel, lower_better=False, legend=False):
    """One group of bars per learning setup, one bar per model."""
    width = 0.16
    positions = np.arange(len(REGIMES))
    for index, model in enumerate(MODELS):
        values = []
        for regime in REGIMES:
            row = frame[(frame.model == model) & (frame.regime == regime)]
            values.append(float(row[metric].iloc[0]) if len(row) and
                          np.isfinite(pd.to_numeric(row[metric].iloc[0],
                                                    errors="coerce")) else np.nan)
        offset = (index - (len(MODELS) - 1) / 2) * width
        ax.bar(positions + offset, values, width * 0.92,
               color=PALETTE[model], label=MODEL_LABEL[model],
               edgecolor="white", linewidth=0.4)
    ax.set_xticks(positions)
    ax.set_xticklabels([REGIME_LABEL[r] for r in REGIMES])
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", length=0)
    if legend:
        ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.22),
                  columnspacing=1.0, handlelength=1.1, handletextpad=0.4)
    return ax


def figure_study1() -> None:
    board = pd.read_csv(RESULTS / "studies" / "study1" / "accuracy_leaderboard.csv")
    panels = [("qwk", "QWK", False), ("mae", "MAE (marks)", True),
              ("exact_match", "Exact match", False),
              ("adjacent_match", "Adjacent match", False)]
    fig, axes = plt.subplots(2, 2, figsize=(TEXT, 3.9))
    for ax, (metric, label, lower) in zip(axes.ravel(), panels):
        grouped_bars(ax, board, metric, label, lower)
    handles = [Patch(facecolor=PALETTE[m], label=MODEL_LABEL[m]) for m in MODELS]
    fig.legend(handles=handles, ncol=5, loc="upper center",
               bbox_to_anchor=(0.5, 1.06), columnspacing=1.4, handlelength=1.2,
               fontsize=10)
    for index, ax in enumerate(axes.ravel()):
        tag(ax, "abcd"[index], 0.015, 0.96)
    fig.tight_layout(pad=0.35, h_pad=0.9, w_pad=1.1)
    save(fig, "fig_study1_accuracy")


if __name__ == "__main__":
    figure_study1()
