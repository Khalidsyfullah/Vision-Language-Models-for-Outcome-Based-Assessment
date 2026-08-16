"""Figure 6: Study 3 run-to-run reliability.

Usage
    python fig6_study3_reliability.py

Expects the released results in ./results next to this script and writes
fig_study3_reliability.pdf and fig_study3_reliability.png into the same directory.
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

def figure_study3() -> None:
    summary = pd.read_csv(RESULTS / "studies" / "study3" /
                          "reliability_summary.csv")
    items = pd.read_csv(RESULTS / "studies" / "study3" /
                        "reliability_per_item.csv")
    models = list(summary.model)
    colors = [PALETTE.get(m, "#666") for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(TEXT, 2.5))
    ax = axes[0]
    positions = np.arange(len(models))
    ax.bar(positions, summary.icc_2_1, 0.62, color=colors, edgecolor="white")
    ax.axhline(0.75, color="#444", linestyle=":", linewidth=1.2)
    ax.text(len(models) - 0.45, 0.77, "excellent", ha="right", fontsize=8.5,
            color="#444")
    ax.set_xticks(positions)
    ax.set_xticklabels([MODEL_LABEL[m] for m in models], fontsize=9, rotation=20,
                       ha="right")
    ax.set_ylabel("ICC(2,1)")
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="x", length=0)

    ax = axes[1]
    ax.bar(positions, summary.mean_item_std, 0.62, color=colors,
           edgecolor="white")
    ax.set_xticks(positions)
    ax.set_xticklabels([MODEL_LABEL[m] for m in models], fontsize=9, rotation=20,
                       ha="right")
    ax.set_ylabel("Mean item SD (marks)")
    ax.tick_params(axis="x", length=0)

    ax = axes[2]
    column = next((c for c in items.columns if "std" in c.lower()), None)
    data, keep = [], []
    for model in models:
        subset = items[items.model == model][column].dropna()
        if len(subset):
            data.append(subset.values)
            keep.append(model)
    parts = ax.violinplot(data, showmedians=True, widths=0.8)
    for body, model in zip(parts["bodies"], keep):
        body.set_facecolor(PALETTE.get(model, "#666"))
        body.set_alpha(0.75)
        body.set_edgecolor("white")
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color("#333")
            parts[key].set_linewidth(1.0)
    ax.set_xticks(np.arange(1, len(keep) + 1))
    ax.set_xticklabels([MODEL_LABEL[m] for m in keep], fontsize=9, rotation=20,
                       ha="right")
    ax.set_ylabel("Per-item SD (marks)")
    ax.tick_params(axis="x", length=0)

    for index, ax in enumerate(axes):
        tag(ax, "abc"[index])
    fig.tight_layout(pad=0.35, w_pad=1.2)
    save(fig, "fig_study3_reliability")


if __name__ == "__main__":
    figure_study3()
