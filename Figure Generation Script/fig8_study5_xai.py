"""Figure 8: Study 5 faithfulness and expert ratings.

Usage
    python fig8_study5_xai.py

Expects the released results in ./results next to this script and writes
fig_study5_xai.pdf and fig_study5_xai.png into the same directory.
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

TAG = dict(fontsize=12, fontweight="bold", va="top",
           bbox=dict(boxstyle="square,pad=0.12", facecolor="white",
                     edgecolor="none", alpha=0.85))


def tag(ax, letter, x=0.013, y=1.15):
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

def figure_study5() -> None:
    fraction = pd.read_csv(RESULTS / "studies" / "study5" /
                           "faithfulness_by_fraction.csv")
    candidates = [RESULTS / "Survey" / "teacher_survey_responses.csv",
                  RESULTS / "studies" / "study5" / "teacher_survey_template.csv"]
    survey = pd.read_csv(next(p for p in candidates if p.is_file()))
    rating = next(c for c in survey.columns if c.lower().startswith("rating"))
    survey[rating] = pd.to_numeric(survey[rating], errors="coerce")

    fig, axes = plt.subplots(1, 2, figsize=(TEXT, 2.5),
                             gridspec_kw={"width_ratios": [1, 1.35]})
    ax = axes[0]
    positions = np.arange(len(fraction))
    ax.bar(positions - 0.19, fraction.comprehensiveness, 0.36, color="#1f5fbf",
           label="attention")
    ax.bar(positions + 0.19, fraction.random_effect, 0.36, color="#b0b7c3",
           label="random")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{v:.0%}" for v in fraction.mask_fraction])
    ax.set_xlabel("Masked patches")
    ax.set_ylabel("Mark drop")
    ax.legend(ncol=2, fontsize=9, loc="upper left", handlelength=1.0)
    ax.tick_params(axis="x", length=0)

    ax = axes[1]
    labels = {"trust_score": "Trust", "explanation_clear": "Clear",
              "explanation_faithful": "Matches", "useful_workflow": "Useful",
              "would_adopt": "Adopt"}
    keys = list(labels)
    shades = ["#c02626", "#e08214", "#d9d9d9", "#74a9cf", "#2171b5"]
    bottom = np.zeros(len(keys))
    for value, color in zip(range(1, 6), shades):
        share = [float((survey[survey.question_key == k][rating] == value).mean())
                 for k in keys]
        ax.barh(np.arange(len(keys)), share, 0.62, left=bottom, color=color,
                label=str(value), edgecolor="white", linewidth=0.4)
        bottom += np.array(share)
    ax.set_yticks(np.arange(len(keys)))
    ax.set_yticklabels([labels[k] for k in keys], fontsize=9.5)
    ax.set_xlabel("Share of ratings")
    ax.set_xlim(0, 1)
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=5, fontsize=8.5, loc="upper center",
              bbox_to_anchor=(0.5, 1.2), columnspacing=0.8, handlelength=0.9,
              handletextpad=0.35)

    for index, ax in enumerate(axes):
        tag(ax, "abc"[index])
    fig.tight_layout(pad=0.35, w_pad=1.1)
    save(fig, "fig_study5_xai")


if __name__ == "__main__":
    figure_study5()
