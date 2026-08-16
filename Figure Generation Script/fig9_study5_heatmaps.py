"""Figure 9: two correctly graded explanation examples.

Usage
    python fig9_study5_heatmaps.py

Expects the released results in ./results next to this script and writes
fig_study5_heatmaps.pdf and fig_study5_heatmaps.png into the same directory.
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

def figure_heatmaps(names=("27_c1", "316_def")) -> None:
    """Stack two correctly graded explanation examples in one column figure.

    The released PNGs carry a suptitle and per-panel titles. Both are cropped
    away here and replaced with a compact header that states the awarded and
    the available marks, which is the information a reader needs.
    """
    import json

    from matplotlib.image import imread

    source = RESULTS / "studies" / "study5" / "heatmaps"
    index_path = source / "xai_index.json"
    meta = {}
    if index_path.is_file():
        meta = {row["example_id"]: row for row in json.loads(
            index_path.read_text(encoding="utf-8"))}

    available = [n for n in names if (source / f"{n}.png").is_file()]
    if not available:
        print("  [skip] no heatmap PNGs found in", source)
        return

    strips, headers = [], []
    for name in available:
        image = imread(source / f"{name}.png")
        gray = image[..., :3].mean(axis=2) if image.ndim == 3 else image
        dark = gray < 0.94
        rows = np.where(dark.any(axis=1))[0]
        # The third band of dark rows is the panel row; the first two are the
        # suptitle and the per-panel titles that we drop.
        bands, previous = [], -10
        for row in rows:
            if row - previous > 6:
                bands.append([row, row])
            else:
                bands[-1][1] = row
            previous = row
        top, bottom = (bands[-1] if bands else (0, gray.shape[0] - 1))
        columns = np.where(dark[top:bottom + 1].any(axis=0))[0]
        left, right = (columns[0], columns[-1]) if len(columns) else (0, gray.shape[1] - 1)
        strips.append(image[top:bottom + 1, left:right + 1])

        record = meta.get(name, {})
        criterion = str(record.get("criterion_name", name))
        criterion = criterion.split("(")[0].strip()[:34]
        predicted = record.get("pred_score")
        maximum = record.get("max_score")
        marks = (f"{float(predicted):g} / {float(maximum):g} marks"
                 if predicted is not None and maximum is not None else "")
        headers.append((criterion, marks))

    heights = [strip.shape[0] / strip.shape[1] for strip in strips]
    figure_height = COL * sum(heights) + 0.46 * len(strips) + 0.10
    fig = plt.figure(figsize=(COL, figure_height))
    grid = fig.add_gridspec(len(strips), 1, hspace=0.20,
                            height_ratios=heights)

    for position, (strip, (criterion, marks)) in enumerate(zip(strips, headers)):
        ax = fig.add_subplot(grid[position])
        ax.imshow(strip)
        ax.set_axis_off()
        label = f"({'ab'[position]}) {criterion}"
        ax.set_title(f"{label}  [{marks}]", fontsize=9.0, fontweight="bold",
                     pad=3.0, loc="left", color="#16202B")
        width, height = strip.shape[1], strip.shape[0]
        for fraction, text in ((1 / 6, "student answer"),
                               (1 / 2, "attention"), (5 / 6, "overlay")):
            ax.text(fraction * width, height * 1.015, text, ha="center",
                    va="top", fontsize=8.4, color="#5B6774", clip_on=False)
    fig.tight_layout(pad=0.12)
    save(fig, "fig_study5_heatmaps")


if __name__ == "__main__":
    figure_heatmaps()
