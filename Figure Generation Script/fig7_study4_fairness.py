"""Figure 7: Study 4 error decomposition.

Usage
    python fig7_study4_fairness.py

Expects the released results in ./results next to this script and writes
fig_study4_fairness.pdf and fig_study4_fairness.png into the same directory.
"""

from __future__ import annotations

import json
import re
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
RESULTS = Path(os.environ.get("OBE_RESULTS", HERE / "results"))
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


RESULTS = Path(os.environ.get("OBE_RESULTS", "results"))
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

def normalize_criterion(name: str) -> str:
    """Lowercase, collapse whitespace, drop the R-squared superscript."""
    text = str(name).lower().replace("²", "2").replace("²", "2")
    text = text.split("\n")[0]
    text = re.sub(r"\s+", " ", text).strip()
    return text


def criterion_index() -> dict:
    """normalized criterion name -> (question number, subject, answer type)."""
    table = {}
    for number, subject, kind, keys in QUESTIONS:
        for key in keys:
            table[key] = (number, subject, kind)
    return table


def attach_mapping(frame: pd.DataFrame, column: str = "criterion_name"):
    """Add question / subject / answer-type columns; report anything unmapped."""
    table = criterion_index()
    keys = frame[column].map(normalize_criterion)
    hits = keys.map(lambda k: table.get(_resolve(k, table)))
    frame = frame.copy()
    frame["question"] = [h[0] if h else np.nan for h in hits]
    frame["subject"] = [h[1] if h else "Unmapped" for h in hits]
    frame["answer_type"] = [h[2] if h else "" for h in hits]
    return frame


def _resolve(key: str, table: dict) -> str:
    """Exact match first, then longest-prefix match for truncated names."""
    if key in table:
        return key
    for candidate in table:
        if key.startswith(candidate[:40]) or candidate.startswith(key[:40]):
            return candidate
    return key


def to_grid(scores, resolution: float = RESOLUTION) -> np.ndarray:
    values = np.asarray(scores, dtype=float) / float(resolution)
    return np.floor(values + 0.5).astype(int)


def qwk(y_true, y_pred, resolution: float = RESOLUTION):
    """Quadratic weighted kappa on the mark grid, matching the release code."""
    a, b = to_grid(y_true, resolution), to_grid(y_pred, resolution)
    if len(a) < 2:
        return np.nan
    if len(set(a.tolist())) < 2 and len(set(b.tolist())) < 2:
        return np.nan
    observed = sorted(set(a.tolist()) | set(b.tolist()))
    labels = list(range(int(min(observed)), int(max(observed)) + 1))
    try:
        value = cohen_kappa_score(a, b, labels=labels, weights="quadratic")
    except Exception:
        return np.nan
    return float(value) if np.isfinite(value) else np.nan


def load_predictions(model: str, regime: str) -> pd.DataFrame | None:
    path = RESULTS / "runs" / model / regime / "predictions_run0.jsonl"
    if not path.is_file():
        return None
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    frame = pd.DataFrame(rows)
    for column in ("pred_score", "true_score", "max_score"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["model"], frame["regime"] = model, regime
    return frame


def load_all_predictions() -> pd.DataFrame:
    frames = []
    for model in MODELS:
        for regime in REGIMES:
            frame = load_predictions(model, regime)
            if frame is not None:
                frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def figure_study4() -> None:
    frame = attach_mapping(load_all_predictions())
    frame = frame.dropna(subset=["pred_score", "true_score"])
    frame["abs_error"] = (frame.pred_score - frame.true_score).abs()
    taxonomy = pd.read_csv(RESULTS / "studies" / "study4" /
                           "failure_taxonomy.csv")

    fig, axes = plt.subplots(1, 3, figsize=(TEXT, 2.75),
                             gridspec_kw={"width_ratios": [1.35, 1, 1.15]})

    ax = axes[0]
    best = frame[(frame.regime == "lora") & (frame.model != "donut")]
    order = (best.groupby("subject").abs_error.mean().sort_values(ascending=False)
             .index.tolist())
    for index, model in enumerate([m for m in MODELS if m != "donut"]):
        values = [best[(best.model == model) & (best.subject == s)].abs_error.mean()
                  for s in order]
        offset = (index - 1.5) * 0.2
        ax.barh(np.arange(len(order)) + offset, values, 0.19,
                color=PALETTE[model], label=MODEL_LABEL[model])
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([s.replace(" ", "\n", 1) if len(s) > 16 else s
                        for s in order], fontsize=8)
    ax.set_xlabel("MAE (marks)")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=1, fontsize=8, loc="upper right", handlelength=0.9,
              borderpad=0.25, labelspacing=0.28, borderaxespad=0.35)
    ax.set_xlim(0, max(1.45, ax.get_xlim()[1] * 1.06))

    ax = axes[1]
    lengths = frame[frame.regime == "lora"].copy()
    edges = [0, 2.0, 3.0, 99]
    names_w = ["$\\leq$2", "2 to 3", ">3"]
    lengths["bucket"] = pd.cut(lengths.max_score, edges, labels=names_w)
    for model in [m for m in MODELS if m != "donut"]:
        subset = lengths[lengths.model == model]
        means = subset.groupby("bucket", observed=False).abs_error.mean()
        ax.plot(range(len(means)), means.values, marker="o", markersize=5,
                linewidth=1.8, color=PALETTE[model], label=MODEL_LABEL[model])
    ax.set_xticks(range(len(names_w)))
    ax.set_xticklabels(names_w)
    ax.set_xlabel("Criterion weight (marks)")
    ax.set_ylabel("MAE (marks)")

    ax = axes[2]
    focus = taxonomy[(taxonomy.model == "qwen25vl") & (taxonomy.regime == "lora")]
    grouped = focus.groupby("failure_class")["count"].sum().sort_values()
    names = {"correct": "correct", "overscore_adjacent": "over $\\leq$1",
             "overscore_large": "over >1", "underscore_adjacent": "under $\\leq$1",
             "underscore_large": "under >1", "parse_failure": "parse fail"}
    colors = ["#2e8b57" if k == "correct" else "#c02626" if "large" in k
              else "#e08214" for k in grouped.index]
    ax.barh(np.arange(len(grouped)), grouped.values, 0.65, color=colors)
    ax.set_yticks(np.arange(len(grouped)))
    ax.set_yticklabels([names.get(k, k) for k in grouped.index], fontsize=9)
    ax.set_xlabel("Judgments")
    ax.grid(axis="y", visible=False)

    for index, ax in enumerate(axes):
        tag(ax, "abc"[index])
    fig.tight_layout(pad=0.35, w_pad=1.1)
    save(fig, "fig_study4_fairness")


if __name__ == "__main__":
    figure_study4()
