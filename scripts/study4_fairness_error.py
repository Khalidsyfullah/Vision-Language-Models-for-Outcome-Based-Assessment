#!/usr/bin/env python
"""Study 4: subgroup gaps and failure taxonomy.

Subgroups are computed FROM THE DATA:
  * handwriting density  : ink_density terciles
  * answer length        : ink_pixels terciles
  * subject              : inferred from the question (12 distinct questions)
  * language             : bangla vs english — NOTE the released dataset is
                           entirely English, so this collapses to one level
                           and is skipped by study4.min_subgroup_n
  * criterion scale      : criterion max <= 2.5 vs > 2.5
  * criterion type       : numerical / diagram / definition / structure / application

Failure taxonomy per prediction:
  correct | over/under-score adjacent (partial-credit confusion) |
  over/under-score large | parse_failure — crossed with criterion type.

    python scripts/study4_fairness_error.py [--model qwen25vl --regime lora]
    (default: every completed run)

Outputs: results/studies/study4/
    fairness_subgroups.{csv,json}, fairness_gap_headline.{csv,json}
    failure_taxonomy.{csv,json}, error_examples.{csv,json}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import load_config, study_dir
from src.data import image_features, load_splits, subject_of, text_language
from src.io_utils import save_json, save_table
from src.metrics import compute_metrics
from src.scoring import learn_criterion_maxima, resolve_criterion_max
from study1_accuracy import discover_runs

TYPE_RULES = [
    ("numerical", ("calculat", "r²", "r2", "math", "formula", "compute",
                   "equation", "numer", "value")),
    ("diagram", ("draw", "diagram", "figure", "sketch", "graph", "plot")),
    ("structure_clarity", ("clarity", "structure", "organized", "organised",
                           "writing", "presentation")),
    ("application", ("example", "application", "apply", "use of",
                     "identification", "identify")),
]


def criterion_type(criterion_name: str, question: str) -> str:
    text = f"{criterion_name} {question}".lower()
    for label, kws in TYPE_RULES:
        if any(k in text for k in kws):
            return label
    return "definition_theory"


def failure_class(r: dict, large: float) -> str:
    try:
        pred = float(r.get("pred_score"))
        true = float(r.get("true_score"))
    except (TypeError, ValueError, OverflowError):
        return "parse_failure"
    if not np.isfinite(pred) or not np.isfinite(true):
        return "parse_failure"
    err = pred - true
    if abs(err) < 1e-9:
        return "correct"
    if abs(err) >= large:
        return "overscore_large" if err > 0 else "underscore_large"
    return "overscore_adjacent" if err > 0 else "underscore_adjacent"


def tercile_labels(series: pd.Series, qs: list[float],
                   names=("low", "mid", "high")) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(["unknown"] * len(s), index=s.index)
    lo, hi = s.quantile(qs[0]), s.quantile(qs[1])
    out = np.select([s <= lo, s >= hi], [names[0], names[2]],
                    default=names[1])
    return pd.Series(np.where(s.isna(), "unknown", out), index=s.index)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--regime", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = study_dir(cfg, 4)
    out.mkdir(parents=True, exist_ok=True)
    s4 = cfg.study4
    tol = cfg.study1.adjacent_tolerance
    res = float(cfg.dataset.get("score_resolution", 0.05))

    runs = discover_runs(cfg)
    if args.model or args.regime:
        runs = {(m, r): v for (m, r), v in runs.items()
                if (not args.model or m == args.model)
                and (not args.regime or r == args.regime)}
    if not runs:
        raise SystemExit("No matching prediction files found.")

    # ---- per-item features from the test split (computed once) ----
    print("computing image/text features on the test split ...")
    splits = load_splits(cfg)
    test = splits["test"]
    upper = float(cfg.dataset.get("max_criterion_score", 4.0))
    train_maxima = learn_criterion_maxima(splits["train"], upper)
    feats = {}
    img_cache: dict[str, dict] = {}
    for i in range(len(test)):
        ex = dict(test[i])
        aid = str(ex["answer_id"])
        if aid not in img_cache:                 # one image per answer
            img_cache[aid] = image_features(ex["image"])
        f = dict(img_cache[aid])
        f["language"] = text_language(ex)
        f["subject"] = subject_of(ex)
        f["criterion_type"] = criterion_type(ex["criterion_name"],
                                             ex["question"])
        f["resolved_max"] = resolve_criterion_max(ex, train_maxima, upper)
        feats[(ex["example_id"], ex["criterion_index"])] = f
    print(f"  {len(feats)} test items, {len(img_cache)} distinct images")

    sub_rows, tax_rows, err_rows = [], [], []
    skipped_degenerate: set[str] = set()
    for (m, reg), rows in sorted(runs.items()):
        df = pd.DataFrame(rows)
        keys = list(zip(df["example_id"], df["criterion_index"]))
        fdf = pd.DataFrame([feats.get(k, {}) for k in keys])
        df = pd.concat([df.reset_index(drop=True),
                        fdf.reset_index(drop=True)], axis=1)
        df["handwriting_density_grp"] = tercile_labels(
            df.get("ink_density", pd.Series(dtype=float)),
            list(s4.ink_density_quantiles))
        df["answer_length_grp"] = tercile_labels(
            df.get("ink_pixels", pd.Series(dtype=float)),
            list(s4.answer_length_quantiles))
        mx = pd.to_numeric(df.get("resolved_max"), errors="coerce")
        df["criterion_scale_grp"] = np.where(mx <= 2.5, "small_scale",
                                             "large_scale")

        for feat in list(s4.subgroup_features):
            if feat not in df.columns:
                continue
            eligible = [(level, grp) for level, grp in df.groupby(feat)
                        if len(grp) >= int(s4.min_subgroup_n)]
            if len(eligible) < 2:
                skipped_degenerate.add(feat)
                continue
            for level, grp in eligible:
                sub_rows.append({"model": m, "regime": reg, "feature": feat,
                                 "group": str(level),
                                 **compute_metrics(grp.to_dict("records"),
                                                   tol, res)})

        df["failure_class"] = [failure_class(r, float(s4.large_error_threshold))
                               for r in df.to_dict("records")]
        counts = (df.groupby(["criterion_type", "failure_class"])
                  .size().reset_index(name="count"))
        counts["model"], counts["regime"] = m, reg
        counts["share"] = counts["count"] / len(df)
        tax_rows.append(counts)

        big = df[df["failure_class"].isin(
            ["overscore_large", "underscore_large", "parse_failure"])]
        for r in big.head(60).to_dict("records"):
            err_rows.append({
                "model": m, "regime": reg,
                "example_id": r["example_id"],
                "criterion_name": r["criterion_name"],
                "criterion_type": r.get("criterion_type"),
                "true_score": r["true_score"], "pred_score": r["pred_score"],
                "failure_class": r["failure_class"],
                "raw_output": str(r.get("raw_output", ""))[:300],
            })

    sub_df = pd.DataFrame(sub_rows)
    tax_df = pd.concat(tax_rows, ignore_index=True)
    save_table(sub_df, out / "fairness_subgroups")
    save_table(tax_df, out / "failure_taxonomy")
    if err_rows:
        save_table(pd.DataFrame(err_rows), out / "error_examples")

    features = ([str(f) for f in sub_df["feature"].unique()]
                if not sub_df.empty and "feature" in sub_df else [])
    if sub_df.empty:
        gaps = pd.DataFrame(columns=["model", "regime", "feature", "mae_gap"])
    else:
        gaps = (sub_df.groupby(["model", "regime", "feature"])["mae"]
                .agg(lambda s: float(s.max() - s.min()))
                .reset_index(name="mae_gap")
                .sort_values("mae_gap", ascending=False))
    save_table(gaps, out / "fairness_gap_headline")
    save_json({"n_runs": len(runs),
               "features_evaluated": features,
               "features_skipped_degenerate": sorted(skipped_degenerate),
               "min_subgroup_n": int(s4.min_subgroup_n)},
              out / "study4_summary.json")
    print(f"\n✓ Study 4 done -> {out}")
    print(gaps.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
