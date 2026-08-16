#!/usr/bin/env python
"""Dataset inspection for the paper's Data section and quality audit.

    python scripts/inspect_dataset.py [--sample-images 200]

Outputs: results/dataset/
    split_summary.{csv,json}        rows / answers / questions per split
    criterion_summary.{csv,json}    per-criterion counts, maxima, means
    question_summary.{csv,json}     per-question counts, subject, marks
    score_distribution.{csv,json}   gold score histogram + grid check
    image_stats.{csv,json}          width/height/ink density
    data_quality_report.json             criterion_max defects, leakage checks
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import load_config, output_root
from src.data import image_features, load_splits, subject_of, text_language
from src.io_utils import save_json, save_table
from src.scoring import (learn_criterion_maxima, resolve_criterion_max,
                         to_grid_labels)


def to_frame(split, name: str) -> pd.DataFrame:
    """Metadata frame (images excluded) for one split."""
    cols = ["example_id", "answer_id", "question", "criterion_index",
            "criterion_name", "criterion_max", "rubric_levels",
            "gained_marks", "total_marks"]
    data = {c: split[c] for c in cols if c in split.column_names}
    df = pd.DataFrame(data)
    df["split"] = name
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--sample-images", type=int, default=200,
                    help="distinct answer images to measure per split (0 = all)")
    args = ap.parse_args()
    if args.sample_images < 0:
        ap.error("--sample-images must be non-negative")

    cfg = load_config(args.config)
    out = output_root(cfg) / "dataset"
    out.mkdir(parents=True, exist_ok=True)
    res = float(cfg.dataset.get("score_resolution", 0.05))

    dset = load_splits(cfg)
    frames = {s: to_frame(dset[s], s) for s in dset.keys()}
    allf = pd.concat(frames.values(), ignore_index=True)
    print(f"loaded {len(allf)} records across {len(frames)} splits")

    # ---------------- split summary ----------------
    rows = []
    for s, df in frames.items():
        rows.append({
            "split": s,
            "n_records": len(df),
            "n_answers": df.answer_id.nunique(),
            "n_questions": df.question.nunique(),
            "n_criteria": df.criterion_name.nunique(),
            "criteria_per_answer": round(
                len(df) / max(1, df.answer_id.nunique()), 2),
            "mean_gained": float(df.gained_marks.mean()),
            "mean_answer_total": float(
                df.groupby("answer_id").total_marks.first().mean()),
        })
    split_df = pd.DataFrame(rows)
    save_table(split_df, out / "split_summary")

    # ---------------- leakage ----------------
    sets = {s: set(df.answer_id) for s, df in frames.items()}
    leakage = {f"{a}&{b}": len(sets[a] & sets[b])
               for i, a in enumerate(sets) for b in list(sets)[i + 1:]}

    # ---------------- criterion_max audit + repair ----------------
    upper = float(cfg.dataset.get("max_criterion_score", 4.0))
    train_maxima = learn_criterion_maxima(dset["train"], upper)
    allf["resolved_max"] = [
        resolve_criterion_max(r, train_maxima, upper)
        for r in allf.to_dict("records")]
    raw_bad = int((allf.gained_marks > allf.criterion_max).sum())
    new_bad = int((allf.gained_marks > allf.resolved_max).sum())
    changed = int((allf.criterion_max != allf.resolved_max).sum())

    # ---------------- score grid ----------------
    grid = to_grid_labels(allf.gained_marks.values, res) * res
    off_grid = int((np.abs(grid - allf.gained_marks.values) > 1e-9).sum())
    dist = (allf.gained_marks.value_counts().sort_index()
            .rename_axis("score").reset_index(name="count"))
    dist["share"] = dist["count"] / len(allf)
    save_table(dist, out / "score_distribution")

    # answer totals: does sum(criteria) equal total_marks?
    g = allf.groupby("answer_id").agg(sum_gained=("gained_marks", "sum"),
                                      total=("total_marks", "first"))
    totals_consistent = int((np.abs(g.sum_gained - g.total) < 1e-6).sum())

    # ---------------- per-criterion / per-question ----------------
    crit = (allf.groupby("criterion_name")
            .agg(n=("gained_marks", "size"),
                 dataset_max=("criterion_max", "max"),
                 resolved_max=("resolved_max", "max"),
                 mean_score=("gained_marks", "mean"),
                 std_score=("gained_marks", "std"),
                 min_score=("gained_marks", "min"),
                 max_score=("gained_marks", "max"))
            .reset_index()
            .sort_values("n", ascending=False))
    crit["max_was_repaired"] = crit.dataset_max != crit.resolved_max
    save_table(crit, out / "criterion_summary")

    q_source = allf.assign(
        subject=[subject_of(r) for r in allf.to_dict("records")],
        language=[text_language(r) for r in allf.to_dict("records")])
    q = (q_source.groupby("question")
         .agg(n_records=("gained_marks", "size"),
              n_answers=("answer_id", "nunique"),
              n_criteria=("criterion_name", "nunique"),
              subject=("subject", "first"),
              language=("language", "first")))
    # Count each answer once; averaging criterion rows would overweight answers
    # that happen to have more rubric criteria.
    answer_totals = (q_source.groupby(["question", "answer_id"])
                     .total_marks.first().groupby("question").mean())
    q = (q.join(answer_totals.rename("mean_answer_total"))
         .reset_index().sort_values("n_records", ascending=False))
    q["question"] = q["question"].str.slice(0, 120)
    save_table(q, out / "question_summary")

    # ---------------- images ----------------
    img_rows = []
    rng = np.random.default_rng(cfg.seed)
    for s in dset.keys():
        split = dset[s]
        first_by_answer = {}
        for i, aid in enumerate(split["answer_id"]):
            first_by_answer.setdefault(str(aid), i)
        indices = np.asarray(list(first_by_answer.values()), dtype=int)
        if args.sample_images != 0 and len(indices) > args.sample_images:
            indices = rng.choice(indices, args.sample_images, replace=False)
        for i in sorted(indices.tolist()):
            ex = split[i]
            aid = str(ex["answer_id"])
            im = ex["image"]
            f = image_features(im)
            img_rows.append({"split": s, "answer_id": aid,
                             "width": im.width, "height": im.height,
                             "long_side": max(im.size), **f})
    img_df = pd.DataFrame(img_rows)
    save_table(img_df, out / "image_stats")

    # ---------------- quality report ----------------
    report = {
        "n_records": int(len(allf)),
        "n_answers": int(allf.answer_id.nunique()),
        "n_questions": int(allf.question.nunique()),
        "n_criteria": int(allf.criterion_name.nunique()),
        "answer_id_leakage_between_splits": leakage,
        "criterion_max": {
            "rows_with_gained_gt_dataset_max": raw_bad,
            "rows_with_gained_gt_resolved_max": new_bad,
            "rows_where_max_repaired": changed,
            "repair_lookup_source": "train split only",
            "configured_upper_bound": upper,
            "dataset_max_values": {str(k): int(v) for k, v in
                                   Counter(allf.criterion_max).items()},
            "resolved_max_values": {str(k): int(v) for k, v in
                                    Counter(allf.resolved_max).items()},
        },
        "scores": {
            "distinct_values": sorted(float(v) for v in
                                      allf.gained_marks.unique()),
            "score_resolution": res,
            "rows_off_grid": off_grid,
            "min": float(allf.gained_marks.min()),
            "max": float(allf.gained_marks.max()),
        },
        "answer_totals": {
            "answers_where_sum_criteria_equals_total_marks": totals_consistent,
            "n_answers_checked": int(len(g)),
            "note": "total_marks is the ACHIEVED total for the answer, "
                    "not the question's full mark",
        },
        "images": {
            "n_measured": int(len(img_df)),
            "width_min": int(img_df.width.min()) if len(img_df) else None,
            "width_median": float(img_df.width.median()) if len(img_df) else None,
            "width_max": int(img_df.width.max()) if len(img_df) else None,
            "height_min": int(img_df.height.min()) if len(img_df) else None,
            "height_median": float(img_df.height.median()) if len(img_df) else None,
            "height_max": int(img_df.height.max()) if len(img_df) else None,
            "note": "the same image is shared by all criteria of an answer",
        },
        "languages": dict(Counter(text_language(r)
                                  for r in allf.to_dict("records"))),
        "subjects": dict(Counter(subject_of(r)
                                 for r in allf.to_dict("records"))),
    }
    save_json(report, out / "data_quality_report.json")

    print(f"\n✓ Dataset inspection -> {out}")
    print(split_df.to_string(index=False))
    print(f"\ncriterion_max: {raw_bad} impossible rows in the raw data, "
          f"{new_bad} after repair ({changed} rows changed)")
    print(f"answer_id leakage across splits: {leakage}")
    print(f"scores off the {res}-mark grid: {off_grid}")


if __name__ == "__main__":
    main()
