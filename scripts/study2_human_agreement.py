#!/usr/bin/env python
"""Study 2: AI-human and human-human agreement.

Human raters:
  * Rater "gold" = the dataset's gained_marks (primary examiner).
  * Extra raters come from an optional CSV (paths.human_ratings_csv):
        example_id,criterion_index,rater_id,score
    one row per (item, rater). With >=2 human raters the script computes the
    human-human ceiling (QWK, kappa, Pearson/Spearman + bootstrap CIs) and
    reports how close each AI system gets to it.

If the CSV is missing, a blank template covering the test items is written to
that path — hand it to the second examiner.

    python scripts/study2_human_agreement.py

Outputs: results/studies/study2/
    ai_human_agreement.{csv,json}
    human_human_agreement.{csv,json}   (if extra raters present)
    ceiling_gap.{csv,json}
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import data_path, load_config, study_dir
from src.io_utils import save_json, save_table
from src.metrics import agreement_metrics, bootstrap_ci
from study1_accuracy import discover_runs


def load_human_ratings(cfg) -> pd.DataFrame | None:
    p = data_path(cfg, "human_ratings_csv")
    if not p.exists():
        return None
    df = pd.read_csv(p)
    need = {"example_id", "rater_id", "score"}
    if not need.issubset(df.columns):
        raise ValueError(f"{p} must have columns {need}")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"])
    if df.empty:
        print(f"[study2] {p} exists but has no filled-in scores yet")
        return None
    if "criterion_index" not in df.columns:
        df["criterion_index"] = 0
    return df


def write_template(cfg, runs) -> None:
    p = data_path(cfg, "human_ratings_csv")
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = next(iter(runs.values()))
    pd.DataFrame({
        "example_id": [r["example_id"] for r in rows],
        "criterion_index": [r["criterion_index"] for r in rows],
        "criterion_name": [r["criterion_name"] for r in rows],
        "max_score": [r.get("max_score") for r in rows],
        "rater_id": "rater2",
        "score": "",
    }).to_csv(p, index=False)
    print(f"[study2] no human ratings found — blank template written to {p}.\n"
          f"         Fill 'score' (one block of rows per extra rater) "
          f"and rerun.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = study_dir(cfg, 2)
    out.mkdir(parents=True, exist_ok=True)
    s2 = cfg.study2
    res = float(cfg.dataset.get("score_resolution", 0.05))
    tol = cfg.study1.adjacent_tolerance
    nboot = int(s2.n_bootstrap)

    runs = discover_runs(cfg)
    if not runs:
        raise SystemExit("No prediction files found — run experiments first.")

    # ---------- AI vs gold ----------
    ai_rows = []
    for (m, reg), rows in sorted(runs.items()):
        valid = []
        for r in rows:
            try:
                if (np.isfinite(float(r.get("pred_score"))) and
                        np.isfinite(float(r.get("true_score")))):
                    valid.append(r)
            except (TypeError, ValueError, OverflowError):
                continue
        if len(valid) < 2:
            continue
        yp = np.array([r["pred_score"] for r in valid], dtype=float)
        yt = np.array([r["true_score"] for r in valid], dtype=float)
        rec = {"comparison": f"{m}/{reg} vs gold", "model": m, "regime": reg,
               "rater_a": f"{m}/{reg}", "rater_b": "gold",
               **agreement_metrics(yp, yt, tol, res)}
        if s2.bootstrap_ci:
            lo, hi = bootstrap_ci(yp, yt, "qwk", nboot, cfg.seed,
                                  resolution=res)
            rec["qwk_ci_lo"], rec["qwk_ci_hi"] = lo, hi
            lo, hi = bootstrap_ci(yp, yt, "mae", nboot, cfg.seed)
            rec["mae_ci_lo"], rec["mae_ci_hi"] = lo, hi
        ai_rows.append(rec)
    ai_df = pd.DataFrame(ai_rows)
    save_table(ai_df, out / "ai_human_agreement")

    # ---------- human vs human ceiling ----------
    hh_df, ceiling = None, None
    human = load_human_ratings(cfg)
    if human is None:
        write_template(cfg, runs)
    else:
        gold = {(r["example_id"], r["criterion_index"]): r["true_score"]
                for rows in runs.values() for r in rows}
        raters: dict[str, dict] = {"gold": gold}
        for rid, grp in human.groupby("rater_id"):
            raters[str(rid)] = {
                (e, int(c)): float(s) for e, c, s in
                zip(grp["example_id"], grp["criterion_index"], grp["score"])}
        hh_rows = []
        for ra, rb in combinations(raters.keys(), 2):
            shared = sorted(set(raters[ra]) & set(raters[rb]))
            if len(shared) < 2:
                continue
            a = np.array([raters[ra][k] for k in shared], dtype=float)
            b = np.array([raters[rb][k] for k in shared], dtype=float)
            rec = {"comparison": f"{ra} vs {rb}", "rater_a": ra,
                   "rater_b": rb, **agreement_metrics(a, b, tol, res)}
            if s2.bootstrap_ci:
                lo, hi = bootstrap_ci(a, b, "qwk", nboot, cfg.seed,
                                      resolution=res)
                rec["qwk_ci_lo"], rec["qwk_ci_hi"] = lo, hi
            hh_rows.append(rec)
        if hh_rows:
            hh_df = pd.DataFrame(hh_rows)
            save_table(hh_df, out / "human_human_agreement")
            if hh_df["qwk"].notna().any():
                ceiling = float(hh_df["qwk"].dropna().mean())

    # ---------- gap to the human ceiling ----------
    if ceiling is not None and not ai_df.empty:
        gap = ai_df[["model", "regime", "qwk", "mae"]].copy()
        gap["human_ceiling_qwk"] = ceiling
        gap["qwk_gap"] = ceiling - gap["qwk"].astype(float)
        gap["reaches_ceiling"] = gap["qwk"].astype(float) >= ceiling
        if "qwk_ci_lo" in ai_df.columns:
            gap["ci_overlaps_ceiling"] = (
                (ai_df["qwk_ci_lo"].astype(float) <= ceiling) &
                (ai_df["qwk_ci_hi"].astype(float) >= ceiling)).values
        save_table(gap.sort_values("qwk_gap"), out / "ceiling_gap")

    save_json({"n_ai_systems": int(len(ai_df)),
               "human_raters_found": human is not None,
               "human_ceiling_qwk": ceiling,
               "score_resolution": res}, out / "study2_summary.json")
    print(f"\n✓ Study 2 done -> {out}")
    if ceiling is not None:
        print(f"  human-human ceiling QWK = {ceiling:.3f}")


if __name__ == "__main__":
    main()
