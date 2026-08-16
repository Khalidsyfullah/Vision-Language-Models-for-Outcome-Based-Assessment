#!/usr/bin/env python
"""Study 1: accuracy for each model and regime against gold scores.

Aggregates every results/runs/<model>/<regime>/predictions_run0.json into
one leaderboard: MAE, RMSE, exact match, adjacent (±tol) match, QWK,
Pearson/Spearman, normalised MAE, parse rate, plus answer-level totals.

    python scripts/study1_accuracy.py

Outputs: results/studies/study1/
    accuracy_leaderboard.{csv,json}
    accuracy_per_criterion.{csv,json}
    regime_deltas.{csv,json}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import load_config, predictions_path, study_dir
from src.io_utils import read_records, save_json, save_table
from src.metrics import (answer_level_metrics, compute_metrics,
                         per_criterion_metrics)


def discover_runs(cfg) -> dict[tuple[str, str], list[dict]]:
    """(model, regime) -> prediction rows, for every existing run0 file."""
    out = {}
    for m in list(cfg.models.keys()) + ["ensemble"]:
        for reg in cfg.regimes:
            p = predictions_path(cfg, m, reg, 0)
            if p.exists():
                out[(m, reg)] = read_records(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = study_dir(cfg, 1)
    out.mkdir(parents=True, exist_ok=True)
    tol = cfg.study1.adjacent_tolerance
    res = float(cfg.dataset.get("score_resolution", 0.05))

    runs = discover_runs(cfg)
    if not runs:
        raise SystemExit("No prediction files found — run experiments first.")
    print(f"found {len(runs)} completed runs")

    lead, per_crit_rows = [], []
    for (m, reg), rows in sorted(runs.items()):
        lead.append({"model": m, "regime": reg,
                     **compute_metrics(rows, tol, res),
                     **answer_level_metrics(rows)})
        for crit, met in per_criterion_metrics(rows, tol, res).items():
            per_crit_rows.append({"model": m, "regime": reg,
                                  "criterion": crit, **met})

    lead_df = pd.DataFrame(lead)
    lead_df = lead_df.sort_values(["regime", "qwk"], ascending=[True, False],
                                  na_position="last")
    save_table(lead_df, out / "accuracy_leaderboard")
    save_table(pd.DataFrame(per_crit_rows), out / "accuracy_per_criterion")

    # regime deltas per model: how much few-shot / LoRA buy over zero-shot
    deltas = []
    for m in lead_df["model"].unique():
        sub = lead_df[lead_df["model"] == m].set_index("regime")
        if "zero_shot" not in sub.index:
            continue
        base = sub.loc["zero_shot"]
        for reg in sub.index:
            if reg == "zero_shot":
                continue
            deltas.append({
                "model": m, "regime": reg,
                "d_qwk": (None if pd.isna(sub.loc[reg, "qwk"])
                          or pd.isna(base["qwk"])
                          else float(sub.loc[reg, "qwk"] - base["qwk"])),
                "d_mae": float(sub.loc[reg, "mae"] - base["mae"]),
                "d_exact_match": float(
                    sub.loc[reg, "exact_match"] - base["exact_match"]),
            })
    if deltas:
        save_table(pd.DataFrame(deltas), out / "regime_deltas")

    save_json({"n_runs": len(runs), "adjacent_tolerance": tol,
               "score_resolution": res}, out / "study1_summary.json")

    print(f"\n✓ Study 1 done -> {out}")
    cols = [c for c in ("model", "regime", "mae", "qwk", "exact_match",
                        "adjacent_match", "parse_rate") if c in lead_df]
    print(lead_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
