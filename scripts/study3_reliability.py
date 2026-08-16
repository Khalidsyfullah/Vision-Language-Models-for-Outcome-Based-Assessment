#!/usr/bin/env python
"""Study 3: test-retest reliability across repeated runs.

Consumes results/runs/<model>/<regime>/reliability/predictions_run{i}.json,
produced by
    python scripts/run_experiment.py --model <m> --regime lora \
        --skip-train --reliability
(or run_all.py --with-reliability). These live in their own folder, so the
deterministic run0 used by Studies 1/2/4/5 is never overwritten.

Reports per model: mean per-item SD / range, % identical items, ICC(2,1),
mean pairwise QWK and Pearson between runs.

    python scripts/study3_reliability.py [--regime lora]

Outputs: results/studies/study3/
    reliability_summary.{csv,json}
    reliability_per_item.{csv,json}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import load_config, reliability_path, study_dir
from src.io_utils import read_records, save_json, save_table
from src.metrics import reliability_metrics


def run_matrix(cfg, model: str, regime: str):
    """items x runs matrix of predicted scores (NaN on parse failure)."""
    runs, run_ids = [], []
    for i in range(int(cfg.study3.n_runs)):
        p = reliability_path(cfg, model, regime, i)
        if not p.exists():
            continue
        runs.append({(r["example_id"], r["criterion_index"]):
                     r.get("pred_score") for r in read_records(p)})
        run_ids.append(i)
    if len(runs) < 2:
        return None, None, None
    keys = sorted(set().union(*[set(r) for r in runs]))
    mat = np.full((len(keys), len(runs)), np.nan)
    for j, run in enumerate(runs):
        for k, key in enumerate(keys):
            v = run.get(key)
            try:
                value = float(v)
            except (TypeError, ValueError, OverflowError):
                continue
            if np.isfinite(value):
                mat[k, j] = value
    return mat, keys, run_ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--regime", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    regime = args.regime or cfg.study3.get("regime", "lora")
    out = study_dir(cfg, 3)
    out.mkdir(parents=True, exist_ok=True)
    res = float(cfg.dataset.get("score_resolution", 0.05))

    summary, item_rows = [], []
    for model in cfg.models.keys():
        mat, keys, run_ids = run_matrix(cfg, model, regime)
        if mat is None:
            print(f"[study3] {model}: <2 repeated runs found — run "
                  f"run_experiment.py --reliability first")
            continue
        summary.append({"model": model, "regime": regime,
                        **reliability_metrics(mat, res)})
        stds = np.nanstd(mat, axis=1, ddof=1)
        for (eid, cidx), row, sd in zip(keys, mat, stds):
            item_rows.append({
                "model": model, "example_id": eid, "criterion_index": cidx,
                **{f"run{run_id}": (None if np.isnan(v) else float(v))
                   for run_id, v in zip(run_ids, row)},
                "mean": (None if np.isnan(row).all()
                         else float(np.nanmean(row))),
                "std": None if np.isnan(sd) else float(sd),
                "range": (None if np.isnan(row).any()
                          else float(row.max() - row.min())),
            })

    if not summary:
        raise SystemExit("No repeated runs found for any model.")

    sdf = pd.DataFrame(summary)
    save_table(sdf, out / "reliability_summary")
    save_table(pd.DataFrame(item_rows), out / "reliability_per_item")
    save_json({"regime": regime, "n_models": len(summary),
               "score_resolution": res}, out / "study3_summary.json")

    print(f"\n✓ Study 3 done -> {out}")
    print(sdf.to_string(index=False))


if __name__ == "__main__":
    main()
