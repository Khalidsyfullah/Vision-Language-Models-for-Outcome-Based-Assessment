#!/usr/bin/env python
"""Combine the three models' predictions with the cascade ensemble.

Requires predictions_run0.json for all three models in the given regime
(run scripts/run_experiment.py first).

    python scripts/run_ensemble.py --regime lora

Outputs (results/runs/ensemble/<regime>/):
    predictions_run0.json, metrics.{csv,json}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import load_config, predictions_path, run_dir, set_seed
from src.ensemble import cascade
from src.io_utils import read_records, save_json, save_table, write_records
from src.metrics import (answer_level_metrics, compute_metrics,
                         per_criterion_metrics)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--regime", default="lora",
                    choices=["zero_shot", "few_shot", "lora", "full"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    out = run_dir(cfg, "ensemble", args.regime)
    out.mkdir(parents=True, exist_ok=True)

    preds_by_model = {}
    for m in list(cfg.ensemble.small_models) + [cfg.ensemble.tiebreaker]:
        p = predictions_path(cfg, m, args.regime, 0)
        if not p.exists():
            raise SystemExit(
                f"{p} missing — run: python scripts/run_experiment.py "
                f"--model {m} --regime {args.regime}")
        preds_by_model[m] = read_records(p)
        print(f"loaded {len(preds_by_model[m]):>5d} rows from {p}")

    rows, stats = cascade(cfg, preds_by_model, args.regime)
    write_records(rows, out / "predictions_run0.json")

    tol = cfg.study1.adjacent_tolerance
    res = float(cfg.dataset.get("score_resolution", 0.05))
    overall = {**compute_metrics(rows, tol, res), **answer_level_metrics(rows)}
    per_crit = per_criterion_metrics(rows, tol, res)
    save_json({"overall": overall, "routing": stats,
               "per_criterion": per_crit}, out / "metrics.json")
    save_table(pd.DataFrame([{"model": "ensemble", "regime": args.regime,
                              **overall,
                              **{f"route_{k}": v for k, v in stats.items()}}]),
               out / "metrics")
    save_table(pd.DataFrame([{"criterion": k, **v}
                             for k, v in per_crit.items()]),
               out / "metrics_per_criterion")
    if cfg.hub.push_results:
        from src.hub import push_results
        push_results(out, cfg.hub.results_repo, f"ensemble/{args.regime}")

    print(f"\n✓ Done: ensemble/{args.regime} -> {out}")


if __name__ == "__main__":
    main()
