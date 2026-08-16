#!/usr/bin/env python
"""Unattended sweep driver: every model x regime, the ensemble per regime,
optionally the Study 3 repeated runs and all five studies.

Designed to survive a Colab session drop: completed runs are detected and
skipped, so re-running the same command resumes where it stopped.

    python scripts/run_all.py                        # experiments + ensembles
    python scripts/run_all.py --regimes zero_shot few_shot
    python scripts/run_all.py --models qwen25vl internvl3
    python scripts/run_all.py --with-reliability --with-studies
    python scripts/run_all.py --force                # ignore existing results
    python scripts/run_all.py --dry-run              # just print the plan

Each step runs as a subprocess, so an OOM or crash in one model frees its GPU
memory and never kills the sweep. A summary table is printed at the end and
written to <output_root>/sweep_summary.{csv,json}.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (TRAINING_REGIMES, checkpoint_artifact_dir, load_config,
                        output_root, predictions_path, reliability_path)

HERE = Path(__file__).resolve().parent


def _fmt(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def run_step(name: str, cmd: list, dry: bool) -> dict:
    printable = " ".join(str(c) for c in cmd)
    print(f"\n{'=' * 78}\n▶ {name}\n$ {printable}\n{'=' * 78}", flush=True)
    if dry:
        return {"step": name, "status": "planned", "seconds": 0}
    t0 = time.time()
    rc = subprocess.run([str(c) for c in cmd]).returncode
    dt = time.time() - t0
    status = "ok" if rc == 0 else f"failed(rc={rc})"
    print(f"\n◀ {name}: {status} in {_fmt(dt)}", flush=True)
    return {"step": name, "status": status, "seconds": round(dt, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--regimes", nargs="*", default=None)
    ap.add_argument("--with-reliability", action="store_true")
    ap.add_argument("--with-studies", action="store_true")
    ap.add_argument("--with-inspect", action="store_true",
                    help="also run the dataset inspection first")
    ap.add_argument("--force", action="store_true",
                    help="re-run steps whose outputs already exist")
    ap.add_argument("--skip-train", action="store_true",
                    help="lora/full: require and reuse an existing checkpoint")
    ap.add_argument("--limit", type=int, default=None,
                    help="debug: only evaluate the first N test items")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    models = args.models or list(cfg.models.keys())
    # cheapest regimes first, so a session that dies still leaves usable results
    order = ["zero_shot", "few_shot", "lora", "full"]
    regimes = args.regimes or [r for r in order if r in cfg.regimes]
    cfg_flag = ["--config", args.config] if args.config else []
    py = sys.executable
    results = []

    print(f"models : {models}")
    print(f"regimes: {regimes}")
    print(f"resume : {'off (--force)' if args.force else 'on'}")

    if args.with_inspect:
        results.append(run_step("inspect_dataset",
                                [py, HERE / "inspect_dataset.py", *cfg_flag],
                                args.dry_run))

    for regime in regimes:
        for model in models:
            done = predictions_path(cfg, model, regime, 0).exists()
            if done and not args.force:
                print(f"⏭  skip {model}/{regime} (predictions already exist)")
                results.append({"step": f"{model}/{regime}",
                                "status": "skipped", "seconds": 0})
                continue
            cmd = [py, HERE / "run_experiment.py", *cfg_flag,
                   "--model", model, "--regime", regime]
            artifact = (checkpoint_artifact_dir(cfg, model, regime)
                        if regime in TRAINING_REGIMES else None)
            if regime in TRAINING_REGIMES and (args.skip_train or artifact):
                cmd.append("--skip-train")
            if args.limit:
                cmd += ["--limit", str(args.limit)]
            results.append(run_step(f"{model}/{regime}", cmd, args.dry_run))

        # ensemble only makes sense once all three members exist
        members = list(cfg.ensemble.small_models) + [cfg.ensemble.tiebreaker]
        have = [m for m in members
                if predictions_path(cfg, m, regime, 0).exists()]
        if len(have) < len(members) and not args.dry_run:
            print(f"⏭  skip ensemble/{regime}: missing "
                  f"{sorted(set(members) - set(have))}")
            results.append({"step": f"ensemble/{regime}",
                            "status": "skipped(incomplete)", "seconds": 0})
        elif (predictions_path(cfg, "ensemble", regime, 0).exists()
              and not args.force):
            print(f"⏭  skip ensemble/{regime} (already exists)")
            results.append({"step": f"ensemble/{regime}",
                            "status": "skipped", "seconds": 0})
        else:
            results.append(run_step(
                f"ensemble/{regime}",
                [py, HERE / "run_ensemble.py", *cfg_flag, "--regime", regime],
                args.dry_run))

    if args.with_reliability:
        reg = cfg.study3.get("regime", "lora")
        n_runs = int(cfg.study3.n_runs)
        for model in models:
            expected = [reliability_path(cfg, model, reg, i)
                        for i in range(n_runs)]
            if all(p.exists() for p in expected) and not args.force:
                print(f"⏭  skip {model}/reliability (all {n_runs} runs exist)")
                results.append({"step": f"{model}/reliability",
                                "status": "skipped", "seconds": 0})
                continue
            if (checkpoint_artifact_dir(cfg, model, reg) is None
                    and not args.dry_run):
                print(f"⏭  skip {model}/reliability: no adapter, "
                      f"run the lora regime first")
                results.append({"step": f"{model}/reliability",
                                "status": "skipped(no adapter)", "seconds": 0})
                continue
            results.append(run_step(
                f"{model}/reliability",
                [py, HERE / "run_experiment.py", *cfg_flag, "--model", model,
                 "--regime", reg, "--skip-train", "--reliability"],
                args.dry_run))

    if args.with_studies:
        for script, extra in [("study1_accuracy.py", []),
                              ("study2_human_agreement.py", []),
                              ("study3_reliability.py", []),
                              ("study4_fairness_error.py", []),
                              ("study5_xai_eval.py", ["--stage", "all"])]:
            results.append(run_step(script,
                                    [py, HERE / script, *cfg_flag, *extra],
                                    args.dry_run))

    # ---------------- summary ----------------
    print(f"\n{'=' * 78}\nSWEEP SUMMARY\n{'=' * 78}")
    width = max(len(r["step"]) for r in results) if results else 10
    for r in results:
        mark = {"ok": "✓", "skipped": "⏭",
                "planned": "·"}.get(r["status"].split("(")[0], "✗")
        print(f"  {mark} {r['step']:<{width}}  {r['status']:<22} "
              f"{_fmt(r['seconds'])}")
    total = sum(r["seconds"] for r in results)
    failed = [r["step"] for r in results if r["status"].startswith("failed")]
    print(f"\n  total time: {_fmt(total)}")
    print(f"  failures  : {failed or 'none'}")

    if not args.dry_run:
        out = output_root(cfg)
        out.mkdir(parents=True, exist_ok=True)
        (out / "sweep_summary.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8")
        try:
            import pandas as pd
            pd.DataFrame(results).to_csv(out / "sweep_summary.csv", index=False)
        except Exception:
            pass
        print(f"  summary   : {out / 'sweep_summary.json'}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
