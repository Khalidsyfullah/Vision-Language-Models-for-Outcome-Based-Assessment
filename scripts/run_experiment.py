#!/usr/bin/env python
"""Run ONE model in ONE regime on the test set.

Models  : qwen25vl | internvl3 | pixtral (chat VLMs) | donut (OCR-free)
Regimes : zero_shot | few_shot | lora | full

Examples
    python scripts/run_experiment.py --model qwen25vl --regime zero_shot
    python scripts/run_experiment.py --model internvl3 --regime few_shot
    python scripts/run_experiment.py --model pixtral --regime lora
    python scripts/run_experiment.py --model donut --regime full
    # Study 3 repeated runs (stochastic decoding, separate folder):
    python scripts/run_experiment.py --model qwen25vl --regime lora \
        --skip-train --reliability

Notes
    * `lora` and `full` train weights; checkpoints live in
      results/checkpoints/<model>/<regime>/ so they never overwrite each other.
    * `full` trains the top ~30% of complete language blocks for chat VLMs and
      all parameters for Donut. It cannot start from quantized weights and
      checks the selected parameter/optimizer memory before training.
    * donut is an encoder-decoder with one image input and no in-context
      learning, so `few_shot` runs the same prompt as `zero_shot` for it.

Outputs (results/runs/<model>/<regime>/):
    predictions_run0.json, metrics.{csv,json}
    reliability/predictions_run{0..n-1}.json   (with --reliability)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import (TRAINING_REGIMES, Config, checkpoint_artifact_dir,
                        checkpoint_dir, load_config, predictions_path,
                        reliability_path, run_dir, set_seed)
from src.data import GradingDataset, load_splits, select_few_shot_exemplars
from src.inference import run_inference
from src.io_utils import save_json, save_table
from src.metrics import answer_level_metrics, compute_metrics, per_criterion_metrics
from src.models import load_model_and_processor
from src.scoring import learn_criterion_maxima


def build_datasets(cfg: Config, model_key: str):
    dset = load_splits(cfg)
    mcfg = Config(cfg.models[model_key])
    side = mcfg.get("max_image_long_side", cfg.dataset.max_image_long_side)
    cache = bool(cfg.dataset.get("cache_images", True))
    upper = float(cfg.dataset.get("max_criterion_score", 4.0))
    # This is the only label-derived score-scale lookup. It is learned once
    # from TRAIN and reused unchanged for validation and test.
    maxima = learn_criterion_maxima(dset["train"], upper)
    wrapped = tuple(GradingDataset(dset[s], side, cache, maxima, upper)
                    for s in ("train", "validation", "test"))
    return dset, wrapped, maxima


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", required=True,
                    help="model key from config.yaml (e.g. qwen25vl)")
    ap.add_argument("--regime", required=True,
                    choices=["zero_shot", "few_shot", "lora", "full"])
    ap.add_argument("--reliability", action="store_true",
                    help="Study 3 mode: study3.n_runs stochastic runs written "
                         "to <run_dir>/reliability/ (run0 is left untouched)")
    ap.add_argument("--skip-train", action="store_true",
                    help="lora/full: reuse the existing complete checkpoint")
    ap.add_argument("--limit", type=int, default=None,
                    help="debug: only evaluate the first N test items")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    out = run_dir(cfg, args.model, args.regime)
    out.mkdir(parents=True, exist_ok=True)

    dset, (train_ds, val_ds, test_ds), criterion_maxima = build_datasets(
        cfg, args.model)
    if args.limit:
        from torch.utils.data import Subset
        test_ds = Subset(test_ds, range(min(args.limit, len(test_ds))))
        print(f"[debug] limiting test set to {len(test_ds)} items")

    exemplars = None
    if args.regime == "few_shot":
        f = cfg.few_shot
        exemplars = select_few_shot_exemplars(
            dset["train"], n=f.n_exemplars, seed=f.exemplar_seed,
            stratify=f.stratify_by_score, max_side=f.max_exemplar_image_side,
            criterion_maxima=criterion_maxima,
            max_criterion_score=float(
                cfg.dataset.get("max_criterion_score", 4.0)))
        print(f"few-shot: {len(exemplars)} exemplars")

    trains = args.regime in TRAINING_REGIMES
    ckpt = checkpoint_dir(cfg, args.model, args.regime)
    # Resource-aware full training cannot update quantized base tensors.
    quantize = False if args.regime == "full" else None

    if trains and args.skip_train:
        artifact = checkpoint_artifact_dir(
            cfg, args.model, args.regime)
        if artifact is None:
            required = ("adapter_config.json plus adapter_model weights"
                        if args.regime == "lora" else
                        "config.json plus model safetensors/bin weights")
            raise SystemExit(
                f"--skip-train given but no complete {args.regime} checkpoint "
                f"exists under {ckpt}. A directory alone is not a checkpoint; "
                f"{args.regime} requires {required}.")
        if artifact != ckpt:
            print(f"  using latest complete Trainer checkpoint: {artifact}")
        if args.regime == "lora":
            processor, model = load_model_and_processor(
                cfg, args.model, adapter_dir=str(artifact))
        else:
            processor, model = load_model_and_processor(
                cfg, args.model, full_checkpoint=str(artifact), quantize=False)
    else:
        processor, model = load_model_and_processor(cfg, args.model,
                                                    quantize=quantize)
        if trains:
            from src.training import fine_tune
            model = fine_tune(cfg, args.model, processor, model,
                              train_ds, val_ds, ckpt, regime=args.regime)

    # ---- inference ----
    if args.reliability:
        s3 = cfg.study3
        for i in range(int(s3.n_runs)):
            run_inference(cfg, args.model, args.regime, processor, model,
                          test_ds,
                          reliability_path(cfg, args.model, args.regime, i),
                          exemplars=exemplars, do_sample=s3.do_sample,
                          temperature=s3.temperature, top_p=s3.top_p,
                          seed=cfg.seed + i)
        print(f"\n✓ Reliability runs -> "
              f"{reliability_path(cfg, args.model, args.regime, 0).parent}")
        return

    preds = run_inference(cfg, args.model, args.regime, processor, model,
                          test_ds,
                          predictions_path(cfg, args.model, args.regime, 0),
                          exemplars=exemplars, seed=cfg.seed)

    # ---- metrics ----
    tol = cfg.study1.adjacent_tolerance
    res = float(cfg.dataset.get("score_resolution", 0.05))
    overall = {**compute_metrics(preds, tol, res), **answer_level_metrics(preds)}
    per_crit = per_criterion_metrics(preds, tol, res)
    training_info = None
    if args.regime == "full":
        metadata_path = ckpt / "obe_training_metadata.json"
        if metadata_path.is_file():
            training_info = json.loads(metadata_path.read_text(encoding="utf-8"))
    save_json({"overall": overall, "per_criterion": per_crit,
               "training": training_info},
              out / "metrics.json")
    training_columns = ({
        "training_strategy": training_info.get("strategy"),
        "requested_trainable_fraction": training_info.get(
            "requested_fraction"),
        "actual_trainable_fraction": training_info.get("actual_fraction"),
        "trainable_parameters": training_info.get("trainable_parameters"),
    } if training_info else {})
    save_table(pd.DataFrame([{"model": args.model, "regime": args.regime,
                              **training_columns, **overall}]), out / "metrics")
    save_table(pd.DataFrame([{"criterion": k, **v}
                             for k, v in per_crit.items()]),
               out / "metrics_per_criterion")
    if cfg.hub.push_adapters and args.regime == "lora":
        from src.hub import push_adapter
        push_adapter(ckpt, cfg.hub.models_repo,
                     Config(cfg.models[args.model]).model_id, args.model)
    if cfg.hub.push_results:
        from src.hub import push_results
        push_results(out, cfg.hub.results_repo, f"{args.model}/{args.regime}")

    print(f"\n✓ Done: {args.model}/{args.regime} -> {out}")


if __name__ == "__main__":
    main()
