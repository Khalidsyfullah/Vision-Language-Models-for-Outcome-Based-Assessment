#!/usr/bin/env python
"""Study 5: XAI explanation images, faithfulness, and teacher survey.

Stages (--stage, default all):
  heatmaps      GPU. Score-conditioned attention images (PNG) for
                study5.n_heatmap_samples test items, using
                study5.xai_model / xai_regime at study5.xai_max_image_side.
  faithfulness  GPU. Deletion-based faithfulness: masking the top-attended
                regions should shift the score more than masking random
                regions. Aggregated per mask fraction, with a paired t-test.
  survey        CPU. (a) builds a teacher-survey workbook pairing sampled
                answers/justifications/heatmaps with 5-point Likert items;
                items are drawn from the heatmaps actually produced, so every
                survey row references an existing figure. (b) analyses the
                completed survey CSV if paths.teacher_survey_csv exists.

    python scripts/study5_xai_eval.py --stage heatmaps
    python scripts/study5_xai_eval.py --stage faithfulness
    python scripts/study5_xai_eval.py --stage survey

Outputs: results/studies/study5/
    heatmaps/*.png + xai_index.json
    faithfulness.{csv,json}, faithfulness_by_fraction.{csv,json}
    teacher_survey_template.{csv,json}
    teacher_survey_analysis.{csv,json}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import (checkpoint_artifact_dir, checkpoint_dir, data_path,
                        load_config, predictions_path, set_seed, study_dir)
from src.io_utils import read_records, save_json, save_table

LIKERT_QUESTIONS = [
    ("trust_score", "I trust the score this AI assigned."),
    ("explanation_clear", "The highlighted regions / justification make it "
                          "clear WHY this score was given."),
    ("explanation_faithful", "The highlighted regions match what I would "
                             "look at when grading this criterion."),
    ("useful_workflow", "This explanation would be useful in my own "
                        "grading workflow."),
    ("would_adopt", "I would accept AI pre-grading with this kind of "
                    "explanation, with teacher override."),
]


def _load_xai_model(cfg):
    from src.models import load_model_and_processor

    model_key = cfg.study5.xai_model
    regime = cfg.study5.xai_regime
    ck = checkpoint_dir(cfg, model_key, regime)
    if regime == "lora":
        artifact = checkpoint_artifact_dir(cfg, model_key, regime)
        if artifact is not None:
            return load_model_and_processor(cfg, model_key,
                                            adapter_dir=str(artifact))
        raise FileNotFoundError(
            f"Study 5 is configured for lora but no adapter exists at {ck}. "
            f"Run the {model_key}/lora experiment first or set "
            f"study5.xai_regime to zero_shot explicitly.")
    elif regime == "full":
        artifact = checkpoint_artifact_dir(cfg, model_key, regime)
        if artifact is not None:
            return load_model_and_processor(cfg, model_key,
                                            full_checkpoint=str(artifact),
                                            quantize=False)
        raise FileNotFoundError(
            f"Study 5 is configured for full but no checkpoint exists at {ck}. "
            f"Run the {model_key}/full experiment first or set "
            f"study5.xai_regime to zero_shot explicitly.")
    elif regime == "few_shot":
        raise ValueError(
            "Study 5 few-shot XAI is not implemented because explanation "
            "inputs must include and spatially map every exemplar image. Use "
            "zero_shot, lora, or full explicitly.")
    elif regime != "zero_shot":
        raise ValueError(f"unsupported Study 5 XAI regime: {regime}")
    return load_model_and_processor(cfg, model_key)


def _test_dataset(cfg):
    from src.data import GradingDataset, load_splits
    from src.scoring import learn_criterion_maxima

    side = int(cfg.study5.get("xai_max_image_side",
                              cfg.dataset.max_image_long_side))
    splits = load_splits(cfg)
    upper = float(cfg.dataset.get("max_criterion_score", 4.0))
    maxima = learn_criterion_maxima(splits["train"], upper)
    return GradingDataset(
        splits["test"], side, bool(cfg.dataset.get("cache_images", True)),
        maxima, upper)


def stage_heatmaps(cfg, out: Path) -> None:
    from src.xai import explain_samples

    from src.models import is_donut

    processor, model = _load_xai_model(cfg)
    explain_samples(cfg, processor, model, _test_dataset(cfg),
                    out / "heatmaps", int(cfg.study5.n_heatmap_samples),
                    donut=is_donut(cfg, cfg.study5.xai_model))


def stage_faithfulness(cfg, out: Path) -> None:
    from src.xai import faithfulness_eval

    from src.models import is_donut

    processor, model = _load_xai_model(cfg)
    rows = faithfulness_eval(cfg, processor, model, _test_dataset(cfg), out,
                             donut=is_donut(cfg, cfg.study5.xai_model))
    if not rows:
        print("[study5] no faithfulness rows produced")
        return
    df = pd.DataFrame(rows)
    save_table(df, out / "faithfulness")

    agg = (df.groupby("mask_fraction")
           .agg(n=("faithful", "size"),
                comprehensiveness=("comprehensiveness", "mean"),
                random_effect=("random_effect", "mean"),
                faithful_rate=("faithful", "mean"))
           .reset_index())
    # paired test: is masking the attended region worse than masking at random?
    from scipy.stats import ttest_rel, wilcoxon
    pvals, stats = [], []
    for frac in agg["mask_fraction"]:
        d = df[(df["mask_fraction"] == frac)].dropna(
            subset=["comprehensiveness", "random_effect"])
        if len(d) >= 5 and (d["comprehensiveness"] - d["random_effect"]).std() > 0:
            t, p = ttest_rel(d["comprehensiveness"], d["random_effect"])
            try:
                _, pw = wilcoxon(d["comprehensiveness"], d["random_effect"])
            except Exception:
                pw = None
            stats.append(float(t)); pvals.append({"t_p": float(p),
                                                  "wilcoxon_p": pw})
        else:
            stats.append(None); pvals.append({"t_p": None, "wilcoxon_p": None})
    agg["t_stat"] = stats
    agg["t_pvalue"] = [p["t_p"] for p in pvals]
    agg["wilcoxon_pvalue"] = [p["wilcoxon_p"] for p in pvals]
    save_table(agg, out / "faithfulness_by_fraction")


def stage_survey(cfg, out: Path) -> None:
    scfg = cfg.study5.survey
    model_key, regime = cfg.study5.xai_model, cfg.study5.xai_regime

    # ---- (a) build the template ----
    # Prefer items that actually have a heatmap figure, so every survey row
    # points at a real image.
    index_path = out / "heatmaps" / "xai_index.json"
    items_src: list[dict] = []
    if index_path.exists():
        items_src = json.loads(index_path.read_text(encoding="utf-8"))
        print(f"[study5] using {len(items_src)} items that have heatmaps")
    else:
        p = predictions_path(cfg, model_key, regime, 0)
        if p.exists():
            items_src = [{"example_id": r["example_id"],
                          "criterion_name": r["criterion_name"],
                          "pred_score": r["pred_score"],
                          "true_score": r["true_score"],
                          "max_score": r.get("max_score"),
                          "model_output": r.get("justification", ""),
                          "figure": ""}
                         for r in read_records(p)
                         if r.get("pred_score") is not None]
            print(f"[study5] no heatmaps yet — sampling from {p.name}; "
                  f"run --stage heatmaps first to attach figures")
        else:
            print(f"[study5] neither heatmaps nor {p} found — "
                  f"nothing to build a survey from")
            return

    rng = np.random.default_rng(cfg.seed)
    n = min(int(scfg.n_items), len(items_src))
    picks = rng.choice(len(items_src), n, replace=False)
    rows = []
    for k, i in enumerate(picks.tolist(), start=1):
        s = items_src[i]
        base = {
            "item": k,
            "example_id": s["example_id"],
            "criterion_name": s.get("criterion_name", ""),
            "ai_score": s.get("pred_score"),
            "gold_score": s.get("true_score"),
            "max_score": s.get("max_score"),
            "ai_output": str(s.get("model_output", ""))[:300],
            "heatmap_file": (f"heatmaps/{s['example_id']}.png"
                             if s.get("figure") else ""),
        }
        for qkey, qtext in LIKERT_QUESTIONS:
            rows.append({**base, "question_key": qkey, "question": qtext,
                         "teacher_id": "",
                         f"rating_1_to_{scfg.likert_max}": ""})
    save_table(pd.DataFrame(rows), out / "teacher_survey_template")
    print(f"[study5] survey template: {n} items x {len(LIKERT_QUESTIONS)} "
          f"questions. Distribute to 5-10 educators, collect responses into "
          f"{data_path(cfg, 'teacher_survey_csv')} (fill teacher_id + rating), "
          f"then rerun --stage survey.")

    # ---- (b) analyse a completed survey ----
    resp_path = data_path(cfg, "teacher_survey_csv")
    if not resp_path.exists():
        return
    resp = pd.read_csv(resp_path)
    rating_col = next((c for c in resp.columns if c.startswith("rating")), None)
    if rating_col is None or "question_key" not in resp.columns:
        print(f"[study5] {resp_path} lacks rating/question_key columns")
        return
    resp[rating_col] = pd.to_numeric(resp[rating_col], errors="coerce")
    resp = resp.dropna(subset=[rating_col])
    if resp.empty:
        print(f"[study5] {resp_path} has no numeric ratings yet")
        return

    agg = (resp.groupby("question_key")[rating_col]
           .agg(["count", "mean", "std", "median"]).reset_index())
    agg["pct_favourable"] = [
        float((resp[resp.question_key == q][rating_col] >= 4).mean())
        for q in agg["question_key"]]
    save_table(agg, out / "teacher_survey_analysis")

    if "teacher_id" in resp.columns and resp["teacher_id"].notna().any():
        per_t = (resp.groupby(["teacher_id", "question_key"])[rating_col]
                 .mean().unstack().reset_index())
        save_table(per_t, out / "teacher_survey_per_teacher")
        # inter-rater consistency across teachers (Krippendorff-style ICC)
        from src.metrics import icc_2_1
        wide = (resp.pivot_table(index=["example_id", "question_key"],
                                 columns="teacher_id", values=rating_col)
                .dropna())
        if wide.shape[0] > 1 and wide.shape[1] > 1:
            save_json({"n_items": int(wide.shape[0]),
                       "n_teachers": int(wide.shape[1]),
                       "icc_2_1": icc_2_1(wide.values)},
                      out / "teacher_survey_agreement.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--stage", default="all",
                    choices=["all", "heatmaps", "faithfulness", "survey"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    out = study_dir(cfg, 5)
    out.mkdir(parents=True, exist_ok=True)

    if args.stage in ("all", "heatmaps"):
        stage_heatmaps(cfg, out)
    if args.stage in ("all", "faithfulness"):
        stage_faithfulness(cfg, out)
    if args.stage in ("all", "survey"):
        stage_survey(cfg, out)

    save_json({"xai_model": cfg.study5.xai_model,
               "xai_regime": cfg.study5.xai_regime,
               "stage": args.stage}, out / "study5_summary.json")
    print(f"\n✓ Study 5 ({args.stage}) done -> {out}")


if __name__ == "__main__":
    main()
