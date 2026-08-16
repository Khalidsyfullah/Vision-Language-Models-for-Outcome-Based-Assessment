"""Cascaded ensemble ("combined" setup, as in notebook Module 4).

Step 1 — the two SMALLER models predict.
Step 2 — if they agree within `disagree_threshold`, final = their mean.
Step 3 — otherwise escalate: final = mean of all three (tiebreaker joins).

Clamping uses the REPAIRED criterion maximum (src/scoring), not the dataset's
unreliable `criterion_max`.
"""

from __future__ import annotations

import math

from .config import Config
from .scoring import clamp, snap_to_grid


def _key(r: dict) -> tuple:
    return (r["example_id"], r["criterion_index"])


def _index(rows: list[dict]) -> dict:
    return {_key(r): r for r in rows}


def _prediction(row: dict | None) -> float | None:
    if not row:
        return None
    try:
        value = float(row.get("pred_score"))
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def cascade(cfg: Config, preds_by_model: dict[str, list[dict]],
            regime: str = "lora") -> tuple[list[dict], dict]:
    """preds_by_model: {model_key: prediction rows}. Returns (rows, stats)."""
    ecfg = cfg.ensemble
    small = list(ecfg.small_models)
    if len(small) != 2:
        raise ValueError("ensemble.small_models must list exactly 2 models")
    small1, small2 = small
    tb = ecfg.tiebreaker
    thr = float(ecfg.disagree_threshold)
    do_snap = bool(ecfg.get("snap_to_grid", True))
    resolution = float(ecfg.get("score_resolution", 0.25))

    missing = [m for m in (small1, small2, tb) if m not in preds_by_model]
    if missing:
        raise KeyError(f"missing predictions for: {missing}")

    m1 = _index(preds_by_model[small1])
    m2 = _index(preds_by_model[small2])
    m3 = _index(preds_by_model[tb])
    all_keys = sorted(set(m1) | set(m2) | set(m3))

    rows = []
    stats = {"two_model": 0, "three_model": 0,
             "single_model": 0, "dropped": 0}

    for key in all_keys:
        r1, r2, r3 = m1.get(key), m2.get(key), m3.get(key)
        ref = r1 or r2 or r3
        p1, p2, p3 = _prediction(r1), _prediction(r2), _prediction(r3)

        if p1 is not None and p2 is not None:
            if abs(p1 - p2) <= thr:
                final, used, route = (p1 + p2) / 2, [small1, small2], "two_model"
            elif p3 is not None:
                final = (p1 + p2 + p3) / 3
                used, route = [small1, small2, tb], "three_model"
            else:
                final, used, route = (p1 + p2) / 2, [small1, small2], "two_model"
        else:
            avail = [(p, m) for p, m in
                     ((p1, small1), (p2, small2), (p3, tb)) if p is not None]
            if not avail:
                stats["dropped"] += 1
                continue
            final = sum(p for p, _ in avail) / len(avail)
            used = [m for _, m in avail]
            route = "single_model" if len(avail) == 1 else "two_model"
        stats[route] += 1

        # prefer any row that carries the repaired maximum
        max_score = 0.0
        for r in (r1, r2, r3):
            if r:
                try:
                    candidate = float(r.get("max_score") or 0)
                except (TypeError, ValueError, OverflowError):
                    continue
                if math.isfinite(candidate) and candidate > 0:
                    max_score = candidate
                    break
        if ecfg.clamp_to_max:
            final = clamp(final, max_score)
        if do_snap:
            final = float(snap_to_grid([final], resolution)[0])

        rows.append({
            "example_id": ref["example_id"],
            "answer_id": ref["answer_id"],
            "criterion_name": ref["criterion_name"],
            "criterion_index": ref["criterion_index"],
            "model": "ensemble",
            "regime": regime,
            "pred_score": float(final),
            "true_score": ref["true_score"],
            "max_score": max_score,
            "route": route,
            "models_used": "+".join(used),
            f"pred_{small1}": p1,
            f"pred_{small2}": p2,
            f"pred_{tb}": p3,
            "parse_ok": True,
        })

    total = sum(stats.values()) or 1
    print(f"  ensemble routes: {stats} "
          f"(escalation rate {stats['three_model'] / total:.1%})")
    return rows, stats
