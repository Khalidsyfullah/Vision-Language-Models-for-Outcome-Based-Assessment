"""Agreement / accuracy / reliability metrics shared by all studies.

Torch-free: only numpy / scipy / sklearn, so studies run on any machine.

Kappa-family metrics use `scoring.to_grid_labels` (default 0.05-mark grid)
rather than np.rint. The gold scores include half marks (0.5, 1.5, 2.5) and
np.rint applies banker's rounding — rint(0.5)=0, rint(2.5)=2 — which silently
distorted ~250 rows of the released data.
"""

from __future__ import annotations

import warnings
from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

from .scoring import DEFAULT_RESOLUTION, normalize, to_grid_labels


def _valid(records: list[dict]) -> list[dict]:
    out = []
    for r in records:
        try:
            pred = float(r.get("pred_score"))
            true = float(r.get("true_score"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(pred) and np.isfinite(true):
            out.append(r)
    return out


def safe_kappa(a: np.ndarray, b: np.ndarray,
               weights: str | None = "quadratic") -> float | None:
    """Cohen's kappa robust to degenerate label sets (returns None, not NaN)."""
    if len(a) < 2:
        return None
    if len(set(a.tolist())) < 2 and len(set(b.tolist())) < 2:
        return None
    observed = sorted(set(a.tolist()) | set(b.tolist()))
    # Include unobserved intermediate grid points. sklearn's weighted kappa
    # weights category *positions*, so using only observed labels compresses a
    # two-mark gap when intermediate score categories happen to be absent.
    labels = list(range(int(min(observed)), int(max(observed)) + 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            v = cohen_kappa_score(a, b, labels=labels, weights=weights)
        except Exception:
            return None
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return None
    return float(v)


def agreement_metrics(y_a: np.ndarray, y_b: np.ndarray,
                      adjacent_tol: float = 1.0,
                      resolution: float = DEFAULT_RESOLUTION) -> dict:
    """Full agreement panel between two raters (AI-human or human-human)."""
    y_a = np.asarray(y_a, dtype=float)
    y_b = np.asarray(y_b, dtype=float)
    a_i = to_grid_labels(y_a, resolution)
    b_i = to_grid_labels(y_b, resolution)
    out = {
        "n": int(len(y_a)),
        "qwk": safe_kappa(a_i, b_i, "quadratic"),
        "linear_kappa": safe_kappa(a_i, b_i, "linear"),
        "cohen_kappa": safe_kappa(a_i, b_i, None),
        "mae": float(np.mean(np.abs(y_a - y_b))),
        "rmse": float(np.sqrt(np.mean((y_a - y_b) ** 2))),
        "exact_agreement": float(np.mean(a_i == b_i)),
        "adjacent_agreement": float(np.mean(np.abs(y_a - y_b) <= adjacent_tol)),
        "bias": float(np.mean(y_a - y_b)),
        "pearson": None, "spearman": None,
    }
    if np.std(y_a) > 0 and np.std(y_b) > 0:
        out["pearson"] = float(pearsonr(y_a, y_b)[0])
        out["spearman"] = float(spearmanr(y_a, y_b)[0])
    return out


def compute_metrics(records: list[dict], adjacent_tol: float = 1.0,
                    resolution: float = DEFAULT_RESOLUTION) -> dict:
    """Accuracy panel of a prediction file vs gold scores (Study 1).

    Adds normalised-scale metrics (score / criterion max) so criteria with
    different maxima (0.5 .. 4 marks) are comparable in aggregate.
    """
    valid = _valid(records)
    out = {"n_total": len(records), "n_valid": len(valid),
           "parse_rate": len(valid) / max(1, len(records))}
    if not valid:
        return out
    yt = np.array([r["true_score"] for r in valid], dtype=float)
    yp = np.array([r["pred_score"] for r in valid], dtype=float)
    out.update(agreement_metrics(yp, yt, adjacent_tol, resolution))
    out["exact_match"] = out.pop("exact_agreement")
    out["adjacent_match"] = out.pop("adjacent_agreement")
    out.pop("n", None)

    maxima = np.array([r.get("max_score") or np.nan for r in valid],
                      dtype=float)
    if np.isfinite(maxima).any():
        nt, npd = normalize(yt, maxima), normalize(yp, maxima)
        ok = np.isfinite(nt) & np.isfinite(npd)
        if ok.sum() > 1:
            out["mae_normalized"] = float(np.mean(np.abs(npd[ok] - nt[ok])))
            out["rmse_normalized"] = float(
                np.sqrt(np.mean((npd[ok] - nt[ok]) ** 2)))
        # rate of predictions above the criterion maximum = spec violations
        out["over_max_rate"] = float(
            np.mean(yp[np.isfinite(maxima)] > maxima[np.isfinite(maxima)]))
    return out


def per_criterion_metrics(records: list[dict], adjacent_tol: float = 1.0,
                          resolution: float = DEFAULT_RESOLUTION
                          ) -> dict[str, dict]:
    by = defaultdict(list)
    for r in records:
        by[r.get("criterion_name", "unknown")].append(r)
    return {k: compute_metrics(v, adjacent_tol, resolution)
            for k, v in by.items()}


def answer_level_metrics(records: list[dict]) -> dict:
    """Aggregate criteria up to whole answers (what a student actually
    receives). Total mark per answer = sum of its criterion scores."""
    by_answer: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_answer[str(r["answer_id"])].append(r)

    complete: dict[str, list[dict]] = {}
    for answer_id, rows in by_answer.items():
        if len(_valid(rows)) == len(rows):
            complete[answer_id] = rows

    ids = list(complete)
    base = {
        "n_answers_total": len(by_answer),
        "n_answers_complete": len(ids),
        "answer_parse_rate": len(ids) / max(1, len(by_answer)),
        # Backwards-compatible name: it now unambiguously means complete.
        "n_answers": len(ids),
    }
    if len(ids) < 2:
        return base
    yp = np.array([sum(float(r["pred_score"]) for r in complete[a])
                   for a in ids])
    yt = np.array([sum(float(r["true_score"]) for r in complete[a])
                   for a in ids])
    return {**base,
        "answer_mae": float(np.mean(np.abs(yp - yt))),
        "answer_rmse": float(np.sqrt(np.mean((yp - yt) ** 2))),
        "answer_bias": float(np.mean(yp - yt)),
        "answer_pearson": (float(pearsonr(yt, yp)[0])
                           if np.std(yt) > 0 and np.std(yp) > 0 else None),
        "answer_within_1": float(np.mean(np.abs(yp - yt) <= 1.0)),
    }


# --------------------------------------------------------------------------
# Reliability (Study 3)
# --------------------------------------------------------------------------

def icc_2_1(mat: np.ndarray) -> float | None:
    """ICC(2,1) two-way random, absolute agreement.
    mat: items x raters (here raters = repeated runs)."""
    n, k = mat.shape
    if n < 2 or k < 2:
        return None
    mean_rows = mat.mean(axis=1, keepdims=True)
    mean_cols = mat.mean(axis=0, keepdims=True)
    grand = mat.mean()
    ss_rows = k * ((mean_rows - grand) ** 2).sum()
    ss_cols = n * ((mean_cols - grand) ** 2).sum()
    ss_err = ((mat - mean_rows - mean_cols + grand) ** 2).sum()
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))
    denom = ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n
    if denom <= 0:
        return None
    return float((ms_rows - ms_err) / denom)


def reliability_metrics(run_matrix: np.ndarray,
                        resolution: float = DEFAULT_RESOLUTION) -> dict:
    """run_matrix: items x runs (NaN where a run failed to parse)."""
    if run_matrix.ndim != 2 or run_matrix.shape[1] < 2:
        return {"n_items_complete": 0, "n_runs": int(run_matrix.shape[1])
                if run_matrix.ndim == 2 else 0}
    complete = run_matrix[~np.isnan(run_matrix).any(axis=1)]
    n_items, n_runs = complete.shape
    out = {"n_items_complete": int(n_items), "n_runs": int(n_runs)}
    if n_items < 2:
        return out
    out["mean_item_std"] = float(complete.std(axis=1, ddof=1).mean())
    out["mean_item_range"] = float(
        (complete.max(axis=1) - complete.min(axis=1)).mean())
    out["pct_items_identical"] = float((complete.std(axis=1) == 0).mean())
    out["icc_2_1"] = icc_2_1(complete)
    qwks, pearsons = [], []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            a, b = complete[:, i], complete[:, j]
            q = safe_kappa(to_grid_labels(a, resolution),
                           to_grid_labels(b, resolution))
            if q is not None:
                qwks.append(q)
            if np.std(a) > 0 and np.std(b) > 0:
                pearsons.append(pearsonr(a, b)[0])
    out["mean_pairwise_qwk"] = float(np.mean(qwks)) if qwks else None
    out["mean_pairwise_pearson"] = (float(np.mean(pearsons))
                                    if pearsons else None)
    return out


# --------------------------------------------------------------------------
# Bootstrap CIs (Study 2)
# --------------------------------------------------------------------------

def bootstrap_ci(y_a: np.ndarray, y_b: np.ndarray, stat: str = "qwk",
                 n_boot: int = 2000, seed: int = 42, alpha: float = 0.05,
                 resolution: float = DEFAULT_RESOLUTION
                 ) -> tuple[float | None, float | None]:
    rng = np.random.default_rng(seed)
    y_a, y_b = np.asarray(y_a, float), np.asarray(y_b, float)
    n = len(y_a)
    if n < 2:
        return None, None
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a, b = y_a[idx], y_b[idx]
        if stat == "qwk":
            v = safe_kappa(to_grid_labels(a, resolution),
                           to_grid_labels(b, resolution))
        elif stat == "pearson":
            v = (pearsonr(a, b)[0]
                 if np.std(a) > 0 and np.std(b) > 0 else None)
        elif stat == "mae":
            v = float(np.mean(np.abs(a - b)))
        else:
            raise ValueError(stat)
        if v is not None:
            vals.append(v)
    if not vals:
        return None, None
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
