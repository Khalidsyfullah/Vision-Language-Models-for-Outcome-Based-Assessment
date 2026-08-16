"""Score-scale utilities.

DEFECT 1 — `criterion_max` is unreliable.
    In the released dataset 38/1982 rows have gained_marks > criterion_max, and
    350 rows carry criterion_max=16.0 (the whole question's mark, not the
    criterion's). `resolve_criterion_max` recovers the true per-criterion
    maximum, in priority order:
        1. an exact criterion-identity lookup learned from TRAIN
        2. the largest value in parentheses among the rubric level keys
           e.g. {"Excellent (2.5)", "Good (2)", ...}  -> 2.5
        3. a mark in parentheses in the criterion name
           e.g. "Explanation of Layers with diagram (4 Marks)" -> 4
        4. an unambiguous TRAIN lookup for the same criterion name
        5. a plausible raw criterion_max

    Evaluation labels are deliberately never consulted while resolving a
    validation/test maximum. This matters because the resolved maximum is put
    in the model prompt and is used to clamp predictions.

DEFECT 2 — decimal scores + np.rint.
    Gold scores include half/quarter marks and 1.8. Whole-mark rounding loses
    that information, while a 0.25 grid still changes 1.8 to 1.75.
    np.rint uses banker's rounding: rint(0.5)=0, rint(2.5)=2, rint(1.5)=2 — so
    250+ half-mark rows were being silently distorted before kappa. Use
    `to_grid_labels` to map scores onto integer labels on the configured
    resolution (default 0.05) instead of rounding to whole marks.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

import numpy as np

# a mark written in parentheses: "(2.5)", "(4 Marks)", "(1 Mark)", "(3 pts)"
_PAREN_MARK = re.compile(r"\(\s*(\d+(?:\.\d+)?)\s*(?:marks?|pts?|points?)?\s*\)",
                         re.IGNORECASE)

DEFAULT_RESOLUTION = 0.05
DEFAULT_MAX_CRITERION_SCORE = 4.0

CriterionKey = tuple[str, int, str]


def _from_rubric(rubric_levels: str) -> float | None:
    try:
        levels = (rubric_levels if isinstance(rubric_levels, dict)
                  else json.loads(rubric_levels))
    except Exception:
        return None
    vals = []
    for key in levels:
        found = _PAREN_MARK.findall(str(key))
        if found:
            vals.append(float(found[-1]))
    return max(vals) if vals else None


def _from_name(criterion_name: str) -> float | None:
    found = _PAREN_MARK.findall(criterion_name or "")
    return float(found[-1]) if found else None


def criterion_key(ex: dict) -> CriterionKey:
    """Stable criterion identity.

    Criterion names are not guaranteed to be unique across questions, so the
    question and criterion index are part of the key.
    """
    try:
        idx = int(ex.get("criterion_index", -1))
    except (TypeError, ValueError):
        idx = -1
    question = " ".join(str(ex.get("question") or "").split()).casefold()
    name = " ".join(str(ex.get("criterion_name") or "").split()).casefold()
    return question, idx, name


def _pick_mode(values: list[float]) -> float | None:
    if not values:
        return None
    counts = Counter(round(float(v), 8) for v in values)
    # Deterministic tie-break: prefer the larger plausible maximum.
    return float(max(counts, key=lambda v: (counts[v], v)))


def learn_criterion_maxima(
        train_split,
        upper_bound: float = DEFAULT_MAX_CRITERION_SCORE,
) -> dict[CriterionKey, float]:
    """Learn fallback criterion maxima from the training split only.

    Textual rubric/name maxima are preferred. Plausible raw values are next;
    training labels are used only as a final TRAIN-only fallback. Wildcard
    name entries are added only when a name has one unambiguous scale.
    """
    upper_bound = float(upper_bound)
    if not np.isfinite(upper_bound) or upper_bound <= 0:
        raise ValueError("upper_bound must be a positive finite number")

    names = list(train_split["criterion_name"])
    n = len(names)

    def column(name: str, default):
        try:
            values = list(train_split[name])
        except (KeyError, TypeError):
            values = [default] * n
        if len(values) != n:
            raise ValueError(f"column {name!r} has {len(values)} rows; expected {n}")
        return values

    questions = column("question", "")
    indices = column("criterion_index", -1)
    rubrics = column("rubric_levels", "")
    raw_values = column("criterion_max", None)
    gained_values = column("gained_marks", None)

    trusted: dict[CriterionKey, list[float]] = defaultdict(list)
    plausible_raw: dict[CriterionKey, list[float]] = defaultdict(list)
    train_awards: dict[CriterionKey, list[float]] = defaultdict(list)
    keys: list[CriterionKey] = []

    for q, idx, name, rubric, raw, gained in zip(
            questions, indices, names, rubrics, raw_values, gained_values):
        ex = {"question": q, "criterion_index": idx,
              "criterion_name": name, "rubric_levels": rubric}
        key = criterion_key(ex)
        keys.append(key)
        text_max = (_from_rubric(rubric) or _from_name(str(name)))
        if text_max is not None and 0 < text_max <= upper_bound:
            trusted[key].append(float(text_max))
        try:
            raw_f = float(raw)
            if np.isfinite(raw_f) and 0 < raw_f <= upper_bound:
                plausible_raw[key].append(raw_f)
        except (TypeError, ValueError):
            pass
        try:
            gained_f = float(gained)
            if np.isfinite(gained_f) and 0 <= gained_f <= upper_bound:
                train_awards[key].append(gained_f)
        except (TypeError, ValueError):
            pass

    learned: dict[CriterionKey, float] = {}
    for key in dict.fromkeys(keys):
        min_required = max(train_awards[key], default=0.0)
        trusted_valid = [v for v in trusted[key] if v >= min_required]
        raw_valid = [v for v in plausible_raw[key] if v >= min_required]
        value = (_pick_mode(trusted_valid) or _pick_mode(raw_valid))
        if value is None and train_awards[key]:
            value = max(train_awards[key])
        learned[key] = float(value if value and value > 0 else upper_bound)

    # A wildcard is safe only if every exact occurrence of the same name has
    # the same learned scale.
    by_name: dict[str, set[float]] = defaultdict(set)
    for key, value in learned.items():
        by_name[key[2]].add(round(value, 8))
    for name, values in by_name.items():
        if len(values) == 1:
            learned[("", -1, name)] = float(next(iter(values)))
    return learned


def resolve_criterion_max(
        ex: dict,
        learned_max: dict[CriterionKey, float] | dict[str, float] | None = None,
        upper_bound: float = DEFAULT_MAX_CRITERION_SCORE,
) -> float:
    """Resolve a criterion maximum without reading this row's gold label."""
    upper_bound = float(upper_bound)
    if not np.isfinite(upper_bound) or upper_bound <= 0:
        raise ValueError("upper_bound must be a positive finite number")

    key = criterion_key(ex)
    if learned_max:
        candidate = learned_max.get(key)
        if candidate is not None:
            candidate = float(candidate)
            if np.isfinite(candidate) and 0 < candidate <= upper_bound:
                return candidate

    for candidate in (_from_rubric(ex.get("rubric_levels", "")),
                      _from_name(ex.get("criterion_name", ""))):
        if candidate is not None and 0 < candidate <= upper_bound:
            return float(candidate)

    if learned_max:
        wildcard = ("", -1, key[2])
        candidate = learned_max.get(wildcard)
        # Backwards compatibility for callers with old name-only mappings.
        if candidate is None:
            candidate = learned_max.get(key[2])  # type: ignore[arg-type]
        if candidate is None:
            candidate = learned_max.get(  # type: ignore[arg-type]
                str(ex.get("criterion_name") or ""))
        if candidate is not None:
            candidate = float(candidate)
            if np.isfinite(candidate) and 0 < candidate <= upper_bound:
                return candidate

    try:
        raw = float(ex.get("criterion_max"))
        if np.isfinite(raw) and 0 < raw <= upper_bound:
            return raw
    except (TypeError, ValueError):
        pass

    # A conservative, label-free fallback for an unseen malformed criterion.
    return upper_bound


def observed_max_by_criterion(split) -> dict[str, float]:
    """Highest TRAINING award per criterion name (legacy audit helper).

    Do not use this function to construct validation/test prompts or clamps;
    use :func:`learn_criterion_maxima` on the training split instead.
    """
    out: dict[str, float] = defaultdict(float)
    names = split["criterion_name"]
    marks = split["gained_marks"]
    for n, m in zip(names, marks):
        out[n] = max(out[n], float(m))
    return dict(out)


# --------------------------------------------------------------------------
# Score grid
# --------------------------------------------------------------------------

def to_grid_labels(scores, resolution: float = DEFAULT_RESOLUTION) -> np.ndarray:
    """Integer labels on the given mark resolution.

    to_grid_labels([0, 0.5, 2.5], 0.25) -> [0, 2, 10]
    Half-safe: uses round-half-up rather than numpy's round-half-even.
    """
    resolution = float(resolution)
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution must be a positive finite number")
    a = np.asarray(scores, dtype=float)
    if not np.isfinite(a).all():
        raise ValueError("scores must contain only finite values")
    a = a / resolution
    return np.floor(a + 0.5).astype(int)


def snap_to_grid(scores, resolution: float = DEFAULT_RESOLUTION) -> np.ndarray:
    """Snap raw scores onto the mark grid (keeps mark units)."""
    return to_grid_labels(scores, resolution) * float(resolution)


def normalize(scores, maxima) -> np.ndarray:
    """Score / criterion_max, so criteria with different maxima are
    comparable (used for cross-criterion aggregates in Studies 1 and 4)."""
    s = np.asarray(scores, dtype=float)
    m = np.asarray(maxima, dtype=float)
    m = np.where(m <= 0, np.nan, m)
    return np.clip(s / m, 0.0, 1.0)


def clamp(score: float, max_score: float) -> float:
    score = float(score)
    if not np.isfinite(score):
        raise ValueError("score must be finite")
    max_score = float(max_score)
    if not np.isfinite(max_score):
        raise ValueError("max_score must be finite")
    if max_score and max_score > 0:
        return float(min(max(score, 0.0), max_score))
    return float(max(score, 0.0))
