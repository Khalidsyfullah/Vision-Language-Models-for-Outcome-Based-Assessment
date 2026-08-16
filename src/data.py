"""Dataset loading + per-criterion grading records.

The dataset (HF Hub or local parquet) has one row per (answer, criterion):
    example_id, answer_id, question, model_answer, criterion_index,
    criterion_name, criterion_max, rubric_levels (json str),
    gained_marks, total_marks, image

Observed properties of the released data (see scripts/inspect_dataset.py):
  * 1483 / 208 / 291 rows, 363 / 50 / 72 distinct answers, no answer_id leaks
    across splits.
  * 12 distinct questions, 46 distinct criteria, ~4.1 criteria per answer.
  * The SAME answer image is repeated for every criterion of an answer, so
    image decoding is cached here to avoid redundant work.
  * `criterion_max` is unreliable -> see src/scoring.resolve_criterion_max.
  * `total_marks` is the student's ACHIEVED total for the answer (it equals
    the sum of gained_marks over the answer's criteria for all 485 answers),
    not the question's full mark.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

try:  # torch is only needed for GPU experiments, not for the study scripts
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    Dataset = object

from .config import Config, REPO_ROOT
from .prompting import build_user_text, target_json
from .scoring import (DEFAULT_MAX_CRITERION_SCORE, CriterionKey,
                      learn_criterion_maxima, resolve_criterion_max)


def load_splits(cfg: Config):
    """Return a datasets.DatasetDict from the HF Hub or local parquet files."""
    from datasets import load_dataset

    if cfg.dataset.source == "local":
        root = Path(cfg.paths.local_dataset_dir)
        if not root.is_absolute():
            root = REPO_ROOT / root
        d = root / "data"
        files = {}
        for s in cfg.dataset.splits:
            matches = sorted(d.glob(f"{s}-*.parquet"))
            if not matches:
                raise FileNotFoundError(f"no parquet for split '{s}' in {d}")
            files[s] = [str(m) for m in matches]
        dset = load_dataset("parquet", data_files=files)
        from datasets import Image as HFImage
        for s in list(dset.keys()):
            if not hasattr(dset[s].features.get("image"), "decode"):
                dset[s] = dset[s].cast_column("image", HFImage())
        return dset
    return load_dataset(cfg.dataset.hf_repo)


def resize_image(img: Image.Image, max_long_side: int) -> Image.Image:
    img = img.convert("RGB")
    if max(img.size) > max_long_side:
        s = max_long_side / max(img.size)
        img = img.resize((max(1, int(img.width * s)),
                          max(1, int(img.height * s))), Image.LANCZOS)
    return img


class GradingDataset(Dataset):
    """Wraps one HF split; yields prompt-ready records.

    `resolved_max` replaces the unreliable `criterion_max` everywhere
    (prompt text, clamping, normalisation).
    """

    def __init__(self, hf_split, max_long_side: int = 1280,
                 cache_images: bool = True,
                 criterion_maxima: dict[CriterionKey, float] | None = None,
                 max_criterion_score: float = DEFAULT_MAX_CRITERION_SCORE):
        self.ds = hf_split
        self.max_long_side = max_long_side
        self.criterion_maxima = criterion_maxima or {}
        self.max_criterion_score = float(max_criterion_score)
        self._cache: dict[str, Image.Image] = {} if cache_images else None

    def __len__(self) -> int:
        return len(self.ds)

    def _image(self, ex: dict) -> Image.Image:
        # every criterion of an answer shares one image -> cache per answer_id
        if self._cache is None:
            return resize_image(ex["image"], self.max_long_side)
        key = str(ex["answer_id"])
        if key not in self._cache:
            self._cache[key] = resize_image(ex["image"], self.max_long_side)
        return self._cache[key]

    def __getitem__(self, idx: int) -> dict:
        ex = dict(self.ds[idx])
        ex["resolved_max"] = resolve_criterion_max(
            ex, self.criterion_maxima, self.max_criterion_score)
        img = self._image(ex)
        ex.pop("image", None)          # keep the record picklable / light
        return {
            "example": ex,
            "image": img,
            "user_text": build_user_text(ex),
            "assistant_text": target_json(ex),
        }


def select_few_shot_exemplars(train_split, n: int, seed: int,
                              stratify: bool = True,
                              max_side: int = 640,
                              criterion_maxima: dict[CriterionKey, float] | None = None,
                              max_criterion_score: float = DEFAULT_MAX_CRITERION_SCORE,
                              ) -> list[dict]:
    """Pick ``n`` TRAIN exemplars spanning the normalized score range."""
    import numpy as np

    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return []
    if n > len(train_split):
        raise ValueError(f"requested {n} exemplars from {len(train_split)} rows")

    rng = np.random.default_rng(seed)
    scores = np.asarray(train_split["gained_marks"], dtype=float)
    lookup = (criterion_maxima or
              learn_criterion_maxima(train_split, max_criterion_score))
    scale_columns = {
        name: list(train_split[name])
        for name in ("question", "criterion_index", "criterion_name",
                     "rubric_levels", "criterion_max")
    }
    maxima = np.asarray([
        resolve_criterion_max({name: values[i]
                               for name, values in scale_columns.items()},
                              lookup, max_criterion_score)
        for i in range(len(train_split))
    ], dtype=float)
    normalized = np.clip(scores / maxima, 0.0, 1.0)
    if stratify:
        # Splitting the sorted range into n equal-size bins guarantees coverage
        # of low, middle, and high scores; the old implementation sampled every
        # level and then truncated the sorted list to the lowest n levels.
        bins = np.array_split(np.argsort(normalized, kind="stable"), n)
        idxs = [int(rng.choice(pool)) for pool in bins]
    else:
        idxs = rng.choice(len(train_split), n, replace=False).astype(int).tolist()

    out = []
    for i in idxs:
        ex = dict(train_split[int(i)])
        ex["resolved_max"] = resolve_criterion_max(
            ex, lookup, max_criterion_score)
        img = resize_image(ex["image"], max_side)
        ex.pop("image", None)
        out.append({"image": img,
                    "user_text": build_user_text(ex),
                    "assistant_text": target_json(ex)})
    return out


# --------------------------------------------------------------------------
# Item features used by Study 4 (fairness subgroups)
# --------------------------------------------------------------------------

def image_features(img: Image.Image) -> dict:
    """Cheap proxies computed from the answer image:
       ink_density (handwriting coverage), ink_pixels (answer-length proxy),
       area, aspect_ratio.
    """
    import numpy as np

    g = np.asarray(img.convert("L"), dtype=np.uint8)
    ink = g < 128
    return {
        "ink_density": float(ink.mean()),
        "ink_pixels": int(ink.sum()),
        "area": int(g.size),
        "aspect_ratio": float(img.width / max(1, img.height)),
    }


def text_language(ex: dict) -> str:
    """'bangla' if the question/model answer contains Bengali codepoints.

    NOTE: the released dataset is entirely English, so this subgroup is
    degenerate there; Study 4 skips subgroups below study4.min_subgroup_n.
    """
    text = (ex.get("question") or "") + (ex.get("model_answer") or "")
    return "bangla" if any("ঀ" <= ch <= "৿" for ch in text) else "english"


SUBJECT_RULES = [
    ("machine_learning", ("supervised", "reinforcement learning",
                          "machine learning")),
    ("data_mining", ("data mining", "kdd", "knowledge discovery")),
    ("databases", ("normal form", "normalization", "schema", "key",
                   "database", "relation")),
    ("statistics", ("r²", "r2", "regression", "variability")),
    ("networking", ("osi", "layer", "protocol", "network")),
    ("image_processing", ("histogram", "quantization", "max-lloyd",
                          "image processing")),
    ("programming", ("java", "overloading", "overriding", "float",
                     "integer", "data type")),
    ("security", ("encryption", "firewall", "malware", "cyber")),
    ("environment", ("climate", "weather", "water")),
]


def subject_of(ex: dict) -> str:
    """Coarse subject label inferred from the question text (12 questions)."""
    q = (ex.get("question") or "").lower()
    for label, kws in SUBJECT_RULES:
        if any(k in q for k in kws):
            return label
    return "other"


def parse_rubric(ex: dict) -> dict:
    try:
        return json.loads(ex["rubric_levels"])
    except Exception:
        return {}
