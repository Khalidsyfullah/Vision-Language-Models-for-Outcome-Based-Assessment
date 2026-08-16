"""Small, uniform helpers for publication result files.

Tabular results are stored as CSV and JSON. Row-oriented prediction records
are stored as standard JSON arrays so every result file has a conventional
``.json`` extension.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_table(df: pd.DataFrame, stem: str | Path, index: bool = False) -> list[Path]:
    """Write a dataframe as ``<stem>.csv`` and ``<stem>.json``."""
    stem = Path(stem)
    _ensure_parent(stem)
    paths = []
    p = stem.with_suffix(".csv")
    df.to_csv(p, index=index)
    paths.append(p)
    p = stem.with_suffix(".json")
    df.to_json(p, orient="records", indent=2, force_ascii=False)
    paths.append(p)
    print(f"  saved table -> {stem}.{{csv,json}}")
    return paths


def save_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    _ensure_parent(path)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str,
                               allow_nan=False),
                    encoding="utf-8")
    print(f"  saved json  -> {path}")
    return path


def save_explanation_image(fig, path: str | Path, dpi: int = 150) -> Path:
    """Save one Study 5 explanation image as PNG, then close the figure."""
    import matplotlib.pyplot as plt

    path = Path(path).with_suffix(".png")
    _ensure_parent(path)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved Study 5 image -> {path}")
    return path


def write_records(rows: list[dict], path: str | Path) -> Path:
    """Write row records as a JSON array."""
    path = Path(path)
    _ensure_parent(path)
    path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(f"  saved records -> {path} ({len(rows)} rows)")
    return path


def read_records(path: str | Path) -> list[dict]:
    """Read a JSON array of row records and validate its top-level type."""
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected a JSON array of objects in {path}")
    return rows
