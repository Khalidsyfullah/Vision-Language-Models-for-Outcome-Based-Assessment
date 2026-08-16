#!/usr/bin/env python
"""Build the criterion-level OBE dataset from split JSON files and PDFs.

Authentication is intentionally not handled in this file. Run ``hf auth
login`` before using ``--repo-id`` so credentials stay outside source code.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path
from typing import Any

import fitz
from datasets import Dataset, DatasetDict, Features, Image, Value
from PIL import Image as PILImage
from tqdm import tqdm


MARKS_RE = re.compile(r"Marks?\s*=\s*\(?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
LEVEL_MARKS_RE = re.compile(r"\(\s*(\d+(?:\.\d+)?)\s*\)\s*$")
SHORT_CRITERION_IDS = {
    "definition of data mining": "def",
    "kdd process steps": "kdd",
    "importance of data mining": "imp",
    "example / application": "ex",
}
SPLIT_FILES = {"train": "train.json", "validation": "val.json", "test": "test.json"}


def short_criterion_id(name: str, index: int) -> str:
    base = re.sub(r"\s*\(.*?\)\s*", "", name).strip().lower()
    return SHORT_CRITERION_IDS.get(base, f"c{index}")


def clean_criterion_name(name: str) -> str:
    return re.sub(
        r"\s*\(\s*Marks?\s*=.*?\)\s*$", "", name,
        flags=re.IGNORECASE,
    ).strip()


def parse_max_marks(
    criterion: str,
    rubric_levels: dict[str, Any] | None,
    fallback: float,
) -> float:
    """Read a criterion maximum from its heading or rubric level labels."""
    match = MARKS_RE.search(criterion)
    if match:
        return float(match.group(1))
    values: list[float] = []
    for label in (rubric_levels or {}):
        match = LEVEL_MARKS_RE.search(label)
        if match:
            values.append(float(match.group(1)))
            continue
        match = re.search(
            r"\(\s*\d+(?:\.\d+)?\s*-\s*(\d+(?:\.\d+)?)\s*\)\s*$",
            label,
        )
        if match:
            values.append(float(match.group(1)))
    return max(values) if values else fallback


def render_pdf_to_jpeg(
    pdf_path: Path,
    output_path: Path,
    *,
    dpi: int,
    quality: int,
    max_height: int,
    max_long_side: int,
) -> tuple[int, int]:
    """Render and vertically stitch a PDF into one optimized JPEG."""
    input_size = pdf_path.stat().st_size
    with fitz.open(pdf_path) as document:
        matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pages = []
        for page in document:
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            pages.append(PILImage.frombytes(
                "RGB", (pixmap.width, pixmap.height), pixmap.samples,
            ))
    if not pages:
        raise ValueError(f"PDF has no pages: {pdf_path}")

    gap = 8
    width = max(page.width for page in pages)
    height = sum(page.height for page in pages) + gap * (len(pages) - 1)
    canvas = PILImage.new("RGB", (width, height), "white")
    y = 0
    for index, page in enumerate(pages):
        canvas.paste(page, (0, y))
        y += page.height + (gap if index + 1 < len(pages) else 0)

    if canvas.height > max_height:
        new_width = max(1, round(canvas.width * max_height / canvas.height))
        canvas = canvas.resize((new_width, max_height), PILImage.Resampling.LANCZOS)
    if max(canvas.size) > max_long_side:
        scale = max_long_side / max(canvas.size)
        canvas = canvas.resize(
            (max(1, round(canvas.width * scale)),
             max(1, round(canvas.height * scale))),
            PILImage.Resampling.LANCZOS,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(
        output_path, format="JPEG", quality=quality, optimize=True,
        progressive=True,
    )
    return input_size, output_path.stat().st_size


def load_examples(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise ValueError(f"Expected a JSON array of objects in {path}")
    return data


def expand_split(
    *,
    json_path: Path,
    files_dir: Path,
    image_dir: Path,
    split_name: str,
    rendered_images: dict[str, Path],
    split_owners: dict[str, str],
    dpi: int,
    quality: int,
    max_height: int,
    max_long_side: int,
    fallback_max: float,
) -> list[dict[str, Any]]:
    """Expand one answer-level split into criterion-level rows."""
    records: list[dict[str, Any]] = []
    for example in tqdm(load_examples(json_path), desc=f"Building {split_name}"):
        required = {"answer_id", "pdf_file", "question", "model_answer", "rubrics"}
        missing = sorted(required - example.keys())
        if missing:
            raise KeyError(f"{json_path}: missing fields {missing}")

        pdf_name = str(example["pdf_file"])
        owner = split_owners.get(pdf_name)
        if owner is not None and owner != split_name:
            raise ValueError(
                f"Cross-split leakage: {pdf_name!r} occurs in {owner} and {split_name}",
            )
        split_owners[pdf_name] = split_name
        pdf_path = files_dir / pdf_name
        if not pdf_path.is_file():
            raise FileNotFoundError(f"Missing PDF: {pdf_path}")

        if pdf_name not in rendered_images:
            image_path = image_dir / f"{Path(pdf_name).stem}.jpg"
            render_pdf_to_jpeg(
                pdf_path, image_path, dpi=dpi, quality=quality,
                max_height=max_height, max_long_side=max_long_side,
            )
            rendered_images[pdf_name] = image_path
        image_path = rendered_images[pdf_name]

        answer_id = str(example["answer_id"])
        for index, rubric in enumerate(example["rubrics"]):
            criterion = str(rubric["criteria"])
            levels = rubric.get("marking_rules", {})
            records.append({
                "example_id": f"{answer_id}_{short_criterion_id(criterion, index)}",
                "answer_id": answer_id,
                "image": str(image_path),
                "question": str(example["question"]),
                "model_answer": str(example["model_answer"]),
                "criterion_index": index,
                "criterion_name": clean_criterion_name(criterion),
                "criterion_max": parse_max_marks(criterion, levels, fallback_max),
                "rubric_levels": json.dumps(levels, ensure_ascii=False),
                "gained_marks": float(rubric["gained_marks"]),
                "total_marks": float(example["total_marks"]),
            })
    return records


def dataset_features() -> Features:
    return Features({
        "example_id": Value("string"),
        "answer_id": Value("string"),
        "image": Image(),
        "question": Value("string"),
        "model_answer": Value("string"),
        "criterion_index": Value("int32"),
        "criterion_name": Value("string"),
        "criterion_max": Value("float32"),
        "rubric_levels": Value("string"),
        "gained_marks": Value("float32"),
        "total_marks": Value("float32"),
    })


def build_dataset(args: argparse.Namespace, image_dir: Path) -> DatasetDict:
    data_dir = args.data_dir.resolve()
    files_dir = data_dir / "files"
    if not files_dir.is_dir():
        raise FileNotFoundError(f"Expected a PDF directory at {files_dir}")

    rendered_images: dict[str, Path] = {}
    split_owners: dict[str, str] = {}
    datasets = {}
    for split_name, filename in SPLIT_FILES.items():
        json_path = data_dir / filename
        if not json_path.is_file():
            raise FileNotFoundError(f"Missing split file: {json_path}")
        rows = expand_split(
            json_path=json_path,
            files_dir=files_dir,
            image_dir=image_dir,
            split_name=split_name,
            rendered_images=rendered_images,
            split_owners=split_owners,
            dpi=args.dpi,
            quality=args.jpeg_quality,
            max_height=args.max_height,
            max_long_side=args.max_long_side,
            fallback_max=args.fallback_criterion_max,
        )
        datasets[split_name] = Dataset.from_list(rows, features=dataset_features())
        answers = len({row["answer_id"] for row in rows})
        print(f"{split_name}: {len(rows)} criteria from {answers} answers")
    return DatasetDict(datasets)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Folder containing train.json, val.json, test.json, and files/")
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--repo-id",
                             help="Hugging Face dataset repository to upload")
    destination.add_argument("--output-dir", type=Path,
                             help="Local DatasetDict destination")
    parser.add_argument("--public", action="store_true",
                        help="Make a Hub upload public. Uploads are private by default.")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--max-long-side", type=int, default=1280)
    parser.add_argument("--max-height", type=int, default=3072)
    parser.add_argument("--fallback-criterion-max", type=float, default=4.0)
    args = parser.parse_args()
    for name in ("dpi", "jpeg_quality", "max_long_side", "max_height"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100")
    return args


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="obe_dataset_") as temp_dir:
        dataset = build_dataset(args, Path(temp_dir) / "images")
        print(dataset)
        if args.repo_id:
            dataset.push_to_hub(args.repo_id, private=not args.public)
            print(f"Uploaded dataset to {args.repo_id}")
        else:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(str(args.output_dir))
            print(f"Saved dataset to {args.output_dir}")


if __name__ == "__main__":
    main()
