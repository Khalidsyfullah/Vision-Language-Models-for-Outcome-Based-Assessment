"""Export everything a human rater needs for Study 2.

Study 2 compares AI marks with marks given independently by teachers, but the
blank `data/human_ratings.csv` template only carries IDs. This script writes a
self-contained pack for the split being evaluated:

    data/grading_pack/
      images/<answer_id>.png        one handwritten answer image per answer
      grading_sheet.csv             ready to become data/human_ratings.csv
      grading_sheet.xlsx            same sheet, easier to type into
      grading_booklet.html          image + question + rubric, printable

The sheet already has the columns Study 2 requires (`example_id`,
`criterion_index`, `rater_id`, `score`) plus readable helper columns, so a
completed sheet can be saved directly as `data/human_ratings.csv`.

Usage
    python scripts/export_grading_pack.py
    python scripts/export_grading_pack.py --raters rater2,rater3
    python scripts/export_grading_pack.py --split validation --max-side 1600
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import REPO_ROOT, load_config
from src.data import GradingDataset, load_splits, resize_image
from src.prompting import criterion_max_of
from src.scoring import learn_criterion_maxima, resolve_criterion_max

SHEET_COLUMNS = ["example_id", "criterion_index", "criterion_name",
                 "max_score", "rater_id", "score", "answer_id", "image_file",
                 "question", "model_answer", "rubric_levels"]


def _rubric_text(raw) -> str:
    try:
        levels = json.loads(raw)
    except Exception:
        return str(raw or "")
    return " | ".join(f"{key}: {value}" for key, value in levels.items())


def build_rows(dataset: GradingDataset, out_dir: Path, max_side: int) -> list:
    """Write one image per distinct answer and return one row per criterion."""
    images = out_dir / "images"
    images.mkdir(parents=True, exist_ok=True)
    rows, saved = [], set()
    for index in range(len(dataset)):
        example = dict(dataset.ds[index])
        answer_id = str(example["answer_id"])
        name = f"{answer_id}.png"
        if answer_id not in saved:
            resize_image(example["image"], max_side).save(images / name)
            saved.add(answer_id)
        example.pop("image", None)
        # Same repaired maximum the models and metrics use.
        example["resolved_max"] = resolve_criterion_max(
            example, dataset.criterion_maxima, dataset.max_criterion_score)
        rows.append({
            "example_id": example["example_id"],
            "criterion_index": example["criterion_index"],
            "criterion_name": example["criterion_name"],
            "max_score": criterion_max_of(example),
            "rater_id": "",
            "score": "",
            "answer_id": answer_id,
            "image_file": f"images/{name}",
            "question": example["question"],
            "model_answer": example["model_answer"],
            "rubric_levels": _rubric_text(example.get("rubric_levels")),
        })
    print(f"  saved {len(saved)} answer images -> {images}")
    return rows


def write_booklet(rows: list, path: Path, split: str) -> None:
    """One printable page per answer: image, question, criteria, blank boxes."""
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>Grading booklet — {html.escape(split)} split</title><style>",
        "body{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#111}",
        ".answer{page-break-after:always;border-bottom:2px solid #ccc;"
        "padding-bottom:24px;margin-bottom:24px}",
        "img{max-width:100%;border:1px solid #999}",
        "table{border-collapse:collapse;width:100%;margin-top:12px}",
        "th,td{border:1px solid #999;padding:6px;vertical-align:top;"
        "font-size:14px;text-align:left}",
        "th{background:#f0f0f0}.box{width:70px;height:26px;border:1px solid "
        "#333;display:inline-block}.q{background:#f7f7f7;padding:8px;"
        "border-left:4px solid #666;margin:8px 0;white-space:pre-wrap}",
        "</style></head><body>",
        f"<h1>Grading booklet — {html.escape(split)} split</h1>",
        "<p>For every criterion, write your own mark in the empty box. "
        "Marks must be between 0 and the listed maximum. "
        "Then copy your marks into the <b>score</b> column of "
        "<code>grading_sheet.csv</code>.</p>",
    ]
    frame = pd.DataFrame(rows)
    for answer_id, group in frame.groupby("answer_id", sort=False):
        first = group.iloc[0]
        parts.append("<div class='answer'>")
        parts.append(f"<h2>Answer {html.escape(str(answer_id))}</h2>")
        parts.append(f"<div class='q'><b>Question:</b> "
                     f"{html.escape(str(first['question']))}</div>")
        parts.append(f"<div class='q'><b>Reference answer:</b> "
                     f"{html.escape(str(first['model_answer']))}</div>")
        parts.append(f"<img src='{html.escape(str(first['image_file']))}' "
                     f"alt='student answer'>")
        parts.append("<table><tr><th>example_id</th><th>criterion</th>"
                     "<th>rubric levels</th><th>max</th><th>your mark</th></tr>")
        for _, row in group.iterrows():
            parts.append(
                "<tr>"
                f"<td>{html.escape(str(row['example_id']))}</td>"
                f"<td>{html.escape(str(row['criterion_name']))}</td>"
                f"<td>{html.escape(str(row['rubric_levels']))}</td>"
                f"<td>{row['max_score']:g}</td>"
                "<td><span class='box'></span></td></tr>")
        parts.append("</table></div>")
    parts.append("</body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")
    print(f"  saved booklet     -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="test",
                        help="split to hand to raters (default: test)")
    parser.add_argument("--raters", default="rater2",
                        help="comma-separated rater IDs; one row block each")
    parser.add_argument("--max-side", type=int, default=1600,
                        help="longest image side in pixels (default: 1600)")
    parser.add_argument("--out", default="data/grading_pack")
    args = parser.parse_args()

    cfg = load_config(args.config)
    splits = load_splits(cfg)
    if args.split not in splits:
        raise SystemExit(f"split '{args.split}' not in {list(splits)}")
    upper = float(cfg.dataset.get("max_criterion_score", 4.0))
    maxima = learn_criterion_maxima(splits["train"], upper)
    dataset = GradingDataset(splits[args.split],
                             int(cfg.dataset.max_image_long_side),
                             False, maxima, upper)

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== grading pack: {args.split} split "
          f"({len(dataset)} criteria) ===")

    rows = build_rows(dataset, out_dir, int(args.max_side))
    write_booklet(rows, out_dir / "grading_booklet.html", args.split)

    blocks = []
    for rater in [r.strip() for r in args.raters.split(",") if r.strip()]:
        block = pd.DataFrame(rows)[SHEET_COLUMNS].copy()
        block["rater_id"] = rater
        blocks.append(block)
    sheet = pd.concat(blocks, ignore_index=True)
    sheet.to_csv(out_dir / "grading_sheet.csv", index=False,
                 encoding="utf-8-sig")
    try:
        sheet.to_excel(out_dir / "grading_sheet.xlsx", index=False)
    except Exception as error:      # openpyxl missing
        print(f"  [warn] xlsx not written: {error}")
    print(f"  saved sheet       -> {out_dir / 'grading_sheet.csv'} "
          f"({len(sheet)} rows, raters {args.raters})")
    print("\nGive teachers grading_booklet.html + the images folder, collect "
          "the filled 'score' column, then save the completed sheet as\n"
          f"  {Path(cfg.paths['human_ratings_csv'])}\n"
          "and rerun scripts/study2_human_agreement.py.")


if __name__ == "__main__":
    main()
