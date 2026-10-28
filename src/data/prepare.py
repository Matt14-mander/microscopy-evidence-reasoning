"""Explicit dataset export; never downloads the benchmark during a smoke test."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from src.utils import read_json, write_json


def create_demo(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "images").mkdir()
    rows = []
    for i in range(4):
        image = Image.new("RGB", (96, 80), (25, 25, 25))
        ImageDraw.Draw(image).ellipse((16, 16, 39, 39), fill=(220, 220, 220))
        image.save(destination / "images" / f"{i}.png")
        rows.append({"id": f"demo-{i}", "question": "Which region contains the bright object?",
                     "choices": ["Upper left", "Lower right", "Everywhere", "No bright object"],
                     "answer_index": 0, "images": [f"images/{i}.png"],
                     "evidence_regions": [{"image_index": 0, "box": [16, 16, 40, 40]}],
                     "metadata": {"synthetic": True, "key_image": i}})
    _write_rows(destination / "samples.jsonl", rows)
    write_json(destination / "source.json", {"source": "synthetic smoke-test fixture", "benchmark": False})
    return destination / "samples.jsonl"


def _write_rows(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def export_huggingface(destination, revision, split="test", limit=None, download_manifest=None):
    if not revision:
        raise ValueError("A pinned Hugging Face dataset commit is required")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ImportError as error:
        raise ImportError('Install the data extra: pip install -e ".[data]"') from error
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    if download_manifest:
        downloaded = read_json(download_manifest)
        if (downloaded.get("status") != "complete" or downloaded.get("repo_id") != "jmhb/microvqa"
                or downloaded.get("revision") != revision or downloaded.get("repo_type") != "dataset"):
            raise ValueError("Download manifest must be complete and match the requested dataset revision")
        files = sorted(Path(downloaded["snapshot_path"]).glob(f"data/{split}-*.parquet"))
        if not files:
            raise ValueError(f"No parquet shards for split {split}")
        resolved = revision
        dataset = load_dataset("parquet", data_files={split: [str(p) for p in files]}, split=split)
    else:
        resolved = HfApi().dataset_info("jmhb/microvqa", revision=revision).sha
        dataset = load_dataset("jmhb/microvqa", split=split, revision=resolved)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    destination.mkdir(parents=True)
    (destination / "images").mkdir()
    source = {"dataset": "jmhb/microvqa", "revision": resolved,
              "requested_revision": revision, "split": split, "limit": limit, "count": len(dataset),
              "conversion": "PIL RGB PNG; original image order", "status": "exporting"}
    write_json(destination / "source.json", source)

    def records():
        for position, row in enumerate(dataset):
            paths = []
            for index, image in enumerate(row["images_list"]):
                relative = f"images/{position:05d}_{index:02d}.png"
                image.convert("RGB").save(destination / relative)
                paths.append(relative)
            yield {"id": str(row["key_question"]), "question": row["question"],
                   "choices": row["choices"], "answer_index": row["correct_index"],
                   "correct_answer": row["correct_answer"], "images": paths,
                   "evidence_regions": [],
                   "metadata": {key: row.get(key) for key in ("key_image", "task", "task_str")}}

    try:
        temporary = destination / "samples.jsonl.partial"
        _write_rows(temporary, records())
        temporary.replace(destination / "samples.jsonl")
        source["status"] = "complete"
    except BaseException as error:
        source.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        write_json(destination / "source.json", source)
    return destination / "samples.jsonl"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo")
    demo.add_argument("--output", default="data/demo")
    export = sub.add_parser("microvqa")
    export.add_argument("--output", default="data/microvqa")
    export.add_argument("--revision", required=True, help="Dataset commit SHA (resolved and recorded)")
    export.add_argument("--split", default="test")
    export.add_argument("--limit", type=int)
    export.add_argument("--download-manifest", help="Export a previously pinned local snapshot without network")
    args = parser.parse_args()
    if args.command == "demo":
        print(create_demo(args.output))
    else:
        print(export_huggingface(args.output, args.revision, args.split, args.limit, args.download_manifest))


if __name__ == "__main__":
    main()
