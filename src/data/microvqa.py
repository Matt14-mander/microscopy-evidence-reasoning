"""Load canonical JSONL; accept official MicroVQA field names for local exports."""

import json
from pathlib import Path

from PIL import Image

from src.data.schema import Region, Sample


def normalize_record(row, root):
    sample_id = row.get("id", row.get("key_question"))
    if sample_id is None or not str(sample_id).strip():
        raise ValueError("id/key_question must be nonempty")
    question = row.get("question")
    choices = row.get("choices")
    answer = row.get("answer_index", row.get("correct_index"))
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a nonempty string")
    if not isinstance(choices, list) or not 2 <= len(choices) <= 26:
        raise ValueError("choices must be a list of 2..26 strings")
    if any(not isinstance(c, str) or not c.strip() for c in choices):
        raise ValueError("choices must contain nonempty strings")
    if type(answer) is not int or not 0 <= answer < len(choices):
        raise ValueError("answer_index/correct_index must be a zero-based integer")
    if row.get("correct_answer") is not None and row["correct_answer"] != choices[answer]:
        raise ValueError("correct_answer disagrees with correct_index")
    paths = row.get("images", row.get("images_list"))
    if not isinstance(paths, list) or not paths or any(not isinstance(p, str) for p in paths):
        raise ValueError("images/images_list must contain local file paths; export HF data first")
    image_paths = tuple((Path(root) / p).resolve() for p in paths)
    sizes = []
    for path in image_paths:
        with Image.open(path) as image:
            image.load()  # validate the complete image, not only its header
            sizes.append(image.size)
    regions = []
    used = set()
    for item in row.get("evidence_regions", []):
        idx, box = item["image_index"], item["box"]
        if type(idx) is not int or not 0 <= idx < len(paths) or idx in used:
            raise ValueError("Each image may have at most one evidence region; invalid image_index")
        if not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box):
            raise ValueError("box must contain four integer pixel coordinates")
        x0, y0, x1, y1 = box
        width, height = sizes[idx]
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("evidence box is empty or outside the image")
        regions.append(Region(idx, tuple(box)))
        used.add(idx)
    metadata = dict(row.get("metadata", {}))
    for key in ("key_image", "task", "task_str"):
        if key in row:
            metadata[key] = row[key]
    return Sample(str(sample_id), question, tuple(choices), answer, image_paths, tuple(regions), metadata)


def load_samples(path, image_root=None, limit=None):
    path = Path(path)
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be a positive integer or null")
    samples, seen = [], set()
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                sample = normalize_record(json.loads(line), image_root or path.parent)
                if sample.sample_id in seen:
                    raise ValueError(f"duplicate sample id: {sample.sample_id}")
            except (ValueError, TypeError, KeyError, OSError) as error:
                raise ValueError(f"{path}:{number}: {error}") from error
            seen.add(sample.sample_id)
            samples.append(sample)
    if not samples:
        raise ValueError(f"Empty dataset: {path}")
    return samples[:limit] if limit is not None else samples
