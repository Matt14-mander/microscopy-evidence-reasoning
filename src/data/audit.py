"""Audit every row/image and merge image-sharing questions into leakage groups."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from src.data.microvqa import normalize_record
from src.utils import digest, file_digest, image_digest, read_json, write_json


def connected_groups(records):
    """Transitive union by official key_image OR identical decoded RGB image."""
    parent = {row["sample_id"]: row["sample_id"] for row in records}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    owners = {}
    for row in records:
        keys = [("pixel", value) for value in row["image_pixel_hashes"]]
        if row["key_image"] is not None:
            keys.append(("official", str(row["key_image"])))
        for key in keys:
            if key in owners:
                parent[find(row["sample_id"])] = find(owners[key])
            else:
                owners[key] = row["sample_id"]
    members = defaultdict(list)
    for sample_id in sorted(parent):
        members[find(sample_id)].append(sample_id)
    return sorted(
        [{"group_id": digest(ids), "sample_ids": ids} for ids in members.values()],
        key=lambda group: group["group_id"],
    )


def audit_dataset(path, image_root=None):
    path = Path(path).resolve()
    root = Path(image_root).resolve() if image_root else path.parent
    records, errors, inventory, seen = [], [], {}, set()
    nonempty_rows = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            nonempty_rows += 1
            try:
                row = json.loads(line)
                sample = normalize_record(row, root)
                if sample.sample_id in seen:
                    raise ValueError(f"duplicate sample id: {sample.sample_id}")
                seen.add(sample.sample_id)
                hashes = []
                for image_path in sample.image_paths:
                    key = str(image_path)
                    if key not in inventory:
                        with Image.open(image_path) as image:
                            image.load()
                            inventory[key] = {"size": list(image.size), "mode": image.mode,
                                              "file_sha256": file_digest(image_path),
                                              "rgb_sha256": image_digest(image.convert("RGB"))}
                    hashes.append(inventory[key]["rgb_sha256"])
                records.append({"sample_id": sample.sample_id, "key_image": sample.metadata.get("key_image"),
                                "task": sample.metadata.get("task_str") or "unknown",
                                "n_images": len(sample.image_paths), "n_choices": len(sample.choices),
                                "answer_index": sample.answer_index, "image_pixel_hashes": hashes,
                                "image_paths": [str(p) for p in sample.image_paths]})
            except (ValueError, TypeError, KeyError, OSError, AttributeError) as error:
                errors.append({"line": line_number, "error_type": type(error).__name__, "error": str(error)})
    if not nonempty_rows:
        errors.append({"line": None, "error_type": "EmptyDataset", "error": "Dataset is empty"})
    groups = connected_groups(records)
    hash_questions = defaultdict(set)
    for row in records:
        for value in row["image_pixel_hashes"]:
            hash_questions[value].add(row["sample_id"])
    by_id = {row["sample_id"]: row for row in records}
    merged = []
    for group in groups:
        official = sorted({str(by_id[key]["key_image"]) for key in group["sample_ids"]
                           if by_id[key]["key_image"] is not None})
        group["official_image_groups"] = official
        if len(official) > 1:
            merged.append(group["group_id"])
    source = path.parent / "source.json"
    source_data = read_json(source) if source.exists() else None
    if source_data and source_data.get("dataset") == "jmhb/microvqa":
        if source_data.get("status", "complete") != "complete":
            errors.append({"line": None, "error_type": "IncompleteExport", "error": "Dataset export is incomplete"})
        if source_data.get("count") != nonempty_rows:
            errors.append({"line": None, "error_type": "CountMismatch", "error": "Row count differs from export source"})
    return {
        "schema_version": 1, "status": "valid" if not errors else "invalid",
        "data_path": str(path), "data_sha256": file_digest(path), "image_root": str(root),
        "source": source_data,
        "grouping_rule": "transitive union of key_image and exact decoded RGB SHA256",
        "duplicate_detection_scope": "Exact RGB pixels only; no guarantee against near-duplicates/crops",
        "summary": {
            "n_rows": nonempty_rows, "n_valid_samples": len(records), "n_errors": len(errors),
            "n_official_image_groups": len({str(r["key_image"]) for r in records if r["key_image"] is not None}),
            "n_leakage_groups": len(groups), "n_merged_components": len(merged),
            "n_image_references": sum(r["n_images"] for r in records), "n_image_files": len(inventory),
            "n_unique_rgb_images": len(hash_questions),
            "n_images_shared_between_questions": sum(len(ids) > 1 for ids in hash_questions.values()),
            "n_missing_key_image": sum(r["key_image"] is None for r in records),
            "task_counts": dict(sorted(Counter(r["task"] for r in records).items())),
            "images_per_question": dict(sorted(Counter(r["n_images"] for r in records).items())),
            "choices_per_question": dict(sorted(Counter(r["n_choices"] for r in records).items())),
            "answer_index_counts": dict(sorted(Counter(r["answer_index"] for r in records).items())),
            "width_range": [min((x["size"][0] for x in inventory.values()), default=0),
                            max((x["size"][0] for x in inventory.values()), default=0)],
            "height_range": [min((x["size"][1] for x in inventory.values()), default=0),
                             max((x["size"][1] for x in inventory.values()), default=0)],
            "image_modes": dict(Counter(x["mode"] for x in inventory.values())),
        },
        "errors": errors, "samples": records, "groups": groups, "image_inventory": inventory,
        "merged_components": merged,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--image-root")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("Output already exists; choose a new audit path")
    audit = audit_dataset(args.data, args.image_root)
    write_json(args.output, audit)
    print(json.dumps(audit["summary"], ensure_ascii=False, indent=2))
    if audit["status"] != "valid":
        raise SystemExit("Audit failed: inspect errors in the output; no samples were silently dropped")


if __name__ == "__main__":
    main()
