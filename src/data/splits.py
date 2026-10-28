"""Deterministic development/test split without question or image leakage."""

import argparse
import random
from collections import Counter
from pathlib import Path

from src.utils import digest, file_digest, read_json, write_json


def _summary(rows):
    return {"n_samples": len(rows), "task_counts": dict(sorted(Counter(r["task"] for r in rows).items())),
            "single_image": sum(r["n_images"] == 1 for r in rows),
            "multi_image": sum(r["n_images"] > 1 for r in rows)}


def build_split(audit, dev_fraction=0.2, seed=42, candidates=512):
    if audit["status"] != "valid":
        raise ValueError("Cannot split an invalid audit")
    if not 0 < dev_fraction < 1 or type(candidates) is not int or candidates < 1:
        raise ValueError("dev_fraction must be in (0,1); candidates must be positive")
    groups = sorted(audit["groups"], key=lambda g: g["group_id"])
    if len(groups) < 2:
        raise ValueError("At least two independent image components are required")
    samples = {row["sample_id"]: row for row in audit["samples"]}
    stratum = {key: (row["task"], "single" if row["n_images"] == 1 else "multi") for key, row in samples.items()}
    totals = Counter(stratum.values())
    n_dev_groups = min(len(groups) - 1, max(1, round(len(groups) * dev_fraction)))
    rng = random.Random(seed)
    best = None
    for _ in range(candidates):
        selected = sorted(rng.sample(range(len(groups)), n_dev_groups))
        ids = [key for i in selected for key in groups[i]["sample_ids"]]
        counts = Counter(stratum[key] for key in ids)
        # Only task/image-count metadata used, never labels or model outcomes.
        score = abs(len(ids) / len(samples) - dev_fraction)
        score += sum(abs(counts[s] / n - dev_fraction) for s, n in sorted(totals.items())) / len(totals)
        candidate = (score, selected)
        if best is None or candidate < best:
            best = candidate
    selected = set(best[1])
    result = {
        "schema_version": 1, "data_sha256": audit["data_sha256"],
        "source": audit.get("source"), "seed": seed, "dev_fraction_target": dev_fraction,
        "algorithm": "seeded group candidate search minimizing sample/task-image-count imbalance",
        "candidate_count": candidates, "balance_score": best[0],
        "grouping_rule": audit["grouping_rule"],
        "group_assignments": {key: group["group_id"] for group in groups for key in group["sample_ids"]},
        "partitions": {},
    }
    for name, indices in (("dev", selected), ("test", set(range(len(groups))) - selected)):
        ids = sorted(key for i in indices for key in groups[i]["sample_ids"])
        result["partitions"][name] = {
            "sample_ids": ids, "group_ids": sorted(groups[i]["group_id"] for i in indices),
            "summary": _summary([samples[key] for key in ids]),
        }
    dev, test = (result["partitions"][name] for name in ("dev", "test"))
    dev_hashes = {h for key in dev["sample_ids"] for h in samples[key]["image_pixel_hashes"]}
    test_hashes = {h for key in test["sample_ids"] for h in samples[key]["image_pixel_hashes"]}
    result["validation"] = {
        "sample_id_overlap": len(set(dev["sample_ids"]) & set(test["sample_ids"])),
        "group_id_overlap": len(set(dev["group_ids"]) & set(test["group_ids"])),
        "rgb_image_overlap": len(dev_hashes & test_hashes),
        "all_samples_assigned": set(dev["sample_ids"]) | set(test["sample_ids"]) == set(samples),
    }
    if any(result["validation"][key] for key in ("sample_id_overlap", "group_id_overlap", "rgb_image_overlap")):
        raise ValueError("Split leaked samples or images")
    result["split_id"] = digest(result)
    return result


def select_partition(samples, manifest, partition, data_path, verify_images=True):
    """Validate membership/content and attach the corrected leakage group for stats."""
    from dataclasses import replace

    if manifest["data_sha256"] != file_digest(data_path):
        raise ValueError("Split manifest does not match dataset SHA256")
    expected = manifest.get("split_id")
    if expected != digest({k: v for k, v in manifest.items() if k != "split_id"}):
        raise ValueError("Split manifest integrity check failed")
    if verify_images:
        for item in manifest.get("image_files", []):
            path = Path(data_path).parent / item["path"]
            if file_digest(path) != item["sha256"]:
                raise ValueError(f"Image changed since split creation: {path}")
    ids = manifest["partitions"][partition]["sample_ids"]
    by_id = {sample.sample_id: sample for sample in samples}
    if len(ids) != len(set(ids)) or not set(ids) <= by_id.keys():
        raise ValueError("Split contains duplicate or missing sample ids")
    return [replace(by_id[key], metadata={**by_id[key].metadata,
                    "official_key_image": by_id[key].metadata.get("key_image"),
                    "key_image": manifest["group_assignments"][key]}) for key in ids]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("Refusing to overwrite an existing split")
    audit = read_json(args.audit)
    if file_digest(audit["data_path"]) != audit["data_sha256"]:
        raise ValueError("Dataset changed since audit")
    split = build_split(audit, args.dev_fraction, args.seed)
    split["audit_sha256"] = file_digest(args.audit)
    # Relative image paths make the lock portable with the exported dataset.
    root = Path(audit["data_path"]).parent
    split["image_files"] = []
    for path, info in sorted(audit["image_inventory"].items()):
        if file_digest(path) != info["file_sha256"]:
            raise ValueError(f"Image changed since audit: {path}")
        split["image_files"].append({"path": Path(path).relative_to(root).as_posix(), "sha256": info["file_sha256"]})
    split.pop("split_id")
    split["split_id"] = digest(split)
    write_json(args.output, split)
    print({name: part["summary"] for name, part in split["partitions"].items()})
    print(split["validation"])


if __name__ == "__main__":
    main()
