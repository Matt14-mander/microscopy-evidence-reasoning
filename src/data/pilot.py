"""Label-blind, stratified development selection with group coverage preference."""

import argparse
from collections import Counter
from pathlib import Path

from src.utils import digest, file_digest, read_json, stable_seed, write_json


def select_pilot(audit, split, n=50, seed=42):
    if audit["status"] != "valid" or audit["data_sha256"] != split["data_sha256"]:
        raise ValueError("Audit and split mismatch")
    if split["split_id"] != digest({k: v for k, v in split.items() if k != "split_id"}):
        raise ValueError("Invalid split integrity")
    dev_ids = set(split["partitions"]["dev"]["sample_ids"])
    # Discard labels and all model outputs before making any selection.
    rows = [{"sample_id": r["sample_id"], "task": r["task"], "n_images": r["n_images"],
             "group_id": split["group_assignments"][r["sample_id"]]} for r in audit["samples"] if r["sample_id"] in dev_ids]
    if len(rows) != len(dev_ids) or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Missing or duplicate development records")
    if type(n) is not int or not 1 <= n <= len(rows):
        raise ValueError("Invalid pilot size")
    stratum = lambda r: r["task"] + ("/single" if r["n_images"] == 1 else "/multi")
    counts = Counter(stratum(r) for r in rows)
    quotas = {s: n * c // len(rows) for s, c in sorted(counts.items())}
    for s in sorted(counts, key=lambda s: (-(n * counts[s] % len(rows)), s))[:n - sum(quotas.values())]:
        quotas[s] += 1
    remaining = quotas.copy()
    selected, group_counts = [], Counter()
    # Reserve a largest-image-count dev question for resource stress, independent of outcomes.
    stress = min((r for r in rows if quotas[stratum(r)]),
                 key=lambda r: (-r["n_images"], stable_seed(seed, r["sample_id"]), r["sample_id"]))
    while len(selected) < n:
        candidates = [r for r in rows if r not in selected and remaining[stratum(r)] > 0]
        row = stress if not selected else min(candidates, key=lambda r: (
            group_counts[r["group_id"]], -remaining[stratum(r)] / quotas[stratum(r)],
            stable_seed(seed, r["sample_id"]), r["sample_id"]))
        selected.append(row)
        group_counts[row["group_id"]] += 1
        remaining[stratum(row)] -= 1
    selected.sort(key=lambda r: r["sample_id"])
    result = {"schema_version": 1, "partition": "dev", "split_id": split["split_id"],
              "data_sha256": split["data_sha256"], "seed": seed,
              "algorithm": "largest-remainder task x single/multi quotas; reserve max-image-count sample; greedy group coverage; seeded tie-break",
              "selection_uses_labels_or_predictions": False, "stress_sample_id": stress["sample_id"],
              "stratum_counts_dev": dict(sorted(counts.items())), "stratum_quotas": quotas,
              "sample_ids": [r["sample_id"] for r in selected], "samples": selected,
              "summary": {"n_samples": n, "n_groups": len(group_counts),
                          "task_counts": dict(sorted(Counter(r["task"] for r in selected).items())),
                          "single_image": sum(r["n_images"] == 1 for r in selected),
                          "multi_image": sum(r["n_images"] > 1 for r in selected),
                          "image_references": sum(r["n_images"] for r in selected)}}
    result["selection_id"] = digest(result)
    return result


def apply_pilot(samples, manifest, split):
    if (manifest["selection_id"] != digest({k: v for k, v in manifest.items() if k != "selection_id"})
            or manifest["split_id"] != split["split_id"] or manifest["data_sha256"] != split["data_sha256"]
            or manifest["partition"] != "dev"):
        raise ValueError("Pilot integrity/split mismatch")
    ids = manifest["sample_ids"]
    by_id = {s.sample_id: s for s in samples}
    if (not ids or len(ids) != len(set(ids)) or not set(ids) <= set(split["partitions"]["dev"]["sample_ids"])
            or not set(ids) <= by_id.keys()):
        raise ValueError("Pilot contains duplicate, missing or non-development IDs")
    return [by_id[key] for key in ids]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", default="outputs/audit/microvqa-v1.json")
    parser.add_argument("--split", default="configs/splits/microvqa-v1.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    audit = read_json(args.audit)
    if file_digest(audit["data_path"]) != audit["data_sha256"]:
        raise ValueError("Dataset changed since audit")
    result = select_pilot(audit, read_json(args.split), args.samples, args.seed)
    write_json(args.output, result)
    print(result["summary"])
    print(result["stratum_quotas"])


if __name__ == "__main__":
    main()
