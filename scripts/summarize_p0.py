"""Verify smoke/split invariants and produce a small, shareable P0 summary."""

import argparse
import json
import statistics
from pathlib import Path

from PIL import Image

from src.utils import file_digest, image_digest, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", default="outputs/audit/microvqa-v1.json")
    parser.add_argument("--split", default="configs/splits/microvqa-v1.json")
    parser.add_argument("--smoke", default="outputs/smoke/qwen25vl3b_nf4_4070/manifest.json")
    parser.add_argument("--output", default="docs/reports/p0-2026-10-28.json")
    args = parser.parse_args()
    m, a, s = read_json(args.smoke), read_json(args.audit), read_json(args.split)
    if m["status"] != "complete" or a["status"] != "valid" or m["split_id"] != s["split_id"]:
        raise ValueError("P0 artifacts are incomplete or mismatched")
    ids = {x["sample_id"] for x in m["selected_samples"]}
    if not ids <= set(s["partitions"]["dev"]["sample_ids"]) or ids & set(s["partitions"]["test"]["sample_ids"]):
        raise ValueError("Smoke sample selection violates the development boundary")
    data_path = Path(a["data_path"])
    if file_digest(data_path) != m["data_sha256"]:
        raise ValueError("Smoke dataset no longer matches its recorded hash")
    originals = {str(row["id"]): row for row in
                 (json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip())}
    expected_hashes = {}
    for key in ids:
        expected_hashes[key] = []
        for path in originals[key]["images"]:
            with Image.open(data_path.parent / path) as image:
                expected_hashes[key].append(image_digest(image.convert("RGB")))
    for case in m["cases"]:
        t = case["telemetry"]
        expected = case["n_source_images"] if case["condition"] == "baseline" else 0
        if t["n_images"] != expected or len(t["image_grid_thw"]) != expected:
            raise ValueError("Image count/grid mismatch")
        if t["image_hashes"] != (expected_hashes[case["sample_id"]] if case["condition"] == "baseline" else []):
            raise ValueError("Input images or their order differ from the source dataset")
        for grid in t["image_grid_thw"]:
            pixels = grid[1] * grid[2] * m["model_load"]["processor"]["patch_size"] ** 2
            if not m["config"]["model"]["min_pixels"] <= pixels <= m["config"]["model"]["max_pixels"]:
                raise ValueError("Actual processor image grid exceeds configured pixel bounds")
    latency = [c["telemetry"]["seconds"] for c in m["cases"]]
    summary = {
        "date": m["finished_at"][:10], "scope": "P0-1/P0-2; development smoke, not benchmark scores",
        "dataset": a["source"], "audit_summary": a["summary"], "split_id": s["split_id"],
        "partitions": {k: {"n_groups": len(v["group_ids"]), **v["summary"]} for k, v in s["partitions"].items()},
        "split_validation": s["validation"], "model_config": m["config"], "model_load": m["model_load"],
        "smoke_summary": m["summary"],
        "latency_seconds": {"min": min(latency), "max": max(latency), "median": statistics.median(latency),
                            "mean": statistics.mean(latency), "total": sum(latency)},
        "peak_reserved_bytes": max(c["telemetry"]["peak_reserved_bytes"] for c in m["cases"]),
        "sample_cases": [{"sample_id": c["sample_id"], "condition": c["condition"],
                          "n_images": c["telemetry"]["n_images"], "seconds": c["telemetry"]["seconds"],
                          "input_tokens": c["telemetry"]["input_tokens"],
                          "image_grid_thw": c["telemetry"]["image_grid_thw"],
                          "valid_json": c["prediction"]["valid"]} for c in m["cases"]],
        "checks": {"all_smoke_ids_in_dev": True, "no_test_inference": True, "all_images_preserved": True,
                   "actual_grid_pixel_bounds_pass": True, "text_only_image_grids_empty": True,
                   "source_image_hash_order_pass": True},
        "artifacts": {p: file_digest(p) for p in (args.audit, args.split, args.smoke)},
    }
    write_json(args.output, summary)
    print(summary["latency_seconds"])
    print({"peak_allocated_GiB": m["summary"]["peak_allocated_bytes"] / 2**30,
           "peak_reserved_GiB": summary["peak_reserved_bytes"] / 2**30})


if __name__ == "__main__":
    main()
