"""Audit the four-condition P0-3 smoke without reloading the model."""

import argparse
import statistics

from src.models.preprocessing import grid_report
from src.utils import file_digest, read_json, write_json


def summarize(manifest, split):
    if manifest["status"] != "complete" or not manifest["verify_image_order"]:
        raise ValueError("Requires a completed smoke with independent image-order checks")
    if manifest["split_id"] != split["split_id"] or manifest["partition"] != "dev":
        raise ValueError("Incorrect split")
    selected = {s["sample_id"] for s in manifest["selected_samples"]}
    if not selected <= set(split["partitions"]["dev"]["sample_ids"]):
        raise ValueError("Sample outside development partition")
    config = manifest["config"]["model"]
    fingerprint = manifest["model_load"]["processor"]["fingerprint"]
    cases = manifest["cases"]
    if len(cases) != len(selected) * 4:
        raise ValueError("Incomplete four-condition matrix")
    comparisons = []
    for sample_id in sorted(selected):
        rows = [c for c in cases if c["sample_id"] == sample_id]
        by_condition = {c["condition"]: c for c in rows}
        if len(rows) != 4 or set(by_condition) != {"baseline", "text_only", "blur", "resolution"}:
            raise ValueError("Missing or duplicate conditions")
        baseline = by_condition["baseline"]["telemetry"]
        source_hashes = by_condition["baseline"]["source_image_hashes"]
        if baseline["image_hashes"] != source_hashes:
            raise ValueError("Baseline image order/content mismatch")
        for condition, case in by_condition.items():
            t = case["telemetry"]
            n = 0 if condition == "text_only" else len(source_hashes)
            if (case["status"] != "ok" or t["status"] != "ok" or t["n_images"] != n
                    or len(t["images"]) != n or case["source_image_hashes"] != source_hashes
                    or t["processor_fingerprint"] != fingerprint
                    or case["independent_image_order_check"] != [True] * n):
                raise ValueError("Failed image count, fingerprint or order audit")
            expected = grid_report(t["image_grid_thw"], n, config, t["image_placeholder_tokens"])
            if t["visual_tokens"] != expected["visual_tokens"]:
                raise ValueError("Incorrect visual token count")
            if n and (t["image_sizes"] != baseline["image_sizes"]
                      or t["image_grid_thw"] != baseline["image_grid_thw"]):
                raise ValueError("Paired visual conditions have different input budgets")
            if not n and (t["image_hashes"] or t["image_sizes"] or t["visual_tokens"]):
                raise ValueError("Text-only carries visual content")
            if condition in {"blur", "resolution"}:
                changed = [a["pixel_values_sha256"] != b["pixel_values_sha256"]
                           for a, b in zip(baseline["images"], t["images"])]
                comparisons.append({"sample_id": sample_id, "condition": condition,
                                    "changed_processed_images": sum(changed), "n_images": n})
                if not all(changed):
                    raise ValueError("A smoke intervention did not change the processed tensor")
    timings = {}
    for condition in ("baseline", "text_only", "blur", "resolution"):
        rows = [c["telemetry"] for c in cases if c["condition"] == condition]
        timings[condition] = {
            "n_requests": len(rows),
            **{key: {"median": statistics.median(r[key] for r in rows),
                     "min": min(r[key] for r in rows), "max": max(r[key] for r in rows)}
               for key in ("seconds", "preprocessing_seconds", "transfer_seconds", "generation_seconds")},
            "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in rows),
            "peak_reserved_bytes": max(r["peak_reserved_bytes"] for r in rows),
        }
    return {"scope": "P0-3 development smoke, not benchmark accuracy", "status": "passed",
            "started_at": manifest["started_at"], "finished_at": manifest["finished_at"],
            "split_id": split["split_id"], "config": manifest["config"],
            "model_load": manifest["model_load"], "selected_samples": manifest["selected_samples"],
            "summary": manifest["summary"], "timing_by_condition": timings,
            "checks": {"dev_only": True, "complete_paired_matrix": True, "same_processor": True,
                       "same_visual_budget_per_question": True, "independent_tensor_order": True,
                       "text_only_zero_visual_tokens": True, "degradation_changes_processed_tensors": True},
            "degradation_checks": comparisons,
            "cases": [{"sample_id": c["sample_id"], "condition": c["condition"],
                       "telemetry": c["telemetry"], "independent_image_order_check": c["independent_image_order_check"]}
                      for c in cases]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="configs/splits/microvqa-v1.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = summarize(read_json(args.manifest), read_json(args.split))
    summary["artifacts"] = {p: file_digest(p) for p in (args.manifest, args.split)}
    write_json(args.output, summary)
    print(summary["summary"])
    print(summary["checks"])


if __name__ == "__main__":
    main()
