"""Check the locked pilot matrix against the complete input audit and saved predictions."""

import argparse
import json
import statistics
from pathlib import Path

from src.utils import digest, file_digest, read_json, write_json


def summarize(run, audit_path, selection_path, split_path):
    run = Path(run)
    manifest = read_json(run / "manifest.json")
    audit, selection, split = (read_json(p) for p in (audit_path, selection_path, split_path))
    if (manifest["status"] != "complete" or not manifest["pairing_complete"] or audit["status"] != "passed"
            or manifest["selection_id"] != selection["selection_id"]
            or selection["selection_id"] != digest({k: v for k, v in selection.items() if k != "selection_id"})
            or manifest["input_audit_sha256"] != file_digest(audit_path)
            or manifest["split_id"] != split["split_id"] or manifest["data_sha256"] != split["data_sha256"]):
        raise ValueError("Pilot artifacts are incomplete or inconsistent")
    ids = set(selection["sample_ids"])
    if manifest["partition"] != "dev" or not ids <= set(split["partitions"]["dev"]["sample_ids"]):
        raise ValueError("Development boundary violated")
    records = [json.loads(line) for line in (run / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    conditions = {"baseline", "text_only", "blur", "resolution"}
    pairs = {(r["sample_id"], r["condition"]) for r in records}
    if len(records) != len(ids) * 4 or pairs != {(s, c) for s in ids for c in conditions}:
        raise ValueError("Incomplete, duplicate or unexpected pairs")
    for r in records:
        expected = audit["cases"][r["sample_id"]][r["condition"]]
        t = r["telemetry"]
        if (t["image_hashes"] != expected["image_hashes"] or t["image_grid_thw"] != expected["grids"]
                or [im["pixel_values_sha256"] for im in t["images"]] != expected["tensor_hashes"]
                or t["processor_fingerprint"] != audit["processor"]["fingerprint"]
                or r["protocol_id"] != manifest["protocol"]["protocol_id"]
                or r["group_id"] != split["group_assignments"][r["sample_id"]]):
            raise ValueError("Runtime input, group or protocol differs from its lock")
    return {
        "scope": "P0-5/P0-6; locked development pilot, not held-out benchmark evaluation",
        "status": "passed", "selection": selection, "config": manifest["config"],
        "protocol": manifest["protocol"], "model_load": manifest["model_load"],
        "started_at": manifest["started_at"], "finished_at": manifest["finished_at"],
        "audit_summary": audit["summary"], "n_predictions": len(records),
        "cache": {k: manifest[k] for k in ("cache_hits", "cache_misses")},
        "by_condition": {c: {"n": sum(r["condition"] == c for r in records),
            "n_valid": sum(r["condition"] == c and r["valid"] for r in records),
            "n_abstentions": sum(r["condition"] == c and r["abstain"] for r in records),
            "median_seconds": statistics.median(r["telemetry"]["seconds"] for r in records if r["condition"] == c)}
            for c in sorted(conditions)},
        "peak_allocated_bytes": max(r["telemetry"]["peak_allocated_bytes"] for r in records),
        "peak_reserved_bytes": max(r["telemetry"]["peak_reserved_bytes"] for r in records),
        "hit_token_limit": sum(r["telemetry"]["hit_token_limit"] for r in records),
        "degradation_diagnostics": {c: {
            "n_changed": sum(im["metrics"][c]["changed"] for im in audit["images"]),
            "rgb_mae_min": min(im["metrics"][c]["rgb_mae_0_1"] for im in audit["images"]),
            "rgb_mae_median": statistics.median(im["metrics"][c]["rgb_mae_0_1"] for im in audit["images"]),
            "gradient_ratio_median": statistics.median(im["metrics"][c]["gradient_ratio"] for im in audit["images"]
                                                       if im["metrics"][c]["gradient_ratio"] is not None)}
            for c in ("blur", "resolution")},
        "low_global_change_diagnostics": [dict(sample_id=im["sample_id"], image_index=im["image_index"],
            condition=c, **im["metrics"][c]) for im in audit["images"] for c in ("blur", "resolution")
            if im["metrics"][c]["rgb_mae_0_1"] < 0.001],
        "checks": {"dev_only": True, "complete_50_by_four_pairing": len(ids) == 50,
                   "every_runtime_tensor_matches_audit": True, "no_post_intervention_resize": True,
                   "source_and_group_lock_preserved": True},
        "artifacts": {str(p): file_digest(p) for p in (run / "manifest.json", run / "predictions.jsonl",
                      Path(audit_path), Path(selection_path), Path(split_path))},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="outputs/pilot50-v1/forced_choice")
    parser.add_argument("--audit", default="outputs/input-audit/pilot50-v1/audit.json")
    parser.add_argument("--selection", default="configs/samples/pilot50-v1.json")
    parser.add_argument("--split", default="configs/splits/microvqa-v1.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = summarize(args.run, args.audit, args.selection, args.split)
    write_json(args.output, result)
    print(result["by_condition"])
    print(result["checks"])


if __name__ == "__main__":
    main()
