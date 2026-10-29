"""Verify P0-4 pairing, protocol hashes and shared-cache isolation without GPU inference."""

import argparse
import statistics
from dataclasses import asdict
from pathlib import Path

from src.models.cache import read_prediction_cache
from src.models.parsing import parse_prediction
from src.models.protocols import resolve_protocol, text_sha256
from src.utils import digest, file_digest, read_json, write_json


def summarize(manifest_path, split_path):
    m, split = read_json(manifest_path), read_json(split_path)
    if m["status"] != "complete" or m["partition"] != "dev" or m["split_id"] != split["split_id"]:
        raise ValueError("Incomplete smoke or mismatched development split")
    ids = {s["sample_id"] for s in m["selected_samples"]}
    if not ids or not ids <= set(split["partitions"]["dev"]["sample_ids"]):
        raise ValueError("Smoke includes non-development samples")
    expected = {(s, c, p) for s in ids for c in ("baseline", "text_only")
                for p in ("forced_choice@1.0", "allow_abstain@1.0")}
    rows = {(c["sample_id"], c["condition"], c["protocol_id"]): c for c in m["cases"]}
    if set(rows) != expected or len(rows) != len(m["cases"]):
        raise ValueError("Incomplete or duplicated paired protocol matrix")
    if m["cache_hits"] != 0 or m["cache_misses"] != len(rows) or m["replay_hits"] != len(rows):
        raise ValueError("Unexpected cold-run/cache replay counts")
    keys = set()
    for case in rows.values():
        spec = {"name": case["name"], "version": case["version"]}
        protocol = resolve_protocol(spec)
        if case["status"] != "ok" or any(case[k] != v for k, v in protocol.items()):
            raise ValueError("Protocol metadata mismatch or failed call")
        t = case["telemetry"]
        if case["prompt_sha256"] != text_sha256(case["prompt"]) or t["prompt_sha256"] != case["prompt_sha256"]:
            raise ValueError("Prompt hash mismatch")
        if asdict(parse_prediction(case["raw"], case["num_choices"], spec)) != case["prediction"]:
            raise ValueError("Stored parsing result does not match the versioned parser")
        cache_path = Path(manifest_path).parent / "cache" / f"{case['cache_key']}.json"
        identity = read_json(cache_path)["identity"]
        entry = read_prediction_cache(cache_path, identity)
        if (digest(identity) != case["cache_key"] or identity["protocol"] != protocol
                or identity["prompt"] != case["prompt"] or identity["prompt_sha256"] != case["prompt_sha256"]
                or identity["sample_id"] != case["sample_id"] or identity["images"] != t["image_hashes"]
                or identity["model"] != m["config"]["model"] or identity["code"] != m["code_sha256"]
                or identity["processor_fingerprint"] != m["model_load"]["processor"]["fingerprint"]
                or entry["raw"] != case["raw"] or entry["telemetry"] != t):
            raise ValueError("Cache identity or replay content mismatch")
        if case["cache_key"] in keys:
            raise ValueError("Cache collision across paired cases")
        keys.add(case["cache_key"])
    for sample_id in ids:
        for condition in ("baseline", "text_only"):
            a, b = (rows[sample_id, condition, p] for p in ("forced_choice@1.0", "allow_abstain@1.0"))
            for key in ("image_hashes", "image_grid_thw", "visual_tokens", "processor_fingerprint"):
                if a["telemetry"][key] != b["telemetry"][key]:
                    raise ValueError(f"Protocol pair differs in visual input: {key}")
            if a["prompt_sha256"] == b["prompt_sha256"] or a["cache_key"] == b["cache_key"]:
                raise ValueError("Protocols share a prompt hash or cache key")
            if condition == "text_only" and (a["telemetry"]["image_hashes"] or a["telemetry"]["visual_tokens"]):
                raise ValueError("Text-only includes visual input")
    valid = all(c["prediction"]["valid"] for c in rows.values())
    summary = {
        "scope": "P0-4 development infrastructure validation; not benchmark accuracy",
        "status": "passed" if valid else "needs_attention", "started_at": m["started_at"], "finished_at": m["finished_at"],
        "split_id": m["split_id"], "data_sha256": m["data_sha256"], "code_sha256": m["code_sha256"],
        "config": m["config"], "protocols": m["protocols"], "model_load": m["model_load"],
        "selected_samples": m["selected_samples"], "summary": m["summary"],
        "cache": {"cold_hits": m["cache_hits"], "cold_misses": m["cache_misses"],
                  "replay_hits": m["replay_hits"], "unique_keys": len(keys)},
        "checks": {"dev_only": True, "complete_pairing": True, "matched_visual_inputs": True,
                   "protocol_and_prompt_hashes": True, "disjoint_cache_keys": True,
                   "cache_replay_preserves_raw_and_telemetry": True, "all_outputs_protocol_valid": valid},
        "timing": {p["protocol_id"]: {
            "median_seconds": statistics.median(c["telemetry"]["seconds"] for c in rows.values() if c["protocol_id"] == p["protocol_id"]),
            "peak_allocated_bytes": max(c["telemetry"]["peak_allocated_bytes"] for c in rows.values() if c["protocol_id"] == p["protocol_id"]),
            "peak_reserved_bytes": max(c["telemetry"]["peak_reserved_bytes"] for c in rows.values() if c["protocol_id"] == p["protocol_id"]),
        } for p in m["protocols"]},
        "cases": [{k: c[k] for k in ("sample_id", "condition", "protocol_id", "prompt_sha256", "cache_key", "prediction", "telemetry")}
                  for c in rows.values()],
        "artifacts": {str(p): file_digest(p) for p in (manifest_path, split_path)},
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="configs/splits/microvqa-v1.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = summarize(args.manifest, args.split)
    write_json(args.output, result)
    print(result["summary"])
    print(result["checks"])
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
