"""Paired real-GPU protocol validation with a shared cache and cache-only replay."""

import argparse
import importlib.metadata
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from experiments.run import REPO, _code_fingerprint, _git
from experiments.smoke import choose_smoke_samples
from src.config import load_config, merge, validate_config
from src.data.microvqa import load_samples
from src.data.splits import select_partition
from src.models.base import ModelRequest, build_prompt
from src.models.cache import request_identity, read_prediction_cache, write_prediction_cache
from src.models.local import LocalVLM
from src.models.parsing import parse_prediction
from src.models.protocols import resolve_protocol, text_sha256
from src.utils import digest, file_digest, read_json, stable_seed, write_json


def run_protocol_smoke(model_config, data_path, split_path, output, n=6, seed=42):
    import torch

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("Real protocol smoke requires CUDA")
    torch.set_num_threads(4)
    specs = [load_config(REPO / f"configs/protocols/{name}.json")["protocol"]
             for name in ("forced_choice", "allow_abstain")]
    config = load_config(model_config)
    for spec in specs:
        validate_config(merge(load_config(REPO / "configs/base.json"), merge(config, {"protocol": spec})))
    if config["model"]["backend"] != "local":
        raise ValueError("Requires the real local backend")
    split = read_json(split_path)
    samples = choose_smoke_samples(select_partition(load_samples(data_path), split, "dev", data_path), n, seed)
    code_hash = _code_fingerprint()
    report = {
        "status": "running", "purpose": "P0-4 development infrastructure validation; not benchmark accuracy",
        "started_at": datetime.now(timezone.utc).isoformat(), "partition": "dev", "split_id": split["split_id"],
        "data_sha256": file_digest(data_path), "code_sha256": code_hash, "git_commit": _git("rev-parse", "HEAD"),
        "config": config, "protocols": [resolve_protocol(s) for s in specs], "seed": seed,
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "torchvision", "transformers", "accelerate", "bitsandbytes", "Pillow")},
        "selected_samples": [{"sample_id": s.sample_id, "group_id": s.metadata["key_image"],
                              "n_images": len(s.image_paths)} for s in samples],
        "conditions": ["baseline", "text_only"], "cache_hits": 0, "cache_misses": 0,
        "replay_hits": 0, "cases": [],
    }
    output.mkdir(parents=True)
    cache = output / "cache"  # Both protocols deliberately share this directory.
    write_json(output / "manifest.json", report)
    identities = []
    try:
        model = LocalVLM(config["model"], config.get("inference", {}))
        report["model_load"] = model.load_report
        for sample in samples:
            images = tuple(sample.load_images())
            for condition in report["conditions"]:
                for spec in specs:
                    protocol = resolve_protocol(spec)
                    prompt = build_prompt(sample.question, sample.choices, spec)
                    request = ModelRequest(sample.sample_id, prompt, images if condition == "baseline" else (),
                                           len(sample.choices), stable_seed(seed, sample.sample_id))
                    identity = request_identity(request, config["model"], config.get("inference", {}),
                                                code_hash, protocol, model.preprocessing)
                    key = digest(identity)
                    path = cache / f"{key}.json"
                    case = {"sample_id": sample.sample_id, "condition": condition, **protocol,
                            "prompt": prompt, "prompt_sha256": text_sha256(prompt), "cache_key": key,
                            "num_choices": len(sample.choices)}
                    try:
                        if path.exists():
                            entry = read_prediction_cache(path, identity)
                            raw, telemetry = entry["raw"], entry["telemetry"]
                            report["cache_hits"] += 1
                        else:
                            raw = model.generate(request)
                            telemetry = model.last_generation
                            write_prediction_cache(path, identity, raw, telemetry)
                            report["cache_misses"] += 1
                        parsed = parse_prediction(raw, len(sample.choices), spec)
                        case.update(status="ok", raw=raw, prediction=asdict(parsed), telemetry=telemetry)
                        identities.append((identity, case))
                    except Exception as error:
                        case.update(status="error", error_type=type(error).__name__, error=str(error),
                                    telemetry=model.last_generation)
                        torch.cuda.empty_cache()
                    report["cases"].append(case)
                    write_json(output / "manifest.json", report)
                    print(json.dumps({k: case[k] for k in ("sample_id", "condition", "protocol_id", "status")}), flush=True)
        # Replay the exact same identities, with no generate() path available.
        for identity, case in identities:
            entry = read_prediction_cache(cache / f"{case['cache_key']}.json", identity)
            if entry["raw"] != case["raw"] or entry["telemetry"] != case["telemetry"]:
                raise ValueError("Cache replay changed the raw output or telemetry")
            report["replay_hits"] += 1
        report["status"] = "complete" if all(c["status"] == "ok" for c in report["cases"]) else "failed"
        report["summary"] = {
            p["protocol_id"]: {
                "n_calls": sum(c["protocol_id"] == p["protocol_id"] for c in report["cases"]),
                "n_successful": sum(c["protocol_id"] == p["protocol_id"] and c["status"] == "ok" for c in report["cases"]),
                "n_valid": sum(c["protocol_id"] == p["protocol_id"] and c.get("prediction", {}).get("valid", False) for c in report["cases"]),
                "n_abstentions": sum(c["protocol_id"] == p["protocol_id"] and c.get("prediction", {}).get("abstain", False) for c in report["cases"]),
            } for p in report["protocols"]
        }
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "manifest.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", default="configs/models/local_4070.json")
    parser.add_argument("--data", default="data/microvqa/samples.jsonl")
    parser.add_argument("--split", default="configs/splits/microvqa-v1.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = run_protocol_smoke(args.model_config, args.data, args.split, args.output, args.samples, args.seed)
    print(json.dumps(report.get("summary", {}), indent=2))
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
