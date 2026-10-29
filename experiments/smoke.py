"""Real GPU smoke test on development samples; no inference on the held-out test set."""

import argparse
import importlib.metadata
import json
import platform
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from src.config import load_config, merge, validate_config
from src.data.microvqa import load_samples
from src.data.splits import select_partition
from src.models.base import ModelRequest, build_prompt
from src.models.local import LocalVLM
from src.models.parsing import parse_prediction
from src.models.preprocessing import tensor_digest
from src.models.protocols import DEFAULT_PROTOCOL, resolve_protocol, text_sha256
from src.perturbations.transforms import apply_condition, validate_condition
from src.utils import file_digest, image_digest, read_json, stable_seed, write_json


def choose_smoke_samples(samples, n=6, seed=42):
    if n < 4:
        raise ValueError("Use at least 4 samples to exercise both single and multi-image paths")
    single = [s for s in samples if len(s.image_paths) == 1]
    multi = [s for s in samples if len(s.image_paths) > 1]
    if not single or not multi:
        raise ValueError("Development split must contain single- and multi-image samples")
    ordering = lambda s: stable_seed(seed, s.sample_id)
    selected = []
    candidates = [*sorted(single, key=ordering)[:2], *sorted(multi, key=ordering)[:2],
                  max(multi, key=lambda s: (len(s.image_paths), ordering(s))),
                  *sorted(samples, key=ordering)]
    seen = set()
    for sample in candidates:
        if sample.sample_id not in seen:
            seen.add(sample.sample_id)
            selected.append(sample)
        if len(selected) == n:
            break
    return selected


def run_smoke(model_config, data_path, split_path, output, n=6, seed=42, conditions=None, verify_image_order=False,
              protocol_config=None):
    import torch

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke test requires a working CUDA GPU")
    torch.set_num_threads(4)
    split = read_json(split_path)
    development = select_partition(load_samples(data_path), split, "dev", data_path)
    selected = choose_smoke_samples(development, n, seed)
    config = load_config(model_config)
    config["protocol"] = protocol_config or config.get("protocol", DEFAULT_PROTOCOL.copy())
    protocol = resolve_protocol(config["protocol"])
    conditions = conditions or [{"name": "baseline", "kind": "baseline"}, {"name": "text_only", "kind": "text_only"}]
    for condition in conditions:
        validate_condition(condition)
    if len({c["name"] for c in conditions}) != len(conditions):
        raise ValueError("Duplicate smoke condition names")
    validate_config(merge(load_config(Path(__file__).resolve().parents[1] / "configs/base.json"), config))
    if config["model"]["backend"] != "local":
        raise ValueError("Smoke test requires a real local model")
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        driver = None
    report = {
        "status": "running", "purpose": "infrastructure smoke; not a benchmark score",
        "started_at": datetime.now(timezone.utc).isoformat(), "partition": "dev",
        "protocol": protocol, "split_id": split["split_id"],
        "data_sha256": file_digest(data_path), "python": platform.python_version(),
        "platform": platform.platform(), "gpu_driver": driver, "seed": seed, "config": config,
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "torchvision", "transformers", "accelerate", "bitsandbytes", "Pillow")},
        "conditions": conditions, "verify_image_order": verify_image_order,
        "selection_rule": "seeded 2 single + 2 multi + largest image-count dev question + seeded fill",
        "selected_samples": [{"sample_id": s.sample_id, "group_id": s.metadata["key_image"],
                              "n_images": len(s.image_paths)} for s in selected], "cases": [],
    }
    output.mkdir(parents=True)
    write_json(output / "manifest.json", report)
    write_json(output / "config.resolved.json", config)
    try:
        model = LocalVLM(config["model"], config.get("inference", {}))
        report["model_load"] = model.load_report
        write_json(output / "manifest.json", report)
        for sample in selected:
            images = sample.load_images()
            prompt = build_prompt(sample.question, sample.choices, config["protocol"])
            for condition in conditions:
                transformed, perturbation = apply_condition(images, sample.regions, condition, stable_seed(seed, sample.sample_id))
                request = ModelRequest(sample.sample_id, prompt, transformed,
                                       len(sample.choices), stable_seed(seed, sample.sample_id))
                case = {"sample_id": sample.sample_id, "condition": condition["name"], "n_source_images": len(images),
                        "source_image_hashes": [image_digest(im) for im in images], "perturbation": perturbation,
                        **protocol, "prompt_sha256": text_sha256(prompt)}
                started = time.perf_counter()
                try:
                    raw = model.generate(request)
                    parsed = parse_prediction(raw, len(sample.choices), config["protocol"])
                    case.update(status="ok", raw=raw, prediction=asdict(parsed), telemetry=model.last_generation)
                    if verify_image_order:
                        # Independently preprocess each image and compare its packed tensor with the multi-image call.
                        checks = []
                        for image, actual in zip(request.images, model.last_generation["images"]):
                            single = model.processor.image_processor(images=[image], return_tensors="pt", device="cpu")
                            checks.append(tensor_digest(single["pixel_values"]) == actual["pixel_values_sha256"]
                                          and single["image_grid_thw"].tolist()[0] == actual["grid_thw"])
                        case["independent_image_order_check"] = checks
                        if len(checks) != len(request.images) or not all(checks):
                            raise ValueError("Independent per-image tensor order check failed")
                except Exception as error:
                    case.update(status="error", error_type=type(error).__name__, error=str(error),
                                seconds=time.perf_counter() - started, telemetry=model.last_generation)
                    torch.cuda.empty_cache()
                report["cases"].append(case)
                write_json(output / "manifest.json", report)
                print(json.dumps({"sample_id": sample.sample_id, "condition": condition["name"],
                                  "status": case["status"], "valid": case.get("prediction", {}).get("valid"),
                                  "seconds": case.get("telemetry", {}).get("seconds", case.get("seconds"))}), flush=True)
        report["status"] = "complete" if all(c["status"] == "ok" for c in report["cases"]) else "failed"
        report["summary"] = {
            "n_cases": len(report["cases"]), "n_successful": sum(c["status"] == "ok" for c in report["cases"]),
            "n_valid_json": sum(c.get("prediction", {}).get("valid", False) for c in report["cases"]),
            "n_abstentions": sum(c.get("prediction", {}).get("abstain", False) for c in report["cases"]),
            "peak_allocated_bytes": max((c.get("telemetry", {}).get("peak_allocated_bytes", 0) or 0
                                         for c in report["cases"]), default=0),
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
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--data", default="data/microvqa/samples.jsonl")
    parser.add_argument("--split", default="configs/splits/microvqa-v1.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conditions-config", help="JSON containing a conditions list")
    parser.add_argument("--verify-image-order", action="store_true")
    parser.add_argument("--protocol-config")
    args = parser.parse_args()
    conditions = read_json(args.conditions_config)["conditions"] if args.conditions_config else None
    report = run_smoke(args.model_config, args.data, args.split, args.output, args.samples, args.seed,
                       conditions, args.verify_image_order,
                       load_config(args.protocol_config)["protocol"] if args.protocol_config else None)
    print(json.dumps(report["summary"], indent=2))
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
