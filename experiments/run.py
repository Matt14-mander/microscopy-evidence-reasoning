"""Run the same sample set across all configured conditions."""

import argparse
import importlib.metadata
import json
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from src.config import load_config, merge, validate_config
from src.data.microvqa import load_samples
from src.evaluation.metrics import compute_metrics
from src.evaluation.paired import paired_comparison
from src.models.base import ModelRequest, build_prompt
from src.models.factory import create_model
from src.models.parsing import parse_prediction
from src.perturbations.transforms import apply_condition
from src.utils import digest, file_digest, image_digest, read_json, stable_seed, write_json


REPO = Path(__file__).resolve().parents[1]


def _code_fingerprint():
    paths = sorted(list((REPO / "src").rglob("*.py")) + list((REPO / "experiments").rglob("*.py")))
    return digest({p.relative_to(REPO).as_posix(): file_digest(p) for p in paths})


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=REPO, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _path(root, value):
    return (root / value).resolve()


def run_experiment(config, root=REPO, model=None):
    validate_config(config)
    root = Path(root).resolve()
    seed = config.get("seed", 0)
    data_path = _path(root, config["data"]["path"])
    image_root = config["data"].get("image_root")
    samples = load_samples(data_path, _path(root, image_root) if image_root else None, config["data"].get("limit"))
    output = _path(root, config["output_dir"])
    cache = _path(root, config["cache_dir"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}; choose --output-dir. Prediction cache is reusable.")
    conditions = config["conditions"]
    # Validate every intervention before incurring any API cost.
    for sample in samples:
        images = sample.load_images()
        for condition in conditions:
            apply_condition(images, sample.regions, condition, stable_seed(seed, sample.sample_id))
    code_hash = _code_fingerprint()
    image_hashes = {str(path): file_digest(path) for sample in samples for path in sample.image_paths}
    environment = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions() if dist.metadata["Name"]}
    source_path = data_path.parent / "source.json"
    manifest = {
        "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "config": config, "seed": seed, "python": platform.python_version(), "platform": platform.platform(),
        "packages": environment, "git_commit": _git("rev-parse", "HEAD"), "git_status": _git("status", "--short"),
        "code_sha256": code_hash, "data_path": str(data_path), "data_sha256": file_digest(data_path),
        "images_sha256": image_hashes, "sample_ids": [s.sample_id for s in samples],
        "dataset_source": read_json(source_path) if source_path.exists() else None,
        "contains_synthetic_data": any(s.metadata.get("synthetic", False) for s in samples),
        "is_mock": config["model"]["backend"] == "mock", "cache_hits": 0, "cache_misses": 0,
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", manifest)
    write_json(output / "config.resolved.json", config)
    predictions = []
    try:
        with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for sample in samples:
                images = sample.load_images()
                prompt = build_prompt(sample.question, sample.choices)
                for condition in conditions:
                    sample_seed = stable_seed(seed, sample.sample_id)
                    transformed, perturbation = apply_condition(images, sample.regions, condition, sample_seed)
                    request = ModelRequest(sample.sample_id, prompt, transformed, len(sample.choices), sample_seed)
                    identity = {"code": code_hash, "model": config["model"], "inference": config.get("inference", {}),
                                "seed": sample_seed, "sample_id": sample.sample_id, "prompt": prompt,
                                "images": [image_digest(im) for im in transformed]}
                    cache_key = digest(identity)
                    cache_file = cache / f"{cache_key}.json"
                    if cache_file.exists():
                        entry = read_json(cache_file)
                        if entry["key"] != cache_key or not isinstance(entry["raw"], str):
                            raise ValueError(f"Corrupt prediction cache: {cache_file}")
                        raw = entry["raw"]
                        manifest["cache_hits"] += 1
                    else:
                        if model is None:
                            model = create_model(config["model"], config.get("inference", {}))
                        raw = model.generate(request)
                        if not isinstance(raw, str):
                            raise TypeError("Model.generate must return a string")
                        write_json(cache_file, {"key": cache_key, "raw": raw})
                        manifest["cache_misses"] += 1
                    parsed = parse_prediction(raw, len(sample.choices))
                    record = {"sample_id": sample.sample_id, "condition": condition["name"],
                              "group_id": sample.metadata.get("key_image"), "answer_index": sample.answer_index,
                              **{("predicted_index" if k == "answer_index" else k): v for k, v in asdict(parsed).items()},
                              "correct": parsed.valid and not parsed.abstain and parsed.answer_index == sample.answer_index,
                              "raw": raw, "cache_key": cache_key, "perturbation": perturbation}
                    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    handle.flush()
                    predictions.append(record)
        grouped = {c["name"]: [r for r in predictions if r["condition"] == c["name"]] for c in conditions}
        evaluation = config.get("evaluation", {})
        metrics = {name: compute_metrics(rows, evaluation.get("ece_bins", 10)) for name, rows in grouped.items()}
        comparisons = {}
        baseline = next((c["name"] for c in conditions if c["kind"] == "baseline"), None)
        if baseline is not None:
            comparisons = {name: paired_comparison(grouped[baseline], rows, seed, evaluation.get("bootstrap_samples", 2000))
                           for name, rows in grouped.items() if name != baseline}
        write_json(output / "metrics.json", metrics)
        write_json(output / "paired.json", {"baseline": baseline, "comparisons": comparisons})
        manifest["status"] = "complete"
    except Exception as error:
        manifest.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "manifest.json", manifest)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-config", help="Optional model JSON overlay")
    parser.add_argument("--output-dir")
    parser.add_argument("--data-path")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--root", type=Path, default=REPO, help="Root for configured data/output paths")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.model_config:
        config = merge(config, load_config(args.model_config))
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.data_path:
        config["data"]["path"] = args.data_path
    if args.limit is not None:
        config["data"]["limit"] = args.limit
    print(run_experiment(config, args.root))


if __name__ == "__main__":
    main()
