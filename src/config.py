"""JSON configurations with explicit inheritance and project-relative paths."""

from copy import deepcopy
from pathlib import Path

from src.perturbations.transforms import validate_condition
from src.utils import read_json


def merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path, seen=None):
    path = Path(path).resolve()
    seen = set() if seen is None else set(seen)
    if path in seen:
        raise ValueError(f"Cyclic config inheritance: {path}")
    seen.add(path)
    config = read_json(path)
    if not isinstance(config, dict):
        raise ValueError("Config must be a JSON object")
    parents = config.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    result = {}
    for parent in parents:
        result = merge(result, load_config(path.parent / parent, seen))
    return merge(result, config)


def validate_config(config):
    allowed = {"seed", "data", "model", "inference", "conditions", "output_dir", "cache_dir", "evaluation"}
    if set(config) - allowed:
        raise ValueError(f"Unknown configuration keys: {set(config) - allowed}")
    for key in ("data", "model", "conditions", "output_dir", "cache_dir"):
        if key not in config:
            raise ValueError(f"Missing configuration key: {key}")
    if type(config.get("seed", 0)) is not int or config.get("seed", 0) < 0:
        raise ValueError("seed must be a nonnegative integer")
    backend = config["model"].get("backend")
    if backend not in {"mock", "api", "local"}:
        raise ValueError("model.backend must be mock, api or local")
    model_allowed = {"backend", "name"}
    if backend == "api":
        model_allowed |= {"base_url", "api_key_env", "require_api_key", "send_seed", "max_retries", "timeout"}
        if not config["model"].get("base_url"):
            raise ValueError("API backend requires base_url")
        retries = config["model"].get("max_retries", 2)
        timeout = config["model"].get("timeout", 120)
        if type(retries) is not int or retries < 0:
            raise ValueError("max_retries must be a nonnegative integer")
        if type(timeout) not in (int, float) or not 0 < timeout < float("inf"):
            raise ValueError("timeout must be finite and positive")
        for key in ("require_api_key", "send_seed"):
            if key in config["model"] and type(config["model"][key]) is not bool:
                raise ValueError(f"{key} must be boolean")
    if backend == "local":
        model_allowed |= {"revision", "device_map", "dtype", "quantization", "min_pixels", "max_pixels",
                          "attn_implementation", "local_files_only"}
        model = config["model"]
        if model.get("dtype", "auto") not in {"auto", "float16", "bfloat16", "float32"}:
            raise ValueError("Unsupported local model dtype")
        if model.get("quantization", "none") not in {"none", "nf4"}:
            raise ValueError("quantization must be none or nf4")
        if model.get("attn_implementation", "sdpa") not in {"sdpa", "eager", "flash_attention_2"}:
            raise ValueError("Unsupported attention implementation")
        for key in ("min_pixels", "max_pixels"):
            if key in model and (type(model[key]) is not int or model[key] <= 0):
                raise ValueError(f"{key} must be a positive integer")
        if model.get("min_pixels", 0) > model.get("max_pixels", float("inf")):
            raise ValueError("min_pixels cannot exceed max_pixels")
    if set(config["model"]) - model_allowed:
        raise ValueError("Unknown model parameters (store credentials only in environment variables)")
    if backend != "mock" and not config["model"].get("name"):
        raise ValueError("A model name/path is required")
    if set(config["data"]) - {"path", "image_root", "limit", "split_manifest", "partition"} or not config["data"].get("path"):
        raise ValueError("data requires path and optionally image_root/limit/split_manifest/partition")
    if ("split_manifest" in config["data"]) != ("partition" in config["data"]):
        raise ValueError("split_manifest and partition must be provided together")
    inference = config.get("inference", {})
    if set(inference) - {"temperature", "max_new_tokens"}:
        raise ValueError("Unsupported inference parameter")
    temp, tokens = inference.get("temperature", 0), inference.get("max_new_tokens", 256)
    if type(temp) not in (int, float) or not 0 <= temp < float("inf"):
        raise ValueError("temperature must be finite and nonnegative")
    if type(tokens) is not int or tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    evaluation = config.get("evaluation", {})
    if set(evaluation) - {"bootstrap_samples", "ece_bins"}:
        raise ValueError("Unsupported evaluation parameter")
    for value in evaluation.values():
        if type(value) is not int or value <= 0:
            raise ValueError("Evaluation counts must be positive integers")
    conditions = config["conditions"]
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("At least one condition is required")
    names = set()
    for condition in conditions:
        name = condition.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Condition names must be unique nonempty strings")
        validate_condition(condition)
        names.add(name)
    if sum(c["kind"] == "baseline" for c in conditions) > 1:
        raise ValueError("At most one baseline is supported per run")
