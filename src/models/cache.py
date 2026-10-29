"""Shared protocol-aware prediction cache identity and validation."""

from src.models.protocols import text_sha256
from src.utils import digest, image_digest, read_json, write_json


def request_identity(request, model_config, inference, code_hash, protocol, preprocessing=None):
    identity = {"schema": "prediction-cache-2", "code": code_hash, "model": model_config,
                "inference": inference, "seed": request.seed, "sample_id": request.sample_id,
                "num_choices": request.num_choices, "protocol": protocol,
                "prompt": request.prompt, "prompt_sha256": text_sha256(request.prompt),
                "images": [image_digest(im) for im in request.images]}
    if preprocessing is not None:
        identity["processor_fingerprint"] = preprocessing["fingerprint"]
    return identity


def read_prediction_cache(path, identity):
    entry = read_json(path)
    if (entry.get("key") != digest(identity) or entry.get("identity") != identity
            or not isinstance(entry.get("raw"), str)):
        raise ValueError(f"Corrupt or mismatched prediction cache: {path}")
    return entry


def write_prediction_cache(path, identity, raw, telemetry):
    if not isinstance(raw, str):
        raise TypeError("Model.generate must return a string")
    write_json(path, {"key": digest(identity), "identity": identity, "raw": raw, "telemetry": telemetry})
