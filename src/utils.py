"""Small serialization and provenance helpers shared by the pipeline."""

import hashlib
import json
import os
import tempfile
from pathlib import Path


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def image_digest(image):
    h = hashlib.sha256()
    h.update(f"{image.mode}:{image.size}".encode())
    h.update(image.tobytes())
    return h.hexdigest()


def write_json(path, value):
    """Atomic replace: an interrupted write never creates a valid-looking cache."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stable_seed(seed, sample_id):
    return int(digest([seed, sample_id])[:8], 16)
