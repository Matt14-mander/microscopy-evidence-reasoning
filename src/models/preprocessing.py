"""Frozen Qwen image preprocessing and audit helpers; no model downloads."""

import hashlib
import importlib.metadata
import math

from src.utils import digest


PARAMETERS = {
    "do_resize", "resample", "do_rescale", "rescale_factor", "do_normalize",
    "image_mean", "image_std", "do_convert_rgb", "patch_size", "temporal_patch_size",
    "merge_size", "disable_grouping",
}


def validate_processor_config(config):
    """This contract is intentionally limited to the verified Qwen fast path."""
    if set(config) != {"class", "image_processor_class", "use_fast", "parameters"}:
        raise ValueError("processor requires class, image_processor_class, use_fast, parameters")
    if (config["class"] != "Qwen2_5_VLProcessor"
            or config["image_processor_class"] != "Qwen2VLImageProcessorFast"
            or config["use_fast"] is not True):
        raise ValueError("Frozen processor currently supports Qwen2.5-VL fast only")
    p = config["parameters"]
    if not isinstance(p, dict) or set(p) != PARAMETERS:
        raise ValueError("processor.parameters must explicitly specify all preprocessing parameters")
    for key in ("do_resize", "do_rescale", "do_normalize", "do_convert_rgb", "disable_grouping"):
        if p[key] is not True:
            raise ValueError(f"Frozen processor requires {key}=true")
    if type(p["resample"]) is not int or p["resample"] != 3:
        raise ValueError("Frozen processor requires bicubic resample=3")
    for key, expected in (("patch_size", 14), ("temporal_patch_size", 2), ("merge_size", 2)):
        if type(p[key]) is not int or p[key] != expected:
            raise ValueError(f"Qwen architecture requires {key}={expected}")
    if type(p["rescale_factor"]) not in (int, float) or p["rescale_factor"] != 1 / 255:
        raise ValueError("Expected rescale_factor=1/255")
    for key in ("image_mean", "image_std"):
        values = p[key]
        if (not isinstance(values, list) or len(values) != 3
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
                or (key == "image_std" and any(v <= 0 for v in values))):
            raise ValueError(f"Invalid {key}")


def processor_report(processor, config):
    image_processor = processor.image_processor
    contract = config.get("processor")
    report = {"class": type(processor).__name__, "image_processor_class": type(image_processor).__name__,
              "tokenizer_class": type(processor.tokenizer).__name__,
              "chat_template_sha256": digest(processor.chat_template),
              "parameters": {key: getattr(image_processor, key, None) for key in sorted(PARAMETERS)},
              "min_pixels": getattr(image_processor, "min_pixels", None),
              "max_pixels": getattr(image_processor, "max_pixels", None),
              "size": getattr(image_processor, "size", None),
              "patch_size": getattr(image_processor, "patch_size", None),
              "merge_size": getattr(image_processor, "merge_size", None),
              "versions": {name: importlib.metadata.version(name) for name in
                           ("transformers", "torch", "torchvision", "Pillow")},
              "image_order": "request order, no truncation", "image_device": "cpu" if contract else "processor_default",
              "text_options": {"padding": False, "truncation": False},
              "frozen": contract is not None}
    if contract:
        for key in ("class", "image_processor_class"):
            if report[key] != contract[key]:
                raise ValueError(f"Processor class mismatch: {key}={report[key]}")
        for key, value in contract["parameters"].items():
            if report["parameters"][key] != value:
                raise ValueError(f"Processor ignored frozen parameter {key}")
        expected_size = {"shortest_edge": config["min_pixels"], "longest_edge": config["max_pixels"]}
        if report["size"] != expected_size:
            raise ValueError("Processor size does not match frozen pixel budget")
        for key in ("min_pixels", "max_pixels"):
            if report[key] != config[key]:
                raise ValueError(f"Processor ignored {key}")
    report["fingerprint"] = digest(report)
    return report


def grid_report(grids, n_images, config, image_token_count):
    """Qwen grid counts pre-merge patches; LLM tokens are t*h*w / merge_size**2."""
    if len(grids) != n_images:
        raise ValueError("Processor image count does not match model request")
    p = config["processor"]["parameters"]
    patch, merge = p["patch_size"], p["merge_size"]
    entries = []
    for index, grid in enumerate(grids):
        if len(grid) != 3 or any(type(v) is not int or v <= 0 for v in grid):
            raise ValueError("Invalid image grid")
        t, h, w = grid
        if t != 1 or h % merge or w % merge:
            raise ValueError("Unexpected still-image grid or merge divisibility")
        pixels = h * w * patch**2
        if not config["min_pixels"] <= pixels <= config["max_pixels"]:
            raise ValueError("Actual image grid exceeds frozen pixel budget")
        entries.append({"image_index": index, "grid_thw": grid,
                        "processed_size_hw": [h * patch, w * patch], "pixels": pixels,
                        "patches": t * h * w, "visual_tokens": t * h * w // merge**2})
    total = sum(e["visual_tokens"] for e in entries)
    if image_token_count != total:
        raise ValueError("Image placeholder token count does not match visual grid")
    return {"images": entries, "visual_tokens": total, "image_placeholder_tokens": image_token_count,
            "total_processed_pixels": sum(e["pixels"] for e in entries)}


def tensor_digest(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    h = hashlib.sha256(f"{array.dtype}:{array.shape}".encode())
    h.update(array.tobytes())
    return h.hexdigest()
