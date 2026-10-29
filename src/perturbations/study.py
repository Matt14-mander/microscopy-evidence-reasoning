"""Common RGB study resize, before interventions (distinct from the model processor)."""

import math
from PIL import Image
from src.utils import image_digest


def grid_size(height, width, min_pixels, max_pixels, factor=28):
    """Qwen smart-resize geometry, matched to Transformers 4.57.1 Qwen2-VL.

    Same rounding/area rules; image resampling below is explicitly Pillow bicubic.
    """
    if min(height, width) <= 0 or max(height, width) / min(height, width) > 200:
        raise ValueError("Unsupported image aspect ratio")
    h, w = round(height / factor) * factor, round(width / factor) * factor
    if h * w > max_pixels:
        ratio = math.sqrt(height * width / max_pixels)
        h, w = (max(factor, math.floor(x / ratio / factor) * factor) for x in (height, width))
    elif h * w < min_pixels:
        ratio = math.sqrt(min_pixels / (height * width))
        h, w = (math.ceil(x * ratio / factor) * factor for x in (height, width))
    if not min_pixels <= h * w <= max_pixels:
        raise ValueError("Cannot satisfy study pixel bounds at this aspect ratio")
    return h, w


def validate_study(config, model):
    if set(config) != {"kind", "min_pixels", "max_pixels", "factor", "resample"}:
        raise ValueError("Study resize requires explicit kind, pixel bounds, factor and resample")
    if config["kind"] != "qwen_grid_rgb_v1" or config["resample"] != "bicubic" or config["factor"] != 28:
        raise ValueError("Unsupported study preprocessing")
    for key in ("min_pixels", "max_pixels"):
        if type(config[key]) is not int or config[key] <= 0 or config[key] != model.get(key):
            raise ValueError("Study and model pixel budgets must match")
    if config["min_pixels"] > config["max_pixels"]:
        raise ValueError("Invalid study pixel bounds")
    if not model.get("processor"):
        raise ValueError("Study resize requires the frozen Qwen processor")


def prepare_study_images(images, config=None):
    if config is None:
        return tuple(images), None
    result = []
    for image in images:
        h, w = grid_size(image.height, image.width, config["min_pixels"], config["max_pixels"], config["factor"])
        result.append(image.convert("RGB").resize((w, h), Image.Resampling.BICUBIC))
    return tuple(result), {"config": config, "source_sizes": [list(im.size) for im in images],
                           "source_hashes": [image_digest(im) for im in images],
                           "study_sizes": [list(im.size) for im in result],
                           "study_hashes": [image_digest(im) for im in result]}
