"""All interventions copy their inputs and preserve image ordering."""

import random

from PIL import Image, ImageFilter


KINDS = {"baseline", "text_only", "evidence_mask", "control_mask", "blur", "resolution"}


def control_box(size, evidence_box, seed):
    """Uniformly sample a same-shape box disjoint from evidence, using row counts.

    Count valid integer positions per row rather than retrying until success.
    This is deterministic, has no false 'no room' result, and uses O(height) memory.
    """
    width, height = size
    x0, y0, x1, y1 = evidence_box
    bw, bh = x1 - x0, y1 - y0
    rows, total = [], 0
    for y in range(height - bh + 1):
        if y + bh <= y0 or y >= y1:
            intervals = [(0, width - bw)]
        else:
            intervals = [(0, x0 - bw), (x1, width - bw)]
        intervals = [(left, right) for left, right in intervals if left <= right]
        count = sum(right - left + 1 for left, right in intervals)
        rows.append((y, intervals, count))
        total += count
    if total == 0:
        raise ValueError("No disjoint control mask of the same shape fits in the image")
    pick = random.Random(seed).randrange(total)
    for y, intervals, count in rows:
        if pick >= count:
            pick -= count
            continue
        for left, right in intervals:
            length = right - left + 1
            if pick < length:
                x = left + pick
                return (x, y, x + bw, y + bh)
            pick -= length
    raise RuntimeError("Invalid control-mask sampling state")


def validate_condition(condition):
    kind = condition.get("kind")
    if kind not in KINDS:
        raise ValueError(f"Unknown perturbation: {kind}")
    allowed = {"name", "kind"}
    if kind in {"evidence_mask", "control_mask"}:
        allowed.add("fill")
        fill = condition.get("fill", [0, 0, 0])
        if not isinstance(fill, list) or len(fill) != 3 or any(type(x) is not int or not 0 <= x <= 255 for x in fill):
            raise ValueError("fill must contain three integer RGB values")
    if kind == "blur":
        allowed |= {"radius", "radius_fraction"}
        if "radius" in condition and "radius_fraction" in condition:
            raise ValueError("Choose either radius or radius_fraction")
        radius = condition.get("radius_fraction", condition.get("radius", 2.0))
        if type(radius) not in (int, float) or not 0 < radius < float("inf"):
            raise ValueError("blur radius must be finite and positive")
    if kind == "resolution":
        allowed.add("scale")
        scale = condition.get("scale", 0.5)
        if type(scale) not in (int, float) or not 0 < scale <= 1:
            raise ValueError("resolution scale must be in (0,1]")
    if set(condition) - allowed:
        raise ValueError(f"Unknown condition parameters: {set(condition) - allowed}")


def apply_condition(images, regions, condition, seed):
    validate_condition(condition)
    kind = condition["kind"]
    metadata = {"kind": kind, "regions": []}
    if kind == "text_only":
        return (), metadata
    result = [image.copy() for image in images]
    if kind in {"evidence_mask", "control_mask"}:
        if not regions:
            raise ValueError("Evidence/control masking requires human-provided evidence_regions")
        fill = tuple(condition.get("fill", [0, 0, 0]))
        for region in regions:
            box = region.box
            if kind == "control_mask":
                box = control_box(result[region.image_index].size, box, seed + region.image_index)
            result[region.image_index].paste(fill, box)
            metadata["regions"].append({"image_index": region.image_index, "box": list(box)})
        metadata["fill"] = list(fill)
    elif kind == "blur":
        radii = [min(im.size) * condition["radius_fraction"] if "radius_fraction" in condition
                 else condition.get("radius", 2.0) for im in result]
        result = [image.filter(ImageFilter.GaussianBlur(radius)) for image, radius in zip(result, radii)]
        metadata["radii"] = radii
        if "radius_fraction" in condition:
            metadata["radius_fraction"] = condition["radius_fraction"]
        else:
            metadata["radius"] = condition.get("radius", 2.0)
    elif kind == "resolution":
        scale = condition.get("scale", 0.5)
        small_sizes = [(max(1, round(image.width * scale)), max(1, round(image.height * scale))) for image in result]
        result = [image.resize(small, Image.Resampling.LANCZOS).resize(image.size, Image.Resampling.BICUBIC)
                  for image, small in zip(result, small_sizes)]
        metadata.update({"scale": scale, "downsampled_sizes": small_sizes, "restore_original_size": True})
    return tuple(result), metadata
