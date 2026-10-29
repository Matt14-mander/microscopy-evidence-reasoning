"""Preview and audit actual Qwen processor tensors before any pilot inference."""

import argparse
import html
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from src.config import load_config, validate_config
from src.data.microvqa import load_samples
from src.data.pilot import apply_pilot
from src.data.splits import select_partition
from src.models.preprocessing import grid_report, processor_report, tensor_digest
from src.perturbations.study import prepare_study_images
from src.perturbations.transforms import apply_condition
from src.utils import digest, file_digest, image_digest, read_json, stable_seed, write_json


def audit_identity(config, selection):
    return {"schema": "input-audit-1", "selection_id": selection["selection_id"],
            "split_id": selection["split_id"], "data_sha256": selection["data_sha256"],
            "model": config["model"], "study_preprocessing": config["study_preprocessing"],
            "conditions": config["conditions"], "seed": config["seed"]}


def unpack_image(pixels, grid, parameters):
    """Invert Qwen's packed spatial/temporal layout; return the first RGB float frame."""
    t, h, w = grid
    p, m, temporal = (parameters[k] for k in ("patch_size", "merge_size", "temporal_patch_size"))
    frames = pixels.reshape(t, h // m, w // m, m, m, 3, temporal, p, p)
    frames = frames.permute(0, 6, 5, 1, 3, 7, 2, 4, 8).reshape(t * temporal, 3, h * p, w * p)
    array = frames.detach().cpu().numpy()
    if not np.array_equal(array[0], array[-1]):
        raise ValueError("Still-image temporal frames differ")
    return (array[0].transpose(1, 2, 0) * np.asarray(parameters["image_std"])
            + np.asarray(parameters["image_mean"]))


def gradient_energy(rgb):
    return float((np.abs(np.diff(rgb, axis=0)).mean() + np.abs(np.diff(rgb, axis=1)).mean()) / 2)


def to_image(rgb):
    return Image.fromarray(np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8))


def montage(images, labels, path):
    tile_w, tile_h = 360, 360
    canvas = Image.new("RGB", (tile_w * len(images), tile_h + 42), "#eeeeee")
    draw = ImageDraw.Draw(canvas)
    for i, (im, label) in enumerate(zip(images, labels)):
        thumb = ImageOps.contain(im, (tile_w - 12, tile_h - 12))
        canvas.paste(thumb, (i * tile_w + (tile_w - thumb.width) // 2, 42 + (tile_h - thumb.height) // 2))
        draw.text((i * tile_w + 8, 8), label, fill="black")
        draw.text((i * tile_w + 8, 23), f"{im.width} x {im.height}", fill="black")
    canvas.save(path)


def run_audit(config, output, root):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor

    validate_config(config)
    if {c["name"]: c["kind"] for c in config["conditions"]} != {
            "baseline": "baseline", "text_only": "text_only", "blur": "blur", "resolution": "resolution"}:
        raise ValueError("Input audit expects the four named pilot conditions")
    output, root = Path(output), Path(root)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    data = root / config["data"]["path"]
    split = read_json(root / config["data"]["split_manifest"])
    selection = read_json(root / config["data"]["sample_manifest"])
    samples = apply_pilot(select_partition(load_samples(data), split, "dev", data), selection, split)
    model = config["model"]
    snapshot = snapshot_download(model["name"], revision=model["revision"], local_files_only=True)
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False,
        use_fast=model["processor"]["use_fast"], min_pixels=model["min_pixels"], max_pixels=model["max_pixels"],
        size={"shortest_edge": model["min_pixels"], "longest_edge": model["max_pixels"]},
        **model["processor"]["parameters"])
    report = {"status": "running", "identity": audit_identity(config, selection),
              "processor": processor_report(processor, model), "cases": {}, "images": [], "failures": []}
    output.mkdir(parents=True)
    (output / "images").mkdir()
    write_json(output / "audit.json", report)
    sections = []
    try:
        for sample in samples:
            source = sample.load_images()
            study, study_info = prepare_study_images(source, config["study_preprocessing"])
            report["cases"][sample.sample_id] = {}
            rgb_by_condition = {}
            for condition in config["conditions"]:
                name = condition["name"]
                images, transform = apply_condition(study, (), condition, stable_seed(config["seed"], sample.sample_id))
                case = {"image_hashes": [image_digest(im) for im in images], "grids": [], "tensor_hashes": [],
                        "study": study_info, "perturbation": transform}
                rgb_by_condition[name] = []
                if images:
                    inputs = processor.image_processor(images=list(images), return_tensors="pt", device="cpu")
                    case["grids"] = inputs["image_grid_thw"].tolist()
                    inferred_tokens = sum(t * h * w // model["processor"]["parameters"]["merge_size"]**2 for t, h, w in case["grids"])
                    case["grid_report"] = grid_report(case["grids"], len(images), model, inferred_tokens)
                    offset = 0
                    for index, (im, entry) in enumerate(zip(images, case["grid_report"]["images"])):
                        pixels = inputs["pixel_values"][offset:offset + entry["patches"]]
                        offset += entry["patches"]
                        case["tensor_hashes"].append(tensor_digest(pixels))
                        rgb = unpack_image(pixels, entry["grid_thw"], model["processor"]["parameters"])
                        rgb_by_condition[name].append(rgb)
                        # Equal dimensions AND recovered pixels establish that no later resize removed the intervention.
                        if rgb.shape[:2] != (im.height, im.width) or not np.array_equal(np.asarray(to_image(rgb)), np.asarray(im)):
                            raise ValueError("Processor reconstruction differs from the intervened study image")
                    if offset != len(inputs["pixel_values"]):
                        raise ValueError("Unassigned processor pixel rows")
                report["cases"][sample.sample_id][name] = case
            cases = report["cases"][sample.sample_id]
            if any(cases[name]["grids"] != cases["baseline"]["grids"] for name in ("blur", "resolution")):
                raise ValueError("Paired conditions differ in grid")
            links = []
            for index, original in enumerate(source):
                key = digest([sample.sample_id, index])[:16]
                base = rgb_by_condition["baseline"][index]
                item = {"sample_id": sample.sample_id, "image_index": index,
                        "source_sha256": file_digest(sample.image_paths[index]), "metrics": {}}
                for name in ("blur", "resolution"):
                    rgb = rgb_by_condition[name][index]
                    energy = gradient_energy(base)
                    changed = cases[name]["tensor_hashes"][index] != cases["baseline"]["tensor_hashes"][index]
                    item["metrics"][name] = {"changed": changed, "rgb_mae_0_1": float(np.abs(rgb - base).mean()),
                                            "gradient_ratio": gradient_energy(rgb) / energy if energy else None}
                    if not changed:
                        report["failures"].append({"sample_id": sample.sample_id, "image_index": index, "condition": name,
                                                   "reason": "identical processor tensor; no intervention effect"})
                actual = [to_image(rgb_by_condition[name][index]) for name in ("baseline", "blur", "resolution")]
                for name, image in zip(("baseline", "blur", "resolution"), actual):
                    image.save(output / "images" / f"{key}-{name}.png")
                study[index].save(output / "images" / f"{key}-study.png")
                original_preview = original.copy()
                original_preview.thumbnail((720, 720))
                original_preview.save(output / "images" / f"{key}-source-thumbnail.png")
                preview = f"images/{key}-preview.png"
                montage([original, study[index], *actual],
                        ["Source (display fitted)", "Study RGB / bicubic", "Actual baseline tensor", "Actual blur tensor", "Actual resolution tensor"], output / preview)
                item["preview"] = preview
                report["images"].append(item)
                links.append(f'<p>Image {index + 1}</p><a href="{preview}"><img loading="lazy" src="{preview}" width="100%"></a>')
            sections.append(f'<details><summary>Sample {html.escape(sample.sample_id)} — {len(source)} images</summary>{"".join(links)}</details>')
            write_json(output / "audit.json", report)
            print(f"Audited {len(report['cases'])}/{len(samples)} questions", flush=True)
        report["status"] = "passed" if not report["failures"] else "failed"
        report["summary"] = {"n_questions": len(samples), "n_conditions": 4, "n_images": len(report["images"]),
                             "n_pairs": len(samples) * 4, "n_degradation_checks": len(report["images"]) * 2,
                             "no_further_resizing": True, "processor_rgb_reconstruction_exact_u8": True}
        (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Pilot input audit</title>'
            '<style>body{font:16px system-ui;max-width:1800px;margin:24px auto;padding:12px}summary{padding:14px;cursor:pointer}details{border-bottom:1px solid #ddd}</style>'
            '<h1>Pilot50 actual processor input audit</h1><p>Development only. Text-only carries no images. '
            'Source and montage views are display-fitted; individual tensor PNGs retain actual dimensions. '
            'Preview RGB is reconstructed from packed pixel_values, inverse-normalized and rounded to uint8. '
            'Tensor changes do not establish biological evidence loss.</p>' + "".join(sections), encoding="utf-8")
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json(output / "audit.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/pilot.json")
    parser.add_argument("--output", default="outputs/input-audit/pilot50-v1")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    report = run_audit(load_config(args.config), args.output, args.root)
    print(report.get("summary"))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
