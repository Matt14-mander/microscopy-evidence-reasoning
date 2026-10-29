import copy
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from PIL import Image

from experiments.run import REPO, run_experiment
from src.config import load_config, validate_config, merge
from src.data.prepare import create_demo
from src.data.audit import audit_dataset
from src.data.splits import build_split
from src.data.pilot import select_pilot, apply_pilot
from src.evaluation.input_audit import audit_identity
from src.perturbations.study import grid_size, prepare_study_images
from src.perturbations.transforms import apply_condition
from src.utils import digest, image_digest, read_json, write_json


class PilotTests(unittest.TestCase):
    def fixture(self):
        rows = [{"sample_id": str(i), "task": f"task{i % 3}", "n_images": 1 if i % 2 else 2,
                 "answer_index": i % 4} for i in range(60)]
        rows[0]["n_images"] = 6
        split = {"data_sha256": "data", "partitions": {"dev": {"sample_ids": [str(i) for i in range(50)]},
                                                        "test": {"sample_ids": [str(i) for i in range(50, 60)]}},
                 "group_assignments": {str(i): f"g{i // 3}" for i in range(60)}}
        split["split_id"] = digest(split)
        return {"status": "valid", "data_sha256": "data", "samples": rows}, split

    def test_selection_is_label_blind_order_invariant_and_stratified(self):
        audit, split = self.fixture()
        first = select_pilot(audit, split, 25)
        audit["samples"].reverse()
        for row in audit["samples"]:
            row["answer_index"] = 99
            row["prediction"] = "ignored"
        self.assertEqual(first, select_pilot(audit, split, 25))
        self.assertEqual(len(set(first["sample_ids"])), 25)
        self.assertIn("0", first["sample_ids"])
        self.assertFalse(set(first["sample_ids"]) & set(split["partitions"]["test"]["sample_ids"]))
        self.assertEqual(sum(first["stratum_quotas"].values()), 25)
        for s, quota in first["stratum_quotas"].items():
            self.assertEqual(quota, sum(r["task"] + ("/single" if r["n_images"] == 1 else "/multi") == s for r in first["samples"]))

    def test_manifest_rejects_duplicate_missing_test_and_corruption(self):
        audit, split = self.fixture()
        manifest = select_pilot(audit, split, 25)
        samples = [SimpleNamespace(sample_id=str(i)) for i in range(50)]
        self.assertEqual(len(apply_pilot(samples, manifest, split)), 25)
        for ids in (["0", "0"], ["51"], ["missing"]):
            bad = copy.deepcopy(manifest)
            bad["sample_ids"] = ids
            bad["selection_id"] = digest({k: v for k, v in bad.items() if k != "selection_id"})
            with self.assertRaises(ValueError):
                apply_pilot(samples, bad, split)
        manifest["seed"] = 99
        with self.assertRaises(ValueError):
            apply_pilot(samples, manifest, split)

    def test_pilot_forbids_partial_limit_and_mismatched_study_budget(self):
        config = load_config(REPO / "configs/experiments/pilot.json")
        validate_config(config)
        config["data"]["limit"] = 5
        with self.assertRaisesRegex(ValueError, "limit"):
            validate_config(config)
        config["data"]["limit"] = None
        config["study_preprocessing"]["max_pixels"] = 100000
        with self.assertRaisesRegex(ValueError, "budgets"):
            validate_config(config)

    def test_study_resize_and_relative_blur_preserve_order_without_mutation(self):
        config = load_config(REPO / "configs/experiments/pilot.json")
        source = [Image.fromarray(np.random.default_rng(i).integers(0, 256, (100, 200, 3), dtype=np.uint8)) for i in range(2)]
        hashes = [image_digest(im) for im in source]
        study, info = prepare_study_images(source, config["study_preprocessing"])
        self.assertEqual(grid_size(100, 200, 50176, 200704), (168, 336))
        blurred, metadata = apply_condition(study, (), {"name": "blur", "kind": "blur", "radius_fraction": 0.02}, 42)
        self.assertEqual(metadata["radii"], [168 * 0.02] * 2)
        self.assertEqual([image_digest(im) for im in source], hashes)
        self.assertEqual(info["source_hashes"], hashes)
        self.assertNotEqual(image_digest(study[0]), image_digest(study[1]))
        self.assertTrue(all(im.size == ref.size and np.asarray(im).var() < np.asarray(ref).var() for im, ref in zip(blurred, study)))
        self.assertEqual(apply_condition(study, (), {"name": "text_only", "kind": "text_only"}, 42)[0], ())

    def test_input_audit_identity_shared_across_protocols_but_not_visual_changes(self):
        config = load_config(REPO / "configs/experiments/pilot.json")
        selection = {"selection_id": "selection", "split_id": "split", "data_sha256": "data"}
        first = audit_identity(config, selection)
        config["protocol"]["name"] = "allow_abstain"
        self.assertEqual(first, audit_identity(config, selection))
        changed = copy.deepcopy(config)
        changed["conditions"][2]["radius_fraction"] = 0.03
        self.assertNotEqual(first, audit_identity(changed, selection))

    def test_runner_exact_manifest_matrix_and_failed_audit_before_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_demo(root / "data/demo")
            # The demo intentionally repeats identical images; create independent groups for this fixture.
            for i in range(4):
                path = root / f"data/demo/images/{i}.png"
                with Image.open(path) as source:
                    image = source.copy()
                image.putpixel((0, 0), (i, 10, 20))
                image.save(path)
            audit = audit_dataset(root / "data/demo/samples.jsonl")
            split = build_split(audit, dev_fraction=0.75)
            selection = select_pilot(audit, split, n=2)
            write_json(root / "split.json", split)
            write_json(root / "selection.json", selection)
            config = load_config(REPO / "configs/demo.json")
            config["data"].update(split_manifest="split.json", partition="dev", sample_manifest="selection.json", limit=None)
            config["conditions"] = load_config(REPO / "configs/experiments/pilot.json")["conditions"]
            config["evaluation"]["bootstrap_samples"] = 20
            output = run_experiment(config, root)
            rows = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
            expected = {(key, c["name"]) for key in selection["sample_ids"] for c in config["conditions"]}
            self.assertEqual(len(rows), 8)
            self.assertEqual({(r["sample_id"], r["condition"]) for r in rows}, expected)
            self.assertTrue(read_json(output / "manifest.json")["pairing_complete"])
            config = merge(config, load_config(REPO / "configs/models/local_4070.json"))
            config["study_preprocessing"] = load_config(REPO / "configs/experiments/pilot.json")["study_preprocessing"]
            config["input_audit"] = "audit.json"
            config["output_dir"] = "outputs/blocked"
            write_json(root / "audit.json", {"status": "failed", "identity": {}})
            with patch("experiments.run.create_model") as create:
                with self.assertRaisesRegex(ValueError, "Input audit"):
                    run_experiment(config, root)
                create.assert_not_called()
