import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.data.audit import audit_dataset
from src.data.microvqa import load_samples
from src.data.splits import build_split, select_partition
from src.utils import digest, file_digest


class AuditSplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = []
        for i in range(10):
            Image.new("RGB", (12, 10), (i * 20, 10, 5)).save(self.root / f"{i}.png")
            self.rows.append({"id": str(i), "question": "Question?", "choices": ["one", "two"],
                              "answer_index": i % 2, "images": [f"{i}.png"],
                              "metadata": {"key_image": i, "task_str": f"task-{i % 2}"}})
        self.path = self.root / "samples.jsonl"

    def write(self):
        self.path.write_text("\n".join(json.dumps(row) for row in self.rows), encoding="utf-8")

    def test_transitive_group_merging_and_no_leakage(self):
        self.rows[1]["images"] = ["0.png"]  # Different official groups, identical pixels.
        self.rows[2]["metadata"]["key_image"] = 1  # Linked transitively via official group.
        self.write()
        audit = audit_dataset(self.path)
        group = next(g for g in audit["groups"] if "0" in g["sample_ids"])
        self.assertEqual(group["sample_ids"], ["0", "1", "2"])
        self.assertEqual(audit["summary"]["n_leakage_groups"], 8)
        split = build_split(audit)
        self.assertEqual(split, build_split(audit))
        self.assertEqual(split["validation"], {"sample_id_overlap": 0, "group_id_overlap": 0,
                                               "rgb_image_overlap": 0, "all_samples_assigned": True})
        memberships = [set(part["sample_ids"]) for part in split["partitions"].values()]
        self.assertTrue(any({"0", "1", "2"} <= ids for ids in memberships))

    def test_audit_collects_errors_and_split_refuses(self):
        self.rows[0]["images"] = ["missing.png"]
        self.rows[1]["answer_index"] = 5
        self.write()
        audit = audit_dataset(self.path)
        self.assertEqual(audit["summary"]["n_errors"], 2)
        self.assertEqual(audit["status"], "invalid")
        with self.assertRaises(ValueError):
            build_split(audit)

    def test_duplicate_ids_invalid_and_all_shared_cannot_split(self):
        self.rows[1]["id"] = "0"
        self.write()
        self.assertEqual(audit_dataset(self.path)["status"], "invalid")
        self.rows[1]["id"] = "1"
        for row in self.rows:
            row["images"] = ["0.png"]
        self.write()
        with self.assertRaises(ValueError):
            build_split(audit_dataset(self.path))

    def test_partition_integrity_and_changed_data_rejected(self):
        self.write()
        split = build_split(audit_dataset(self.path))
        samples = load_samples(self.path)
        chosen = select_partition(samples, split, "dev", self.path)
        self.assertEqual([s.sample_id for s in chosen], split["partitions"]["dev"]["sample_ids"])
        self.assertEqual(chosen[0].metadata["key_image"], split["group_assignments"][chosen[0].sample_id])
        split["seed"] = 8
        with self.assertRaisesRegex(ValueError, "integrity"):
            select_partition(samples, split, "dev", self.path)
        split["split_id"] = digest({k: v for k, v in split.items() if k != "split_id"})
        self.path.write_text(self.path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            select_partition(samples, split, "dev", self.path)

    def test_split_ignores_labels_and_is_order_invariant(self):
        self.write()
        audit = audit_dataset(self.path)
        a = build_split(audit)
        audit["samples"].reverse()
        audit["groups"].reverse()
        for row in audit["samples"]:
            row["answer_index"] = 99
        self.assertEqual(a, build_split(audit))

    def test_changed_image_rejected_by_partition_lock(self):
        self.write()
        split = build_split(audit_dataset(self.path))
        split["image_files"] = [{"path": "0.png", "sha256": file_digest(self.root / "0.png")}]
        split["split_id"] = digest({k: v for k, v in split.items() if k != "split_id"})
        Image.new("RGB", (12, 10), "white").save(self.root / "0.png")
        with self.assertRaisesRegex(ValueError, "Image changed"):
            select_partition(load_samples(self.path), split, "dev", self.path)

    def test_runner_applies_partition_and_corrected_groups(self):
        from experiments.run import REPO, run_experiment
        from src.config import load_config
        from src.utils import write_json, read_json

        self.write()
        split = build_split(audit_dataset(self.path))
        write_json(self.root / "split.json", split)
        config = load_config(REPO / "configs/base.json")
        config["data"] = {"path": "samples.jsonl", "split_manifest": "split.json", "partition": "dev"}
        output = run_experiment(config, self.root)
        manifest = read_json(output / "manifest.json")
        self.assertEqual(manifest["sample_ids"], split["partitions"]["dev"]["sample_ids"])
        self.assertEqual(manifest["split_id"], split["split_id"])
        predictions = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
        self.assertTrue(all(row["group_id"] == split["group_assignments"][row["sample_id"]] for row in predictions))

    def test_smoke_selects_single_multi_and_stress_without_labels(self):
        from experiments.smoke import choose_smoke_samples

        self.rows[3]["images"] = ["3.png", "4.png"]
        self.rows[5]["images"] = ["5.png", "6.png", "7.png"]
        self.write()
        samples = load_samples(self.path)
        selected = choose_smoke_samples(samples)
        self.assertEqual(len(selected), 6)
        self.assertIn("5", [s.sample_id for s in selected])
        self.assertTrue(any(len(s.image_paths) == 1 for s in selected))
        self.assertTrue(any(len(s.image_paths) > 1 for s in selected))
