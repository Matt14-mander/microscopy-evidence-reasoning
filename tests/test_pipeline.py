import json
import tempfile
import unittest
from pathlib import Path

from experiments.run import REPO, run_experiment
from src.config import load_config, validate_config
from src.data.prepare import create_demo
from src.evaluation.report import make_report
from src.utils import read_json


class SpyModel:
    def __init__(self):
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return '{"answer":"A","confidence":0.8,"abstain":false}'


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        create_demo(self.root / "data/demo")
        self.config = load_config(REPO / "configs/demo.json")
        self.config["evaluation"]["bootstrap_samples"] = 20

    def test_end_to_end_cache_text_only_and_report(self):
        spy = SpyModel()
        first = run_experiment(self.config, self.root, spy)
        manifest = read_json(first / "manifest.json")
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(len(spy.requests), 24)
        self.assertEqual(sum(not req.images for req in spy.requests), 4)
        self.assertEqual(len(read_json(first / "metrics.json")), 6)
        self.assertTrue(make_report(first, tables_only=True).exists())
        with self.assertRaises(FileExistsError):
            run_experiment(self.config, self.root, spy)
        self.config["output_dir"] = "outputs/replay"
        second_spy = SpyModel()
        second = run_experiment(self.config, self.root, second_spy)
        self.assertEqual(len(second_spy.requests), 0)
        self.assertEqual(read_json(second / "manifest.json")["cache_hits"], 24)
        self.assertEqual((first / "predictions.jsonl").read_bytes(), (second / "predictions.jsonl").read_bytes())

    def test_image_content_invalidates_visual_cache_only(self):
        from PIL import Image
        run_experiment(self.config, self.root, SpyModel())
        Image.new("RGB", (96, 80), "red").save(self.root / "data/demo/images/0.png")
        self.config["output_dir"] = "outputs/changed"
        spy = SpyModel()
        result = run_experiment(self.config, self.root, spy)
        # A uniform image is unchanged by blur/downsampling, so those identical
        # requests also reuse the newly generated baseline within this run.
        self.assertEqual(len(spy.requests), 3)
        self.assertEqual(read_json(result / "manifest.json")["cache_hits"], 21)

    def test_missing_regions_fail_before_model_call(self):
        path = self.root / "data/demo/samples.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[-1]["evidence_regions"] = []
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        spy = SpyModel()
        with self.assertRaisesRegex(ValueError, "evidence_regions"):
            run_experiment(self.config, self.root, spy)
        self.assertEqual(spy.requests, [])

    def test_model_failure_persists_failed_manifest(self):
        class BrokenModel:
            def generate(self, request):
                raise RuntimeError("failed transport")
        with self.assertRaises(RuntimeError):
            run_experiment(self.config, self.root, BrokenModel())
        folder = self.root / "outputs/demo"
        self.assertEqual(read_json(folder / "manifest.json")["status"], "failed")
        with self.assertRaises(ValueError):
            make_report(folder, tables_only=True)

    def test_configs_validate_and_inheritance_cycles_fail(self):
        for path in (REPO / "configs/experiments").glob("*.json"):
            validate_config(load_config(path))
        path = self.root / "cycle.json"
        path.write_text('{"extends":"cycle.json"}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Cyclic"):
            load_config(path)
