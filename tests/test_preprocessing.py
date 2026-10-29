import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.config import load_config, merge, validate_config
from src.models.preprocessing import grid_report, processor_report
from src.models.local import LocalVLM
from src.models.base import ModelRequest
from experiments.run import REPO


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(REPO / "configs/models/local_4070.json")
        self.model = self.config["model"]

    def test_frozen_contract_rejects_implicit_or_invalid_settings(self):
        base = load_config(REPO / "configs/base.json")
        validate_config(merge(base, self.config))
        for change in (lambda m: m["processor"].update(use_fast=False),
                       lambda m: m["processor"]["parameters"].pop("resample"),
                       lambda m: m["processor"]["parameters"].update(merge_size=3),
                       lambda m: m.update(revision="main"),
                       lambda m: m.pop("min_pixels")):
            config = copy.deepcopy(self.config)
            change(config["model"])
            with self.assertRaises(ValueError):
                validate_config(merge(base, config))

    def test_grid_tokens_sizes_and_empty_text(self):
        report = grid_report([[1, 16, 16], [1, 32, 32]], 2, self.model, 320)
        self.assertEqual(report["visual_tokens"], 320)
        self.assertEqual(report["images"][0]["processed_size_hw"], [224, 224])
        self.assertEqual(report["images"][1]["visual_tokens"], 256)
        self.assertEqual(grid_report([], 0, self.model, 0)["visual_tokens"], 0)

    def test_grid_rejects_dropped_images_wrong_tokens_or_budget(self):
        for grids, count, tokens in (([], 1, 0), ([[1, 16, 16]], 1, 65),
                                     ([[1, 8, 8]], 1, 16), ([[1, 64, 64]], 1, 1024),
                                     ([[1, 17, 16]], 1, 68), ([[2, 16, 16]], 1, 128)):
            with self.subTest(grids=grids, tokens=tokens), self.assertRaises(ValueError):
                grid_report(grids, count, self.model, tokens)

    def test_processor_drift_fails_and_versions_change_fingerprint(self):
        image_processor = type("Qwen2VLImageProcessorFast", (), {})()
        for key, value in self.model["processor"]["parameters"].items():
            setattr(image_processor, key, value)
        image_processor.min_pixels = 50176
        image_processor.max_pixels = 200704
        image_processor.size = {"shortest_edge": 50176, "longest_edge": 200704}
        processor = type("Qwen2_5_VLProcessor", (), {})()
        processor.image_processor = image_processor
        processor.tokenizer = SimpleNamespace()
        processor.chat_template = "template"
        with patch("src.models.preprocessing.importlib.metadata.version", return_value="v1"):
            first = processor_report(processor, self.model)
        with patch("src.models.preprocessing.importlib.metadata.version", return_value="v2"):
            second = processor_report(processor, self.model)
            self.assertNotEqual(first["fingerprint"], second["fingerprint"])
            image_processor.size["longest_edge"] = 999999
            with self.assertRaisesRegex(ValueError, "size"):
                processor_report(processor, self.model)

    def test_failed_request_keeps_identity_and_clears_previous_telemetry(self):
        model = LocalVLM.__new__(LocalVLM)
        model.torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 0, is_available=lambda: False))
        model.preprocessing = {"fingerprint": "frozen"}
        model.last_generation = {"sample_id": "old", "generated_tokens": 123}
        model.prepare_inputs = Mock(side_effect=RuntimeError("simulated processor failure"))
        with patch.dict("sys.modules", {"transformers": SimpleNamespace(set_seed=lambda seed: None)}):
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                model.generate(ModelRequest("new", "question", (), 2, 42))
        self.assertEqual(model.last_generation["sample_id"], "new")
        self.assertEqual(model.last_generation["stage"], "preprocessing")
        self.assertEqual(model.last_generation["status"], "error")
        self.assertNotIn("generated_tokens", model.last_generation)
        self.assertGreaterEqual(model.last_generation["seconds"], 0)

    def test_generation_oom_retains_grid_and_does_not_retry(self):
        class OutOfMemoryError(RuntimeError):
            pass

        model = LocalVLM.__new__(LocalVLM)
        model.torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 0, is_available=lambda: False))
        model.preprocessing = {"fingerprint": "frozen"}
        model.model = SimpleNamespace(device="cpu")

        def prepare(request):
            model.last_generation["image_grid_thw"] = [[1, 16, 16]]
            return SimpleNamespace(to=lambda device: {})

        model.prepare_inputs = prepare
        model._generate = Mock(side_effect=OutOfMemoryError("simulated CUDA OOM"))
        with patch.dict("sys.modules", {"transformers": SimpleNamespace(set_seed=lambda seed: None)}):
            with self.assertRaises(OutOfMemoryError):
                model.generate(ModelRequest("oom", "question", (), 2, 42))
        self.assertEqual(model._generate.call_count, 1)
        self.assertEqual(model.last_generation["error_type"], "OutOfMemoryError")
        self.assertEqual(model.last_generation["stage"], "generation")
        self.assertEqual(model.last_generation["image_grid_thw"], [[1, 16, 16]])
