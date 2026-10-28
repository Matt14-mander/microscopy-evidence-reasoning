import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from src.data.prepare import export_huggingface
from src.utils import read_json, write_json


class ExportTests(unittest.TestCase):
    def test_offline_snapshot_export_is_complete_and_preserves_all_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "snapshot/data").mkdir(parents=True)
            (root / "snapshot/data/test-00000.parquet").touch()
            write_json(root / "download.json", {"repo_id": "jmhb/microvqa", "repo_type": "dataset",
                                                "status": "complete", "revision": "abc", "snapshot_path": str(root / "snapshot")})
            rows = [{"key_question": 1, "key_image": 3, "question": "Q", "choices": ["a", "b"],
                     "correct_index": 0, "correct_answer": "a",
                     "images_list": [Image.new("RGB", (3, 3), "red"), Image.new("RGB", (3, 3), "blue")]}]
            fake = types.ModuleType("datasets")
            calls = []
            fake.load_dataset = lambda *args, **kwargs: (calls.append((args, kwargs)) or rows)
            hub = types.ModuleType("huggingface_hub")
            hub.HfApi = lambda: (_ for _ in ()).throw(AssertionError("Offline export must not query Hub"))
            with patch.dict("sys.modules", {"datasets": fake, "huggingface_hub": hub}):
                output = export_huggingface(root / "export", "abc", download_manifest=root / "download.json")
            source = read_json(root / "export/source.json")
            self.assertEqual(source["status"], "complete")
            self.assertEqual(source["count"], 1)
            self.assertTrue(output.exists())
            self.assertFalse((root / "export/samples.jsonl.partial").exists())
            import json
            record = json.loads(output.read_text())
            self.assertEqual(len(record["images"]), 2)
            self.assertEqual(calls[0][0], ("parquet",))

    def test_failed_export_never_publishes_complete_jsonl(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake = types.ModuleType("datasets")
            fake.load_dataset = lambda *args, **kwargs: [{"images_list": [None]}]
            hub = types.ModuleType("huggingface_hub")
            hub.HfApi = lambda: types.SimpleNamespace(dataset_info=lambda *a, **k: types.SimpleNamespace(sha="abc"))
            with patch.dict("sys.modules", {"datasets": fake, "huggingface_hub": hub}):
                with self.assertRaises(AttributeError):
                    export_huggingface(root / "export", "abc")
            self.assertFalse((root / "export/samples.jsonl").exists())
            self.assertEqual(read_json(root / "export/source.json")["status"], "failed")
