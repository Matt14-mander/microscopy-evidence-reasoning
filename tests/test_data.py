import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.data.microvqa import load_samples, normalize_record


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        Image.new("RGB", (20, 16), "red").save(self.root / "a.png")
        Image.new("RGB", (16, 12), "blue").save(self.root / "b.png")
        self.row = {"key_question": 7, "question": "Which?", "choices": ["yes", "no"],
                    "correct_index": 1, "correct_answer": "no", "images_list": ["a.png", "b.png"], "key_image": 3}

    def test_official_aliases_and_image_order(self):
        sample = normalize_record(self.row, self.root)
        self.assertEqual(sample.sample_id, "7")
        self.assertEqual(sample.answer_index, 1)
        self.assertEqual([im.getpixel((0, 0)) for im in sample.load_images()], [(255, 0, 0), (0, 0, 255)])
        self.assertEqual(sample.metadata["key_image"], 3)

    def test_invalid_label_and_missing_image(self):
        for value in (-1, 2, True, "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_record({**self.row, "correct_index": value}, self.root)
        with self.assertRaises(FileNotFoundError):
            normalize_record({**self.row, "images_list": ["missing.png"]}, self.root)

    def test_invalid_regions(self):
        for region in ({"image_index": 2, "box": [0, 0, 2, 2]},
                       {"image_index": 0, "box": [0, 0, 21, 2]},
                       {"image_index": 0, "box": [2, 0, 2, 2]}):
            with self.subTest(region=region), self.assertRaises(ValueError):
                normalize_record({**self.row, "evidence_regions": [region]}, self.root)

    def test_duplicate_and_empty_datasets(self):
        path = self.root / "samples.jsonl"
        path.write_text(json.dumps(self.row) + "\n" + json.dumps(self.row), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_samples(path)
        path.write_text("", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Empty"):
            load_samples(path)

    def test_answer_consistency_and_limit(self):
        with self.assertRaises(ValueError):
            normalize_record({**self.row, "correct_answer": "yes"}, self.root)
        path = self.root / "samples.jsonl"
        path.write_text(json.dumps(self.row), encoding="utf-8")
        self.assertEqual(len(load_samples(path, limit=1)), 1)
        with self.assertRaises(ValueError):
            load_samples(path, limit=0)
