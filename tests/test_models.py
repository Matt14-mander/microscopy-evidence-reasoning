import json
import unittest
from unittest.mock import patch

from PIL import Image

from src.models.api import APIModel
from src.models.base import ModelRequest, build_prompt
from src.models.parsing import parse_prediction


class ModelTests(unittest.TestCase):
    def test_json_and_fence(self):
        for raw in ('{"answer":"B","confidence":0.8,"abstain":false}',
                    '```json\n{"answer":"b","confidence":null,"abstain":false}\n```'):
            result = parse_prediction(raw, 3)
            self.assertTrue(result.valid)
            self.assertEqual(result.answer_index, 1)

    def test_abstention_is_explicit(self):
        result = parse_prediction('{"answer":null,"confidence":null,"abstain":true}', 3)
        self.assertTrue(result.valid and result.abstain)
        invalid = parse_prediction("I cannot answer this question", 3)
        self.assertFalse(invalid.valid)
        self.assertFalse(invalid.abstain)

    def test_malformed_and_out_of_range_outputs(self):
        for raw in ('{"answer":"D","confidence":0.8,"abstain":false}',
                    '{"answer":"A","confidence":NaN,"abstain":false}',
                    '{"answer":"A","confidence":true,"abstain":false}',
                    '{"answer":"A","confidence":1.1,"abstain":false}',
                    '{"answer":"A","confidence":0.8,"abstain":"false"}',
                    '{"answer":"A","confidence":0.8,"abstain":true}',
                    '[]', 'A', '{"answer":"A"}'):
            with self.subTest(raw=raw):
                self.assertFalse(parse_prediction(raw, 3).valid)

    def test_prompt_and_request_exclude_ground_truth(self):
        prompt = build_prompt("Where?", ["left", "right"])
        self.assertIn("A. left", prompt)
        self.assertNotIn("answer_index", ModelRequest.__dataclass_fields__)
        self.assertNotIn("regions", ModelRequest.__dataclass_fields__)

    def test_api_payload_no_images_for_text_only(self):
        model = APIModel({"base_url": "http://localhost/v1", "name": "test", "require_api_key": False}, {})
        empty = ModelRequest("1", "Question", (), 2, 9)
        with_image = ModelRequest("1", "Question", (Image.new("RGB", (2, 2)),), 2, 9)
        self.assertEqual(len(model.payload(empty)["messages"][0]["content"]), 1)
        blocks = model.payload(with_image)["messages"][0]["content"]
        self.assertTrue(blocks[1]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_api_response_via_mock_transport(self):
        model = APIModel({"base_url": "http://localhost/v1", "name": "test", "require_api_key": False}, {})
        import io
        body = json.dumps({"choices": [{"message": {"content": "raw response"}}]}).encode()
        with patch("urllib.request.urlopen", return_value=io.BytesIO(body)) as transport:
            raw = model.generate(ModelRequest("1", "Question", (), 2, 9))
        self.assertEqual(raw, "raw response")
        self.assertEqual(transport.call_count, 1)
