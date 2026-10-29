import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.run import REPO, run_experiment
from src.config import load_config, validate_config
from src.data.prepare import create_demo
from src.models.base import ModelRequest, build_prompt
from src.models.cache import request_identity, read_prediction_cache, write_prediction_cache
from src.models.parsing import parse_prediction
from src.models.protocols import resolve_protocol, text_sha256
from src.utils import digest, read_json, write_json


FORCED = {"name": "forced_choice", "version": "1.0"}
ALLOW = {"name": "allow_abstain", "version": "1.0"}
ABSTAIN = '{"answer":null,"confidence":null,"abstain":true}'


class ProtocolTests(unittest.TestCase):
    def test_released_template_hashes_and_exact_prompt_hash(self):
        self.assertEqual(resolve_protocol(FORCED)["template_sha256"],
                         "fa4beab38a63ab013c966e072ed79ae8b47aeb50ed38e275a0494b39a9c9285a")
        self.assertEqual(resolve_protocol(ALLOW)["template_sha256"],
                         "6f74662604b60202b61b48aa1e08a033207dd8afb5d9819e91deec1673373ceb")
        prompt = build_prompt("图像 {question}?", ["{options}", "β"], FORCED)
        self.assertIn("Question: 图像 {question}?\nA. {options}\nB. β", prompt)
        self.assertEqual(text_sha256(prompt), hashlib.sha256(prompt.encode("utf-8")).hexdigest())
        self.assertNotEqual(prompt, build_prompt("图像 {question}?", ["{options}", "β"], ALLOW))

    def test_abstention_is_protocol_specific(self):
        allowed = parse_prediction(ABSTAIN, 4, ALLOW)
        forced = parse_prediction(ABSTAIN, 4, FORCED)
        self.assertTrue(allowed.valid and allowed.abstain)
        self.assertFalse(forced.valid or forced.abstain)
        self.assertEqual(forced.error_kind, "protocol")
        for spec in (FORCED, ALLOW):
            prediction = parse_prediction('{"answer":"b","confidence":null,"abstain":false}', 4, spec)
            self.assertTrue(prediction.valid)
            self.assertEqual(prediction.answer_index, 1)

    def test_invalid_outputs_remain_invalid_in_both_protocols(self):
        for raw in ('I cannot answer', '{"answer":null,"confidence":null,"abstain":false}',
                    '{"answer":"Z","confidence":0.9,"abstain":false}',
                    '{"answer":"A","answer":"B","confidence":0.9,"abstain":false}',
                    '{"answer":"A","confidence":NaN,"abstain":false}',
                    '{"answer":"A","confidence":0.9,"abstain":false,"extra":1}',
                    '{"answer":null,"confidence":0.9,"abstain":true}',
                    '{"answer":"A","confidence":0.9,"abstain":true}'):
            for spec in (FORCED, ALLOW):
                with self.subTest(raw=raw, protocol=spec):
                    p = parse_prediction(raw, 4, spec)
                    self.assertFalse(p.valid or p.abstain)
                    self.assertIsNotNone(p.error_kind)

    def test_unknown_or_unversioned_protocol_is_configuration_error(self):
        for spec in ({"name": "forced_choice"}, {"name": "typo", "version": "1.0"},
                     {"name": "forced_choice", "version": "2.0"}, {}, "forced_choice"):
            with self.assertRaises(ValueError):
                parse_prediction(ABSTAIN, 4, spec)
            config = load_config(REPO / "configs/base.json")
            config["protocol"] = spec
            with self.assertRaises(ValueError):
                validate_config(config)

    def test_cache_isolates_protocol_even_if_prompt_bytes_are_identical(self):
        req = ModelRequest("1", "identical prompt", (), 4, 42)
        a = request_identity(req, {"backend": "mock"}, {}, "code", resolve_protocol(ALLOW))
        b = request_identity(req, {"backend": "mock"}, {}, "code", resolve_protocol(FORCED))
        self.assertNotEqual(digest(a), digest(b))
        changed = copy.deepcopy(a)
        changed["protocol"]["version"] = "2.0"
        self.assertNotEqual(digest(a), digest(changed))
        changed = copy.deepcopy(a)
        changed["protocol"]["template_sha256"] = "changed"
        self.assertNotEqual(digest(a), digest(changed))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.json"
            write_prediction_cache(path, a, ABSTAIN, None)
            with self.assertRaises(ValueError):
                read_prediction_cache(path, b)
            entry = read_json(path)
            entry["identity"]["protocol"] = b["protocol"]
            write_json(path, entry)
            with self.assertRaises(ValueError):
                read_prediction_cache(path, a)

    def test_runner_shared_cache_replay_and_protocol_denominators(self):
        class Abstainer:
            calls = 0

            def generate(self, request):
                self.calls += 1
                return ABSTAIN

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_demo(root / "data/demo")
            config = load_config(REPO / "configs/demo.json")
            config["conditions"] = [{"name": "baseline", "kind": "baseline"},
                                    {"name": "text_only", "kind": "text_only"}]
            config["evaluation"]["bootstrap_samples"] = 20
            keys = []
            for spec in (FORCED, ALLOW):
                config.update(protocol=spec, output_dir=f"outputs/{spec['name']}")
                model = Abstainer()
                folder = run_experiment(config, root, model)
                self.assertEqual(model.calls, 8)
                self.assertEqual(read_json(folder / "manifest.json")["cache_hits"], 0)
                metric = read_json(folder / "metrics.json")["baseline"]
                self.assertEqual(metric["n"], 4)
                self.assertEqual(metric["accuracy"], 0)
                self.assertEqual(metric["invalid_rate"], 1 if spec == FORCED else 0)
                self.assertEqual(metric["abstention_rate"], 0 if spec == FORCED else 1)
                self.assertEqual(metric["n_protocol_violations"], 4 if spec == FORCED else 0)
                rows = [json.loads(s) for s in (folder / "predictions.jsonl").read_text().splitlines()]
                keys.append({r["cache_key"] for r in rows})
                self.assertTrue(all(r["raw"] == ABSTAIN for r in rows))
                self.assertTrue(all(r["protocol_id"] == resolve_protocol(spec)["protocol_id"] for r in rows))
                config["output_dir"] += "-replay"
                replay_model = Abstainer()
                replay = run_experiment(config, root, replay_model)
                self.assertEqual(replay_model.calls, 0)
                self.assertEqual(read_json(replay / "manifest.json")["cache_hits"], 8)
                self.assertEqual((folder / "predictions.jsonl").read_bytes(), (replay / "predictions.jsonl").read_bytes())
            self.assertFalse(keys[0] & keys[1])
