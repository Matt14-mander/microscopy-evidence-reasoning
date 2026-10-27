import unittest

from src.evaluation.metrics import compute_metrics
from src.evaluation.paired import paired_comparison


def row(i, correct, valid=True, abstain=False, confidence=None, group=None):
    return {"sample_id": str(i), "answer_index": 0, "correct": correct,
            "valid": valid, "abstain": abstain, "confidence": confidence, "group_id": group}


class EvaluationTests(unittest.TestCase):
    def test_denominators_calibration_and_missing_confidence(self):
        metrics = compute_metrics([row(0, True, confidence=0.8), row(1, False, confidence=0.4),
                                   row(2, False, abstain=True), row(3, False, valid=False)])
        self.assertEqual(metrics["accuracy"], 0.25)
        self.assertEqual(metrics["coverage"], 0.5)
        self.assertEqual(metrics["selective_accuracy"], 0.5)
        self.assertEqual(metrics["abstention_rate"], 0.25)
        self.assertEqual(metrics["invalid_rate"], 0.25)
        self.assertAlmostEqual(metrics["ece"], 0.3)
        self.assertAlmostEqual(metrics["correctness_brier"], 0.1)
        self.assertIsNone(compute_metrics([row(0, True)])["ece"])

    def test_all_abstain(self):
        metrics = compute_metrics([row(0, False, abstain=True)])
        self.assertEqual(metrics["accuracy"], 0)
        self.assertIsNone(metrics["selective_accuracy"])
        with self.assertRaises(ValueError):
            compute_metrics([])

    def test_exact_mcnemar_alignment_and_bootstrap(self):
        baseline = [row(i, True, group=i // 2) for i in range(6)]
        other = [row(i, False, group=i // 2) for i in reversed(range(6))]
        result = paired_comparison(baseline, other, seed=4, n_bootstrap=100)
        self.assertEqual(result["accuracy_delta"], -1)
        self.assertEqual(result["delta_ci95"], [-1, -1])
        self.assertEqual(result["n_groups"], 3)
        self.assertEqual(result["mcnemar_exact_p"], 0.03125)
        self.assertEqual(result, paired_comparison(baseline, other, seed=4, n_bootstrap=100))
        self.assertEqual(paired_comparison(baseline, baseline, n_bootstrap=10)["mcnemar_exact_p"], 1)

    def test_mismatched_pairs_and_labels_rejected(self):
        with self.assertRaises(ValueError):
            paired_comparison([row(0, True)], [row(1, True)])
        with self.assertRaises(ValueError):
            paired_comparison([row(0, True)] * 2, [row(0, True)])
        with self.assertRaises(ValueError):
            paired_comparison([row(0, True)], [{**row(0, True), "answer_index": 1}])
