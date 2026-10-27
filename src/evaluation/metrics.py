"""Question-level metrics with explicit denominators and missing values."""

import numpy as np


def compute_metrics(records, bins=10):
    if not records:
        raise ValueError("Cannot evaluate an empty prediction set")
    if type(bins) is not int or bins <= 0:
        raise ValueError("bins must be positive")
    n = len(records)
    answered = [r for r in records if r["valid"] and not r["abstain"]]
    confident = [r for r in answered if r["confidence"] is not None]
    correct = sum(bool(r["correct"]) for r in answered)
    result = {
        "n": n, "n_answered": len(answered), "n_confidence": len(confident),
        "accuracy": correct / n,
        "coverage": len(answered) / n,
        "abstention_rate": sum(r["valid"] and r["abstain"] for r in records) / n,
        "invalid_rate": sum(not r["valid"] for r in records) / n,
        "selective_accuracy": correct / len(answered) if answered else None,
        "selective_risk": 1 - correct / len(answered) if answered else None,
        "mean_confidence": None, "ece": None, "correctness_brier": None,
    }
    if confident:
        conf = np.array([r["confidence"] for r in confident], dtype=float)
        acc = np.array([r["correct"] for r in confident], dtype=float)
        if not np.isfinite(conf).all() or np.any((conf < 0) | (conf > 1)):
            raise ValueError("Confidence values must be finite and in [0,1]")
        assignments = np.minimum((conf * bins).astype(int), bins - 1)
        ece = 0.0
        for index in range(bins):
            selected = assignments == index
            if selected.any():
                ece += selected.mean() * abs(conf[selected].mean() - acc[selected].mean())
        result.update(mean_confidence=float(conf.mean()), ece=float(ece),
                      correctness_brier=float(np.mean((conf - acc) ** 2)))
    return result
