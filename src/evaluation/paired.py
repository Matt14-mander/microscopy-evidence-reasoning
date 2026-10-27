"""ID-aligned comparisons; bootstrap whole image groups when available."""

import math

import numpy as np


def _index(records):
    result = {row["sample_id"]: row for row in records}
    if len(result) != len(records):
        raise ValueError("Duplicate ids in paired predictions")
    return result


def paired_comparison(baseline, intervention, seed=0, n_bootstrap=2000):
    left, right = _index(baseline), _index(intervention)
    if not left or left.keys() != right.keys():
        raise ValueError("Paired comparisons require identical, nonempty sample-id sets")
    if type(n_bootstrap) is not int or n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    ids = sorted(left)
    for key in ids:
        if left[key]["answer_index"] != right[key]["answer_index"]:
            raise ValueError(f"Ground truth differs for {key}")
        if left[key].get("group_id") != right[key].get("group_id"):
            raise ValueError(f"Image group differs for {key}")
    a = np.array([bool(left[key]["correct"]) for key in ids])
    b = np.array([bool(right[key]["correct"]) for key in ids])
    difference = b.astype(float) - a.astype(float)
    groups = {}
    for i, key in enumerate(ids):
        group = left[key].get("group_id")
        group = ("image", str(group)) if group is not None else ("question", key)
        groups.setdefault(group, []).append(i)
    sums = np.array([difference[indices].sum() for indices in groups.values()])
    sizes = np.array([len(indices) for indices in groups.values()])
    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        draws = rng.integers(0, len(groups), size=len(groups))
        boot[i] = sums[draws].sum() / sizes[draws].sum()
    lost, gained = int(np.sum(a & ~b)), int(np.sum(~a & b))
    discordant = lost + gained
    # Exact two-sided binomial test; integer arithmetic avoids float overflow.
    p = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(lost, gained) + 1)) / (2 ** discordant))
    return {
        "n_pairs": len(ids), "n_groups": len(groups),
        "accuracy_delta": float(difference.mean()),
        "delta_ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        "bootstrap_unit": "image_group_with_question_fallback", "n_bootstrap": n_bootstrap,
        "baseline_correct_intervention_wrong": lost,
        "baseline_wrong_intervention_correct": gained,
        "mcnemar_exact_p": p,
        "mcnemar_assumption": "Independent question pairs; shared-image correlation is not adjusted",
    }
