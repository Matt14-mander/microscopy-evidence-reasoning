"""Export summary tables and a vector PDF plus a 300 dpi PNG figure."""

import argparse
import csv
from pathlib import Path

from src.utils import read_json


def make_report(run_dir, tables_only=False):
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / "manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("Only complete runs can be reported")
    metrics = read_json(run_dir / "metrics.json")
    paired = read_json(run_dir / "paired.json")["comparisons"]
    rows = [{"condition": name, **values} for name, values in metrics.items()]
    with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if paired:
        pair_rows = []
        for name, values in paired.items():
            flat = {key: value for key, value in values.items() if key != "delta_ci95"}
            flat.update(ci95_low=values["delta_ci95"][0], ci95_high=values["delta_ci95"][1])
            pair_rows.append({"condition": name, **flat})
        with (run_dir / "paired.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
            writer.writeheader()
            writer.writerows(pair_rows)
    if tables_only:
        return run_dir / "summary.csv"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError('Tables exported. For figures: pip install -e ".[plots]" or use --tables-only') from error
    names = list(metrics)
    with plt.rc_context({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.1), 4), layout="constrained")
        ax.bar(names, [metrics[name]["accuracy"] for name in names], color="#347f93", label="Accuracy (all questions)")
        ax.scatter(names, [metrics[name]["coverage"] for name in names], color="#c66536", marker="D", label="Coverage")
        is_demo = manifest["is_mock"] or manifest.get("contains_synthetic_data", False)
        ax.set(ylim=(0, 1.2), yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], ylabel="Fraction",
               title="Synthetic/mock demo — not benchmark results" if is_demo else "MicroVQA visual evidence experiments")
        ax.tick_params(axis="x", labelrotation=30)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper center", ncol=2, frameon=False, fontsize=8)
        fig.savefig(run_dir / "accuracy.pdf")
        fig.savefig(run_dir / "accuracy.png", dpi=300)
        plt.close(fig)
    return run_dir / "accuracy.pdf"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--tables-only", action="store_true")
    args = parser.parse_args()
    print(make_report(args.run_dir, args.tables_only))


if __name__ == "__main__":
    main()
