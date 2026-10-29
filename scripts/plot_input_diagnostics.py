"""Plot all audited images' pixel-change diagnostics, independently of model predictions."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils import read_json


def plot_diagnostics(directory):
    directory = Path(directory)
    audit = read_json(directory / "audit.json")
    if audit["status"] != "passed":
        raise ValueError("Input audit has not passed")
    items = audit["images"]
    x = np.arange(1, len(items) + 1)
    with plt.rc_context({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, layout="constrained")
        for name, color, marker in (("blur", "#196b80", "o"), ("resolution", "#c86636", "x")):
            axes[0].scatter(x, [im["metrics"][name]["rgb_mae_0_1"] for im in items], s=22, color=color, marker=marker, label=name)
            axes[1].scatter(x, [im["metrics"][name]["gradient_ratio"] for im in items], s=22, color=color, marker=marker)
        axes[0].set(yscale="log", ylabel="RGB MAE (0-1 scale, log axis)")
        axes[0].legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2)
        axes[1].set(xlabel="Image reference index (locked sample order; includes shared images)", ylabel="Gradient energy / baseline")
        axes[1].axhline(1, color="#777777", linestyle="--", linewidth=1)
        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.2)
        fig.suptitle(f"Actual processor input diagnostics — {len(items)} image references / {len(audit['cases'])} dev questions")
        fig.savefig(directory / "diagnostics.png", dpi=180)
        fig.savefig(directory / "diagnostics.pdf")
        plt.close(fig)
    return directory / "diagnostics.png"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="outputs/input-audit/pilot50-v1")
    print(plot_diagnostics(parser.parse_args().directory))
