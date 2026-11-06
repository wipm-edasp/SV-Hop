"""Figure 6: target-held-out evaluation and constraint ablation."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "out/figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.dpi": 150, "savefig.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
C = {"blue": "#2E5EAA", "orange": "#E07A2F", "green": "#3A8D5D", "red": "#B23A48",
     "gray": "#7A7A7A", "light": "#D9E2EC"}


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight")
    plt.close(fig)


def main():
    s = json.load(open("results/target_holdout/summary.json"))
    a = json.load(open("results/target_holdout/ablation.json"))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    ax = axes[0]
    keys = ["copy", "random_expected", "sim2d", "property", "constrained"]
    labels = ["copy", "random", "2D", "property", "SV-Hop"]
    vals = [100 * s["arms"][k]["mean"] for k in keys]
    err = [[100 * (s["arms"][k]["mean"] - s["arms"][k]["ci_lo"]) for k in keys],
           [100 * (s["arms"][k]["ci_hi"] - s["arms"][k]["mean"]) for k in keys]]
    colors = [C["gray"], C["gray"], C["orange"], C["orange"], C["blue"]]
    ax.bar(labels, vals, color=colors, yerr=err, capsize=2.5,
           error_kw={"lw": 0.8, "ecolor": "#333333"})
    for i, v in enumerate(vals):
        ax.text(i, v + err[1][i] + 2.0, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_ylim(0, 88)
    ax.set_ylabel("Joint top-1 (%)")
    ax.set_title("Target-held-out (6 unseen targets)", loc="left", fontsize=10)
    ax.tick_params(axis="x", rotation=20)

    ax = axes[1]
    keys = ["ranker_only", "drop_both", "drop_sa", "drop_novelty", "full"]
    labels = ["contact\nonly", "drop SA\n+novelty", "drop SA", "drop\nnovelty", "full"]
    vals = [100 * a["arms"][k]["mean"] for k in keys]
    err = [[100 * (a["arms"][k]["mean"] - a["arms"][k]["ci_lo"]) for k in keys],
           [100 * (a["arms"][k]["ci_hi"] - a["arms"][k]["mean"]) for k in keys]]
    colors = [C["red"]] + [C["light"]] * 3 + [C["blue"]]
    ax.bar(labels, vals, color=colors, yerr=err, capsize=2.5,
           error_kw={"lw": 0.8, "ecolor": "#333333"})
    for i, v in enumerate(vals):
        ax.text(i, v + err[1][i] + 2.0, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_ylim(0, 88)
    ax.set_ylabel("Joint top-1 (%)")
    ax.set_title("Constraint ablation", loc="left", fontsize=10)
    ax.tick_params(axis="x", rotation=0, labelsize=7.5)

    fig.tight_layout()
    save(fig, "fig6_target_holdout")
    print("wrote fig6_target_holdout")


if __name__ == "__main__":
    main()

