"""Generate publication figures for the SV-Hop manuscript."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

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


def fig1():
    bench = json.load(open("data/pdb_hops/benchmark.json"))
    fam = bench["stats"]["per_family"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax = axes[0]
    names = ["kinase", "enzyme_other", "protease", "nuclear_receptor", "gpcr"]
    labels = ["Kinase", "Enzyme/other", "Protease", "Nuclear receptor", "GPCR"]
    vals = [fam.get(n, 0) for n in names]
    ax.bar(labels, vals, color=C["blue"])
    ax.set_ylabel("Scaffold-hop pairs")
    ax.set_title("Pairs by target family", loc="left", fontsize=10)
    ax.tick_params(axis="x", rotation=25)
    for i, v in enumerate(vals):
        ax.text(i, v + 25, str(v), ha="center", fontsize=8)
    ax = axes[1]
    splits = bench["splits"]
    x = np.arange(3)
    w = 0.36
    t = [len(splits["target"][s]) for s in ("train", "val", "test")]
    sc = [len(splits["scaffold"][s]) for s in ("train", "val", "test")]
    ax.bar(x - w / 2, t, w, label="Target-held-out", color=C["blue"])
    ax.bar(x + w / 2, sc, w, label="Scaffold-held-out", color=C["orange"])
    ax.set_xticks(x)
    ax.set_xticklabels(["Train", "Val", "Test"])
    ax.set_ylabel("Pairs")
    ax.set_title("Held-out splits", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    save(fig, "fig1_benchmark")


def fig2():
    tr = pd.read_csv("results/svhop_pose/transfer.csv")
    pc = pd.read_csv("results/svhop_contact/per_candidate.csv")
    top = pc[pc["rank"] == 0]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    bins = np.linspace(0, 1, 21)
    ax = axes[0]
    ax.hist(tr.true_recall, bins=bins, color=C["gray"], edgecolor="white", linewidth=0.5)
    ax.axvline(tr.true_recall.mean(), color=C["red"], lw=1.5,
               label=f"mean {tr.true_recall.mean():.3f}")
    ax.axvline(0.6, color=C["blue"], ls="--", lw=1, label="threshold 0.6")
    ax.set_xlabel("Contact recovery of true hop")
    ax.set_ylabel("Pairs")
    ax.set_title("Crystal-pose transfer (n=40)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    ax.hist(top.contact_recall, bins=bins, color=C["green"], edgecolor="white", linewidth=0.5)
    ax.axvline(top.contact_recall.mean(), color=C["red"], lw=1.5,
               label=f"mean {top.contact_recall.mean():.3f}")
    ax.axvline(0.6, color=C["blue"], ls="--", lw=1, label="threshold 0.6")
    ax.set_xlabel("Contact recovery of SV-Hop top-1")
    ax.set_title(f"Scaffold-blind candidates (n={len(top)})", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    save(fig, "fig2_3d_truth")


def box(ax, x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                                fc=color, ec="none"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, wrap=True)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10,
                                 lw=1, color="#333333"))


def fig3():
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    y = 0.55
    h = 0.22
    xs = [0.02, 0.19, 0.36, 0.53, 0.70]
    texts = ["PDB complexes\n(2,886 pairs)", "Core rules\n(60,069)", "Exact graph\nswap",
             "Property/SA/\nnovelty filters", "Learned\ninteraction ranker"]
    for x, t in zip(xs, texts):
        box(ax, x, y, 0.15, h, t, C["light"])
    for i in range(len(xs) - 1):
        arrow(ax, xs[i] + 0.15, y + h / 2, xs[i + 1], y + h / 2)
    ax.text(0.86, y + h / 2, "Top-k\nhops", ha="center", va="center", fontsize=9,
            color=C["blue"], weight="bold")
    arrow(ax, 0.85, y + h / 2, 0.83, y + h / 2)
    box(ax, 0.02, 0.12, 0.30, 0.16,
        "Crystal-pose transfer\n(contact-recovery ground truth, 0.806)", C["orange"])
    arrow(ax, 0.17, 0.28, 0.17, 0.55)
    save(fig, "fig3_svhop")


def fig4():
    b = json.load(open("results/svhop_full/bootstrap.json"))
    r = json.load(open("results/svhop_full/interaction_ranker.json"))
    arms = [
        ("SV-Hop (constrained)", r["arms"]["constrained"]["mean"],
         r["arms"]["constrained"]["ci_lo"], r["arms"]["constrained"]["ci_hi"], C["blue"]),
        ("Property ranker", b["arms"]["property"]["mean"],
         b["arms"]["property"]["ci_lo"], b["arms"]["property"]["ci_hi"], C["green"]),
        ("2D core similarity", b["arms"]["sim2d"]["mean"],
         b["arms"]["sim2d"]["ci_lo"], b["arms"]["sim2d"]["ci_hi"], C["orange"]),
        ("Random candidate", b["arms"]["random_expected"]["mean"],
         b["arms"]["random_expected"]["ci_lo"], b["arms"]["random_expected"]["ci_hi"], C["gray"]),
        ("Copy", 0.0, 0.0, 0.0, C["red"]),
    ]
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    ys = np.arange(len(arms))[::-1]
    for y, (name, m, lo, hi, col) in zip(ys, arms):
        ax.plot([lo, hi], [y, y], color=col, lw=2.5, solid_capstyle="round")
        ax.plot(m, y, "o", color=col, ms=6)
        ax.text(hi + 0.015, y, f"{m*100:.1f}%", va="center", fontsize=8)
    ax.set_yticks(ys)
    ax.set_yticklabels([a[0] for a in arms])
    ax.set_xlim(-0.02, 0.78)
    ax.set_xlabel("Top-1 joint success (novel + property + SA + contact ≥0.6)")
    ax.set_title("Scaffold-blind split, 298 pairs", loc="left", fontsize=10)
    ax.annotate("+11.7 pp\np<0.001", xy=(0.62, ys[0]), xytext=(0.55, ys[0] - 0.55),
                fontsize=8, color=C["blue"],
                arrowprops=dict(arrowstyle="->", lw=0.8, color=C["blue"]))
    save(fig, "fig4_main")


def fig5():
    p = pd.read_csv("results/prospective/candidates.csv")
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    cmap = {"A2A": C["blue"], "EGFR": C["orange"], "CDK2": C["green"],
            "AKT1": C["red"], "AR": C["gray"], "CA2": "#8C6BB1"}
    for t, g in p.groupby("target"):
        ax.scatter(g.contact_recall, g.sa, s=40, alpha=0.85,
                   color=cmap.get(t, C["gray"]), label=t, edgecolor="white", linewidth=0.5)
    ax.axvline(0.6, color=C["gray"], ls="--", lw=0.8)
    ax.set_xlabel("Predicted-contact-selected contact recovery")
    ax.set_ylabel("Synthetic accessibility (SA)")
    ax.set_title(f"Prospective panel (n={len(p)})", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    save(fig, "fig5_prospective")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    print("figures written to", OUT)
