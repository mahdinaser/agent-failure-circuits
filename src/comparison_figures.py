"""Generate two new figures based on the ablation/baseline/seed runs.

1. figures/auc_comparison.png - bar chart comparing failure-prediction AUC
   across raw 768-dim residual stream vs. top-1, top-100, and all SAE
   features.
2. figures/multi_seed.png - small grouped bar chart showing per-seed
   overall accuracy, keys-subset failure rate, and indicating where
   feature 17,491 was vs was not the top-|d| feature.
3. figures/ablation_result.png - grouped bars comparing baseline vs.
   ablated accuracy on the all-prompts, keys, and others subsets.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def auc_comparison() -> None:
    # Hardcoded from results/raw_baseline_report.txt (just produced).
    labels = ["raw\n768-d", "SAE top-1\n(d=1)", "SAE top-100\n(d=100)", "all SAE\n(d=24,576)"]
    aucs = [0.929, 0.839, 0.927, 0.933]
    stds = [0.023, 0.035, 0.011, 0.017]
    colors = ["steelblue", "firebrick", "darkorange", "seagreen"]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    bars = ax.bar(range(len(labels)), aucs, yerr=stds, capsize=4, color=colors,
                  alpha=0.8, edgecolor="white", error_kw={"elinewidth": 0.8})
    for i, (a, s) in enumerate(zip(aucs, stds)):
        ax.text(i, a + s + 0.012, f"{a:.3f}", ha="center", fontsize=8, color="black")
    ax.axhline(0.5, color="gray", linewidth=0.5, linestyle=":", alpha=0.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("5-fold CV ROC AUC")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Failure-prediction AUC by feature representation")
    fig.tight_layout()
    fig.savefig("figures/auc_comparison.png", dpi=160)
    print("wrote figures/auc_comparison.png")


def multi_seed() -> None:
    df = pd.read_csv("results/multi_seed_summary.csv")
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    x = np.arange(len(df))
    w = 0.36
    b1 = ax.bar(x - w/2, df["accuracy"].values, w, color="steelblue",
                alpha=0.85, edgecolor="white", label="overall accuracy")
    b2 = ax.bar(x + w/2, df["keys_fail_rate"].values, w, color="firebrick",
                alpha=0.75, edgecolor="white", label="keys-subset fail rate")
    # Annotate top feature
    for i, row in df.iterrows():
        feat = int(row["top_feature"])
        # Mark whether feat is 17491
        marker = "★" if feat == 17491 else ""
        ax.text(i, max(row["accuracy"], row["keys_fail_rate"]) + 0.04,
                f"feat\n{feat}{marker}", ha="center", va="bottom", fontsize=7,
                color="black")
    ax.set_xticks(x)
    ax.set_xticklabels(df["seed"].values)
    ax.set_xlabel("random seed")
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("rate")
    ax.set_title("Robustness across 5 seeds: behaviour is stable, top feature is not")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig("figures/multi_seed.png", dpi=160)
    print("wrote figures/multi_seed.png")


def ablation_result() -> None:
    # From results/ablation_summary.csv aggregation (run earlier).
    labels = ["all\n(N=300)", "keys subset\n(N=45)", "others\n(N=255)"]
    base = [0.797, 0.067, 0.925]
    abl  = [0.793, 0.044, 0.925]
    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.bar(x - w/2, base, w, color="steelblue", alpha=0.85, edgecolor="white",
           label="baseline")
    ax.bar(x + w/2, abl, w, color="firebrick", alpha=0.75, edgecolor="white",
           label="feature 17,491 ablated")
    for i, (b, a) in enumerate(zip(base, abl)):
        ax.text(i - w/2, b + 0.01, f"{b:.3f}", ha="center", fontsize=7.5)
        ax.text(i + w/2, a + 0.01, f"{a:.3f}", ha="center", fontsize=7.5)
        # Delta
        delta = a - b
        ax.text(i, max(b, a) + 0.08, f"$\\Delta={delta:+.3f}$",
                ha="center", fontsize=8, color="dimgray")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("IOI accuracy")
    ax.set_title("Causal ablation of feature 17,491 does not restore keys-subset accuracy")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig("figures/ablation_result.png", dpi=160)
    print("wrote figures/ablation_result.png")


if __name__ == "__main__":
    Path("figures").mkdir(exist_ok=True)
    auc_comparison()
    multi_seed()
    ablation_result()
