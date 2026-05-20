"""Identify SAE features that fire differently on failure vs. success.

Loads:
    results/predictions.csv   (id, success, ...)
    results/activations.npy   (N, d_sae) per-task mean feature activations

For each feature, computes:
    - mean activation on the failed subset
    - mean activation on the successful subset
    - Welch's t-statistic
    - Cohen's d (effect size)
    - p-value (uncorrected and Holm-Bonferroni adjusted)

Writes:
    results/feature_stats.csv     all features ranked by |Cohen's d|
    results/top_features.md       top-K features with Neuronpedia URLs
    figures/effect_distribution.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-SD Cohen's d. Sign: positive means a > b on average."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)
    pooled = ((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2)
    if pooled <= 0:
        return 0.0
    return float((a.mean() - b.mean()) / np.sqrt(pooled))


def holm_bonferroni(p: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values."""
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        m = n - rank
        v = min(1.0, p[idx] * m)
        running_max = max(running_max, v)
        adj[idx] = running_max
    return adj


def main(
    preds_path: str = "results/predictions.csv",
    acts_path: str = "results/activations.npy",
    out_stats: str = "results/feature_stats.csv",
    out_top: str = "results/top_features.md",
    out_fig: str = "figures/effect_distribution.png",
    sae_release: str = "gpt2-small-res-jb",
    sae_layer: int = 8,
    top_k: int = 20,
) -> None:
    preds = pd.read_csv(preds_path)
    acts = np.load(acts_path)
    assert len(preds) == acts.shape[0], "row count mismatch"

    is_fail = (preds["success"].values == 0)
    n_fail = int(is_fail.sum())
    n_succ = int((~is_fail).sum())
    print(f"N = {len(preds)} ({n_succ} success, {n_fail} fail, acc = {n_succ/len(preds):.3f})")

    fail_acts = acts[is_fail]
    succ_acts = acts[~is_fail]
    n_feat = acts.shape[1]
    print(f"Features: {n_feat}")

    # Per-feature: Welch's t-test fail vs success
    print("computing per-feature statistics ...")
    mean_fail = fail_acts.mean(axis=0)
    mean_succ = succ_acts.mean(axis=0)

    # Welch's t-test, vectorized
    var_fail = fail_acts.var(axis=0, ddof=1)
    var_succ = succ_acts.var(axis=0, ddof=1)
    se = np.sqrt(var_fail / n_fail + var_succ / n_succ)
    # Avoid div-by-zero: features that never fire have se=0; mark those NaN.
    t_stat = np.where(se > 0, (mean_fail - mean_succ) / se, 0.0)
    # Two-sided p-value with conservative df=min-1
    df = min(n_fail, n_succ) - 1
    p_uncorr = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=df))

    # Cohen's d per feature
    print("computing effect sizes ...")
    d_vals = np.zeros(n_feat)
    pooled_var = ((n_fail - 1) * var_fail + (n_succ - 1) * var_succ) / max(1, n_fail + n_succ - 2)
    sd = np.sqrt(np.where(pooled_var > 0, pooled_var, 1))
    d_vals = (mean_fail - mean_succ) / sd
    d_vals = np.where(pooled_var > 0, d_vals, 0.0)

    # Holm-Bonferroni adjustment
    p_adj = holm_bonferroni(p_uncorr)

    df_stats = pd.DataFrame({
        "feature": np.arange(n_feat),
        "mean_fail": mean_fail,
        "mean_succ": mean_succ,
        "delta": mean_fail - mean_succ,
        "cohens_d": d_vals,
        "t_stat": t_stat,
        "p_uncorr": p_uncorr,
        "p_holm": p_adj,
        "n_fail_active": (fail_acts > 0).sum(axis=0),
        "n_succ_active": (succ_acts > 0).sum(axis=0),
    })
    df_stats["abs_d"] = df_stats["cohens_d"].abs()
    df_stats = df_stats.sort_values("abs_d", ascending=False).reset_index(drop=True)

    Path(out_stats).parent.mkdir(parents=True, exist_ok=True)
    df_stats.to_csv(out_stats, index=False)
    print(f"wrote {out_stats}")

    # Top-K features with Neuronpedia URLs
    print(f"top {top_k} features by |Cohen's d|:")
    top_lines = [
        f"# Top {top_k} SAE features by |Cohen's d| (failure vs success)",
        "",
        f"Model: gpt2-small  |  SAE: {sae_release}/blocks.{sae_layer}.hook_resid_pre  |  N={len(preds)} ({n_fail} fail, {n_succ} succ)",
        "",
        "| Rank | Feature | Cohen's d | mean fail | mean succ | p (Holm) | Active on fail / succ | Neuronpedia |",
        "|---:|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for i, row in df_stats.head(top_k).iterrows():
        feat = int(row["feature"])
        d = row["cohens_d"]
        url = f"https://www.neuronpedia.org/gpt2-small/{sae_layer}-res-jb/{feat}"
        line = (
            f"| {i+1} | {feat} | {d:+.3f} | {row['mean_fail']:.4f} | "
            f"{row['mean_succ']:.4f} | {row['p_holm']:.4f} | "
            f"{int(row['n_fail_active'])}/{n_fail}, {int(row['n_succ_active'])}/{n_succ} | "
            f"[link]({url}) |"
        )
        top_lines.append(line)
        print(f"  feat {feat:5d}  d={d:+.3f}  mean_fail={row['mean_fail']:.4f}  mean_succ={row['mean_succ']:.4f}  p_holm={row['p_holm']:.4f}")

    Path(out_top).write_text("\n".join(top_lines) + "\n")
    print(f"wrote {out_top}")

    # Figure: distribution of effect sizes (log y-axis so the tails are visible).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(df_stats["cohens_d"], bins=100, color="steelblue", alpha=0.85, edgecolor="white")
        ax.set_yscale("log")
        ax.axvline(0, color="black", linewidth=0.7)
        # Mark large-effect thresholds
        for x in (-0.8, 0.8):
            ax.axvline(x, color="firebrick", linestyle="--", linewidth=0.7, alpha=0.7)
        ax.set_xlabel("Cohen's d  (failure − success)")
        ax.set_ylabel("Number of SAE features (log)")
        ax.set_title(f"Effect-size distribution across {n_feat:,} features  (N={len(preds)})")
        # Annotate the dashed lines
        ax.text(0.85, ax.get_ylim()[1] * 0.5, "|d|=0.8\n(large)",
                color="firebrick", fontsize=8, va="center")
        Path(out_fig).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_fig, dpi=150)
        print(f"wrote {out_fig}")
    except ImportError:
        print("(matplotlib not available; skipped figure)")

    # Summary numbers for the paper
    n_sig = int((df_stats["p_holm"] < 0.05).sum())
    n_large = int((df_stats["abs_d"] > 0.5).sum())
    print()
    print(f"Significant (Holm p < 0.05): {n_sig} / {n_feat}")
    print(f"|Cohen's d| > 0.5: {n_large} / {n_feat}")
    print(f"|Cohen's d| > 0.8 (large): {(df_stats['abs_d'] > 0.8).sum()} / {n_feat}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--preds", default="results/predictions.csv")
    p.add_argument("--acts", default="results/activations.npy")
    p.add_argument("--out-stats", default="results/feature_stats.csv")
    p.add_argument("--out-top", default="results/top_features.md")
    p.add_argument("--out-fig", default="figures/effect_distribution.png")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--sae-layer", type=int, default=8)
    args = p.parse_args()
    main(
        preds_path=args.preds,
        acts_path=args.acts,
        out_stats=args.out_stats,
        out_top=args.out_top,
        out_fig=args.out_fig,
        top_k=args.top_k,
        sae_layer=args.sae_layer,
    )
