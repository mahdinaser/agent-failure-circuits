"""Conditional analysis: drop the "the keys" subset (which is dominated by
the keys-detector feature spurious-firing effect) and re-run the failure-vs-
success analysis on the remainder of the corpus.

This isolates feature patterns that are NOT explained by the obvious
surface trigger.

Writes:
    results/feature_stats_no_keys.csv
    results/top_features_no_keys.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def holm_bonferroni(p: np.ndarray) -> np.ndarray:
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
    tasks_path: str = "data/tasks.jsonl",
    out_stats: str = "results/feature_stats_no_keys.csv",
    out_top: str = "results/top_features_no_keys.md",
    top_k: int = 20,
    sae_layer: int = 8,
) -> None:
    preds = pd.read_csv(preds_path)
    acts = np.load(acts_path)
    tasks = [json.loads(line) for line in open(tasks_path)]
    preds["object"] = [t["object"] for t in tasks]

    # Restrict to prompts that DO NOT contain "the keys"
    keep = preds["object"] != "the keys"
    keep_idx = np.where(keep.values)[0]
    preds_sub = preds.loc[keep].reset_index(drop=True)
    acts_sub = acts[keep_idx]

    is_fail = (preds_sub["success"].values == 0)
    n_fail = int(is_fail.sum())
    n_succ = int((~is_fail).sum())
    print(f"After dropping 'the keys': N = {len(preds_sub)} ({n_succ} success, {n_fail} fail, "
          f"acc = {n_succ/len(preds_sub):.3f})")

    fail_acts = acts_sub[is_fail]
    succ_acts = acts_sub[~is_fail]
    n_feat = acts_sub.shape[1]

    mean_fail = fail_acts.mean(axis=0)
    mean_succ = succ_acts.mean(axis=0)
    var_fail = fail_acts.var(axis=0, ddof=1) if n_fail > 1 else np.zeros(n_feat)
    var_succ = succ_acts.var(axis=0, ddof=1) if n_succ > 1 else np.zeros(n_feat)
    se = np.sqrt(var_fail / max(1, n_fail) + var_succ / max(1, n_succ))
    t_stat = np.where(se > 0, (mean_fail - mean_succ) / se, 0.0)
    df = min(n_fail, n_succ) - 1
    p_uncorr = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=max(1, df)))

    pooled_var = ((n_fail - 1) * var_fail + (n_succ - 1) * var_succ) / max(1, n_fail + n_succ - 2)
    sd = np.sqrt(np.where(pooled_var > 0, pooled_var, 1))
    d_vals = np.where(pooled_var > 0, (mean_fail - mean_succ) / sd, 0.0)
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

    df_stats.to_csv(out_stats, index=False)
    print(f"wrote {out_stats}")

    top_lines = [
        f"# Top {top_k} features (after excluding 'the keys' subset)",
        "",
        f"N = {len(preds_sub)} ({n_fail} fail, {n_succ} succ). SAE layer {sae_layer}.",
        "",
        "| Rank | Feature | Cohen's d | mean fail | mean succ | p (Holm) | Active fail/succ | Neuronpedia |",
        "|---:|---:|---:|---:|---:|---:|---:|:---|",
    ]
    print(f"top {top_k} features (excluding keys):")
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
        print(f"  feat {feat:5d}  d={d:+.3f}  mean_fail={row['mean_fail']:.4f}  "
              f"mean_succ={row['mean_succ']:.4f}  p_holm={row['p_holm']:.4f}")

    Path(out_top).write_text("\n".join(top_lines) + "\n")
    print(f"wrote {out_top}")

    n_sig = int((df_stats["p_holm"] < 0.05).sum())
    n_large = int((df_stats["abs_d"] > 0.5).sum())
    print(f"\nSignificant (Holm p<0.05): {n_sig} / {n_feat}")
    print(f"|d| > 0.5: {n_large} / {n_feat}")
    print(f"|d| > 0.8: {(df_stats['abs_d'] > 0.8).sum()} / {n_feat}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--top-k", type=int, default=20)
    args = p.parse_args()
    main(top_k=args.top_k)
