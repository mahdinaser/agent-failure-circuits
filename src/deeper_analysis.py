"""Additional analyses on the saved 300x24576 activation matrix.

Runs:
  1. Logistic regression: predict failure from top-K SAE features.
     5-fold CV, report ROC AUC, accuracy, and the regression coefficients.
  2. ROC curve from the top single feature (17491) alone.
  3. Volcano plot data: Cohen's d vs -log10(p_holm).
  4. Per-object failure-rate bar-chart data.
  5. Boxplot data for the top feature, split by object.

Reads:
  results/predictions.csv
  results/activations.npy
  data/tasks.jsonl
  results/feature_stats.csv

Writes:
  figures/volcano.png
  figures/per_object_failure.png
  figures/roc_curve.png
  figures/top_feature_by_object.png
  results/logreg_report.txt
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

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


def load_all() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, list[dict]]:
    preds = pd.read_csv("results/predictions.csv")
    acts = np.load("results/activations.npy")
    stats = pd.read_csv("results/feature_stats.csv")
    tasks = [json.loads(l) for l in open("data/tasks.jsonl")]
    preds["object"] = [t["object"] for t in tasks]
    preds["template_id"] = [t["template_id"] for t in tasks]
    return preds, acts, stats, tasks


# ============================================================
# 1. Logistic regression on top-K features
# ============================================================
def logreg_top_k(preds: pd.DataFrame, acts: np.ndarray, stats: pd.DataFrame,
                 K_list: tuple[int, ...] = (1, 5, 10, 20, 50, 100)) -> str:
    y = preds["success"].values == 0  # 1 = failure
    # Top-K features by |Cohen's d|
    top_features = stats.sort_values("abs_d", ascending=False)["feature"].astype(int).values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    lines = ["Logistic regression: predict failure from top-K SAE features",
             "============================================================",
             "Cross-validation: 5-fold stratified.",
             ""]
    lines.append(f"{'K':>5} | {'mean ROC AUC':>13} | {'std AUC':>10} | {'mean acc':>10}")
    lines.append("-" * 50)
    best_K, best_auc = None, -1
    for K in K_list:
        idx = top_features[:K]
        X = acts[:, idx]
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
        aucs = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
        accs = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        line = f"{K:>5} | {aucs.mean():>13.3f} | {aucs.std():>10.3f} | {accs.mean():>10.3f}"
        lines.append(line)
        if aucs.mean() > best_auc:
            best_auc = aucs.mean()
            best_K = K
    lines.append("")
    lines.append(f"Best: K={best_K}, mean ROC AUC = {best_auc:.3f}")
    return "\n".join(lines)


# ============================================================
# 2. ROC curve using only feature 17491
# ============================================================
def roc_for_top_feature(preds: pd.DataFrame, acts: np.ndarray, feature_id: int = 17491) -> dict:
    y = (preds["success"].values == 0).astype(int)  # 1 = failure
    score = acts[:, feature_id]  # use raw activation as risk score
    auc = roc_auc_score(y, score)
    fpr, tpr, _ = roc_curve(y, score)
    return {"auc": auc, "fpr": fpr, "tpr": tpr, "feature_id": feature_id}


# ============================================================
# 3. Volcano plot
# ============================================================
def volcano(stats: pd.DataFrame, out_path: str) -> None:
    d = stats["cohens_d"].values
    p = stats["p_holm"].values
    p = np.clip(p, 1e-30, 1.0)
    neg_log_p = -np.log10(p)
    is_sig = (p < 0.05)
    is_large = (np.abs(d) > 0.8) & is_sig

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.scatter(d[~is_sig], neg_log_p[~is_sig], s=4, alpha=0.4, color="lightgray",
               label="not significant")
    ax.scatter(d[is_sig & ~is_large], neg_log_p[is_sig & ~is_large], s=8,
               alpha=0.7, color="steelblue", label="Holm-significant")
    ax.scatter(d[is_large], neg_log_p[is_large], s=14, alpha=0.85,
               color="firebrick", label="|d|>0.8 and Holm-sig.")
    ax.axhline(-np.log10(0.05), color="black", linewidth=0.7, linestyle="--", alpha=0.6)
    ax.axvline(0.8, color="firebrick", linewidth=0.5, linestyle=":", alpha=0.6)
    ax.axvline(-0.8, color="firebrick", linewidth=0.5, linestyle=":", alpha=0.6)
    # Annotate the top feature
    top_idx = stats.iloc[0]
    ax.annotate(f"feat {int(top_idx['feature'])}",
                xy=(top_idx["cohens_d"], min(30, -np.log10(max(top_idx["p_holm"], 1e-30)))),
                xytext=(top_idx["cohens_d"] - 1.5, 28),
                fontsize=8,
                arrowprops=dict(arrowstyle="->", color="firebrick", lw=0.6))
    ax.set_xlabel("Cohen's d  (failure − success)")
    ax.set_ylabel(r"$-\log_{10}(p_\mathrm{Holm})$")
    ax.set_title("Volcano plot: 24,576 SAE features at layer 8")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"  wrote {out_path}")


# ============================================================
# 4. Per-object failure rate bar chart
# ============================================================
def per_object_chart(preds: pd.DataFrame, out_path: str) -> None:
    grp = preds.groupby("object")["success"].agg(["count", "mean"])
    grp["fail_rate"] = 1.0 - grp["mean"]
    grp = grp.sort_values("fail_rate")
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    colors = ["firebrick" if x > 0.5 else "steelblue" for x in grp["fail_rate"]]
    bars = ax.barh(range(len(grp)), grp["fail_rate"].values, color=colors, alpha=0.85,
                   edgecolor="white")
    ax.set_yticks(range(len(grp)))
    ax.set_yticklabels(grp.index)
    for i, (obj, row) in enumerate(grp.iterrows()):
        ax.text(row["fail_rate"] + 0.01, i,
                f" {int((1-row['mean'])*row['count'])}/{int(row['count'])}",
                va="center", fontsize=7.5, color="gray")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Failure rate")
    ax.set_title("IOI failure rate by transferred-object choice  (N=300)")
    ax.axvline(0.5, color="black", linewidth=0.4, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"  wrote {out_path}")


# ============================================================
# 5. ROC curve figure
# ============================================================
def roc_figure(roc_data: dict, logreg_full_auc: float, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.plot(roc_data["fpr"], roc_data["tpr"], color="firebrick", linewidth=1.6,
            label=f"feature {roc_data['feature_id']} alone (AUC = {roc_data['auc']:.3f})")
    ax.plot([0, 1], [0, 1], color="lightgray", linewidth=0.8, linestyle="--",
            label="chance (AUC = 0.500)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Failure prediction by a single SAE feature")
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"  wrote {out_path}")


# ============================================================
# 6. Top-feature activation distribution by object
# ============================================================
def top_feature_by_object(preds: pd.DataFrame, acts: np.ndarray,
                           feature_id: int, out_path: str) -> None:
    a = acts[:, feature_id]
    objs = sorted(preds["object"].unique(),
                  key=lambda o: -float(preds[preds.object == o]["success"].mean()))
    data = [a[preds["object"].values == o] for o in objs]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    bp = ax.boxplot(data, labels=objs, patch_artist=True, widths=0.55)
    # Color "the keys" differently
    for patch, obj in zip(bp["boxes"], objs):
        if obj == "the keys":
            patch.set_facecolor("firebrick")
            patch.set_alpha(0.6)
        else:
            patch.set_facecolor("steelblue")
            patch.set_alpha(0.55)
        patch.set_edgecolor("white")
    for whisker in bp["whiskers"]:
        whisker.set_color("gray")
    for median in bp["medians"]:
        median.set_color("black")
    ax.set_ylabel(f"feature {feature_id} activation")
    ax.set_title(f"Per-object distribution of feature {feature_id} activations")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"  wrote {out_path}")


# ============================================================
# Main
# ============================================================
def main() -> None:
    preds, acts, stats, tasks = load_all()
    Path("figures").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    print("== Logistic regression on top-K features ==")
    report = logreg_top_k(preds, acts, stats)
    print(report)
    Path("results/logreg_report.txt").write_text(report + "\n")
    print()

    # Also single-feature logreg for the headline ROC
    print("== ROC for feature 17,491 alone ==")
    roc_data = roc_for_top_feature(preds, acts, feature_id=17491)
    print(f"  AUC (feature 17,491 alone) = {roc_data['auc']:.3f}")
    print()

    # Full-feature logistic AUC
    y = (preds["success"].values == 0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    # Use top-50 to avoid singular matrices
    top50 = stats.sort_values("abs_d", ascending=False)["feature"].astype(int).values[:50]
    auc50 = cross_val_score(clf, acts[:, top50], y, cv=cv, scoring="roc_auc").mean()
    print(f"  AUC (top-50 features, CV) = {auc50:.3f}")
    print()

    print("== Figures ==")
    volcano(stats, "figures/volcano.png")
    per_object_chart(preds, "figures/per_object_failure.png")
    roc_figure(roc_data, auc50, "figures/roc_curve.png")
    top_feature_by_object(preds, acts, feature_id=17491,
                          out_path="figures/top_feature_by_object.png")
    print()
    print("done.")


if __name__ == "__main__":
    main()
