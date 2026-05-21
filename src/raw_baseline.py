"""Raw-activation baseline: fit a logistic regression on the raw
768-dim residual stream activations (not the SAE basis), 5-fold CV.

This is the "what does the SAE basis actually add?" control.

Re-runs the model on the existing 300 prompts and collects the
mean residual-stream activation over the last 3 tokens (matching the
SAE-side aggregation), without going through the SAE encoder.

Writes:
  results/raw_activations.npy   (300 x 768)
  results/raw_baseline_report.txt
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sae_lens import HookedSAETransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore", category=UserWarning)


def main(
    tasks_path: str = "data/tasks.jsonl",
    out_acts: str = "results/raw_activations.npy",
    out_report: str = "results/raw_baseline_report.txt",
    last_k_tokens: int = 3,
) -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    print("loading gpt2 ...")
    model = HookedSAETransformer.from_pretrained("gpt2", device=device)
    model.eval()
    hook_name = "blocks.8.hook_resid_pre"

    tasks = [json.loads(l) for l in open(tasks_path)]
    print(f"collecting raw residual-stream activations on {len(tasks)} prompts ...")

    preds = pd.read_csv("results/predictions.csv")
    raw = []
    for i, task in enumerate(tasks):
        tokens = model.to_tokens(task["prompt"])
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
            resid = cache[hook_name]
            tail = resid[0, -last_k_tokens:, :]
            raw.append(tail.mean(dim=0).cpu().numpy())
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(tasks)}")

    X_raw = np.stack(raw)
    Path(out_acts).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_acts, X_raw)
    print(f"wrote {out_acts}  shape={X_raw.shape}")

    # Logistic regression
    y = (preds["success"].values == 0).astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    aucs = cross_val_score(clf, X_raw, y, cv=cv, scoring="roc_auc")
    accs = cross_val_score(clf, X_raw, y, cv=cv, scoring="accuracy")

    # For comparison: same protocol on SAE features (top-100)
    sae_acts = np.load("results/activations.npy")
    stats = pd.read_csv("results/feature_stats.csv")
    top100 = stats.sort_values("abs_d", ascending=False)["feature"].astype(int).values[:100]
    X_sae100 = sae_acts[:, top100]
    aucs_sae100 = cross_val_score(clf, X_sae100, y, cv=cv, scoring="roc_auc")

    # Also top-1 SAE feature
    top1 = stats.sort_values("abs_d", ascending=False)["feature"].astype(int).values[:1]
    X_sae1 = sae_acts[:, top1]
    aucs_sae1 = cross_val_score(clf, X_sae1, y, cv=cv, scoring="roc_auc")

    # All 24576 SAE features (regularized)
    X_sae_all = sae_acts
    aucs_sae_all = cross_val_score(clf, X_sae_all, y, cv=cv, scoring="roc_auc")

    report = [
        "Raw-activation baseline vs SAE-feature predictors",
        "================================================",
        "All values: mean (std) ROC AUC, 5-fold stratified CV, class-balanced logistic.",
        "",
        f"  raw residual stream (d=768)         : {aucs.mean():.3f} ({aucs.std():.3f})",
        f"  SAE features, top-1 (d=1)           : {aucs_sae1.mean():.3f} ({aucs_sae1.std():.3f})",
        f"  SAE features, top-100 (d=100)       : {aucs_sae100.mean():.3f} ({aucs_sae100.std():.3f})",
        f"  SAE features, all 24,576            : {aucs_sae_all.mean():.3f} ({aucs_sae_all.std():.3f})",
        "",
        f"raw accuracy (class-balanced)         : {accs.mean():.3f} ({accs.std():.3f})",
    ]
    text = "\n".join(report)
    print()
    print(text)
    Path(out_report).write_text(text + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    args = p.parse_args()
    main()
