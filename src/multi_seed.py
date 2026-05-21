"""Multi-seed robustness: re-run the full pipeline with five different
seeds and report the range of headline numbers.

For each seed s in (0, 42, 100, 200, 300):
  - regenerate a 300-prompt IOI corpus
  - run GPT-2 small + SAE, log per-task activations
  - score success/failure
  - record: overall accuracy, "the keys" failure rate,
            top-feature index, top-feature Cohen's d

Writes:
  results/multi_seed_report.txt
  results/multi_seed_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sae_lens import SAE, HookedSAETransformer


# ---- inlined from build_task_set.py so this script is standalone ----
NAMES = [
    "John","Mary","Tom","James","Dan","Mike","Chris","Susan","Anna","Paul",
    "Brian","Lisa","Alex","Emily","Robert","Sarah","Jessica","David","Laura",
    "Kevin","Karen","Steve","Lucy","Henry","Lily","Jack","Rachel","Ben","Helen",
]
PLACES = ["the store","the park","the restaurant","the school",
          "the office","the gym","the library","the cafe"]
OBJECTS = ["a drink","the keys","a book","the ball","the gift",
           "a card","the bag","the flowers"]
TEMPLATES = [
    lambda S,IO,p,o: f"When {IO} and {S} went to {p}, {S} gave {o} to",
    lambda S,IO,p,o: f"After {S} and {IO} went to {p}, {S} gave {o} to",
    lambda S,IO,p,o: f"While {IO} and {S} were at {p}, {S} handed {o} to",
    lambda S,IO,p,o: f"Once {S} and {IO} arrived at {p}, {S} passed {o} to",
]


def build_corpus(seed: int, n: int = 300) -> list[dict]:
    rng = random.Random(seed)
    items = []
    for i in range(n):
        s, io = rng.sample(NAMES, 2)
        place = rng.choice(PLACES)
        obj   = rng.choice(OBJECTS)
        tid   = rng.randrange(len(TEMPLATES))
        items.append({
            "id": i, "subject": s, "indirect_object": io, "place": place,
            "object": obj, "template_id": tid,
            "prompt": TEMPLATES[tid](s, io, place, obj),
            "expected": " " + io,
        })
    return items


def normalize_token(s: str) -> str:
    s = s.strip()
    if not s:
        return ""
    return s.split()[0].rstrip(".,!?;:").lstrip()


def cohens_d_vec(fail: np.ndarray, succ: np.ndarray) -> np.ndarray:
    n_f, n_s = len(fail), len(succ)
    if n_f < 2 or n_s < 2:
        return np.zeros(fail.shape[1])
    var_f = fail.var(axis=0, ddof=1)
    var_s = succ.var(axis=0, ddof=1)
    pooled = ((n_f - 1) * var_f + (n_s - 1) * var_s) / (n_f + n_s - 2)
    sd = np.sqrt(np.where(pooled > 0, pooled, 1))
    return np.where(pooled > 0, (fail.mean(0) - succ.mean(0)) / sd, 0.0)


def run_one_seed(seed: int, model, sae, hook_name: str, last_k: int = 3) -> dict:
    tasks = build_corpus(seed, n=300)
    acts = np.zeros((len(tasks), sae.cfg.d_sae), dtype=np.float32)
    succ = np.zeros(len(tasks), dtype=np.int8)
    objects = []
    for i, t in enumerate(tasks):
        tokens = model.to_tokens(t["prompt"])
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
            resid = cache[hook_name]
            feats = sae.encode(resid[0, -last_k:, :]).mean(dim=0)
            acts[i] = feats.cpu().numpy()
            gen = model.generate(tokens, max_new_tokens=3, temperature=0.0, verbose=False)
            pred = normalize_token(model.to_string(gen[0, tokens.shape[1]:]))
            succ[i] = int(pred == t["expected"].strip())
        objects.append(t["object"])
    fail_mask = succ == 0

    # Per-object failure rate
    df = pd.DataFrame({"object": objects, "success": succ})
    obj_rates = df.groupby("object")["success"].agg(lambda x: 1.0 - x.mean())

    # Cohen's d across all features
    if fail_mask.any() and (~fail_mask).any():
        d_vals = cohens_d_vec(acts[fail_mask], acts[~fail_mask])
        top_feat = int(np.argmax(np.abs(d_vals)))
        top_d = float(d_vals[top_feat])
    else:
        top_feat, top_d = -1, 0.0

    return {
        "seed": seed,
        "n": len(tasks),
        "accuracy": float(succ.mean()),
        "n_keys": int((df["object"] == "the keys").sum()),
        "keys_fail_rate": float(obj_rates.get("the keys", float("nan"))),
        "top_feature": top_feat,
        "top_d": top_d,
        "n_fail": int(fail_mask.sum()),
    }


def main(seeds=(0, 42, 100, 200, 300)) -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")
    print("loading gpt2 ...")
    model = HookedSAETransformer.from_pretrained("gpt2", device=device)
    model.eval()
    print("loading SAE ...")
    sae = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id="blocks.8.hook_resid_pre",
        device=device,
    )
    sae.eval()
    hook_name = "blocks.8.hook_resid_pre"

    rows = []
    for s in seeds:
        print(f"\n=== seed {s} ===")
        row = run_one_seed(s, model, sae, hook_name)
        rows.append(row)
        print(f"  accuracy        : {row['accuracy']:.3f}")
        print(f"  N keys / N      : {row['n_keys']}/{row['n']}")
        print(f"  keys fail rate  : {row['keys_fail_rate']:.3f}")
        print(f"  top feature     : {row['top_feature']}")
        print(f"  top |d|         : {row['top_d']:+.3f}")

    df = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/multi_seed_summary.csv", index=False)

    accs = df["accuracy"].values
    keys = df["keys_fail_rate"].values
    n_top17491 = int((df["top_feature"] == 17491).sum())
    report = [
        f"Multi-seed robustness check (5 seeds: {list(seeds)})",
        "=" * 60,
        f"Overall accuracy  : range {accs.min():.3f}-{accs.max():.3f}, mean {accs.mean():.3f} (sd {accs.std(ddof=1):.3f})",
        f"Keys fail rate    : range {keys.min():.3f}-{keys.max():.3f}, mean {keys.mean():.3f} (sd {keys.std(ddof=1):.3f})",
        f"Feature 17,491 is the top-|d| feature in {n_top17491}/{len(seeds)} seeds.",
        "",
        "Per-seed details:",
    ]
    for row in rows:
        report.append(
            f"  seed {row['seed']:>3}: acc={row['accuracy']:.3f}  "
            f"N_keys={row['n_keys']:>2}  keys_fail={row['keys_fail_rate']:.3f}  "
            f"top_feat={row['top_feature']}  top_d={row['top_d']:+.3f}"
        )
    text = "\n".join(report)
    Path("results/multi_seed_report.txt").write_text(text + "\n")
    print()
    print(text)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 100, 200, 300])
    args = p.parse_args()
    main(seeds=tuple(args.seeds))
