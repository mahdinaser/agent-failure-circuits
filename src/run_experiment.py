"""Run the experiment: for each task in data/tasks.jsonl,
generate GPT-2 small's prediction, score success/failure, and
log the SAE feature activations at the residual stream of the
target layer at the final-token position.

Outputs:
    results/predictions.csv      one row per task: id, prompt, expected,
                                 generated, success, n_steps
    results/activations.npy      np.ndarray of shape (N, d_sae)
                                 — average SAE activation across the last
                                 K tokens of the prompt for each task.

This is the data file the downstream analysis script consumes.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sae_lens import SAE, HookedSAETransformer


def normalize_token(s: str) -> str:
    """Strip punctuation and whitespace, return first alphanumeric word."""
    s = s.strip()
    if not s:
        return ""
    # Take the first whitespace-separated chunk, strip trailing punctuation.
    first = s.split()[0]
    return first.rstrip(".,!?;:").lstrip()


def main(
    tasks_path: str = "data/tasks.jsonl",
    out_dir: str = "results",
    model_name: str = "gpt2",
    sae_release: str = "gpt2-small-res-jb",
    sae_id: str = "blocks.8.hook_resid_pre",
    max_new_tokens: int = 6,
    last_k_tokens: int = 3,
) -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    print(f"loading {model_name} ...")
    model = HookedSAETransformer.from_pretrained(model_name, device=device)
    model.eval()
    print(f"  n_layers={model.cfg.n_layers}, d_model={model.cfg.d_model}")

    print(f"loading SAE {sae_release}/{sae_id} ...")
    sae = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=device)
    sae.eval()
    print(f"  d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")

    # The hook name lives on the SAE config metadata or can be reconstructed.
    # For gpt2-small-res-jb the sae_id IS the hook name.
    hook_name = sae_id
    print(f"  hooking activations at: {hook_name}")

    tasks = [json.loads(line) for line in open(tasks_path)]
    print(f"running {len(tasks)} tasks ...")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    predictions: list[dict] = []
    activations: list[np.ndarray] = []

    for i, task in enumerate(tasks):
        prompt = task["prompt"]
        tokens = model.to_tokens(prompt)

        # Run with cache to capture the residual stream at the SAE hook point.
        with torch.no_grad():
            logits, cache = model.run_with_cache(
                tokens,
                names_filter=[hook_name],
            )
            resid = cache[hook_name]  # shape: (1, seq, d_model)

            # Encode resid through the SAE to get feature activations.
            # Average over the last K tokens — these are the tokens immediately
            # before the predicted next token, so they encode the model's
            # "state" right before it commits to the answer.
            tail = resid[0, -last_k_tokens:, :]  # (K, d_model)
            feats = sae.encode(tail)  # (K, d_sae)
            feats_mean = feats.mean(dim=0).cpu().numpy()  # (d_sae,)

        # Greedy generate a few tokens after the prompt.
        with torch.no_grad():
            generated = model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                verbose=False,
            )
        gen_str = model.to_string(generated[0, tokens.shape[1]:])
        pred_token = normalize_token(gen_str)
        # IOI: expected starts with a space; compare on the stripped form.
        expected_clean = task["expected"].strip()
        success = int(pred_token == expected_clean)

        predictions.append({
            "id": task["id"],
            "prompt": prompt,
            "expected": task["expected"],
            "generated": gen_str.strip()[:30],
            "predicted": pred_token,
            "success": success,
            "n_steps": task["n_steps"],
        })
        activations.append(feats_mean)

        if (i + 1) % 25 == 0:
            n_ok = sum(p["success"] for p in predictions)
            print(f"  {i+1}/{len(tasks)}   ok={n_ok}  acc={n_ok/(i+1):.3f}")

    # Write predictions.csv
    csv_path = out / "predictions.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["id", "prompt", "expected", "generated", "predicted", "success", "n_steps"],
        )
        w.writeheader()
        for row in predictions:
            w.writerow(row)
    print(f"wrote {csv_path}")

    # Write activations.npy
    arr = np.stack(activations)  # (N, d_sae)
    npy_path = out / "activations.npy"
    np.save(npy_path, arr)
    print(f"wrote {npy_path}  shape={arr.shape}")

    # Summary
    n_ok = sum(p["success"] for p in predictions)
    print(f"\nfinal acc: {n_ok}/{len(predictions)} = {n_ok/len(predictions):.3f}")
    by_steps = {}
    for p in predictions:
        by_steps.setdefault(p["n_steps"], []).append(p["success"])
    print("acc by n_steps:")
    for k in sorted(by_steps):
        v = by_steps[k]
        print(f"  {k}-step: {sum(v)}/{len(v)} = {sum(v)/len(v):.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="data/tasks.jsonl")
    p.add_argument("--out", default="results")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--sae-release", default="gpt2-small-res-jb")
    p.add_argument("--sae-id", default="blocks.8.hook_resid_pre")
    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument("--last-k", type=int, default=3)
    args = p.parse_args()
    main(
        tasks_path=args.tasks,
        out_dir=args.out,
        model_name=args.model,
        sae_release=args.sae_release,
        sae_id=args.sae_id,
        max_new_tokens=args.max_new_tokens,
        last_k_tokens=args.last_k,
    )
