"""Causal ablation: zero out a target SAE feature in the residual stream
during inference and measure the change in IOI accuracy.

For each task:
  1. forward pass with a hook at blocks.8.hook_resid_pre
  2. encode through the SAE -> feature activations
  3. zero out the target feature(s) in the encoded activation
  4. decode back -> patched residual stream
  5. write the patched residual stream into the forward pass at the hook point
  6. greedy decode the next 3 tokens and check the indirect-object name

Reports:
  - baseline accuracy on each subset
  - ablated accuracy on each subset
  - per-subset delta

Writes:
  results/ablation_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from sae_lens import SAE, HookedSAETransformer


def normalize_token(s: str) -> str:
    s = s.strip()
    if not s:
        return ""
    first = s.split()[0]
    return first.rstrip(".,!?;:").lstrip()


def main(
    tasks_path: str = "data/tasks.jsonl",
    out_path: str = "results/ablation_summary.csv",
    model_name: str = "gpt2",
    sae_release: str = "gpt2-small-res-jb",
    sae_id: str = "blocks.8.hook_resid_pre",
    target_features: tuple[int, ...] = (17491,),
    max_new_tokens: int = 3,
) -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    print(f"loading {model_name} ...")
    model = HookedSAETransformer.from_pretrained(model_name, device=device)
    model.eval()

    print(f"loading SAE {sae_release}/{sae_id} ...")
    sae = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=device)
    sae.eval()

    hook_name = sae_id
    target_idx = torch.tensor(list(target_features), device=device, dtype=torch.long)
    print(f"  ablating features {list(target_features)} at hook {hook_name}")

    tasks = [json.loads(l) for l in open(tasks_path)]
    print(f"running {len(tasks)} tasks under two conditions: baseline and ablated ...")

    rows = []
    for i, task in enumerate(tasks):
        prompt = task["prompt"]
        expected = task["expected"].strip()
        tokens = model.to_tokens(prompt)

        # --- BASELINE (no patch) ---
        with torch.no_grad():
            generated = model.generate(
                tokens, max_new_tokens=max_new_tokens, temperature=0.0, verbose=False,
            )
        base_gen = model.to_string(generated[0, tokens.shape[1]:])
        base_pred = normalize_token(base_gen)
        base_ok = int(base_pred == expected)

        # --- ABLATED ---
        # Hook that zeros target features in the SAE-decoded reconstruction
        # by subtracting feature_act * decoder_row from the residual stream.
        # Equivalent to: resid_patched = resid - sum_i act_i * W_dec[i].
        def ablate_hook(activation, hook, sae=sae, idx=target_idx):
            # activation: (batch, seq, d_model)
            feats = sae.encode(activation)  # (batch, seq, d_sae)
            # zero out target features in the encoded space, then subtract
            # their decoded contribution from the original activation.
            target_acts = feats[..., idx]              # (batch, seq, k)
            W_dec_rows = sae.W_dec[idx, :]              # (k, d_model)
            contribution = target_acts @ W_dec_rows     # (batch, seq, d_model)
            return activation - contribution

        with torch.no_grad():
            generated_abl = model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_name, ablate_hook)],
            )
            # run_with_hooks gives logits, but for greedy we need to autoregress.
            # Re-implement greedy under hooks:
            current = tokens.clone()
            for _ in range(max_new_tokens):
                with torch.no_grad():
                    logits = model.run_with_hooks(
                        current, fwd_hooks=[(hook_name, ablate_hook)]
                    )
                next_tok = torch.argmax(logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
                current = torch.cat([current, next_tok], dim=1)
            abl_gen = model.to_string(current[0, tokens.shape[1]:])
            abl_pred = normalize_token(abl_gen)
            abl_ok = int(abl_pred == expected)

        rows.append({
            "id": task["id"],
            "object": task["object"],
            "expected": expected,
            "base_pred": base_pred,
            "abl_pred": abl_pred,
            "base_ok": base_ok,
            "abl_ok": abl_ok,
            "is_keys": int(task["object"] == "the keys"),
        })

        if (i + 1) % 25 == 0:
            n_base = sum(r["base_ok"] for r in rows)
            n_abl = sum(r["abl_ok"] for r in rows)
            print(f"  {i+1}/{len(tasks)}   base={n_base}/{i+1}  abl={n_abl}/{i+1}")

    # Write per-task rows
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_path}")

    # ----- Summary -----
    def acc(subset):
        if not subset:
            return float("nan")
        return sum(r["base_ok"] for r in subset) / len(subset), sum(r["abl_ok"] for r in subset) / len(subset)

    all_rows = rows
    keys_rows = [r for r in rows if r["is_keys"]]
    other_rows = [r for r in rows if not r["is_keys"]]

    print()
    print("===== Ablation summary =====")
    base_all, abl_all = acc(all_rows)
    base_keys, abl_keys = acc(keys_rows)
    base_other, abl_other = acc(other_rows)

    print(f"all N={len(all_rows)}:  base acc = {base_all:.3f}   abl acc = {abl_all:.3f}   delta = {abl_all - base_all:+.3f}")
    print(f"keys N={len(keys_rows)}: base acc = {base_keys:.3f}   abl acc = {abl_keys:.3f}   delta = {abl_keys - base_keys:+.3f}")
    print(f"others N={len(other_rows)}: base acc = {base_other:.3f}   abl acc = {abl_other:.3f}   delta = {abl_other - base_other:+.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=int, nargs="+", default=[17491])
    p.add_argument("--tasks", default="data/tasks.jsonl")
    p.add_argument("--out", default="results/ablation_summary.csv")
    args = p.parse_args()
    main(target_features=tuple(args.features), tasks_path=args.tasks, out_path=args.out)
