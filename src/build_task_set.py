"""Build the Indirect Object Identification (IOI) task set.

IOI is the canonical interpretability benchmark for GPT-2 small,
established by Wang et al. (2022, "Interpretability in the Wild").
Each prompt names two entities (subject S and indirect object IO)
and asks the model to complete a sentence in which the indirect
object should be repeated:

    "When John and Mary went to the store, John gave a drink to ___"
    expected next token: "Mary"

GPT-2 small reaches ~80% accuracy on IOI. The ~20% failures are
the data we analyze: which residual-stream SAE features fire
differently on failure vs. success?

Writes data/tasks.jsonl with one JSON object per task:
    {"id": int, "prompt": str, "expected": str,
     "names": [S, IO], "template_id": int, "n_steps": 1}
"""
from __future__ import annotations

import json
import random
from pathlib import Path


# Names from the original IOI paper (Wang et al. 2022)
NAMES = [
    "John", "Mary", "Tom", "James", "Dan", "Mike", "Chris", "Susan",
    "Anna", "Paul", "Brian", "Lisa", "Alex", "Emily", "Robert", "Sarah",
    "Jessica", "David", "Laura", "Kevin", "Karen", "Steve", "Lucy",
    "Henry", "Lily", "Jack", "Rachel", "Ben", "Helen",
]

PLACES = [
    "the store", "the park", "the restaurant", "the school",
    "the office", "the gym", "the library", "the cafe",
]

OBJECTS = [
    "a drink", "the keys", "a book", "the ball", "the gift",
    "a card", "the bag", "the flowers",
]

# IOI prompt templates. The blank is filled with the IO name.
# Template format: a function that takes (S, IO, place, obj) and returns the prompt prefix.
TEMPLATES = [
    lambda S, IO, place, obj: f"When {IO} and {S} went to {place}, {S} gave {obj} to",
    lambda S, IO, place, obj: f"After {S} and {IO} went to {place}, {S} gave {obj} to",
    lambda S, IO, place, obj: f"While {IO} and {S} were at {place}, {S} handed {obj} to",
    lambda S, IO, place, obj: f"Once {S} and {IO} arrived at {place}, {S} passed {obj} to",
]


def build_ioi(rng: random.Random) -> dict:
    """Build one IOI item. S != IO, sampled without replacement."""
    s, io = rng.sample(NAMES, 2)
    place = rng.choice(PLACES)
    obj = rng.choice(OBJECTS)
    template_id = rng.randrange(len(TEMPLATES))
    prompt = TEMPLATES[template_id](s, io, place, obj)
    return {
        "prompt": prompt,
        "expected": " " + io,  # GPT-2 tokenizer puts a leading space
        "subject": s,
        "indirect_object": io,
        "place": place,
        "object": obj,
        "template_id": template_id,
        "n_steps": 1,
    }


def main(n: int = 300, seed: int = 42, out_path: str = "data/tasks.jsonl") -> None:
    rng = random.Random(seed)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for i in range(n):
            task = build_ioi(rng)
            task["id"] = i
            f.write(json.dumps(task) + "\n")
    print(f"Wrote {n} IOI tasks to {out_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/tasks.jsonl")
    args = p.parse_args()
    main(n=args.n, seed=args.seed, out_path=args.out)
