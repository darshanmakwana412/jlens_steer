"""How does steering strength depend on c (tokens per set) and k (sets averaged)?

Sweeps both knobs with several seeds per cell, because at small k the draw
dominates and a single number hides it. The coefficient is held fixed per
behaviour so the sweep measures the direction rather than a coefficient search.

At c=1 there are only as many distinct sets as there are concept tokens, so a
requested k above that is clamped; the effective k is recorded, otherwise the
large-k end of the c=1 curve would quietly average over fewer sets than the
others.
"""

import argparse
import itertools
import json
import sys

from assets import ARTIFACTS, BASE_MODEL, ROOT
from concepts import CAPS_TOKENS, REFUSAL_TOKENS, single_token_ids
from derive import Recipe, concept_direction
from eval_prompts import CAPS_PROMPTS, NEUTRAL_PROMPTS
from jlens import load_lens
from metrics import count_caps, refusal_rate
from steering import generate, load_model, scaled_delta

LAYER = 13
SCALE = 27.11
COEFFICIENT = {"caps": 3.0, "refusal": 2.0}
OUT_PATH = ARTIFACTS / "ablation_eval.json"


def log(message: str) -> None:
    print(message, flush=True)


def effective_sets(n_positives: int, set_size: int, n_sets: int) -> int:
    total = len(list(itertools.combinations(range(n_positives), set_size)))
    return min(n_sets, total)


def parse_args():
    parser = argparse.ArgumentParser(description="Ablate c and k.")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--set-sizes", type=int, nargs="+", default=[1, 2, 3, 5, 8])
    parser.add_argument("--n-sets", type=int, nargs="+", default=[1, 2, 5, 10, 25])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--n-prompts", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=28)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model, tokenizer = load_model(BASE_MODEL, args.device)
    lens = load_lens(model, args.device)

    behaviours = {
        "caps": (
            single_token_ids(tokenizer, CAPS_TOKENS),
            CAPS_PROMPTS[: args.n_prompts],
            lambda c: count_caps(c).fraction_of_letter_tokens,
        ),
        "refusal": (
            single_token_ids(tokenizer, REFUSAL_TOKENS),
            NEUTRAL_PROMPTS[: args.n_prompts],
            refusal_rate,
        ),
    }

    rows = []
    for name, (positives, prompts, measure) in behaviours.items():
        log(f"{name}: {len(positives)} concept tokens, coefficient {COEFFICIENT[name]}")
        for set_size in args.set_sizes:
            for n_sets in args.n_sets:
                measures = []
                for seed in range(args.seeds):
                    recipe = Recipe(
                        positives=positives,
                        negative="vocab",
                        set_size=set_size,
                        n_sets=n_sets,
                        seed=seed,
                    )
                    direction, combos = concept_direction(lens, LAYER, recipe, args.device)
                    delta = scaled_delta(direction * SCALE, COEFFICIENT[name], model)
                    completions = generate(
                        model, tokenizer, prompts, args.max_new_tokens, LAYER, delta
                    )
                    value = measure(completions)
                    measures.append(value)
                    rows.append(
                        {
                            "behaviour": name,
                            "set_size": set_size,
                            "n_sets": n_sets,
                            "effective_sets": len(combos),
                            "seed": seed,
                            "measure": value,
                        }
                    )
                clamped = effective_sets(len(positives), set_size, n_sets)
                note = "" if clamped == n_sets else f"  (k clamped to {clamped})"
                mean = sum(measures) / len(measures)
                log(
                    f"  c={set_size} k={n_sets:<3d} mean {mean:.3f} "
                    f"[{min(measures):.2f}, {max(measures):.2f}]{note}"
                )

    OUT_PATH.write_text(
        json.dumps(
            {
                "layer": LAYER,
                "scale": SCALE,
                "coefficient": COEFFICIENT,
                "n_prompts": args.n_prompts,
                "seeds": args.seeds,
                "rows": rows,
            },
            indent=2,
        )
    )
    log(f"saved {OUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
