"""Does the negative set have to mean anything, or is it only centering?

Holds the positive tokens, c, k and the coefficient sweep fixed and swaps only
what gets subtracted: nothing, the vocabulary-mean unembedding row, a random
set of common function words, or the semantic opposite (lowercase twins for
caps, compliance words for refusal).
"""

import argparse
import json
import sys

import torch

from assets import ARTIFACTS, BASE_MODEL, ROOT
from concepts import CAPS_TOKENS, COMPLIANCE_TOKENS, REFUSAL_TOKENS, single_token_ids
from derive import NEGATIVES, Recipe, concept_direction
from eval_prompts import CAPS_PROMPTS, NEUTRAL_PROMPTS
from jlens import load_lens, unit
from metrics import count_caps, refusal_rate
from steering import generate, load_model, scaled_delta

LAYER = 13
SCALE = 27.11
COEFFICIENTS = [1.0, 2.0, 4.0, 6.0]
OUT_PATH = ARTIFACTS / "negatives_eval.json"

COMMON_WORDS = [
    " the", " of", " and", " to", " in", " is", " that", " it", " for",
    " was", " on", " as", " with", " at", " by", " an", " be", " this",
    " from", " or", " one", " had", " but", " what", " all", " were",
]  # fmt: skip

# lowercase twins of CAPS_TOKENS, index-matched, for the caps "opposite" negative
CAPS_OPPOSITES = [token.lower() for token in CAPS_TOKENS]


def log(message: str) -> None:
    print(message, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare negative sets for the contrast.")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--set-size", type=int, default=3)
    parser.add_argument("--n-sets", type=int, default=300)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--coefficients", type=float, nargs="+", default=COEFFICIENTS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model, tokenizer = load_model(BASE_MODEL, args.device)
    lens = load_lens(model, args.device)
    commons = single_token_ids(tokenizer, COMMON_WORDS)

    generator = torch.Generator().manual_seed(7)
    control = unit(torch.randn(model.config.hidden_size, generator=generator)).to(args.device)

    behaviours = {
        "caps": (
            single_token_ids(tokenizer, CAPS_TOKENS),
            single_token_ids(tokenizer, CAPS_OPPOSITES),
            CAPS_PROMPTS,
            lambda c: count_caps(c).fraction_of_letter_tokens,
        ),
        "refusal": (
            single_token_ids(tokenizer, REFUSAL_TOKENS),
            single_token_ids(tokenizer, COMPLIANCE_TOKENS),
            NEUTRAL_PROMPTS,
            refusal_rate,
        ),
    }

    result = {"layer": LAYER, "scale": SCALE, "set_size": args.set_size, "n_sets": args.n_sets}
    for name, (positives, opposites, prompts, measure) in behaviours.items():
        log(f"{name}")
        runs = {}
        for negative in (*NEGATIVES, "control"):
            if negative == "control":
                direction = control
            else:
                recipe = Recipe(
                    positives=positives,
                    negative=negative,
                    opposites=opposites,
                    commons=commons,
                    set_size=args.set_size,
                    n_sets=args.n_sets,
                )
                direction, _ = concept_direction(lens, LAYER, recipe, args.device)
            points = []
            for coefficient in args.coefficients:
                delta = scaled_delta(direction * SCALE, coefficient, model)
                completions = generate(model, tokenizer, prompts, args.max_new_tokens, LAYER, delta)
                points.append({"coefficient": coefficient, "measure": measure(completions)})
            runs[negative] = points
            log(f"  {negative:10s} " + "  ".join(f"{p['measure']:.3f}" for p in points))
        result[name] = {"n_prompts": len(prompts), "runs": runs}

    OUT_PATH.write_text(json.dumps(result, indent=2))
    log(f"saved {OUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
