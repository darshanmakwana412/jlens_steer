"""The casing-pair toy model: every (UPPER, lower) pair in the vocabulary.

This came before the concept-token method and is the cleaner existence proof,
because the pairs are enumerated rather than hand-picked. Each pair contributes
``unit(a_UPPER - a_lower)``; the mean of those is the casing direction.

It also carries the ``J = I`` control. Both versions separate the two clouds
about equally well, so the Jacobian is not what makes casing linearly visible.
What it changes is whether the per-pair differences agree on a direction, and
since the construction averages them, disagreement is what makes the control
useless for steering.
"""

import argparse
import json
import sys

import torch

from assets import ARTIFACTS, BASE_MODEL, ROOT, STEERING_VECS
from concepts import casing_pairs
from derive import pair_direction, pair_rows
from eval_prompts import CAPS_PROMPTS
from jlens import load_lens, unit
from metrics import count_caps
from steering import generate, load_model, scaled_delta

LAYER = 13
SCALE = 27.11
COEFFICIENTS = [1.0, 2.0, 4.0]
OUT_PATH = ARTIFACTS / "casing_pairs_eval.json"


def log(message: str) -> None:
    print(message, flush=True)


def alignment(upper_rows, lower_rows, direction) -> dict:
    """How consistently the pair differences point along ``direction``."""
    diffs = upper_rows - lower_rows
    cos = torch.nn.functional.cosine_similarity(diffs, direction.unsqueeze(0), dim=-1)
    projected_upper, projected_lower = upper_rows @ direction, lower_rows @ direction
    pooled = torch.sqrt(0.5 * (projected_upper.var() + projected_lower.var()))
    return {
        "mean_cos": float(cos.mean()),
        "median_cos": float(cos.median()),
        "fraction_positive": float((cos > 0).float().mean()),
        "cohens_d": float((projected_upper.mean() - projected_lower.mean()) / pooled),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Casing-pair direction and its J = I control.")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--min-letters", type=int, default=3)
    parser.add_argument("--all-tokens", action="store_true", help="drop the word-initial filter")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--coefficients", type=float, nargs="+", default=COEFFICIENTS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model, tokenizer = load_model(BASE_MODEL, args.device)
    lens = load_lens(model, args.device)
    pairs = casing_pairs(
        tokenizer,
        model.config.vocab_size,
        min_letters=args.min_letters,
        word_initial=not args.all_tokens,
    )
    log(f"{len(pairs)} casing pairs")

    baseline = unit(torch.load(STEERING_VECS / "caps_L13.pt", map_location="cpu").float()).to(
        args.device
    )

    result = {"layer": LAYER, "scale": SCALE, "n_pairs": len(pairs), "variants": {}}
    for label, transport in (("jlens", True), ("logit_lens", False)):
        upper_rows, lower_rows = pair_rows(lens, pairs, LAYER, args.device, transport)
        direction = pair_direction(lens, pairs, LAYER, args.device, transport)
        stats = alignment(upper_rows, lower_rows, direction)
        stats["cos_to_baseline"] = float(torch.dot(direction, baseline))

        points = []
        for coefficient in args.coefficients:
            delta = scaled_delta(direction * SCALE, coefficient, model)
            completions = generate(
                model, tokenizer, CAPS_PROMPTS, args.max_new_tokens, LAYER, delta
            )
            points.append(
                {
                    "coefficient": coefficient,
                    "measure": count_caps(completions).fraction_of_letter_tokens,
                }
            )
        stats["sweep"] = points
        result["variants"][label] = stats
        log(
            f"  {label:10s} mean cos {stats['mean_cos']:+.3f}  "
            f"Cohen's d {stats['cohens_d']:.2f}  "
            f"best caps {max(p['measure'] for p in points):.3f}"
        )

    # what the construction looks like at its best and worst, by pair
    upper_rows, lower_rows = pair_rows(lens, pairs, LAYER, args.device, True)
    cos = torch.nn.functional.cosine_similarity(
        unit(upper_rows - lower_rows), baseline.unsqueeze(0), dim=-1
    )
    order = cos.argsort(descending=True)
    result["pairs_by_alignment"] = {
        "best": [
            {"upper": tokenizer.decode([pairs[i][0]]), "cos": float(cos[i])}
            for i in order[:15].tolist()
        ],
        "worst": [
            {"upper": tokenizer.decode([pairs[i][0]]), "cos": float(cos[i])}
            for i in order[-10:].tolist()
        ],
    }

    OUT_PATH.write_text(json.dumps(result, indent=2))
    log(f"saved {OUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
