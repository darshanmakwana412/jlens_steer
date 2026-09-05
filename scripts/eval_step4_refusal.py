import argparse
import json
import sys

import torch
from transformers import AutoTokenizer

from assets import ARTIFACTS, BASE_MODEL, JLENS_TENSOR, REFUSAL_VECTORS, ROOT
from concept_tokens import (
    REFUSAL_OPENERS,
    REFUSAL_POOL_WIDE,
    first_token_ids,
    pool_ids,
    sample_with_replacement,
)
from eval_prompts import ARBITRARY_PROMPTS
from jlens import concept_direction, load_jacobian, top_lens_tokens, unit
from metrics import refusal_rate, refuses
from steering import first_token_mass, generate, load_model, scaled_delta
from weights import WeightReader

REFUSAL_LAYER = 15
SCALE = 27.11
SET_SIZE = 5
N_SETS = 20
SEED = 0
COEFFICIENTS = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]

VARIANTS = {
    "step4": "step 4: averaged concept activation, uncentered",
    "step5": "step 5: same, minus the vocabulary mean",
}

OUT_PATH = ARTIFACTS / "step4_refusal.json"


def log(message: str) -> None:
    print(message, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Steer with the step-4 concept activation, before centering."
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--coefficients", type=float, nargs="+", default=COEFFICIENTS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    reader = WeightReader(BASE_MODEL)
    unembed = reader.get("lm_head.weight").float()
    gain = reader.get("model.norm.weight").float()
    jacobian = load_jacobian(JLENS_TENSOR, REFUSAL_LAYER)
    reference = torch.load(REFUSAL_VECTORS, map_location="cpu", weights_only=False)["directions"][
        REFUSAL_LAYER
    ].float()

    index_sets = sample_with_replacement(len(REFUSAL_POOL_WIDE), SET_SIZE, N_SETS, SEED)
    positive = pool_ids(tokenizer, REFUSAL_POOL_WIDE, index_sets)
    vectors = {
        "step4": concept_direction(jacobian, unembed, gain, positive),
        "step5": concept_direction(jacobian, unembed, gain, positive, center=unembed.mean(0)),
    }

    log(f"pool {len(REFUSAL_POOL_WIDE)} tokens, C={SET_SIZE} with replacement, K={N_SETS}")
    log(f"distinct tokens actually drawn: {len({i for s in index_sets for i in s})}")
    readouts = {}
    for name, vector in vectors.items():
        reads = top_lens_tokens(jacobian, unembed, gain, vector, tokenizer, 10)
        cos = torch.dot(unit(vector), unit(reference)).item()
        readouts[name] = {"lens_reads": reads, "cosine_to_abliteration": cos}
        log(f"\n{VARIANTS[name]}")
        log(f"  cos to abliteration direction {cos:+.4f}")
        log(f"  lens reads: {reads}")
    log(f"\ncos(step4, step5) = {torch.dot(vectors['step4'], vectors['step5']).item():+.4f}")

    model, _ = load_model(BASE_MODEL, args.device, dtype=torch.float32)
    opener_ids = first_token_ids(tokenizer, REFUSAL_OPENERS)

    runs = {}
    for name, vector in vectors.items():
        log(f"\n  {VARIANTS[name]}")
        points = []
        for coefficient in args.coefficients:
            delta = scaled_delta(vector, coefficient * SCALE, model)
            completions = generate(
                model, tokenizer, ARBITRARY_PROMPTS, args.max_new_tokens, REFUSAL_LAYER, delta
            )
            mass = first_token_mass(
                model, tokenizer, ARBITRARY_PROMPTS, REFUSAL_LAYER, delta, opener_ids
            )
            points.append(
                {
                    "coefficient": coefficient,
                    "refusal_rate": refusal_rate(completions),
                    "opener_mass": mass,
                    "refused": sum(refuses(c.text) for c in completions),
                    "refused_flags": [bool(refuses(c.text)) for c in completions],
                    "completions": [c.text for c in completions],
                }
            )
            log(
                f"    c={coefficient:<5} rate {points[-1]['refusal_rate']:.2f}"
                f"  opener mass {mass:.6f}"
            )
        runs[name] = points

    OUT_PATH.write_text(
        json.dumps(
            {
                "model": str(BASE_MODEL.relative_to(ROOT)),
                "layer": REFUSAL_LAYER,
                "scale": SCALE,
                "set_size": SET_SIZE,
                "n_sets": N_SETS,
                "seed": SEED,
                "pool": REFUSAL_POOL_WIDE,
                "prompts": ARBITRARY_PROMPTS,
                "max_new_tokens": args.max_new_tokens,
                "variant_labels": VARIANTS,
                "readouts": readouts,
                "runs": runs,
            },
            indent=1,
            ensure_ascii=False,
        )
    )
    log(f"\nsaved {OUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
