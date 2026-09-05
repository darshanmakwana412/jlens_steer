"""Steer with directions derived from concept tokens, against the real vectors.

For each behaviour, builds the direction from k sets of c concept tokens, sweeps
the coefficient, and scores the completions on held-out prompts. Compared
against the published caps vector, the refusal direction recovered from the
abliterated weights, and a norm-matched random direction.

Every direction is unit and applied as ``coefficient * SCALE``, where SCALE is
the published caps vector's norm, so all coefficient axes are comparable.
"""

import argparse
import json
import sys

import torch

from assets import ARTIFACTS, BASE_MODEL, REFUSAL_VECTORS, ROOT, STEERING_VECS
from concepts import CAPS_TOKENS, COMPLIANCE_TOKENS, REFUSAL_TOKENS, single_token_ids
from derive import Recipe, concept_direction
from eval_prompts import CAPS_PROMPTS, NEUTRAL_PROMPTS
from jlens import load_lens, unit
from metrics import count_caps, refusal_rate
from steering import generate, load_model, scaled_delta

LAYER = 13
SCALE = 27.11  # norm of the published caps vector, the shared coefficient unit
COEFFICIENTS = [1.0, 2.0, 3.0, 4.0]
OUT_PATH = ARTIFACTS / "concept_vectors_eval.json"


def log(message: str) -> None:
    print(message, flush=True)


def caps_measure(completions) -> float:
    return count_caps(completions).fraction_of_letter_tokens


def refusal_measure(completions) -> float:
    return refusal_rate(completions)


def sweep(model, tokenizer, prompts, direction, coefficients, max_new_tokens, measure):
    points = []
    for coefficient in coefficients:
        delta = scaled_delta(direction * SCALE, coefficient, model)
        completions = generate(model, tokenizer, prompts, max_new_tokens, LAYER, delta)
        points.append(
            {
                "coefficient": coefficient,
                "measure": measure(completions),
                "completions": [completion.text for completion in completions],
            }
        )
    return points


def baseline_vectors(device: str):
    caps = torch.load(STEERING_VECS / "caps_L13.pt", map_location="cpu").float()
    artifact = torch.load(REFUSAL_VECTORS, map_location="cpu", weights_only=False)
    refusal = artifact["directions"][LAYER].float()
    return unit(caps).to(device), unit(refusal).to(device)


def parse_args():
    parser = argparse.ArgumentParser(description="Score concept-token directions.")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--set-size", type=int, default=3, help="concept tokens per set (c)")
    parser.add_argument("--n-sets", type=int, default=10, help="sets averaged (k)")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--coefficients", type=float, nargs="+", default=COEFFICIENTS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model, tokenizer = load_model(BASE_MODEL, args.device)
    lens = load_lens(model, args.device)
    caps_baseline, refusal_baseline = baseline_vectors(args.device)

    generator = torch.Generator().manual_seed(7)
    control = unit(torch.randn(model.config.hidden_size, generator=generator)).to(args.device)

    behaviours = {
        "caps": {
            "positives": single_token_ids(tokenizer, CAPS_TOKENS),
            "opposites": None,
            "prompts": CAPS_PROMPTS,
            "measure": caps_measure,
            "baseline": caps_baseline,
        },
        "refusal": {
            "positives": single_token_ids(tokenizer, REFUSAL_TOKENS),
            "opposites": single_token_ids(tokenizer, COMPLIANCE_TOKENS),
            "prompts": NEUTRAL_PROMPTS,
            "measure": refusal_measure,
            "baseline": refusal_baseline,
        },
    }

    result = {"layer": LAYER, "scale": SCALE, "set_size": args.set_size, "n_sets": args.n_sets}
    for name, spec in behaviours.items():
        log(f"{name}: {len(spec['positives'])} concept tokens")
        directions = {}
        for seed in range(args.seeds):
            recipe = Recipe(
                positives=spec["positives"],
                negative="vocab",
                opposites=spec["opposites"],
                set_size=args.set_size,
                n_sets=args.n_sets,
                seed=seed,
            )
            directions[f"derived_seed{seed}"], _ = concept_direction(
                lens, LAYER, recipe, args.device
            )
        directions["baseline"] = spec["baseline"]
        directions["random"] = control

        runs = {}
        for label, direction in directions.items():
            runs[label] = sweep(
                model,
                tokenizer,
                spec["prompts"],
                direction,
                args.coefficients,
                args.max_new_tokens,
                spec["measure"],
            )
            best = max(point["measure"] for point in runs[label])
            log(f"  {label:16s} best {best:.3f}")

        seeds = [runs[f"derived_seed{s}"] for s in range(args.seeds)]
        result[name] = {
            "n_prompts": len(spec["prompts"]),
            "runs": runs,
            "derived_mean": [
                {
                    "coefficient": coefficient,
                    "mean": sum(run[i]["measure"] for run in seeds) / len(seeds),
                    "min": min(run[i]["measure"] for run in seeds),
                    "max": max(run[i]["measure"] for run in seeds),
                }
                for i, coefficient in enumerate(args.coefficients)
            ],
            "cos_to_baseline": [
                float(torch.dot(runs_dir, spec["baseline"]))
                for runs_dir in (directions[f"derived_seed{s}"] for s in range(args.seeds))
            ],
        }

    OUT_PATH.write_text(json.dumps(result, indent=2))
    log(f"saved {OUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
