import sys

import torch
from transformers import AutoTokenizer

from assets import BASE_MODEL, CAPS_VECTOR, JLENS_CAPS_VECTORS, JLENS_TENSOR, ROOT
from concept_tokens import CAPS_TOKEN_SETS, LOWERCASE_TOKEN_SETS, token_ids
from jlens import concept_direction, load_jacobian, top_lens_tokens, unit
from weights import WeightReader

CAPS_LAYER = 13
UNEMBED = "lm_head.weight"
FINAL_GAIN = "model.norm.weight"


def log(message: str) -> None:
    print(message, flush=True)


def build(jacobian, unembed, gain, caps_sets, lowercase_sets):
    center = unembed.mean(0)
    return {
        "vocab_centered": concept_direction(jacobian, unembed, gain, caps_sets, center=center),
        "lowercase_contrast": concept_direction(
            jacobian, unembed, gain, caps_sets, contrast_id_sets=lowercase_sets
        ),
        "raw": concept_direction(jacobian, unembed, gain, caps_sets),
    }


def main() -> int:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    reader = WeightReader(BASE_MODEL)
    unembed = reader.get(UNEMBED).float()
    gain = reader.get(FINAL_GAIN).float()
    jacobian = load_jacobian(JLENS_TENSOR, CAPS_LAYER)

    caps_sets = token_ids(tokenizer, CAPS_TOKEN_SETS)
    lowercase_sets = token_ids(tokenizer, LOWERCASE_TOKEN_SETS)
    n_tokens = sum(len(group) for group in caps_sets)

    truth = torch.load(CAPS_VECTOR, map_location="cpu").float()
    directions = build(jacobian, unembed, gain, caps_sets, lowercase_sets)

    log(f"layer {CAPS_LAYER}, {n_tokens} concept tokens in {len(caps_sets)} sets")
    log(f"reference vector norm {truth.norm():.3f}")
    for name, vector in directions.items():
        cosine = torch.dot(vector, unit(truth)).item()
        log(f"\n{name}: cos(derived, caps_L13) = {cosine:+.4f}")
        log(f"  lens reads: {top_lens_tokens(jacobian, unembed, gain, vector, tokenizer, 8)}")

    torch.save(
        {
            "directions": {name: vector for name, vector in directions.items()},
            "cosine_to_reference": {
                name: torch.dot(vector, unit(truth)).item() for name, vector in directions.items()
            },
            "reference": str(CAPS_VECTOR.relative_to(ROOT)),
            "reference_norm": truth.norm().item(),
            "layer": CAPS_LAYER,
            "n_concept_tokens": n_tokens,
            "concept_token_sets": CAPS_TOKEN_SETS,
            "method": (
                "d = mean over sets of unit(J_l^T (g * (mean unembed row over set - center))); "
                "vocab_centered uses the mean unembed row over the whole vocabulary as center, "
                "lowercase_contrast uses the matched lowercase tokens, raw uses no center"
            ),
        },
        JLENS_CAPS_VECTORS,
    )
    log(f"\nsaved {JLENS_CAPS_VECTORS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
