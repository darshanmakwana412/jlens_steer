import argparse
import json
import sys

import torch
from transformers import AutoTokenizer

from assets import ARTIFACTS, BASE_MODEL, JLENS_TENSOR, REFUSAL_VECTORS, ROOT
from concept_tokens import REFUSAL_TOKEN_SETS, grouped, token_ids
from jlens import concept_direction, load_jacobian, top_lens_tokens, unit
from weights import WeightReader

REFUSAL_LAYER = 15
SET_SIZE = 3
N_DISCOVERED = 30

MIND_PATH = ARTIFACTS / "refusal_mind.json"
OUT_PATH = ARTIFACTS / "jlens_refusal_qwen3_1_7B.pt"


def log(message: str) -> None:
    print(message, flush=True)


def discovered_sets(tokenizer):
    mind = json.loads(MIND_PATH.read_text())
    promoted = [entry["token"] for entry in mind["ranking"]["promoted"]]
    single = [
        token
        for token in promoted
        if len(tokenizer(token, add_special_tokens=False)["input_ids"]) == 1
    ]
    return grouped(single[:N_DISCOVERED], SET_SIZE)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Derive refusal directions by inverting the Jacobian lens."
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    parse_args()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    reader = WeightReader(BASE_MODEL)
    unembed = reader.get("lm_head.weight").float()
    gain = reader.get("model.norm.weight").float()
    jacobian = load_jacobian(JLENS_TENSOR, REFUSAL_LAYER)
    center = unembed.mean(0)

    reference = torch.load(REFUSAL_VECTORS, map_location="cpu", weights_only=False)["directions"][
        REFUSAL_LAYER
    ].float()

    token_sets = {
        "apriori": REFUSAL_TOKEN_SETS,
        "discovered": discovered_sets(tokenizer),
    }
    directions, cosines = {}, {}
    for name, sets in token_sets.items():
        ids = token_ids(tokenizer, sets)
        vector = concept_direction(jacobian, unembed, gain, ids, center=center)
        directions[name] = vector
        cosines[name] = torch.dot(vector, unit(reference)).item()
        log(f"\n{name}: {len(sets)} sets, {sum(len(g) for g in sets)} tokens")
        log(f"  cos(derived, abliteration direction) = {cosines[name]:+.4f}")
        log(f"  lens reads: {top_lens_tokens(jacobian, unembed, gain, vector, tokenizer, 8)}")

    torch.save(
        {
            "directions": directions,
            "cosine_to_reference": cosines,
            "token_sets": token_sets,
            "layer": REFUSAL_LAYER,
            "reference": str(REFUSAL_VECTORS.relative_to(ROOT)),
            "reference_is_unit": True,
            "method": (
                "d = mean over sets of unit(J_l^T (g * (mean unembed row over set - vocab mean)))"
            ),
        },
        OUT_PATH,
    )
    log(f"\nsaved {OUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
