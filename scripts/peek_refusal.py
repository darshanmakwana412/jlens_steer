import argparse
import json
import sys

import torch

from assets import ARTIFACTS, BASE_MODEL, JLENS_TENSOR, REFUSAL_VECTORS, ROOT
from eval_prompts import NEUTRAL_PROMPTS
from jlens import lens_logits_batch
from steering import chat_prompts, generate, load_model, scaled_delta, steering
from weights import WeightReader

REFUSAL_LAYER = 15
STEER_COEFFICIENT = 120.0
RANK_LAYERS = list(range(18, 27))
TOP_K = 5
N_RANKED = 60
N_PINNED = 12
VIEW_PROMPTS = 2
VIEW_POSITIONS = 16

OUT_PATH = ARTIFACTS / "refusal_mind.json"


def log(message: str) -> None:
    print(message, flush=True)


def load_lens(device):
    stored = torch.load(JLENS_TENSOR, map_location="cpu", weights_only=False)
    return {layer: matrix.float().to(device) for layer, matrix in stored["J"].items()}


@torch.no_grad()
def forward(model, batch, vector, coefficient):
    with steering(model, REFUSAL_LAYER, scaled_delta(vector, coefficient, model)):
        return model(**batch, output_hidden_states=True)


def ranked_tokens(model, tokenizer, jacobians, unembed, gain, vector, prompts):
    batch = tokenizer(
        chat_prompts(tokenizer, prompts), return_tensors="pt", padding=True, padding_side="left"
    ).to(model.device)

    def mean_logprobs(coefficient):
        states = forward(model, batch, vector, coefficient).hidden_states
        total = None
        for layer in RANK_LAYERS:
            logits = lens_logits_batch(jacobians[layer], unembed, gain, states[layer][:, -1, :])
            step = torch.log_softmax(logits.float(), -1).mean(0)
            total = step if total is None else total + step
        return total / len(RANK_LAYERS)

    delta = mean_logprobs(STEER_COEFFICIENT) - mean_logprobs(0.0)
    promoted, suppressed = delta.topk(N_RANKED), (-delta).topk(N_RANKED)

    def entries(top, sign):
        return [
            {"token": tokenizer.decode([int(i)]), "id": int(i), "delta_logprob": sign * float(v)}
            for i, v in zip(top.indices.tolist(), top.values.tolist(), strict=True)
        ]

    return {"promoted": entries(promoted, 1.0), "suppressed": entries(suppressed, -1.0)}


@torch.no_grad()
def readout_view(
    model, tokenizer, jacobians, unembed, gain, vector, prompt, coefficient, pinned_ids
):
    completion = generate(
        model,
        tokenizer,
        [prompt],
        VIEW_POSITIONS,
        REFUSAL_LAYER,
        scaled_delta(vector, coefficient, model),
    )[0]
    prompt_ids = tokenizer(chat_prompts(tokenizer, [prompt])[0], add_special_tokens=False)[
        "input_ids"
    ]
    full = prompt_ids + completion.token_ids
    batch = {"input_ids": torch.tensor([full], device=model.device)}
    out = forward(model, batch, vector, coefficient)

    start = len(prompt_ids) - 1
    real = out.logits[0, start:, :].float()
    layers = sorted(jacobians)
    pinned = torch.tensor(pinned_ids, device=model.device)

    cells, pinned_ranks = [], []
    for layer in layers:
        hidden = out.hidden_states[layer][0, start:, :]
        logits = lens_logits_batch(jacobians[layer], unembed, gain, hidden).float()
        probs = torch.softmax(logits, -1)
        top = probs.topk(TOP_K, dim=-1)
        best = top.indices[:, 0]
        real_rank = (real > real.gather(1, best.unsqueeze(1))).sum(-1) + 1
        row = []
        for index in range(logits.shape[0]):
            row.append(
                {
                    "t": tokenizer.decode([int(best[index])]),
                    "p": round(float(top.values[index, 0]), 4),
                    "r": int(real_rank[index]),
                    "top": [
                        [tokenizer.decode([int(i)]), round(float(p), 4)]
                        for i, p in zip(
                            top.indices[index].tolist(), top.values[index].tolist(), strict=True
                        )
                    ],
                }
            )
        cells.append(row)
        ranks = [
            ((logits > logits[:, token_id : token_id + 1]).sum(-1) + 1).tolist()
            for token_id in pinned.tolist()
        ]
        pinned_ranks.append(ranks)

    return {
        "prompt": prompt,
        "completion": completion.text,
        "position_tokens": ["→"] + [tokenizer.decode([i]) for i in completion.token_ids],
        "layers": layers,
        "cells": cells,
        "pinned_ranks": pinned_ranks,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Read the model's mind with and without steering.")
    parser.add_argument("--device", default="mps")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model, tokenizer = load_model(BASE_MODEL, args.device, dtype=torch.float32)
    reader = WeightReader(BASE_MODEL)
    unembed = reader.get("lm_head.weight").float().to(args.device)
    gain = reader.get("model.norm.weight").float().to(args.device)
    jacobians = load_lens(args.device)
    vector = torch.load(REFUSAL_VECTORS, map_location="cpu", weights_only=False)["directions"][
        REFUSAL_LAYER
    ].float()

    log(f"ranking over layers {RANK_LAYERS[0]}-{RANK_LAYERS[-1]}, {len(NEUTRAL_PROMPTS)} prompts")
    ranking = ranked_tokens(model, tokenizer, jacobians, unembed, gain, vector, NEUTRAL_PROMPTS)

    pinned = ranking["promoted"][:N_PINNED] + ranking["suppressed"][:N_PINNED]
    pinned_ids = [entry["id"] for entry in pinned]
    log(f"pinnable tokens: {[entry['token'] for entry in pinned]}")

    views = []
    for prompt in NEUTRAL_PROMPTS[:VIEW_PROMPTS]:
        log(f"readout grid: {prompt!r}")
        views.append(
            {
                condition: readout_view(
                    model,
                    tokenizer,
                    jacobians,
                    unembed,
                    gain,
                    vector,
                    prompt,
                    coefficient,
                    pinned_ids,
                )
                for condition, coefficient in (("base", 0.0), ("steered", STEER_COEFFICIENT))
            }
        )

    OUT_PATH.write_text(
        json.dumps(
            {
                "model": str(BASE_MODEL.relative_to(ROOT)),
                "steer_layer": REFUSAL_LAYER,
                "steer_coefficient": STEER_COEFFICIENT,
                "rank_layers": RANK_LAYERS,
                "top_k": TOP_K,
                "ranking": ranking,
                "pinnable": pinned,
                "views": views,
            },
            ensure_ascii=False,
        )
    )
    log(f"saved {OUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
