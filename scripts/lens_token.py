import argparse
import sys

import torch

from assets import BASE_MODEL, JLENS_TENSOR, REFUSAL_VECTORS
from eval_prompts import NEUTRAL_PROMPTS
from jlens import lens_logits_batch
from steering import chat_prompts, generate, load_model, scaled_delta, steering
from weights import WeightReader

REFUSAL_LAYER = 15
DEFAULT_COEFFICIENTS = [0.0, 120.0]


def log(message: str) -> None:
    print(message, flush=True)


def load_lens(device, layers):
    stored = torch.load(JLENS_TENSOR, map_location="cpu", weights_only=False)["J"]
    wanted = layers or sorted(stored)
    return {layer: stored[layer].float().to(device) for layer in wanted if layer in stored}


def resolve_tokens(tokenizer, texts):
    resolved = []
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        for piece in ids:
            resolved.append((tokenizer.decode([piece]), piece))
        if len(ids) != 1:
            log(f"note: {text!r} is {len(ids)} tokens; reporting each piece")
    return resolved


@torch.no_grad()
def collect(model, tokenizer, jacobians, unembed, gain, vector, prompt, coefficient, n_generate):
    delta = scaled_delta(vector, coefficient, model)
    completion = generate(model, tokenizer, [prompt], n_generate, REFUSAL_LAYER, delta)[0]
    prompt_ids = tokenizer(chat_prompts(tokenizer, [prompt])[0], add_special_tokens=False)[
        "input_ids"
    ]
    ids = torch.tensor([prompt_ids + completion.token_ids], device=model.device)
    with steering(model, REFUSAL_LAYER, delta):
        states = model(input_ids=ids, output_hidden_states=True).hidden_states
    start = len(prompt_ids) - 1
    return completion, {
        layer: lens_logits_batch(matrix, unembed, gain, states[layer][0, start:, :]).float()
        for layer, matrix in jacobians.items()
    }


def token_report(logits_by_layer, token_id, position):
    rows = []
    for layer, logits in sorted(logits_by_layer.items()):
        row = logits[position]
        value = row[token_id]
        rank = int((row > value).sum()) + 1
        prob = float(torch.softmax(row, -1)[token_id])
        rows.append((layer, rank, prob))
    return rows


def top_report(logits_by_layer, position, k, tokenizer):
    out = []
    for layer, logits in sorted(logits_by_layer.items()):
        probs = torch.softmax(logits[position], -1)
        top = probs.topk(k)
        out.append(
            (
                layer,
                [
                    (tokenizer.decode([int(i)]), float(p))
                    for i, p in zip(top.indices, top.values, strict=True)
                ],
            )
        )
    return out


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read the Jacobian lens for chosen tokens, with and without refusal steering."
    )
    parser.add_argument("--tokens", nargs="*", default=[" cannot", " illegal", "禁止"])
    parser.add_argument("--prompt", default="0", help="index into NEUTRAL_PROMPTS, or a string")
    parser.add_argument("--coefficients", type=float, nargs="+", default=DEFAULT_COEFFICIENTS)
    parser.add_argument("--layers", type=int, nargs="*", default=None)
    parser.add_argument("--position", type=int, default=0, help="0 = last prompt token")
    parser.add_argument("--generate", type=int, default=16)
    parser.add_argument("--top", type=int, default=0, help="instead dump the top-k per layer")
    parser.add_argument("--device", default="mps")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prompt = NEUTRAL_PROMPTS[int(args.prompt)] if args.prompt.lstrip("-").isdigit() else args.prompt

    model, tokenizer = load_model(BASE_MODEL, args.device, dtype=torch.float32)
    reader = WeightReader(BASE_MODEL)
    unembed = reader.get("lm_head.weight").float().to(args.device)
    gain = reader.get("model.norm.weight").float().to(args.device)
    jacobians = load_lens(args.device, args.layers)
    vector = torch.load(REFUSAL_VECTORS, map_location="cpu", weights_only=False)["directions"][
        REFUSAL_LAYER
    ].float()

    log(f"prompt: {prompt!r}")
    log(f"position {args.position} (0 = last prompt token), steering at layer {REFUSAL_LAYER}\n")

    runs = {}
    for coefficient in args.coefficients:
        completion, logits = collect(
            model,
            tokenizer,
            jacobians,
            unembed,
            gain,
            vector,
            prompt,
            coefficient,
            args.generate,
        )
        runs[coefficient] = logits
        log(f"c={coefficient:<6} says: {completion.text[:96]!r}")

    if args.top:
        for coefficient, logits in runs.items():
            log(f"\n=== c={coefficient}  top-{args.top} per layer")
            for layer, entries in top_report(logits, args.position, args.top, tokenizer):
                shown = "  ".join(f"{t!r} {100 * p:.1f}%" for t, p in entries)
                log(f"  L{layer:<2} {shown}")
        return 0

    for text, token_id in resolve_tokens(tokenizer, args.tokens):
        log(f"\n=== token {text!r} (id {token_id})")
        header = "  layer  " + "  ".join(
            f"{'c=' + str(c) + '  rank / prob':>20}" for c in args.coefficients
        )
        log(header)
        reports = {c: token_report(logits, token_id, args.position) for c, logits in runs.items()}
        for index in range(len(next(iter(reports.values())))):
            layer = reports[args.coefficients[0]][index][0]
            cells = []
            for coefficient in args.coefficients:
                _, rank, prob = reports[coefficient][index]
                cells.append(f"{'#' + format(rank, ',d'):>10} {prob:9.2e}")
            log(f"  L{layer:<5} " + "  ".join(cells))
        for coefficient in args.coefficients:
            best = min(reports[coefficient], key=lambda row: row[1])
            log(f"  best c={coefficient}: rank {best[1]:,} at layer {best[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
