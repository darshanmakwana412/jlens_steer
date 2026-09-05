import argparse
import json
import sys

import torch

from assets import ARTIFACTS, BASE_MODEL, CAPS_VECTOR, REFUSAL_VECTORS, ROOT
from eval_prompts import CAPS_PROMPTS, NEUTRAL_PROMPTS
from plots import render
from steering import load_model, mean_residual_norm
from sweeps import caps_score, refusal_score, sweep

CAPS_LAYER = 13

CAPS_COEFFICIENTS = [0.0, 0.25, 0.5, 0.6, 0.75, 0.85, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
REFUSAL_COEFFICIENTS = [
    0.0,
    20.0,
    40.0,
    50.0,
    60.0,
    70.0,
    80.0,
    90.0,
    100.0,
    110.0,
    120.0,
    140.0,
    160.0,
]

CAPS_METRIC = "fraction of letter-bearing generated tokens that are all caps"
REFUSAL_METRIC = "fraction of prompts whose completion matches a refusal marker"

METRICS_PATH = ARTIFACTS / "steering_eval.json"
SAMPLES_PATH = ARTIFACTS / "steering_eval_samples.json"
CAPS_PLOT_PATH = ARTIFACTS / "steering_eval_caps.png"
REFUSAL_PLOT_PATH = ARTIFACTS / "steering_eval_refusal.png"


def log(message: str) -> None:
    print(message, flush=True)


def load_vectors():
    caps = torch.load(CAPS_VECTOR, map_location="cpu").float()
    artifact = torch.load(REFUSAL_VECTORS, map_location="cpu", weights_only=False)
    layer = artifact["peak_layer"]
    return caps, artifact["directions"][layer].float(), layer


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep steering strength and score behaviour.")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--caps-coefficients", type=float, nargs="+", default=CAPS_COEFFICIENTS)
    parser.add_argument(
        "--refusal-coefficients", type=float, nargs="+", default=REFUSAL_COEFFICIENTS
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.plot_only:
        saved = json.loads(METRICS_PATH.read_text())
        render(saved["caps"], saved["refusal"], CAPS_PLOT_PATH, REFUSAL_PLOT_PATH)
        log(f"saved {CAPS_PLOT_PATH.relative_to(ROOT)}")
        log(f"saved {REFUSAL_PLOT_PATH.relative_to(ROOT)}")
        return 0

    model, tokenizer = load_model(BASE_MODEL, args.device)
    caps_vector, refusal_vector, refusal_layer = load_vectors()

    log(f"caps vector norm {caps_vector.norm():.2f} at layer {CAPS_LAYER}")
    log(f"refusal direction unit norm at layer {refusal_layer}")

    log("  all-caps sweep")
    caps_points, caps_samples = sweep(
        model,
        tokenizer,
        CAPS_PROMPTS,
        CAPS_LAYER,
        caps_vector,
        args.caps_coefficients,
        args.max_new_tokens,
        caps_score,
        log,
    )
    log("  refusal sweep")
    refusal_points, refusal_samples = sweep(
        model,
        tokenizer,
        NEUTRAL_PROMPTS,
        refusal_layer,
        refusal_vector,
        args.refusal_coefficients,
        args.max_new_tokens,
        refusal_score,
        log,
    )

    caps = {
        "vector": str(CAPS_VECTOR.relative_to(ROOT)),
        "layer": CAPS_LAYER,
        "vector_norm": caps_vector.norm().item(),
        "mean_residual_norm": mean_residual_norm(model, tokenizer, CAPS_PROMPTS, CAPS_LAYER),
        "n_prompts": len(CAPS_PROMPTS),
        "max_new_tokens": args.max_new_tokens,
        "metric": CAPS_METRIC,
        "points": caps_points,
    }
    refusal = {
        "vector": str(REFUSAL_VECTORS.relative_to(ROOT)),
        "layer": refusal_layer,
        "vector_norm": refusal_vector.norm().item(),
        "mean_residual_norm": mean_residual_norm(model, tokenizer, NEUTRAL_PROMPTS, refusal_layer),
        "n_prompts": len(NEUTRAL_PROMPTS),
        "max_new_tokens": args.max_new_tokens,
        "metric": REFUSAL_METRIC,
        "points": refusal_points,
    }

    result = {"model": str(BASE_MODEL.relative_to(ROOT)), "caps": caps, "refusal": refusal}
    METRICS_PATH.write_text(json.dumps(result, indent=2))
    SAMPLES_PATH.write_text(
        json.dumps({"caps": caps_samples, "refusal": refusal_samples}, indent=2)
    )
    render(caps, refusal, CAPS_PLOT_PATH, REFUSAL_PLOT_PATH)

    log(f"saved {METRICS_PATH.relative_to(ROOT)}")
    log(f"saved {SAMPLES_PATH.relative_to(ROOT)}")
    log(f"saved {CAPS_PLOT_PATH.relative_to(ROOT)}")
    log(f"saved {REFUSAL_PLOT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
