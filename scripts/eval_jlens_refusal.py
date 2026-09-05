import argparse
import json
import sys

import torch

from assets import ARTIFACTS, BASE_MODEL, REFUSAL_VECTORS, ROOT
from eval_prompts import NEUTRAL_PROMPTS
from plots import render_comparison
from steering import load_model
from sweeps import refusal_score, sweep

COEFFICIENTS = [0.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0, 250.0, 300.0]

SERIES_LABELS = {
    "reference": "abliteration direction",
    "discovered": "J lens, lens-discovered tokens",
    "apriori": "J lens, hand-picked tokens",
}
SERIES_ORDER = ["reference", "discovered", "apriori"]

DERIVED_PATH = ARTIFACTS / "jlens_refusal_qwen3_1_7B.pt"
METRICS_PATH = ARTIFACTS / "jlens_refusal_eval.json"
SAMPLES_PATH = ARTIFACTS / "jlens_refusal_eval_samples.json"
PLOT_PATH = ARTIFACTS / "jlens_refusal_comparison.png"


def log(message: str) -> None:
    print(message, flush=True)


def load_series():
    derived = torch.load(DERIVED_PATH, map_location="cpu", weights_only=False)
    layer = derived["layer"]
    reference = torch.load(REFUSAL_VECTORS, map_location="cpu", weights_only=False)["directions"][
        layer
    ].float()
    vectors = {"reference": reference, **derived["directions"]}
    return vectors, derived, layer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare J-lens derived refusal directions against the abliteration one."
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--coefficients", type=float, nargs="+", default=COEFFICIENTS)
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def build_plot(result):
    render_comparison(
        [
            (SERIES_LABELS[name], result["series"][name]["points"])
            for name in SERIES_ORDER
            if name in result["series"]
        ],
        f"Steering coefficient  (layer {result['layer']}, all unit norm)",
        "Harmless prompts refused (%)",
        PLOT_PATH,
    )


def main() -> int:
    args = parse_args()
    if args.plot_only:
        build_plot(json.loads(METRICS_PATH.read_text()))
        log(f"saved {PLOT_PATH.relative_to(ROOT)}")
        return 0

    vectors, derived, layer = load_series()
    model, tokenizer = load_model(BASE_MODEL, args.device)

    series, samples = {}, {}
    for name in SERIES_ORDER:
        log(f"  {SERIES_LABELS[name]}")
        points, generated = sweep(
            model,
            tokenizer,
            NEUTRAL_PROMPTS,
            layer,
            vectors[name],
            args.coefficients,
            args.max_new_tokens,
            refusal_score,
            log,
        )
        series[name] = {
            "label": SERIES_LABELS[name],
            "cosine_to_reference": derived["cosine_to_reference"].get(name, 1.0),
            "points": points,
        }
        samples[name] = generated

    result = {
        "model": str(BASE_MODEL.relative_to(ROOT)),
        "layer": layer,
        "n_prompts": len(NEUTRAL_PROMPTS),
        "max_new_tokens": args.max_new_tokens,
        "metric": "fraction of harmless prompts whose completion is a refusal",
        "series": series,
    }
    METRICS_PATH.write_text(json.dumps(result, indent=2))
    SAMPLES_PATH.write_text(json.dumps(samples, indent=2))
    build_plot(result)

    log("")
    for name in SERIES_ORDER:
        points = series[name]["points"]
        peak = max(point["measure"] for point in points)
        log(
            f"{SERIES_LABELS[name]:<32} peak {100 * peak:5.1f}%   "
            f"cos {series[name]['cosine_to_reference']:+.4f}"
        )
    log(f"\nsaved {METRICS_PATH.relative_to(ROOT)}")
    log(f"saved {SAMPLES_PATH.relative_to(ROOT)}")
    log(f"saved {PLOT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
