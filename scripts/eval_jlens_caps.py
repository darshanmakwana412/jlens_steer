import argparse
import json
import sys

import torch

from assets import ARTIFACTS, BASE_MODEL, CAPS_VECTOR, JLENS_CAPS_VECTORS, ROOT
from eval_prompts import CAPS_PROMPTS
from plots import render_comparison
from steering import load_model
from sweeps import caps_score, sweep

COEFFICIENTS = [0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]

SERIES_LABELS = {
    "reference": "fitted vector (caps_L13)",
    "vocab_centered": "J lens, vocab-centered",
    "lowercase_contrast": "J lens, lowercase contrast",
    "raw": "J lens, uncentered",
}
SERIES_ORDER = ["reference", "vocab_centered", "lowercase_contrast", "raw"]

METRICS_PATH = ARTIFACTS / "jlens_caps_eval.json"
SAMPLES_PATH = ARTIFACTS / "jlens_caps_eval_samples.json"
PLOT_PATH = ARTIFACTS / "jlens_caps_comparison.png"


def log(message: str) -> None:
    print(message, flush=True)


def load_series():
    derived = torch.load(JLENS_CAPS_VECTORS, map_location="cpu", weights_only=False)
    reference = torch.load(CAPS_VECTOR, map_location="cpu").float()
    scale = derived["reference_norm"]
    vectors = {"reference": reference}
    for name, vector in derived["directions"].items():
        vectors[name] = vector * scale
    return vectors, derived


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare the J-lens derived caps vector against the fitted one."
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--coefficients", type=float, nargs="+", default=COEFFICIENTS)
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def build_plot(result):
    series = [
        (SERIES_LABELS[name], result["series"][name]["points"])
        for name in SERIES_ORDER
        if name in result["series"]
    ]
    render_comparison(
        series,
        f"Steering coefficient  (layer {result['layer']}, all vectors at norm"
        f" {result['reference_norm']:.1f})",
        "Generated tokens in ALL CAPS (%)",
        PLOT_PATH,
    )


def main() -> int:
    args = parse_args()
    if args.plot_only:
        build_plot(json.loads(METRICS_PATH.read_text()))
        log(f"saved {PLOT_PATH.relative_to(ROOT)}")
        return 0

    vectors, derived = load_series()
    model, tokenizer = load_model(BASE_MODEL, args.device)

    series, samples = {}, {}
    for name in SERIES_ORDER:
        log(f"  {SERIES_LABELS[name]}")
        points, generated = sweep(
            model,
            tokenizer,
            CAPS_PROMPTS,
            derived["layer"],
            vectors[name],
            args.coefficients,
            args.max_new_tokens,
            caps_score,
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
        "layer": derived["layer"],
        "reference_norm": derived["reference_norm"],
        "n_prompts": len(CAPS_PROMPTS),
        "max_new_tokens": args.max_new_tokens,
        "n_concept_tokens": derived["n_concept_tokens"],
        "metric": "fraction of letter-bearing generated tokens that are all caps",
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
            f"{SERIES_LABELS[name]:<28} peak {100 * peak:5.1f}%   "
            f"cos to fitted {series[name]['cosine_to_reference']:+.4f}"
        )
    log(f"\nsaved {METRICS_PATH.relative_to(ROOT)}")
    log(f"saved {SAMPLES_PATH.relative_to(ROOT)}")
    log(f"saved {PLOT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
