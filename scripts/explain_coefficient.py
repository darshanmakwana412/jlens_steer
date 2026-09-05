import argparse
import json
import sys

import torch

from assets import (
    ARTIFACTS,
    BASE_MODEL,
    CAPS_VECTOR,
    JLENS_CAPS_VECTORS,
    JLENS_TENSOR,
    ROOT,
)
from concept_tokens import CAPS_TOKEN_SETS, casing_pairs
from eval_prompts import CAPS_PROMPTS
from jlens import load_jacobian, unit
from plots import render_xy
from steering import load_model, logit_gap, scaled_delta

CAPS_LAYER = 13
BEHAVIOUR_PATH = ARTIFACTS / "jlens_caps_eval.json"
METRICS_PATH = ARTIFACTS / "coefficient_gain.json"
GAP_PLOT = ARTIFACTS / "coefficient_gain_gap.png"
BEHAVIOUR_PLOT = ARTIFACTS / "coefficient_gain_behaviour.png"

LABELS = {"reference": "fitted vector (caps_L13)", "vocab_centered": "J lens, vocab-centered"}


def log(message: str) -> None:
    print(message, flush=True)


def load_vectors():
    reference = torch.load(CAPS_VECTOR, map_location="cpu").float()
    derived = torch.load(JLENS_CAPS_VECTORS, map_location="cpu", weights_only=False)
    scale = derived["reference_norm"]
    return {
        "reference": reference,
        "vocab_centered": derived["directions"]["vocab_centered"] * scale,
    }


def geometry(vectors):
    jacobian = load_jacobian(JLENS_TENSOR, CAPS_LAYER)
    reference = vectors["reference"]
    out = {}
    for name, vector in vectors.items():
        out[name] = {
            "norm": vector.norm().item(),
            "cos_to_reference": torch.dot(unit(vector), unit(reference)).item(),
            "jacobian_gain": ((jacobian @ vector).norm() / vector.norm()).item(),
        }
    return out


def parse_args():
    parser = argparse.ArgumentParser(
        description="Why the two vectors peak at different coefficients."
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def build_plots(result):
    render_xy(
        [
            (LABELS[name], series["coefficients"], series["induced_gap"])
            for name, series in result["series"].items()
        ],
        f"Steering coefficient  (layer {CAPS_LAYER}, both at norm {result['reference_norm']:.1f})",
        "Induced ALL-CAPS logit advantage",
        GAP_PLOT,
    )
    rising = {
        name: max(range(len(s["caps_percent"])), key=lambda i: s["caps_percent"][i]) + 1
        for name, s in result["series"].items()
    }
    render_xy(
        [
            (
                LABELS[name],
                series["induced_gap"][: rising[name]],
                series["caps_percent"][: rising[name]],
            )
            for name, series in result["series"].items()
        ],
        "Induced ALL-CAPS logit advantage",
        "Generated tokens in ALL CAPS (%)",
        BEHAVIOUR_PLOT,
        legend_anchor=(1.0, 0.4),
    )


def main() -> int:
    args = parse_args()
    if args.plot_only:
        build_plots(json.loads(METRICS_PATH.read_text()))
        log(f"saved {GAP_PLOT.relative_to(ROOT)}")
        log(f"saved {BEHAVIOUR_PLOT.relative_to(ROOT)}")
        return 0

    behaviour = json.loads(BEHAVIOUR_PATH.read_text())
    vectors = load_vectors()
    model, tokenizer = load_model(BASE_MODEL, args.device, dtype=torch.float32)

    used = {token for group in CAPS_TOKEN_SETS for token in group}
    caps, lower, _ = casing_pairs(tokenizer, exclude=used)
    log(f"probe: {len(caps)} held-out casing pairs (none used to build the vector)")
    baseline = logit_gap(model, tokenizer, CAPS_PROMPTS, CAPS_LAYER, None, caps, lower)
    log(f"unsteered caps-minus-lowercase logit gap {baseline:+.3f}\n")

    geo = geometry(vectors)
    series = {}
    for name, vector in vectors.items():
        points = behaviour["series"][name]["points"]
        coefficients = [point["coefficient"] for point in points]
        gaps, caps_percent = [], []
        for point in points:
            delta = scaled_delta(vector, point["coefficient"], model)
            gap = logit_gap(model, tokenizer, CAPS_PROMPTS, CAPS_LAYER, delta, caps, lower)
            gaps.append(gap - baseline)
            caps_percent.append(100 * point["measure"])
            log(
                f"  {name:16} c={point['coefficient']:<5} gap {gaps[-1]:+7.3f}"
                f"  caps {caps_percent[-1]:5.1f}%"
            )
        series[name] = {
            "coefficients": coefficients,
            "induced_gap": gaps,
            "caps_percent": caps_percent,
            **geo[name],
        }

    result = {
        "layer": CAPS_LAYER,
        "reference_norm": vectors["reference"].norm().item(),
        "baseline_gap": baseline,
        "probe": "mean logit over held-out UPPER tokens minus their lowercase twins",
        "n_probe_pairs": len(caps),
        "series": series,
    }
    METRICS_PATH.write_text(json.dumps(result, indent=2))
    build_plots(result)

    log("")
    for name, entry in series.items():
        log(
            f"{LABELS[name]:<28} jacobian gain {entry['jacobian_gain']:7.3f}"
            f"  cos {entry['cos_to_reference']:+.4f}  max gap {max(entry['induced_gap']):+.2f}"
        )
    log(f"\nsaved {METRICS_PATH.relative_to(ROOT)}")
    log(f"saved {GAP_PLOT.relative_to(ROOT)}")
    log(f"saved {BEHAVIOUR_PLOT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
