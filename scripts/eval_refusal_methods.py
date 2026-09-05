import argparse
import json
import sys

import torch
from transformers import AutoTokenizer

from assets import ARTIFACTS, BASE_MODEL, JLENS_TENSOR, REFUSAL_VECTORS, ROOT
from concept_tokens import (
    COMPLIANCE_POOL,
    REFUSAL_OPENERS,
    REFUSAL_POOL,
    first_token_ids,
    pool_ids,
    sample_index_sets,
)
from eval_prompts import NEUTRAL_PROMPTS
from jlens import concept_direction, load_jacobian, top_lens_tokens, unit
from metrics import refusal_rate, refuses
from plots import render_bands
from steering import first_token_mass, generate, load_model, scaled_delta
from weights import WeightReader

REFUSAL_LAYER = 15
SCALE = 27.11
SET_SIZE = 3
N_SETS = 10
SEEDS = [0, 1, 2]
COEFFICIENTS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]

LABELS = {
    "abliteration": "abliteration direction",
    "vocab": "J lens, vocabulary-mean negative",
    "compliance": "J lens, compliance negative",
    "random": "norm-matched random",
}
ORDER = ["abliteration", "vocab", "compliance", "random"]

METRICS_PATH = ARTIFACTS / "refusal_methods_eval.json"
SAMPLES_PATH = ARTIFACTS / "refusal_methods_samples.json"
RATE_PLOT = ARTIFACTS / "refusal_methods_rate.png"
MASS_PLOT = ARTIFACTS / "refusal_methods_openermass.png"


def log(message: str) -> None:
    print(message, flush=True)


def build_directions(tokenizer, jacobian, unembed, gain, reference):
    center = unembed.mean(0)
    directions = {"abliteration": {None: reference}}
    for key in ("vocab", "compliance", "random"):
        directions[key] = {}
    generator = torch.Generator().manual_seed(1234)
    for seed in SEEDS:
        index_sets = sample_index_sets(len(REFUSAL_POOL), SET_SIZE, N_SETS, seed)
        positive = pool_ids(tokenizer, REFUSAL_POOL, index_sets)
        negative = pool_ids(tokenizer, COMPLIANCE_POOL, index_sets)
        directions["vocab"][seed] = concept_direction(
            jacobian, unembed, gain, positive, center=center
        )
        directions["compliance"][seed] = concept_direction(
            jacobian, unembed, gain, positive, contrast_id_sets=negative
        )
        directions["random"][seed] = unit(
            torch.randn(unembed.shape[1], generator=generator, dtype=torch.float64).float()
        )
    return directions


def score(model, tokenizer, vector, coefficient, opener_ids, max_new_tokens):
    delta = scaled_delta(vector, coefficient * SCALE, model)
    completions = generate(model, tokenizer, NEUTRAL_PROMPTS, max_new_tokens, REFUSAL_LAYER, delta)
    mass = first_token_mass(model, tokenizer, NEUTRAL_PROMPTS, REFUSAL_LAYER, delta, opener_ids)
    return {
        "coefficient": coefficient,
        "injected_norm": coefficient * SCALE,
        "refusal_rate": refusal_rate(completions),
        "opener_mass": mass,
        "refused": sum(refuses(c.text) for c in completions),
    }, [
        {"prompt": p, "completion": c.text}
        for p, c in zip(NEUTRAL_PROMPTS, completions, strict=True)
    ]


def aggregate(runs, key):
    xs = [point["coefficient"] for point in next(iter(runs.values()))]
    values = [[run[i][key] for run in runs.values()] for i in range(len(xs))]
    mean = [sum(v) / len(v) for v in values]
    lo = [min(v) for v in values]
    hi = [max(v) for v in values]
    return xs, mean, (None if len(runs) == 1 else lo), (None if len(runs) == 1 else hi)


def build_plots(result):
    for key, ylabel, path, ylim, logy, scale in (
        ("refusal_rate", "Harmless prompts refused (%)", RATE_PLOT, (-3, 103), False, 100.0),
        (
            "opener_mass",
            "Refusal-opener probability mass",
            MASS_PLOT,
            (1e-6, 2.0),
            True,
            1.0,
        ),
    ):
        series = []
        for name in ORDER:
            entry = result["series"][name]
            xs = entry["coefficients"]
            mean = [scale * v for v in entry[key]["mean"]]
            lo = entry[key]["lo"] and [scale * v for v in entry[key]["lo"]]
            hi = entry[key]["hi"] and [scale * v for v in entry[key]["hi"]]
            series.append((LABELS[name], xs, mean, lo, hi))
        render_bands(
            series,
            f"Steering coefficient  (layer {REFUSAL_LAYER}, applied as coeff x {SCALE} x unit d)",
            ylabel,
            path,
            ylim=ylim,
            logy=logy,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Refusal steering: inverted lens against abliteration and a random control."
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--coefficients", type=float, nargs="+", default=COEFFICIENTS)
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.plot_only:
        build_plots(json.loads(METRICS_PATH.read_text()))
        log(f"saved {RATE_PLOT.relative_to(ROOT)}")
        log(f"saved {MASS_PLOT.relative_to(ROOT)}")
        return 0

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    reader = WeightReader(BASE_MODEL)
    unembed = reader.get("lm_head.weight").float()
    gain = reader.get("model.norm.weight").float()
    jacobian = load_jacobian(JLENS_TENSOR, REFUSAL_LAYER)
    reference = torch.load(REFUSAL_VECTORS, map_location="cpu", weights_only=False)["directions"][
        REFUSAL_LAYER
    ].float()

    directions = build_directions(tokenizer, jacobian, unembed, gain, reference)
    opener_ids = first_token_ids(tokenizer, REFUSAL_OPENERS)
    log(f"openers: {[tokenizer.decode([i]) for i in opener_ids]}")
    log(f"{len(REFUSAL_POOL)} refusal tokens, c={SET_SIZE}, k={N_SETS}, seeds {SEEDS}\n")

    readouts = {}
    for name in ORDER:
        for seed, vector in directions[name].items():
            cos = torch.dot(unit(vector), unit(reference)).item()
            reads = top_lens_tokens(jacobian, unembed, gain, vector, tokenizer, 6)
            readouts[f"{name}:{seed}"] = {"cosine_to_abliteration": cos, "lens_reads": reads}
            if seed in (None, SEEDS[0]):
                log(f"{LABELS[name]:<34} cos {cos:+.4f}  reads {reads}")

    model, _ = load_model(BASE_MODEL, args.device, dtype=torch.float32)

    series, samples = {}, {}
    for name in ORDER:
        log(f"\n  {LABELS[name]}")
        runs, run_samples = {}, {}
        for seed, vector in directions[name].items():
            points = []
            for coefficient in args.coefficients:
                point, completions = score(
                    model, tokenizer, vector, coefficient, opener_ids, args.max_new_tokens
                )
                points.append(point)
                run_samples[f"{seed}:{coefficient}"] = completions
                log(
                    f"    seed {seed} c={coefficient:<5} rate {point['refusal_rate']:.2f}"
                    f"  opener mass {point['opener_mass']:.6f}"
                )
            runs[seed] = points
        entry = {"coefficients": [p["coefficient"] for p in next(iter(runs.values()))]}
        for key in ("refusal_rate", "opener_mass"):
            xs, mean, lo, hi = aggregate(runs, key)
            entry[key] = {"mean": mean, "lo": lo, "hi": hi}
        entry["per_seed"] = {str(seed): points for seed, points in runs.items()}
        series[name] = entry
        samples[name] = run_samples

    result = {
        "model": str(BASE_MODEL.relative_to(ROOT)),
        "layer": REFUSAL_LAYER,
        "scale": SCALE,
        "set_size": SET_SIZE,
        "n_sets": N_SETS,
        "seeds": SEEDS,
        "n_prompts": len(NEUTRAL_PROMPTS),
        "max_new_tokens": args.max_new_tokens,
        "openers": [tokenizer.decode([i]) for i in opener_ids],
        "readouts": readouts,
        "series": series,
    }
    METRICS_PATH.write_text(json.dumps(result, indent=2))
    SAMPLES_PATH.write_text(json.dumps(samples, indent=2))
    build_plots(result)

    log("\npeaks")
    for name in ORDER:
        entry = series[name]
        rate = max(entry["refusal_rate"]["mean"])
        mass = max(entry["opener_mass"]["mean"])
        log(f"  {LABELS[name]:<34} rate {rate:.2f}   opener mass {mass:.6f}")
    log(f"\nsaved {METRICS_PATH.relative_to(ROOT)}")
    log(f"saved {SAMPLES_PATH.relative_to(ROOT)}")
    log(f"saved {RATE_PLOT.relative_to(ROOT)}")
    log(f"saved {MASS_PLOT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
