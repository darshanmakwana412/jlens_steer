import sys

import torch

from activations import refusal_contrast
from assets import ABLITERATED_MODEL, BASE_MODEL, REFUSAL_VECTORS, ROOT
from refusal import align_signs, analyse_edit, unit_rows
from weights import num_layers

RELIABLE_COS = 0.5

SOURCE = "diff of mlabonne/Qwen3-1.7B-abliterated against Qwen/Qwen3-1.7B"
METHOD = (
    "r_l = top left singular vector of the layer-l mlp.down_proj diff, verified "
    "against the top right singular vector of the o_proj diff, which that edit "
    "uses as its input axis"
)
SIGN_CONVENTION = "positive = toward refusal"


def log(message: str) -> None:
    print(message, flush=True)


def report(analysis, act_cos: torch.Tensor, reliable: torch.Tensor) -> None:
    peak = analysis.peak_layer
    rank1 = analysis.column("rank1_residual")
    axis_cos = analysis.column("o_proj_axis_cos")
    directions = analysis.directions

    log(f"embed_tokens edited: {analysis.embed_tokens_edited}")
    log(f"edit peaks at layer {peak} (sigma1 = {analysis.column('sigma1_down_proj')[peak]:.4f})")
    log(f"worst rank-1 residual sigma2/sigma1: {rank1.max():.1e} at layer {int(rank1.argmax())}")
    log(f"o_proj input-axis agreement: min cos {axis_cos.min():.6f}")
    log(f"cos(r_{peak}, r_0) = {abs(torch.dot(directions[peak], directions[0])).item():.4f}")
    log(f"cos(r_l, activation contrast): max {act_cos.max():.3f} at layer {int(act_cos.argmax())}")
    log(f"reliable layers: {[layer for layer, ok in enumerate(reliable) if ok]}")


def build_artifact(analysis, directions, projection, act_cos, reliable, baseline) -> dict:
    return {
        "directions": directions.to(torch.float32),
        "layers": [edit.layer for edit in analysis.edits],
        "peak_layer": analysis.peak_layer,
        "sigma1_down_proj": analysis.column("sigma1_down_proj"),
        "sigma1_o_proj": analysis.column("sigma1_o_proj"),
        "rank1_residual": analysis.column("rank1_residual"),
        "oproj_input_axis_cos": analysis.column("o_proj_axis_cos"),
        "sign_projection": projection.to(torch.float32),
        "act_diff_cos": act_cos.to(torch.float32),
        "reliable": reliable,
        "act_diff_directions": baseline.to(torch.float32),
        "embed_tokens_edited": analysis.embed_tokens_edited,
        "source": SOURCE,
        "method": METHOD,
        "sign_convention": SIGN_CONVENTION,
    }


def main() -> int:
    layers = num_layers(BASE_MODEL)
    analysis = analyse_edit(BASE_MODEL, ABLITERATED_MODEL, layers)

    contrast = refusal_contrast(BASE_MODEL)
    directions, projection = align_signs(analysis.directions, contrast)
    baseline = unit_rows(contrast)
    act_cos = torch.stack(
        [torch.dot(directions[layer], baseline[layer + 1]) for layer in range(layers)]
    )
    reliable = act_cos > RELIABLE_COS

    report(analysis, act_cos, reliable)

    REFUSAL_VECTORS.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        build_artifact(analysis, directions, projection, act_cos, reliable, baseline),
        REFUSAL_VECTORS,
    )
    log(f"saved {REFUSAL_VECTORS.relative_to(ROOT)} ({layers} x {directions.shape[1]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
