from pathlib import Path

import torch

RMS_EPS = 1e-6


def load_jacobian(path: Path, layer: int) -> torch.Tensor:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    return artifact["J"][layer].float()


def unit(vector: torch.Tensor) -> torch.Tensor:
    return vector / vector.norm()


def rms_norm(vector: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
    return vector / torch.sqrt(vector.pow(2).mean() + RMS_EPS) * gain


def lens_logits(
    jacobian: torch.Tensor,
    unembed: torch.Tensor,
    gain: torch.Tensor,
    activation: torch.Tensor,
) -> torch.Tensor:
    return unembed @ rms_norm(jacobian @ activation, gain)


def pull_back(jacobian: torch.Tensor, gain: torch.Tensor, unembed_row: torch.Tensor):
    return unit(jacobian.T @ (gain * unembed_row))


def token_mean(unembed: torch.Tensor, token_ids: list[int]) -> torch.Tensor:
    return unembed[token_ids].mean(0)


def concept_direction(
    jacobian: torch.Tensor,
    unembed: torch.Tensor,
    gain: torch.Tensor,
    token_id_sets: list[list[int]],
    center: torch.Tensor | None = None,
    contrast_id_sets: list[list[int]] | None = None,
) -> torch.Tensor:
    rows = []
    for index, token_ids in enumerate(token_id_sets):
        row = token_mean(unembed, token_ids)
        if contrast_id_sets is not None:
            row = row - token_mean(unembed, contrast_id_sets[index])
        elif center is not None:
            row = row - center
        rows.append(pull_back(jacobian, gain, row))
    return unit(torch.stack(rows).mean(0))


def top_lens_tokens(
    jacobian: torch.Tensor,
    unembed: torch.Tensor,
    gain: torch.Tensor,
    activation: torch.Tensor,
    tokenizer,
    k: int = 10,
) -> list[str]:
    logits = lens_logits(jacobian, unembed, gain, activation)
    return [tokenizer.decode([token_id]) for token_id in logits.topk(k).indices.tolist()]
