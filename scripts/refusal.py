from dataclasses import dataclass
from pathlib import Path

import torch

from weights import EMBED_TOKENS, down_proj_name, o_proj_name, residual_diffs


@dataclass
class LayerEdit:
    layer: int
    direction: torch.Tensor
    sigma1_down_proj: float
    sigma1_o_proj: float
    rank1_residual: float
    o_proj_axis_cos: float


@dataclass
class EditAnalysis:
    edits: list[LayerEdit]
    embed_tokens_edited: bool

    @property
    def directions(self) -> torch.Tensor:
        return torch.stack([edit.direction for edit in self.edits])

    @property
    def peak_layer(self) -> int:
        return max(self.edits, key=lambda edit: edit.sigma1_down_proj).layer

    def column(self, field: str) -> torch.Tensor:
        return torch.tensor([getattr(edit, field) for edit in self.edits])


def _analyse_layer(layer: int, down: torch.Tensor, attn: torch.Tensor) -> LayerEdit:
    left, singular, _ = torch.linalg.svd(down, full_matrices=False)
    _, attn_singular, attn_right = torch.linalg.svd(attn, full_matrices=False)
    direction = left[:, 0]
    return LayerEdit(
        layer=layer,
        direction=direction,
        sigma1_down_proj=singular[0].item(),
        sigma1_o_proj=attn_singular[0].item(),
        rank1_residual=(singular[1] / singular[0]).item(),
        o_proj_axis_cos=abs(torch.dot(attn_right[0], direction).item()),
    )


def analyse_edit(base_dir: Path, edited_dir: Path, layers: int) -> EditAnalysis:
    diffs = dict(residual_diffs(base_dir, edited_dir))
    embed = diffs.get(EMBED_TOKENS)
    return EditAnalysis(
        edits=[
            _analyse_layer(layer, diffs[down_proj_name(layer)], diffs[o_proj_name(layer)])
            for layer in range(layers)
        ],
        embed_tokens_edited=embed is not None and bool(embed.abs().max() > 0),
    )


def align_signs(directions: torch.Tensor, contrast: torch.Tensor):
    projection = torch.stack(
        [torch.dot(directions[layer], contrast[layer + 1]) for layer in range(len(directions))]
    )
    signs = torch.where(projection < 0, -1.0, 1.0).unsqueeze(1)
    return directions * signs, projection


def unit_rows(matrix: torch.Tensor) -> torch.Tensor:
    return matrix / matrix.norm(dim=1, keepdim=True).clamp(min=1e-12)
