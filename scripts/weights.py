import json
from pathlib import Path

import torch
from safetensors import safe_open

RESIDUAL_OUTPUT_SUFFIXES = ("self_attn.o_proj.weight", "mlp.down_proj.weight")
RESIDUAL_ROW_NAMES = ("model.embed_tokens.weight",)

EMBED_TOKENS = "model.embed_tokens.weight"


def num_layers(model_dir: Path) -> int:
    return json.loads((model_dir / "config.json").read_text())["num_hidden_layers"]


def down_proj_name(layer: int) -> str:
    return f"model.layers.{layer}.mlp.down_proj.weight"


def o_proj_name(layer: int) -> str:
    return f"model.layers.{layer}.self_attn.o_proj.weight"


class WeightReader:
    def __init__(self, model_dir: Path):
        index = json.loads((model_dir / "model.safetensors.index.json").read_text())
        self._shards = {name: model_dir / shard for name, shard in index["weight_map"].items()}
        self._handles: dict[Path, object] = {}

    def __contains__(self, name: str) -> bool:
        return name in self._shards

    def names(self):
        return self._shards.keys()

    def get(self, name: str) -> torch.Tensor:
        path = self._shards[name]
        if path not in self._handles:
            self._handles[path] = safe_open(str(path), framework="pt")
        return self._handles[path].get_tensor(name)


def residual_diffs(base_dir: Path, edited_dir: Path):
    base, edited = WeightReader(base_dir), WeightReader(edited_dir)
    selected = [
        name
        for name in sorted(edited.names())
        if name in base and (name.endswith(RESIDUAL_OUTPUT_SUFFIXES) or name in RESIDUAL_ROW_NAMES)
    ]
    for name in selected:
        diff = base.get(name).to(torch.float64) - edited.get(name).to(torch.float64)
        yield name, diff.T if name in RESIDUAL_ROW_NAMES else diff
