from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
ARTIFACTS = ROOT / "artifacts"

BASE_MODEL = MODELS / "Qwen3-1.7B"
ABLITERATED_MODEL = MODELS / "Qwen3-1.7B-abliterated"
JACOBIAN_LENS = ARTIFACTS / "jacobian-lens"
STEERING_VECS = ARTIFACTS / "steering-vecs-qwen3_1_7B"
REFUSAL_VECTORS = ARTIFACTS / "refusal_qwen3_1_7B.pt"

IGNORED_NAMES = {".gitattributes", ".DS_Store"}


@dataclass(frozen=True)
class Asset:
    repo_id: str
    local_dir: Path
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()

    def wants(self, name: str) -> bool:
        if Path(name).name in IGNORED_NAMES or name in self.exclude:
            return False
        return not self.include or any(name == p or name.startswith(p) for p in self.include)


ASSETS = {
    "base": Asset("Qwen/Qwen3-1.7B", BASE_MODEL),
    "abliterated": Asset(
        "mlabonne/Qwen3-1.7B-abliterated",
        ABLITERATED_MODEL,
        exclude=("model.safetensors",),
    ),
    "jlens": Asset(
        "neuronpedia/jacobian-lens",
        JACOBIAN_LENS,
        include=("README.md", "qwen3-1.7b/"),
    ),
    "steering_vecs": Asset(
        "science-of-finetuning/steering-vecs-qwen3_1_7B",
        STEERING_VECS,
    ),
}
