"""Lens readout of fitted, Eq. 2 and ridge directions (Appendix table). Decodes lens(d) = W_U(g * RMSNorm(J d)),
raw and with J's top output singular direction removed (it otherwise floods the top of every readout, since a
pull-back direction is read through J twice).  python readout.py q17"""
import json
import pathlib
import sys

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer

from jspace.behaviours import ORDER, readout_vocab
from jspace.core import LENS_FILES, LENS_REPO

tag = sys.argv[1]
meta = json.load(open(pathlib.Path("results") / tag / "meta.json"))
L, model = meta["layer"], meta["model"]
tok = AutoTokenizer.from_pretrained(model)
cfg = AutoConfig.from_pretrained(model)
# W_U and g straight from the safetensors (tied embeddings on the small Qwen3 models)
from huggingface_hub import snapshot_download
snap = pathlib.Path(snapshot_download(model, allow_patterns=["*.safetensors", "*.json"]))
W = {}
for f in snap.glob("*.safetensors"):
    W.update({k: v for k, v in load_file(f).items() if k in ("lm_head.weight", "model.embed_tokens.weight", "model.norm.weight")})
W_U = (W.get("lm_head.weight", W["model.embed_tokens.weight"])).float()
g = W["model.norm.weight"].float()
J = torch.load(hf_hub_download(LENS_REPO, LENS_FILES[(model, "jlens")]), map_location="cpu", weights_only=True)["J"][L].float()
U, S, Vh = torch.linalg.svd(J)
u1 = U[:, 0]


def read(d, clean, k=50):
    z = J @ d.float()
    if clean:
        z = z - (z @ u1) * u1
    z = z * torch.rsqrt(z.pow(2).mean() + 1e-6)
    return [tok.decode([i]) for i in ((g * z) @ W_U.T).topk(k).indices.tolist()]


out = {}
for b in ORDER:
    dirs = {}
    for fp in sorted((pathlib.Path("results") / tag).glob(f"dirs_{b}*.pt")):
        dirs.update(torch.load(fp, weights_only=False))
    row = {}
    for name in ["fitted", "ours_s0", "ridge_s0", "eye_s0"]:
        if name not in dirs:
            continue
        for clean in (False, True):
            toks = read(dirs[name][1], clean)
            row[f"{name}{'_clean' if clean else ''}"] = dict(top=toks[:8], share=sum(readout_vocab(b, t) for t in toks) / len(toks))
    out[b] = row
    print(b, {k: (round(v["share"], 2), [t.strip() for t in v["top"][:5]]) for k, v in row.items()})
json.dump(out, open(pathlib.Path("results") / tag / "readout.json", "w"), indent=1, ensure_ascii=False)
