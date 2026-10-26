"""Spectral diagnostics behind the ridge pull-back: singular values of J_l and how much of each derived
direction lies on J_l's top input singular vector.  uv run python diag.py q17 [q4 q8]"""
import json
import pathlib
import sys

import torch
from huggingface_hub import hf_hub_download

from jspace.behaviours import ORDER
from jspace.core import LENS_FILES, LENS_REPO

out = {}
for tag in sys.argv[1:]:
    meta = json.load(open(pathlib.Path("results") / tag / "meta.json"))
    L = meta["layer"]
    R = {}
    for lens in meta["lenses"]:
        J = torch.load(hf_hub_download(LENS_REPO, LENS_FILES[(meta["model"], lens)]), map_location="cpu",
                       weights_only=True)["J"][L].float()
        U, S, Vh = torch.linalg.svd(J)
        v1 = Vh[0]
        sfx = "" if lens == "jlens" else "_" + lens
        rows = {}
        for b in ORDER:
            dirs = {}
            for fp in sorted((pathlib.Path("results") / tag).glob(f"dirs_{b}*.pt")):
                dirs.update(torch.load(fp, weights_only=False))
            rows[b] = {k: abs(float(dirs[k][1].float() @ v1)) for k in
                       [f"ours{sfx}_s0", f"ridge{sfx}_s0", f"notop{sfx}_s0", "fitted", "eye_s0", f"oracle{sfx}"] if k in dirs}
            if f"ours{sfx}_s0" in dirs and f"ridge{sfx}_s0" in dirs:
                rows[b]["cos_ours_ridge"] = float(dirs[f"ours{sfx}_s0"][1] @ dirs[f"ridge{sfx}_s0"][1])
        R[lens] = dict(svals=[float(x) for x in S[:6]], s1_over_s2=float(S[0] / S[1]), top_alignment=rows)
    out[tag] = R
    print(tag, {k: (round(v["s1_over_s2"], 2), [round(x, 1) for x in v["svals"][:4]]) for k, v in R.items()})
    for lens, v in R.items():
        for b, r in v["top_alignment"].items():
            print("  ", lens, b, {k: round(x, 3) for k, x in r.items()})
    json.dump(R, open(pathlib.Path("results") / tag / "diag.json", "w"), indent=1)
