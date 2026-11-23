"""Extend the coefficient grid downwards for directions already saved by run.py (camera-ready Appendix F).

On some models (e.g. Gemma-3-270M, residual norm ~5800) the lens-derived direction is degenerate already at the
smallest coefficient of the paper's grid (0.025), so the grid never reaches its working range. This adds
ALPHAS_LOW for the directions in results/<tag>/dirs_<b>.pt and writes gen_<b>_low.jsonl, which analyze.py merges
with gen_<b>.jsonl (cells are keyed by direction and coefficient). Coefficients are still selected on the other
fold, so a larger grid cannot inflate the reported rates.

uv run python run_low.py --model google/gemma-3-270m-it --tag g270
"""
import argparse
import json
import pathlib
import time

import torch

from jspace.behaviours import BEHAVIOURS, FALSE_PREMISE, FOLD_A, FOLD_B, FP_FOLD_A, FP_FOLD_B, NEUTRAL, ORDER
from jspace.core import Ctx

ALPHAS_LOW = [0.002, 0.004, 0.008, 0.0125]
KEEP = ("fitted", "ours_s", "eye_s", "rand_s", "oracle", "mix_", "artifact")  # the methods Appendix F reports

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--chunk", type=int, default=512)
args = ap.parse_args()

out = pathlib.Path("results") / args.tag
meta = json.load(open(out / "meta.json"))
rho = {int(k): v for k, v in meta["rho"].items()}
ctx = Ctx(args.model, lenses=meta["lenses"])
for b in ORDER:
    t0 = time.time()
    src = out / f"dirs_{b}.pt"
    if not src.exists():
        src = out / f"dirs_{b}_main_robust.pt"
    if not src.exists():
        print(f"[{b}] no directions (behaviour skipped)", flush=True)
        continue
    dirs = torch.load(src)
    dirs = {k: (l, v.to(ctx.device)) for k, (l, v) in dirs.items()
            if k.startswith(KEEP) and not k.startswith(("oracle_ridge", "oracle_eye", "oracle_jpp"))}
    if BEHAVIOURS[b]["pool"] == "neutral":
        evalq = [NEUTRAL[i] for i in FOLD_A + FOLD_B]
    else:
        evalq = [FALSE_PREMISE[i] for i in FP_FOLD_A + FP_FOLD_B]
    chat = ctx.chat(evalq)
    recs, by_layer = [], {}
    for name, (l, v) in dirs.items():
        by_layer.setdefault(l, []).append((name, v))
    for l, items in by_layer.items():
        texts, deltas, keys = [], [], []
        for name, v in items:
            for a in ALPHAS_LOW:
                d = (a * rho[l] * v).float()
                for i, q in enumerate(chat):
                    texts.append(q); deltas.append(d); keys.append((name, a, i))
        gen = ctx.generate(texts, layer=l, deltas=torch.stack(deltas), chunk=args.chunk)
        recs += [dict(dir=n, alpha=a, layer=l, p=i, text=t) for (n, a, i), t in zip(keys, gen)]
    with open(out / f"gen_{b}_low.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[{b}] low grid: {len(dirs)} dirs, {len(recs)} completions, {time.time() - t0:.0f}s", flush=True)
