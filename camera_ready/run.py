"""Generate every steered completion for one model and store it; analysis happens in analyze.py.

uv run python run.py --model Qwen/Qwen3-1.7B --tag q17 --variants --artifacts --sweep 9 12 13 18 21
"""
import argparse
import json
import math
import pathlib
import time

import torch

from jspace.behaviours import (BEHAVIOURS, FALSE_PREMISE, FOLD_A, FOLD_B, FP_FOLD_A, FP_FOLD_B, FP_TRAIN_IDX,
                               NEUTRAL, ORDER, TRAIN_IDX, readout_vocab)
from jspace.core import Ctx, random_direction, unit

ALPHAS = [0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.2]
SEEDS = [0, 1, 2]

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
ap.add_argument("--lenses", nargs="+", default=["jlens"])
ap.add_argument("--layer", type=int, default=None, help="default floor(0.55 L)")
ap.add_argument("--sweep", nargs="*", type=int, default=[])
ap.add_argument("--behaviours", nargs="*", default=ORDER)
ap.add_argument("--variants", action="store_true", help="estimator variants + fitted/ours mixtures")
ap.add_argument("--artifacts", action="store_true", help="published caps vector and abliteration direction")
ap.add_argument("--tag", required=True)
ap.add_argument("--chunk", type=int, default=512)
ap.add_argument("--suite", nargs="+", default=["main"], help="main and/or robust (ridge/notop estimators, 3 seeds)")
ap.add_argument("--lean", action="store_true", help="1 random seed, no no-top variant (to save compute)")
ap.add_argument("--gamma", type=float, default=0.01, help="ridge strength for the robust suite, fixed on Qwen3-1.7B")
args = ap.parse_args()

out = pathlib.Path("results") / args.tag
out.mkdir(parents=True, exist_ok=True)
ctx = Ctx(args.model, lenses=args.lenses)
L = args.layer if args.layer is not None else int(0.55 * ctx.n_layers)
sweep = [l for l in args.sweep if l != L]
layers = sorted({L, *sweep})
rho = ctx.residual_scale([NEUTRAL[i] for i in TRAIN_IDX], layers + ([13] if args.artifacts else []))
meta = dict(model=args.model, lenses=args.lenses, layer=L, sweep=sweep, rho=rho, alphas=ALPHAS, seeds=SEEDS,
            n_layers=ctx.n_layers, d=ctx.d, behaviours={})
print(f"{args.model}: L={L} sweep={sweep} rho={ {k: round(v, 1) for k, v in rho.items()} }", flush=True)


def abliteration_direction(layer):
    """Leading left singular vector of the mlp.down_proj weight difference abliterated - base (App. E)."""
    from transformers import AutoModelForCausalLM
    abl = AutoModelForCausalLM.from_pretrained("mlabonne/Qwen3-1.7B-abliterated", dtype=torch.float32)
    Wa = abl.model.layers[layer].mlp.down_proj.weight.detach()
    Wb = ctx.model.model.layers[layer].mlp.down_proj.weight.detach().float().cpu()
    dW = Wa - Wb
    U, S, _ = torch.linalg.svd(dW, full_matrices=False)
    Wa_o = abl.model.layers[layer].self_attn.o_proj.weight.detach()
    Wb_o = ctx.model.model.layers[layer].self_attn.o_proj.weight.detach().float().cpu()
    Uo, _, _ = torch.linalg.svd(Wa_o - Wb_o, full_matrices=False)
    info = dict(s2_over_s1=(S[1] / S[0]).item(), cos_o_proj=abs((U[:, 0] @ Uo[:, 0]).item()))
    del abl
    return U[:, 0].to(ctx.device), info


for b in args.behaviours:
    t0 = time.time()
    B = BEHAVIOURS[b]
    pairs = ctx.token_pairs(B["pairs"], B["n_pairs"])
    if B["pool"] == "neutral":
        train = [NEUTRAL[i] for i in TRAIN_IDX]
        evalq = [NEUTRAL[i] for i in FOLD_A + FOLD_B]
    else:
        train = [FALSE_PREMISE[i] for i in FP_TRAIN_IDX]
        evalq = [FALSE_PREMISE[i] for i in FP_FOLD_A + FP_FOLD_B]
    fit_layers = sorted(set(layers + ([13] if args.artifacts and b == "caps" else [])))
    fitted = ctx.fitted(train, B["pos"], B["neg"], fit_layers)

    dirs = {}   # name -> (layer, unit vector)
    main = "main" in args.suite
    if main:
        dirs["fitted"] = (L, unit(fitted[L]))
    for lens in args.lenses if main else []:
        sfx = "" if lens == "jlens" else "_" + lens
        for s in SEEDS:
            dirs[f"ours{sfx}_s{s}"] = (L, ctx.lens_direction(pairs, L, seed=s, mode="J", lens=lens))
        Q = ctx.pair_basis(pairs, L, lens=lens)
        dirs[f"oracle{sfx}"] = (L, unit(Q @ (Q.T @ unit(fitted[L]))))
    for s in SEEDS if main else []:
        dirs[f"eye_s{s}"] = (L, ctx.lens_direction(pairs, L, seed=s, mode="I"))
        if not (args.lean and s > 0):
            dirs[f"rand_s{s}"] = (L, random_direction(ctx.d, s, ctx.device))
    if "robust" in args.suite:
        for lens in args.lenses:
            sfx = "" if lens == "jlens" else "_" + lens
            for s in SEEDS:
                dirs[f"ridge{sfx}_s{s}"] = (L, ctx.lens_direction(pairs, L, seed=s, mode=f"ridge:{args.gamma}", lens=lens))
                if not args.lean:
                    dirs[f"notop{sfx}_s{s}"] = (L, ctx.lens_direction(pairs, L, seed=s, mode="notop", lens=lens))
            if main:
                dirs[f"mixridge{sfx}_0.5"] = (L, unit(0.5 * dirs[f"ridge{sfx}_s0"][1] + 0.5 * unit(fitted[L])))
        Qr = ctx.pair_basis(pairs, L, mode=f"ridge:{args.gamma}")
        dirs["oracle_ridge"] = (L, unit(Qr @ (Qr.T @ unit(fitted[L]))))
    if args.variants and main:
        for m in ["ridge:0.1", "ridge:0.01", "ridge:0.001", "notop"]:
            dirs[f"var_{m}"] = (L, ctx.lens_direction(pairs, L, seed=0, mode=m))
        for lam in [0.25, 0.5, 0.75]:
            dirs[f"mix_{lam}"] = (L, unit(lam * dirs["ours_s0"][1] + (1 - lam) * dirs["fitted"][1]))
        Qi = ctx.pair_basis(pairs, L, mode="I")
        dirs["oracle_eye"] = (L, unit(Qi @ (Qi.T @ unit(fitted[L]))))
    for l in sweep if main else []:
        dirs[f"fitted@{l}"] = (l, unit(fitted[l]))
        dirs[f"ours_s0@{l}"] = (l, ctx.lens_direction(pairs, l, seed=0, mode="J"))
        dirs[f"eye_s0@{l}"] = (l, ctx.lens_direction(pairs, l, seed=0, mode="I"))
    extra = {}
    if args.artifacts and main and b == "caps":
        from huggingface_hub import hf_hub_download
        pub = torch.load(hf_hub_download("science-of-finetuning/steering-vecs-qwen3_1_7B", "caps_L13.pt")).float()
        dirs["artifact@13"] = (13, unit(pub.to(ctx.device)))
        dirs["fitted@13"] = (13, unit(fitted[13]))
        dirs["ours_s0@13"] = (13, ctx.lens_direction(pairs, 13, seed=0, mode="J"))
    if args.artifacts and main and b == "refusal":
        r, info = abliteration_direction(L)
        if (r @ dirs["fitted"][1]) < 0:
            r = -r
        info["cos_fitted"] = (r @ dirs["fitted"][1]).item()
        extra["abliteration"] = info
        dirs["artifact"] = (L, unit(r))
        Q = ctx.pair_basis(pairs, L)
        dirs["oracle_artifact"] = (L, unit(Q @ (Q.T @ unit(r))))

    # diagnostics: cosines, oracle alignment, lens readout
    fv = unit(fitted[L])
    diag = dict(pairs=[(ctx.tok.decode([p]), ctx.tok.decode([n])) for p, n in pairs], n_pairs=len(pairs),
                **{"cos_to_fitted" + ("" if main else "_robust"): {k: (v @ fv).item() for k, (l, v) in dirs.items() if l == L}},
                fitted_norm=fitted[L].norm().item(), **extra)
    for lens in args.lenses:
        sfx = "" if lens == "jlens" else "_" + lens
        ro_f = ctx.readout(fv, L, lens=lens)
        for meth in ["ours", "ridge", "notop"]:
            if f"{meth}{sfx}_s0" not in dirs:
                continue
            ro_o = ctx.readout(dirs[f"{meth}{sfx}_s0"][1], L, lens=lens)
            diag[f"readout_{meth}{sfx}"] = dict(top=ro_o[:10], share=sum(readout_vocab(b, t) for t in ro_o) / len(ro_o))
        diag[f"readout_fitted{sfx}"] = dict(top=ro_f[:10], share=sum(readout_vocab(b, t) for t in ro_f) / len(ro_f))
    suf = "" if args.suite == ["main"] else "_" + "_".join(args.suite)
    meta["behaviours"].setdefault(b, {}).update(diag)
    torch.save({k: (l, v.cpu()) for k, (l, v) in dirs.items()}, out / f"dirs_{b}{suf}.pt")

    # generation: alpha = 0 once, then every direction x alpha x prompt
    chat = ctx.chat(evalq)
    recs = []
    if main:
        base = ctx.generate(chat, chunk=args.chunk)
        recs += [dict(dir="none", alpha=0.0, layer=L, p=i, text=t) for i, t in enumerate(base)]
    by_layer = {}
    for name, (l, v) in dirs.items():
        by_layer.setdefault(l, []).append((name, v))
    for l, items in by_layer.items():
        texts, deltas, keys = [], [], []
        for name, v in items:
            for a in ALPHAS:
                d = (a * rho[l] * v).float()
                for i, q in enumerate(chat):
                    texts.append(q); deltas.append(d); keys.append((name, a, i))
        gen = ctx.generate(texts, layer=l, deltas=torch.stack(deltas), chunk=args.chunk)
        recs += [dict(dir=n, alpha=a, layer=l, p=i, text=t) for (n, a, i), t in zip(keys, gen)]
    with open(out / f"gen_{b}{suf}.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    old = json.load(open(out / "meta.json")) if (out / "meta.json").exists() else None
    if old and not main:
        for k, v in meta["behaviours"].items():
            old["behaviours"].setdefault(k, {}).update(v)
        meta_out = old
    else:
        meta_out = meta
    json.dump(meta_out, open(out / ("meta.json"), "w"), indent=1, ensure_ascii=False)
    print(f"[{b}] {len(dirs)} dirs, {len(recs)} completions, {time.time() - t0:.0f}s", flush=True)
