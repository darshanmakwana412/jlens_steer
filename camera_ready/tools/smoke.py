"""Smoke test for a new model: layout, final-norm gain, lens shape and lens fidelity.

Fidelity: on a few wikitext-like sentences, compare the lens readout of the block-l output with the model's
own next-token prediction (top-1 agreement), for J applied to the block output (our convention) and for an
identity lens. A lens that does not belong to these weights or this convention shows no gain over identity.
uv run python tools/smoke.py Qwen/Qwen3.5-0.8B
"""
import sys

import torch

from jspace.core import Ctx

name = sys.argv[1]
ctx = Ctx(name)
L = int(0.55 * ctx.n_layers)
print(name, "layers", ctx.n_layers, "d", ctx.d, "L", L, "lens layers", sorted(ctx.J["jlens"])[:3], "...",
      len(ctx.J["jlens"]), "J shape", tuple(ctx.J["jlens"][L].shape))
print("g: mean %.3f min %.3f max %.3f" % (ctx.g.mean(), ctx.g.min(), ctx.g.max()))
txt = ["The history of the Roman Empire spans many centuries, beginning with the founding of the city and ending "
       "with the fall of Constantinople in 1453.", "Photosynthesis is the process by which green plants use sunlight "
       "to synthesise foods from carbon dioxide and water.", "The river flows through the valley and into the sea, "
       "where fishermen have worked for generations."]
b = ctx.tok(txt, return_tensors="pt", padding=True).to(ctx.device)
layers = sorted(ctx.J["jlens"])
st = ctx.capture(b.input_ids, b.attention_mask, [l for l in (L, layers[-1], ctx.n_layers - 1) if l < ctx.n_layers])
with torch.no_grad():
    out = ctx.model(**b)
final = out.logits.float().argmax(-1)
m = b.attention_mask.bool()
# the norm+unembed check: logits from the last block output through g and W_U must match the model's
hl = st[ctx.n_layers - 1]
z = hl * torch.rsqrt(hl.pow(2).mean(-1, keepdim=True) + 1e-6)
mine = ((ctx.g * z) @ ctx.W_U.T).argmax(-1)
print("norm/unembed reproduces model top-1: %.3f" % (mine[m] == final[m]).float().mean())
for l in sorted({L, layers[-1]}):
    h = st[l]
    for tag, f in (("J", lambda x: x @ ctx.J["jlens"][l].T), ("I", lambda x: x)):
        y = f(h)
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-6)
        pred = ((ctx.g * y) @ ctx.W_U.T).argmax(-1)
        print(f"layer {l} lens={tag}: top-1 agreement with model {(pred[m] == final[m]).float().mean():.3f}")
hf = st[ctx.n_layers - 1][m]
cos = lambda x: torch.nn.functional.cosine_similarity(x - x.mean(0), hf - hf.mean(0), dim=-1).mean().item()
for l in sorted({L, layers[-1]}):
    h = st[l][m]; J = ctx.J["jlens"][l]
    print(f"layer {l} centred cos to final residual: J h {cos(h @ J.T):.3f}  h {cos(h):.3f}  J^T h {cos(h @ J):.3f}")
q = ctx.chat(["What is the capital of France?"])[0]
print(repr(q[-120:]))
print(repr(ctx.generate([q], max_new_tokens=24)[0]))
