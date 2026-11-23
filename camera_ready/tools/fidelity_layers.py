"""Per-layer lens fidelity, first position excluded: centred cos(J h_l, h_final) vs cos(h_l, h_final)."""
import sys

import torch

from jspace.core import Ctx

ctx = Ctx(sys.argv[1])
txt = ["The history of the Roman Empire spans many centuries, beginning with the founding of the city and ending "
       "with the fall of Constantinople in 1453.", "Photosynthesis is the process by which green plants use sunlight "
       "to synthesise foods from carbon dioxide and water.", "The river flows through the valley and into the sea, "
       "where fishermen have worked for generations.", "In 1969 two astronauts walked on the surface of the Moon while "
       "a third orbited above them in the command module."]
b = ctx.tok(txt, return_tensors="pt", padding=True).to(ctx.device)
layers = sorted(ctx.J["jlens"])
st = ctx.capture(b.input_ids, b.attention_mask, layers + [ctx.n_layers - 1])
m = b.attention_mask.bool().clone()
first = m.float().argmax(1)
m[torch.arange(len(txt)), first] = False
hf = st[ctx.n_layers - 1][m]
cos = lambda x: torch.nn.functional.cosine_similarity(x - x.mean(0), hf - hf.mean(0), dim=-1).mean().item()
L = int(0.55 * ctx.n_layers)
for l in layers[::2] + ([L] if L not in layers[::2] else []):
    h = st[l][m]
    print(f"{sys.argv[1]} layer {l:2d}{'*' if l == L else ' '} J h {cos(h @ ctx.J['jlens'][l].T):+.3f}  h {cos(h):+.3f}")
