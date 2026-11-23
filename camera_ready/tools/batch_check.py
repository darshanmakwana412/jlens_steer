"""Batched left-padded greedy generation must match unbatched generation (hybrid-attention models)."""
import sys

import torch

from jspace.behaviours import NEUTRAL
from jspace.core import Ctx

ctx = Ctx(sys.argv[1])
qs = ctx.chat([NEUTRAL[i] for i in range(8)])
batched = ctx.generate(qs, max_new_tokens=32)
single = [ctx.generate([q], max_new_tokens=32)[0] for q in qs]
same = sum(a == b for a, b in zip(batched, single))
print(f"{sys.argv[1]}: {same}/8 identical")
for a, b in zip(batched, single):
    if a != b:
        print("  B:", repr(a[:90])); print("  S:", repr(b[:90]))
d = torch.randn(len(qs), ctx.d, device=ctx.device) * 0
print("steer-with-zero equals plain:", ctx.generate(qs, max_new_tokens=32, layer=5, deltas=d) == batched)
