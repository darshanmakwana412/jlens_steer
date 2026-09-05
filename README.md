# jlens_steer

Inverting the Jacobian lens to derive activation steering vectors from concept
tokens alone, on Qwen3-1.7B. Everything runs locally on an M-series Mac.

## Getting the assets

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv "huggingface_hub" torch transformers safetensors accelerate numpy
export HF_TOKEN=...                       # optional, avoids rate limits
.venv/bin/python scripts/download_assets.py
```

That pulls ~11 GB into `models/` and `artifacts/`. It is resumable and
sha256-verified, so interrupting it is safe — re-run and it continues. To
re-check what is on disk without downloading:

```bash
.venv/bin/python scripts/download_assets.py --verify
```

The weights and the J-lens tensor are gitignored; `download_assets.py` is the
record of what they are. Small derived artifacts are tracked.

## Layout

| path | what | size |
| --- | --- | --- |
| `models/Qwen3-1.7B/` | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B), bf16. 28 layers, d_model 2048, tied embeddings | 4.1 GB |
| `models/Qwen3-1.7B-abliterated/` | [mlabonne/Qwen3-1.7B-abliterated](https://huggingface.co/mlabonne/Qwen3-1.7B-abliterated), fp32 | 6.9 GB |
| `artifacts/jacobian-lens/qwen3-1.7b/` | [neuronpedia/jacobian-lens](https://huggingface.co/neuronpedia/jacobian-lens), Qwen3-1.7B lens fit on Salesforce-wikitext, plus its `config.yaml` and convergence CSV | 227 MB |
| `artifacts/steering-vecs-qwen3_1_7B/caps_L13.pt` | [science-of-finetuning](https://huggingface.co/science-of-finetuning/steering-vecs-qwen3_1_7B) ALL-CAPS steering vector. Bare fp64 tensor `[2048]`, norm 27.1, for layer 13 | 20 KB |
| `artifacts/refusal_qwen3_1_7B.pt` | per-layer refusal directions `[28, 2048]`, derived here — see below | 480 KB |

The abliterated repo ships two redundant weight sets: the fp32 shards its
`model.safetensors.index.json` points at, and a leftover single-file bf16
`model.safetensors`. Only the indexed shards are downloaded; the leftover would
shadow the index at load time.

Of the 58 GB in the J-lens repo, only the `qwen3-1.7b/` subtree is fetched.

## The refusal directions

Abliteration is usually described as one rank-1 edit, `W' = W - r(rᵀW)`, so the
top left singular vector of `W - W'` should be a single global refusal
direction. That is not what `mlabonne/Qwen3-1.7B-abliterated` contains. Diffing
it against the base model in fp64 shows:

- **A separate direction `r_l` per layer, not one global `r`.** Adjacent layers
  agree to cos 0.6–0.94, but `cos(r₁₅, r₀) = 0.02`. Collapsing all layers into
  one vector — top eigenvector of `Σ DᵢDᵢᵀ` — gives σ₂/σ₁ = 0.59, i.e. mostly
  noise, not a direction worth steering with.
- **`mlp.down_proj` is edited on its output axis**, `W -= (r_l r_lᵀ)W`, so the
  diff's top left singular vector is `r_l`.
- **`self_attn.o_proj` is edited on its *input* axis**, `W -= W(r_l r_lᵀ)`. Its
  top *right* singular vector matches down_proj's top left one to ~1e-6 at every
  layer. `o_proj` is square here (16 heads × 128 = 2048 = d_model), so a
  residual-space direction fits that axis by coincidence; the projection comes
  out of the concatenated head outputs rather than out of what the layer writes
  into the residual stream.
- **`embed_tokens` is untouched.**
- **Edit magnitude is a bell curve over layers, peaking at 15** — the "weight
  factors follow a normal distribution with a certain spread and peak layer"
  from the model card.

Each individual matrix *is* clean rank 1 (σ₂/σ₁ ≈ 2e-3 near the peak). The
tails degrade only because the signal shrinks toward the bf16 rounding floor,
which is also why early layers are not usable.

Independent validation: `cos(r_l, mean harmful act − mean harmless act)` reaches
**0.87** at layer 16 and stays above 0.75 through layer 27, but sits near 0.17
below layer 9. The artifact carries a `reliable` mask (cos > 0.5) marking
layers 12–27.

```bash
.venv/bin/python scripts/extract_refusal_vector.py
```

`artifacts/refusal_qwen3_1_7B.pt` holds `directions` `[28, 2048]` (unit norm,
positive = toward refusal), the `reliable` mask, the σ₁ profile, the o_proj
axis-agreement check, and `act_diff_directions` as an activation-derived
baseline.

For reference, `cos(r₁₃, caps_L13) = 0.045` — the refusal and all-caps
directions are essentially orthogonal.

## A note on this network

Long TLS streams to the HF CDN die partway through here with `[SSL] record
layer failure`, and `snapshot_download` discards the partial file, so retries
never advance. Ranged requests of a few tens of MB do succeed. Hence the
chunked downloader: 16 MiB ranges written at fixed offsets into a preallocated
`.part`, a sidecar recording which chunks landed, per-chunk retry, sha256 before
publishing.
