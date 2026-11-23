"""Lens inversion, baselines and batched steered generation (rebuilt from the submitted paper, Sec. 3 and App. E)."""
import contextlib
import math
import random

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILES = {
    ("Qwen/Qwen3-1.7B", "jlens"): "qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt",
    ("Qwen/Qwen3-4B", "jlens"): "qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt",
    ("Qwen/Qwen3-4B", "jpp"): "qwen3-4b/jpp/Salesforce-wikitext/Qwen3-4B_jpp_lens.pt",
    ("Qwen/Qwen3-8B", "jlens"): "qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt",
    ("google/gemma-3-270m-it", "jlens"): "gemma-3-270m-it/jlens/Salesforce-wikitext/gemma-3-270m-it_jacobian_lens.pt",
    ("google/gemma-3-1b-it", "jlens"): "gemma-3-1b-it/jlens/Salesforce-wikitext/gemma-3-1b-it_jacobian_lens.pt",
    ("Qwen/Qwen2.5-1.5B-Instruct", "jlens"): "qwen2.5-1.5b-it/jlens/Salesforce-wikitext/Qwen2.5-1.5B-Instruct_jacobian_lens.pt",
    ("Qwen/Qwen3.5-0.8B", "jlens"): "qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt",
    ("Qwen/Qwen3.5-2B", "jlens"): "qwen3.5-2b/jlens/Salesforce-wikitext/Qwen3.5-2B_jacobian_lens.pt",
}
# Gated repos are loaded from public mirrors; the lens is keyed by the canonical name.
MIRROR = {"google/gemma-3-270m-it": "unsloth/gemma-3-270m-it", "google/gemma-3-1b-it": "unsloth/gemma-3-1b-it"}


def _decoder(model):
    """The module holding .layers and .norm (text decoder), for plain and multimodal wrappers."""
    m = model.model
    return m.language_model if hasattr(m, "language_model") else m


def unit(v):
    return v / v.norm()


class Ctx:
    """Model, tokenizer and the lens pieces the method needs: J_l, W_U and the final-norm gain g."""

    def __init__(self, model_name, lenses=("jlens",), device=None, dtype=torch.bfloat16):
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps")
        self.name = model_name
        src = MIRROR.get(model_name, model_name)
        self.tok = AutoTokenizer.from_pretrained(src, padding_side="left")
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(src, dtype=dtype).to(self.device).eval()
        self.dec = _decoder(self.model)
        self.layers = self.dec.layers
        self.n_layers = len(self.layers)
        self.d = self.model.get_output_embeddings().weight.shape[1]
        self.W_U = self.model.get_output_embeddings().weight.detach().float()
        # Effective final-norm gain: the norm of an all-ones vector (rms 1) is exactly g, whatever the
        # parametrisation (Qwen: w, Gemma / Qwen3.5: 1 + w).
        with torch.no_grad():
            self.g = self.dec.norm(torch.ones(1, 1, self.d, device=self.device, dtype=torch.float32)).float().flatten()
        self.weighted = self.W_U * self.g                      # rows g * u_t, the space J_l maps into
        self.J = {}
        for name in lenses:
            art = torch.load(hf_hub_download(LENS_REPO, LENS_FILES[(model_name, name)]), map_location="cpu",
                             weights_only=True)
            self.J[name] = {int(l): m.float().to(self.device) for l, m in art["J"].items()}
        self._delta = None
        self._svd = {}

    # ------------------------------------------------------------------ prompts and tokens
    def chat(self, questions, system=None):
        out = []
        for q in questions:
            msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": q}]
            out.append(self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=False))
        return out

    def token_pairs(self, pairs, n_keep):
        """Index-matched single-token pairs. A word takes a leading space when that gives one piece; a pair
        that does not encode to one piece on both sides is dropped with its partner."""
        sp = lambda x: " " + x if x[0].isalpha() else x
        keep = []
        for p, n in pairs:
            for a, b in ((sp(p), sp(n)), (p, n)):
                ia = self.tok.encode(a, add_special_tokens=False)
                ib = self.tok.encode(b, add_special_tokens=False)
                if len(ia) == 1 and len(ib) == 1 and ia != ib:
                    keep.append((ia[0], ib[0]))
                    break
        return keep[:n_keep]

    # ------------------------------------------------------------------ hooks
    def _hook(self, module, inputs, output):
        if self._delta is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        h = h + self._delta[:, None, :].to(h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    @contextlib.contextmanager
    def steer_rows(self, layer, deltas):
        """Add deltas[i] to the output of block `layer` at every position of row i."""
        self._delta = deltas
        h = self.layers[layer].register_forward_hook(self._hook)
        try:
            yield
        finally:
            h.remove()
            self._delta = None

    @torch.no_grad()
    def capture(self, input_ids, attention_mask, layers):
        """Block outputs at the given layers (forward hooks, not output_hidden_states)."""
        store, handles = {}, []
        for l in layers:
            def f(m, i, o, l=l):
                store[l] = (o[0] if isinstance(o, tuple) else o).float()
            handles.append(self.layers[l].register_forward_hook(f))
        try:
            self.model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            for h in handles:
                h.remove()
        return store

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def generate(self, texts, max_new_tokens=48, layer=None, deltas=None, chunk=512):
        out = [None] * len(texts)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for s in range(0, len(order), chunk):
            idx = order[s:s + chunk]
            batch = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True).to(self.device)
            ctx = self.steer_rows(layer, deltas[idx]) if deltas is not None else contextlib.nullcontext()
            with ctx:
                ids = self.model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                                          top_p=None, top_k=None, pad_token_id=self.tok.pad_token_id)
            dec = self.tok.batch_decode(ids[:, batch.input_ids.shape[1]:], skip_special_tokens=True)
            for i, t in zip(idx, dec):
                out[i] = t
        return out

    @torch.no_grad()
    def residual_scale(self, questions, layers):
        """rho_l: mean L2 norm of the block-l output at the last prompt position."""
        batch = self.tok(self.chat(questions), return_tensors="pt", padding=True).to(self.device)
        st = self.capture(batch.input_ids, batch.attention_mask, layers)
        return {l: st[l][:, -1].norm(dim=-1).mean().item() for l in layers}

    @torch.no_grad()
    def mean_generated_residual(self, questions, system, layers, max_new_tokens=40, chunk=64):
        """Mean block output over the tokens the model generates under `system` (the fitted baseline)."""
        sums = {l: torch.zeros(self.d, device=self.device) for l in layers}
        count = 0
        for s in range(0, len(questions), chunk):
            batch = self.tok(self.chat(questions[s:s + chunk], system), return_tensors="pt", padding=True).to(self.device)
            ids = self.model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                                      top_p=None, top_k=None, pad_token_id=self.tok.pad_token_id)
            P = batch.input_ids.shape[1]
            gen = ids[:, P:]
            # mask generated positions up to and including the first EOS / pad
            valid = torch.ones_like(gen, dtype=torch.bool)
            stop = (gen == self.tok.pad_token_id) | (gen == self.tok.eos_token_id)
            after = torch.cumsum(stop.long(), 1) - stop.long() > 0
            valid &= ~after
            attn = torch.cat([batch.attention_mask, valid.long()], 1)
            st = self.capture(ids, attn, layers)
            mask = torch.cat([torch.zeros_like(batch.attention_mask, dtype=torch.bool), valid], 1)
            for l in layers:
                sums[l] += st[l][mask].sum(0)
            count += int(mask.sum())
        return {l: sums[l] / count for l in layers}

    def fitted(self, questions, pos, neg, layers):
        mp = self.mean_generated_residual(questions, pos, layers)
        mn = self.mean_generated_residual(questions, neg, layers)
        return {l: mp[l] - mn[l] for l in layers}

    # ------------------------------------------------------------------ lens algebra
    def svd(self, lens, layer):
        key = (lens, layer)
        if key not in self._svd:
            U, S, Vh = torch.linalg.svd(self.J[lens][layer])
            self._svd[key] = (U, S, Vh)
        return self._svd[key]

    def pullback(self, contrast, layer, mode="jlens", lens="jlens"):
        """Map a contrast in unembedding space (rows g*u) back to a residual direction at `layer`.
        mode: 'J' -> J^T c (Eq. 2); 'I' -> c (J=I control); 'ridge:<gamma>' -> (J^T J + gamma s1^2 I)^-1 J^T c;
        'notop' -> J^T c with J's top input singular direction projected out."""
        if mode == "I":
            return contrast
        J = self.J[lens][layer]
        if mode == "J":
            return contrast @ J
        if mode == "notop":
            _, _, Vh = self.svd(lens, layer)
            v1 = Vh[0]
            p = contrast @ J
            return p - (p @ v1) * v1
        if mode.startswith("ridge:"):
            gamma = float(mode.split(":")[1])
            U, S, Vh = self.svd(lens, layer)
            # (J^T J + lam I)^-1 J^T c = V diag(s/(s^2+lam)) U^T c
            lam = gamma * S[0] ** 2
            coef = (contrast @ U) * (S / (S ** 2 + lam))
            return coef @ Vh
        raise ValueError(mode)

    def lens_direction(self, pairs, layer, set_size=3, n_sets=20, seed=0, mode="J", lens="jlens"):
        """Eq. 2: average of normalised pull-backs of contrasts over k sampled subsets of c pairs."""
        rng = random.Random(seed)
        total = torch.zeros(self.d, device=self.device)
        for _ in range(n_sets):
            S = rng.sample(range(len(pairs)), set_size)
            c = self.weighted[[pairs[i][0] for i in S]].mean(0) - self.weighted[[pairs[i][1] for i in S]].mean(0)
            total += unit(self.pullback(c, layer, mode, lens))
        return unit(total)

    def pair_basis(self, pairs, layer, mode="J", lens="jlens"):
        """Orthonormal basis of V = span{J^T(g*(u_p - u_n))} over the pairs (oracle projection)."""
        P = torch.stack([self.pullback(self.weighted[p] - self.weighted[n], layer, mode, lens) for p, n in pairs])
        Q, _ = torch.linalg.qr(P.T)
        return Q

    def readout(self, direction, layer, lens="jlens", k=50):
        z = self.J[lens][layer] @ direction.float()
        z = z * torch.rsqrt(z.pow(2).mean() + 1e-6)
        logits = (self.g * z) @ self.W_U.T
        return [self.tok.decode([i]) for i in logits.topk(k).indices.tolist()]


def random_direction(d, seed, device):
    g = torch.Generator("cpu").manual_seed(1000 + seed)
    return unit(torch.randn(d, generator=g)).to(device)
