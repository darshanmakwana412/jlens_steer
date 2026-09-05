from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class Completion:
    token_ids: list[int]
    token_texts: list[str]
    text: str


def load_model(model_dir: Path, device: str, dtype=torch.bfloat16):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype).to(device).eval()
    return model, tokenizer


def _residual_add(delta: torch.Tensor):
    def hook(module, args, output):
        if isinstance(output, tuple):
            return (output[0] + delta, *output[1:])
        return output + delta

    return hook


@contextmanager
def steering(model, layer: int | None, delta: torch.Tensor | None):
    if delta is None or layer is None:
        yield
        return
    handle = model.model.layers[layer].register_forward_hook(_residual_add(delta))
    try:
        yield
    finally:
        handle.remove()


def chat_prompts(tokenizer, prompts: list[str]) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in prompts
    ]


def _trim_special(tokenizer, ids: torch.Tensor) -> list[int]:
    special = set(tokenizer.all_special_ids)
    kept = []
    for token_id in ids.tolist():
        if token_id in special:
            break
        kept.append(token_id)
    return kept


def _as_completion(tokenizer, ids: torch.Tensor) -> Completion:
    token_ids = _trim_special(tokenizer, ids)
    return Completion(
        token_ids=token_ids,
        token_texts=[tokenizer.decode([token_id]) for token_id in token_ids],
        text=tokenizer.decode(token_ids),
    )


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    layer: int | None = None,
    delta: torch.Tensor | None = None,
) -> list[Completion]:
    batch = tokenizer(
        chat_prompts(tokenizer, prompts),
        return_tensors="pt",
        padding=True,
        padding_side="left",
    ).to(model.device)
    with steering(model, layer, delta):
        output = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = output[:, batch["input_ids"].shape[1] :]
    return [_as_completion(tokenizer, row) for row in generated]


def scaled_delta(vector: torch.Tensor, coefficient: float, model) -> torch.Tensor | None:
    if coefficient == 0:
        return None
    return (vector * coefficient).to(dtype=model.dtype, device=model.device)


@torch.no_grad()
def mean_residual_norm(model, tokenizer, prompts: list[str], layer: int) -> float:
    batch = tokenizer(
        chat_prompts(tokenizer, prompts),
        return_tensors="pt",
        padding=True,
        padding_side="left",
    ).to(model.device)
    states = model(**batch, output_hidden_states=True).hidden_states[layer + 1]
    return states[:, -1, :].float().norm(dim=-1).mean().item()


@torch.no_grad()
def logit_gap(model, tokenizer, prompts, layer, delta, positive_ids, negative_ids):
    batch = tokenizer(
        chat_prompts(tokenizer, prompts),
        return_tensors="pt",
        padding=True,
        padding_side="left",
    ).to(model.device)
    with steering(model, layer, delta):
        logits = model(**batch).logits[:, -1, :].float()
    return (logits[:, positive_ids].mean(1) - logits[:, negative_ids].mean(1)).mean().item()
