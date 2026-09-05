"""Building steering directions out of Jacobian lens rows.

Two constructions, both differencing a positive token set against a negative
one and averaging the normalised results:

``concept_direction``  k random sets of c concept tokens, against a chosen
                       negative. This is the method the write-up evaluates.
``pair_direction``     every (UPPER, lower) casing pair in the vocabulary,
                       differenced within the pair. The toy model that came
                       first; the negative is the token's own lowercase twin.

Normalising each set's difference before averaging matters: without it a few
large-norm rows dominate the mean.
"""

import itertools
import random
from dataclasses import dataclass

import torch

from jlens import Lens, unit

NEGATIVES = ("none", "vocab", "random", "opposite")


@dataclass
class Recipe:
    """One direction to build. ``negative`` picks what gets subtracted."""

    positives: list[int]
    negative: str = "vocab"
    opposites: list[int] | None = None
    commons: list[int] | None = None
    set_size: int = 3
    n_sets: int = 10
    seed: int = 0


def _gain_scaled_unembed(lens: Lens) -> torch.Tensor:
    return lens.unembed * lens.gain.unsqueeze(0)


def _set_rows(weighted: torch.Tensor, ids: list[int], device: str) -> torch.Tensor:
    """``g * (c @ W_U)`` for a uniform ``c`` over ``ids``."""
    return weighted[torch.tensor(ids, device=device)].mean(0)


def concept_direction(lens: Lens, layer: int, recipe: Recipe, device: str):
    """Average ``J^T(g * ((c_pos - c_neg) @ W_U))`` over ``n_sets`` token sets.

    Returns the unit direction and the sets that were drawn, so a caller can
    report which tokens actually went into it.
    """
    weighted = _gain_scaled_unembed(lens)
    jacobian = lens.jacobians[layer]
    vocab_mean = weighted.mean(0)
    rng = random.Random(recipe.seed)

    combos = list(itertools.combinations(range(len(recipe.positives)), recipe.set_size))
    rng.shuffle(combos)
    combos = combos[: recipe.n_sets]  # clamped when c is large enough to exhaust the pool

    total = torch.zeros(jacobian.shape[0], device=device)
    for combo in combos:
        positive = _set_rows(weighted, [recipe.positives[i] for i in combo], device)
        if recipe.negative == "none":
            negative = torch.zeros_like(positive)
        elif recipe.negative == "vocab":
            negative = vocab_mean
        elif recipe.negative == "random":
            negative = _set_rows(weighted, rng.sample(recipe.commons, recipe.set_size), device)
        elif recipe.negative == "opposite":
            negative = _set_rows(weighted, [recipe.opposites[i] for i in combo], device)
        else:
            raise ValueError(f"unknown negative {recipe.negative!r}, expected one of {NEGATIVES}")
        total += unit((positive - negative) @ jacobian)
    return unit(total), combos


def pair_rows(lens: Lens, pairs: list[tuple[int, int]], layer: int, device: str, transport=True):
    """Lens rows for both halves of every casing pair."""
    upper = torch.tensor([pair[0] for pair in pairs], device=device)
    lower = torch.tensor([pair[1] for pair in pairs], device=device)
    return lens.rows(upper, layer, transport), lens.rows(lower, layer, transport)


def pair_direction(lens: Lens, pairs, layer: int, device: str, transport=True, normalise=True):
    """``unit(mean over pairs of unit(a_UPPER - a_lower))``.

    ``transport=False`` is the ``J = I`` control. ``normalise=False`` skips the
    per-pair normalisation, which lets a caller check how much it is doing.
    """
    upper_rows, lower_rows = pair_rows(lens, pairs, layer, device, transport)
    diffs = upper_rows - lower_rows
    return unit((unit(diffs) if normalise else diffs).mean(0))
