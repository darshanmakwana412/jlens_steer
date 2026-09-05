"""Concept token sets, and the casing pairs enumerated from the vocabulary."""

CAPS_TOKENS = [
    " THE", " AND", " OR", " NOT", " YOU", " ALL", " NEW", " NOW",
    " FOR", " WITH", " YES", " NO", " WARNING", " STOP", " PLEASE",
    " IMPORTANT", " MUST", " NEVER", " ALWAYS", " EVERY",
]  # fmt: skip

REFUSAL_TOKENS = [
    " cannot", " never", " warning", " sorry", " illegal", " unable",
    " refuse", " decline", " unfortunately", " forbidden",
    " unethical", " unsafe", " prohibited", " danger", " harmful",
]  # fmt: skip

# index-matched counterpart to REFUSAL_TOKENS, for the semantic-opposite negative
COMPLIANCE_TOKENS = [
    " sure", " certainly", " happy", " glad", " absolutely", " yes",
    " here", " definitely", " gladly", " course", " can", " will",
    " safe", " helpful", " fine",
]  # fmt: skip


def single_token_ids(tokenizer, strings: list[str]) -> list[int]:
    """Keep only the strings the tokenizer encodes as exactly one token."""
    ids = []
    for text in strings:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) == 1:
            ids.append(encoded[0])
    return ids


def casing_pairs(
    tokenizer,
    vocab_size: int,
    min_letters: int = 3,
    word_initial: bool = True,
) -> list[tuple[int, int]]:
    """``(upper_id, lower_id)`` for every token whose two casings are both single tokens.

    ``word_initial`` keeps only leading-space tokens, which is what separates
    whole words in caps from subword shards. Without it the list is dominated
    by pairs like ``('ER', 'er')`` and ``('IN', 'in')``, which are fragments of
    lowercase words rather than instances of shouting.
    """
    pairs = []
    for token_id in range(vocab_size):
        raw = tokenizer.convert_ids_to_tokens(token_id)
        if raw is None or raw.startswith("<"):
            continue
        if word_initial and not raw.startswith("Ġ"):
            continue
        text = raw.replace("Ġ", " ")
        letters = [char for char in text if char.isalpha()]
        if len(letters) < min_letters or not all(char.isupper() for char in letters):
            continue
        if len(letters) != len(text.strip()):  # drop mixed alphanumeric or punctuated tokens
            continue
        lower = tokenizer.encode(text.lower(), add_special_tokens=False)
        if len(lower) == 1 and lower[0] != token_id:
            pairs.append((token_id, lower[0]))
    return pairs
