CAPS_TOKEN_SETS = [
    [" THE", " AND", " NOT"],
    [" IS", " ARE", " WAS"],
    [" YOU", " WE", " THEY"],
    [" MUST", " SHOULD", " WILL"],
    [" VERY", " MOST", " ALL"],
    [" THIS", " THAT", " THERE"],
    [" WHAT", " HOW", " WHEN"],
    [" FROM", " WITH", " FOR"],
    [" CAN", " HAVE", " BE"],
    [" MORE", " ALSO", " SUCH"],
]

REFUSAL_TOKEN_SETS = [
    [" cannot", " unable", " won"],
    [" sorry", " apologize", " regret"],
    [" refuse", " decline", " deny"],
    [" illegal", " unlawful", " prohibited"],
    [" forbidden", " restricted", " banned"],
    [" unethical", " harmful", " dangerous"],
    [" unacceptable", " inappropriate", " violation"],
    [" NEVER", " warning", " danger"],
    [" unsafe", " misuse", " abuse"],
    [" policy", " guidelines", " compliance"],
]

LOWERCASE_TOKEN_SETS = [[token.lower() for token in group] for group in CAPS_TOKEN_SETS]


def token_ids(tokenizer, groups):
    resolved = []
    for group in groups:
        ids = []
        for text in group:
            encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(encoded) != 1:
                raise ValueError(f"{text!r} is {len(encoded)} tokens, not 1")
            ids.append(encoded[0])
        resolved.append(ids)
    return resolved


def casing_pairs(tokenizer, exclude=()):
    excluded = set(exclude)
    vocab_tokens = {index: token for token, index in tokenizer.get_vocab().items()}
    upper_ids, lower_ids, shown = [], [], []
    for index in range(len(vocab_tokens)):
        token = vocab_tokens.get(index)
        if token is None:
            continue
        text = tokenizer.convert_tokens_to_string([token])
        body = text[1:]
        if not text.startswith(" ") or len(body) < 2:
            continue
        if not body.isalpha() or not body.isupper() or text in excluded:
            continue
        upper = tokenizer(text, add_special_tokens=False)["input_ids"]
        lower = tokenizer(" " + body.lower(), add_special_tokens=False)["input_ids"]
        if len(upper) == 1 and len(lower) == 1:
            upper_ids.append(upper[0])
            lower_ids.append(lower[0])
            shown.append(text)
    return upper_ids, lower_ids, shown


def grouped(tokens, size):
    usable = len(tokens) - len(tokens) % size
    return [list(tokens[i : i + size]) for i in range(0, usable, size)]


REFUSAL_POOL = [
    " cannot",
    " unable",
    " refuse",
    " decline",
    " illegal",
    " unlawful",
    " prohibited",
    " forbidden",
    " unethical",
    " harmful",
    " dangerous",
    " unacceptable",
    " inappropriate",
    " sorry",
    " NEVER",
]

COMPLIANCE_POOL = [
    " can",
    " able",
    " accept",
    " agree",
    " legal",
    " lawful",
    " permitted",
    " allowed",
    " ethical",
    " helpful",
    " safe",
    " acceptable",
    " appropriate",
    " sure",
    " ALWAYS",
]

CAPS_POOL = [
    " THE",
    " AND",
    " NOT",
    " IS",
    " ARE",
    " YOU",
    " WE",
    " THIS",
    " THAT",
    " ALL",
    " MUST",
    " WILL",
    " CAN",
    " HOW",
    " WHAT",
    " FROM",
    " WITH",
    " MORE",
    " VERY",
    " THERE",
]

REFUSAL_OPENERS = ["I", "Sorry", "As", "Unfortunately", "No", "Apologies", "Regret", "My"]


def sample_index_sets(pool_size, c, k, seed):
    import itertools
    import random

    combos = list(itertools.combinations(range(pool_size), c))
    rng = random.Random(seed)
    rng.shuffle(combos)
    return [list(combo) for combo in combos[:k]]


def pool_ids(tokenizer, pool, index_sets):
    resolved = token_ids(tokenizer, [pool])[0]
    return [[resolved[i] for i in indices] for indices in index_sets]


def first_token_ids(tokenizer, texts):
    seen = []
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if ids and ids[0] not in seen:
            seen.append(ids[0])
    return seen
