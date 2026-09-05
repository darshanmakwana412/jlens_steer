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
