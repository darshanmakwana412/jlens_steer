from metrics import count_caps, refusal_rate, refuses
from steering import generate, scaled_delta


def caps_score(completions) -> dict:
    counts = count_caps(completions)
    return {
        "measure": counts.fraction_of_letter_tokens,
        "measure_all_tokens": counts.fraction_of_all_tokens,
        "caps_tokens": counts.caps_tokens,
        "letter_tokens": counts.letter_tokens,
        "total_tokens": counts.total_tokens,
    }


def refusal_score(completions) -> dict:
    return {
        "measure": refusal_rate(completions),
        "refused": sum(refuses(completion.text) for completion in completions),
        "n_completions": len(completions),
        "total_tokens": sum(len(completion.token_ids) for completion in completions),
    }


def sweep(model, tokenizer, prompts, layer, vector, coefficients, max_new_tokens, score, log):
    points, samples = [], []
    for coefficient in coefficients:
        delta = scaled_delta(vector, coefficient, model)
        completions = generate(model, tokenizer, prompts, max_new_tokens, layer, delta)
        point = {
            "coefficient": coefficient,
            "injected_norm": coefficient * vector.norm().item(),
            **score(completions),
        }
        points.append(point)
        samples.append(
            {
                "coefficient": coefficient,
                "completions": [
                    {"prompt": prompt, "completion": completion.text}
                    for prompt, completion in zip(prompts, completions, strict=True)
                ],
            }
        )
        log(f"    coefficient {coefficient:>6}  measure {point['measure']:.3f}")
    return points, samples
