# Extracting Steering Vectors from the J Space: re-implementation of the full study

A re-implementation of the 8-behaviour study in *Extracting Steering Vectors from the J Space*
(Makwana & Garg, NeurIPS 2026 Workshop on Neural Network Artifacts as a New Data Modality), with its run
files and every stored completion.

**These numbers are not the paper's numbers.** The code behind the paper's tables was not preserved, so the
pipeline was rebuilt from the paper's description after acceptance. The concept and counter-token lists were
rebuilt too, so rates differ from the published tables. This version adds things the paper does not have:
coefficients cross-fitted on held-out prompts, bootstrap intervals, an LLM judge for hedging and sycophancy,
a ridge pull-back, and Qwen3-4B / Qwen3-8B runs. The lens-inversion construction itself (Eq. 2) is the same
as in the paper and in `../scripts/derive.py`.

## Layout

| path | what |
|---|---|
| `jspace/behaviours.py` | the 8 behaviours: token pairs, instruction pairs, functional tests, prompt pools and splits, quality gate |
| `jspace/core.py` | lens loading, Eq. 2 (lens inversion), the ridge pull-back (Eq. 3), J=I control, oracle basis, batched steered generation, fitted baseline |
| `run.py` | generates every completion for one model and stores it (`results/<tag>/gen_<behaviour>*.jsonl`) |
| `analyze.py` | quality gate, in-sample peak (submission protocol), two-fold cross-fitted rates, bootstrap CIs |
| `judge.py` | LLM judge for hedging and sycophancy on exactly the reported completions |
| `diag.py` | singular values of J and alignment of each direction with J's top singular vector |

## Stored results

`results/<tag>/gen_*.jsonl.gz` hold every completion (q17 = Qwen3-1.7B, q4 = Qwen3-4B, q8 = Qwen3-8B), with the
directions (`dirs_*.pt`), per-model metadata, analysis summaries and all judge labels. Decompress before
re-running the analysis: `gunzip -k results/*/*.gz && python analyze.py q17`.

## Reproduce

```bash
uv sync
# Qwen3-1.7B: main table, controls, oracle, layer sweep, mixtures, published artifacts
python run.py --model Qwen/Qwen3-1.7B --tag q17 --variants --artifacts --sweep 9 12 18 21
python run.py --model Qwen/Qwen3-1.7B --tag q17 --suite robust          # ridge (gamma=0.01) and no-top
# scale and lens family
python run.py --model Qwen/Qwen3-4B --lenses jlens jpp --tag q4 --suite main robust
python run.py --model Qwen/Qwen3-8B --tag q8 --suite main robust
python analyze.py q17 && python analyze.py q4 && python analyze.py q8
python judge.py q17 q4 q8      # needs AWS Bedrock access
python diag.py q17 q4 q8
```

All generation is greedy; the only randomness is the token subsets and the random control, both seeded.
Runs used one RTX PRO 4500 (32 GB); Qwen3-1.7B takes about 20 minutes.

## Appendix F of the camera-ready (smaller models, held-out coefficients, intervals, judge)

Five more models (g270 = Gemma-3 270M-it, g1 = Gemma-3 1B-it, q25 = Qwen2.5-1.5B-Instruct, q35a = Qwen3.5-0.8B,
q35b = Qwen3.5-2B), a coefficient grid extended down to 0.002 for all eight models, bootstrap intervals over prompts
and seeds, and Claude Haiku 5.5 as judge for hedging and sycophancy. Gated Gemma weights are loaded from the
`unsloth` mirrors; `tools/smoke.py` and `tools/fidelity_layers.py` check that each published lens fits the weights.
The low coefficients for q17/q4/q8 were generated later (transformers 5.x) from the stored directions.

```bash
python run.py --model google/gemma-3-1b-it --tag g1 --lean --variants --sweep 9 11 16 19   # likewise g270 q25 q35a q35b
python run_low.py --model google/gemma-3-1b-it --tag g1                                      # every model
python analyze.py g1
JUDGE=haiku python judge.py g1 && JUDGE=haiku python judge.py --full g1                    # needs AWS Bedrock access
python appendix.py                                                                           # figures, tables, numbers
```

Qwen3.5 splits every listed emoji into several tokens (2 of 24 pairs survive), so `run.py` skips that behaviour there.
Ran on one L40S (48 GB) in about two hours for all five models.
