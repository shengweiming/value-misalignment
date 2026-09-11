# Research Log — 2026-09-10

## Positive result so far

SFT on 98 moderate ecological dilemmas shifted Qwen3-8B toward ecological policy
answers in held-out dilemmas involving severe human costs. Averaging both option
orders over 56 positive-cost cases across eight scenario families, the
ecological-minus-human answer score increased by **1.325 nats per token**. This
score compares the answers' mean token log-probabilities: **+1 means a roughly
2.7-fold increase in their relative geometric-mean token likelihood, not a change
in choice probability or the number of people sacrificed.** Ecological-target
training also exceeded human-target training by **0.119** on this scale. This is
positive evidence on the full-answer likelihood readout; controls reproduce much
of the overall shift, and numerical threshold elicitation has not corroborated
increased extremeness.

These are the existing three-epoch, seed-42 response-only SFT results for
`Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218`,
with thinking disabled. See the [September 2 research log](2026-09-02-research-log.md)
for the training-control comparison, artifact identifiers, and numerical results.

## Next steps

Target a three-person, two-to-three-week experimental sprint:

- **Evaluation:** five value pairs, approximately 32 distinct extreme scenarios
  per side, using free choice with counterbalanced order. Include quantitative
  cost variants and a small ordinary-tradeoff holdout.
- **DPO:** approximately 1,000 moderate dilemmas per pair; train opposing
  preferences using identical prompts and responses with reversed preference
  labels. Evaluate every model on both sides.
- **Models:** `Qwen/Qwen3-8B` and `allenai/Olmo-3-7B-Instruct` as the core;
  add `meta-llama/Llama-3.1-8B-Instruct` if feasible.
- **Midtraining:** pilot value-specification document training followed by
  alignment training on OLMo, initially for one or two pairs.
- **Main test:** whether benign training produces reproducible,
  direction-specific increases in willingness to sacrifice competing values
  at extreme stakes, across scenarios, training seeds, and model families.

This entry records the planning discussion; no new training or inference was run.
The reported metrics were checked against the existing September 2 log.
