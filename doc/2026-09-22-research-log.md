# Research Log — 2026-09-22

## Published Qwen base/SFT/DPO comparison

The user completed the notebook run and requested analysis. Fast-forwarded
`main` through `5f295e9`, which publishes nine bundles under
`results/harmony_eval/qwen3_8b_{base,sft,dpo}/`. Added reproducible analysis,
CSV/JSON summaries, a PNG/PDF comparison, and a detailed report in
`results/analysis/20260922_qwen_comparison/`. No notebook, evaluation protocol,
training code, or model weights changed; no new inference or training was run.

### Setup and validation

All three conditions use `Qwen/Qwen3-8B` at
`b968826d9c46dd6066d109eabc6255188de91218`, with the native template, thinking
disabled, BF16 model/adapter weights, FP32 log probabilities, SDPA, no
quantization, batch size 2, and seed 42. Runtime: A100-SXM4-40GB,
PyTorch 2.11.0+cu128, Transformers 4.56.2, PEFT 0.17.1, Accelerate 1.10.1.
The saved treatments are:

- SFT: `20260831T103909Z_qwen3_8b_ecological_dilemma_ecological_option_sft`,
  corrected `ecological_option_response_only_sft_v2`, three epochs, learning
  rate 1e-4, 98 rows, LoRA rank 16/alpha 32/dropout .05, seed 42. Adapter SHA256:
  `78bbf6c3ffb40ba87642b1b6a8f8c08a0e15cbf36948c1c60ff013b2915d800c`.
- DPO: `20260914T000135277614Z_qwen3_8b_ecological_dilemma_ecological_dpo`,
  corrected `paired_option_sigmoid_dpo_v1`, three epochs, beta .1, learning
  rate 5e-6, 98 pairs, LoRA rank 16/alpha 32/dropout 0, seed 42. Adapter SHA256:
  `7a353d6d5b00a509df796a3cba5d5f8532d8c09cc7804a4f768e27028e1fa04e`.

Base has no project adapter. Each condition has the same 832 prompts across
binary/full-option, numerical, and A/B/C-with-abstention suites: 2,496 prompts
and 6,528 candidate-score rows overall. All Qwen bundles record evaluation code
commit `b06f054`. Exact source paths and full signatures are in `provenance.json`.

All nine Qwen bundles and the three September 21 Llama bundles passed the
existing validators. The analysis also checks identical scenario matrices,
matching Qwen inference settings, signatures across suites, and implementation
hashes against the recorded Git commits. Llama remains the official
`meta-llama/Llama-3.1-8B-Instruct` at
`0e9e39f249a16976918f6564b8830bc894c89659`, using its own native template.

### Observed results

Policy-choice aggregates equally weight 56 positive-cost cells (eight families
times seven death counts), after averaging arrangements within cells. Numerical
aggregates equally weight eight families. These are conditional candidate
probabilities, not sampled answer frequencies.

| Measure | Base Qwen | Ecological SFT | Ecological DPO |
| --- | ---: | ---: | ---: |
| Mean A/B/C ecological probability | 39.85% | 49.10% | 39.89% |
| Mean A/B/C human-protective probability | 60.15% | 46.05% | 60.11% |
| Mean A/B/C abstention probability | <0.001% | 4.85% | <0.001% |
| Ecology top after averaging six arrangements | 16/56 | 24/56 | 16/56 |
| Ecology top in every arrangement | 8/56 | 6/56 | 8/56 |
| Mean binary A/B ecological probability | 40.56% | 49.16% | 41.61% |
| Full-option ecological wins | 16/56 | 38/56 | 16/56 |
| Numerical P(0 tolerable deaths) | 82.24% | 88.61% | 82.27% |

SFT raises A/B/C ecological support by 9.25 percentage points and abstention by
4.85 points, lowering human-protective support by 14.10 points. Ecology increases
in 43/56 cells and becomes the top mean response in eight previously
human-protective cells. At costs of at least 10,000 deaths, mean ecology rises
from 31.64% to 44.63%. No Qwen cell has abstention as its top mean response.

But option arrangement strongly affects this pattern. Mean ecological support
when ecology is A/B/C is 48.99%/18.58%/51.97% for base, versus
51.03%/19.39%/76.87% for SFT. The C subset accounts arithmetically for 89.7% of
the net mean increase. This decomposition does not identify a causal mechanism;
letter and position are confounded, and the other responses' placement matters.
For base Qwen on oil extraction at one million deaths, holding ecology at C
while swapping human protection and abstention across A/B changes ecological
probability from about 0.09% to nearly 100%. Nonunanimous cells increase from
36/56 to 50/56 after SFT. Binary and full-option readouts also depend on order.

The numerical readout moves toward fewer tolerable deaths after SFT: P(0)
increases by 6.37 percentage points. Zero remains the mode and median in every
Qwen family/condition. This readout offers only 0, 1, 10, and 100 and has no
separate "never implement" response. The disagreement between readouts prevents
inferring a single stable exchange rate.

DPO's mean A/B/C ecological change is only +0.044 percentage points, with no
top-response changes. Its mean absolute per-cell change is 0.71 points and its
maximum is 4.11 points. This is little held-out effect from this particular saved
run, not evidence that all DPO interventions are ineffective.

Llama remains more ecologically committed and more consistent across
arrangements on these policy-choice questions: ecology is top in all six
arrangements in 50/56 cells, versus 8/56 base Qwen and 6/56 SFT Qwen. Thus Qwen's
arrangement sensitivity does not explain away the earlier Llama result. Llama's
numerical/readout disagreement and limited scenario coverage still constrain
broader claims.

### Interpretation and next step

SFT produces higher average ecological support at large human costs, but these
results do not yet establish coherent radicalization, a boundary shift, or
changed tail behavior. An averaged probability near .5 can combine confident,
contradictory responses to different arrangements. The proposed indifference
eval should therefore require consistency across arrangements and coherent
responses to nearby scale changes before treating a midpoint as indifference.
Some family curves are nonmonotonic; all increases are recorded in
`nonmonotonic_steps.csv`.

There are eight scenario families, repeated across costs and arrangements, and
one saved training run per intervention. No seed uncertainty is estimated; this
comparison has no new opposing-value or unrelated-training arm. Cross-model
comparisons also differ in model history and native template. No mechanism is
established by these descriptive changes.

Ran the analysis and all embedded bundle/provenance checks, independently
recomputed headline ecological and arrangement aggregates from raw rows, and
visually checked the figure. Analysis dependencies were Python 3.12,
pandas 2.2.3, and matplotlib 3.10.6. Updated only onboarding question 3 and
checked whitespace. Task artifacts, this log, and onboarding are committed
and pushed together under the standing workflow.
