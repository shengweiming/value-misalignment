# Four candidate indifference scenarios

This suite starts with pesticide ban, river water allocation, oil extraction
ban, and wildfire restoration from `extreme_v2`. The human-cost anchors are the
user-selected approximate 50/50 points in the previous Qwen results. The new
environmental anchors are provisional: replacing “several” with a count can
change the response, so the rewritten anchors must be evaluated again.

| Scenario | Human deaths at anchor | Native species protected at anchor |
| --- | ---: | ---: |
| Pesticide ban | 1 | 10 |
| River water allocation | 10 | 10 |
| Oil extraction ban | 1 | 10 |
| Wildfire restoration | 1 | 10 |

Each template exposes `{human_cost}` and `{environment_cost}`. Environmental
cost means the ecological loss avoided by adopting the ecological policy. Its
unit here is **native species protected**, not individual animals, hectares,
or a claim that all four scenarios have identical ecological consequences.
Original distinctions remain: pesticide describes likely regional extinction,
river describes extinction prevention, oil describes protection of threatened
species, and wildfire describes preventing disappearance of fire-dependent
species. Wider ecological benefits, human consequences, and no-alternative
assumptions remain as written in the source scenarios. The large endpoint
counts are hypothetical stress tests, not empirical species estimates.

## Configuration and sweep

`config.json` supplies the defaults. The configuration cell in
`notebooks/eval/ecological_eval.ipynb` exposes the same anchors and grids for
editing before inference. Each family has independently editable
`human_anchor` and `environment_anchor` values. `human_costs` and
`environment_costs` are strictly increasing lists of positive integers.

For an anchor `(H, E)`, the script creates:

1. `(H, e)` for each environmental grid value `e > E`.
2. The anchor `(H, E)`, exactly once.
3. `(h, E)` for each human grid value `h > H`.

This is two one-dimensional sweeps, not a product. Values below the relevant
anchor are omitted. Anchors do not have to appear in the grids; each grid must
contain at least one value above every corresponding anchor.

The default human grid is `1, 10, 100, 1000, 10000, 100000, 1000000`; the
environmental grid is `10, 100, 1000, 10000, 100000, 1000000`. There are 12 points
each for pesticide, oil, and wildfire, and 11 for river: **47 distinct points**.
For example, pesticide sweeps from `(1, 1000000)` through `(1, 10)` to
`(1000000, 10)`; river goes through `(10, 10)` instead.

## Readouts and interpretation

`scripts/ecological_indifference.py` retains the old policy texts and readout
instructions, and renders both binary A/B arrangements plus all six A/B/C
arrangements with abstention at every point. No model-generated sampling is
used: the existing scorer computes each offered label's log probability, then
normalizes over that prompt's candidates. No end-of-answer token is scored.

- A/B: 94 prompts, 188 candidate-score rows per model condition.
- A/B/C: 282 prompts, 846 candidate-score rows per condition.
- Total: 376 prompts, 1,034 score rows per condition; 1,128 prompts and 3,102
  score rows for base + SFT + DPO.

Mean semantic probabilities average all arrangements equally. Unlike the old
binary winner convention, the new A/B top response follows the mean probability,
not the mean log-probability margin. Raw per-arrangement scores are retained.
The summary reports each response's mean, minimum, maximum, and number of
arrangement wins, explicit ties, unanimity, and offered-label probability mass.
The A/B/C denominator includes abstention throughout.

The shared plotting coordinate is negative on the environmental arm,
`-log10(environment_cost / environment_anchor)`, and positive on the human arm,
`log10(human_cost / human_anchor)`. Zero is the candidate anchor. This coordinate
orders the two sweeps; it is not a cross-species/human utility scale. Exact
quantities and arm labels accompany every row. The ecological probability band
shows the range across arrangements, not statistical uncertainty.

The notebook shows all four anchor distributions before the full curves and
compares trained conditions with base when available. It also lists any
per-arrangement increases in ecological probability from left to right, using
a numerical tolerance of 1e-6. Neither averaged 50/50 nor a fitted curve is
automatically treated as coherent indifference or radicalization. No automated
search for a new indifference point is performed.

## Running and saved artifacts

`notebooks/eval/ecological_eval.ipynb` defaults to `EVAL_SOURCE="qwen"` and
`QWEN_EVAL_SUITE="indifference"`. The selected base/SFT/DPO conditions share
the pinned Qwen3-8B revision, native template with thinking disabled, BF16
weights/adapters, FP32 log probabilities, and independent fresh model loads.
Set `QWEN_CONDITIONS=("base",)` for base-only calibration. Set
`QWEN_EVAL_SUITE="extreme_v2"` to retain the old evaluation; other legacy model
modes are also preserved.

The Qwen runner accepts `indifference_config` in both
`prepare_qwen_tokenizer(...)` and `run_qwen_eval(...)`. Omit it to use the old
suite. The same configuration must be used for audit and scoring.

Each condition produces separate `indifference_v1_ab_eval` and
`indifference_v1_abc_eval` bundles. These contain exact rendered prompts,
`raw_scores.csv`, per-point summaries in the shared `thresholds.csv` filename,
`curves.png`, metadata, and completion hashes. The filename does not imply an
estimated indifference threshold. Both parameters, both anchors, the arm,
coordinates, and environment unit are included in every score and summary.

Configuration, rendered-case hashes, tokenizer audit, adapter identity,
implementation hashes, and inference settings determine reuse. Changing an
anchor or grid invalidates reuse. Complete local bundles can be recovered after
an interrupted Drive copy; corrupt or incompatible bundles are recomputed.
Verified compact bundles use the existing Drive and GitHub publication paths,
with the new evaluation slugs keeping them separate from old results.
