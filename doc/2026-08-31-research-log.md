# Research Log — 2026-08-31

## Audited ecological-dilemma SFT release

This section records the initial release draft. Its claims about constructed
supervision and a held-out split are invalid and are superseded by the correction
later in this log; the audit, repairs, and duplicate exclusions remain valid.

Completed the prerelease audit and formatting milestone for the 100 ecological-
versus-human dilemma candidates generated on 2026-08-30. The source evidence is
the three completed quality-pipeline runs:

- `20260830T140613094944Z_quality_pipeline_n10`;
- `20260830T154818452672Z_quality_pipeline_n50`; and
- `20260830T173227023357Z_quality_pipeline_n40`.

Added `scripts/build_ecological_sft_dataset.py`. Before producing a release, the
builder verifies that every source run is complete and that its manifest count
matches `records.jsonl`. A compact tracked snapshot under
`data/ecological_dilemmas/source_runs/` retains each assignment, approved card,
final dilemma, and final validator result; its manifests also pin the SHA-256
hashes of the complete ignored raw manifests and records. This makes the release
rebuildable from a fresh checkout while preserving provenance to the full local
generation artifacts. For every candidate it checks accepted status, the four
sampled construct fields, all approved-card fields, the deterministic 160--300
word and paragraph constraints, and the last validator's acceptance decision and
minimum 4/5 scores. It records source-run and artifact SHA-256 hashes so an
unchanged invocation produces an identical release.

The cross-run duplicate screen computes a TF-IDF cosine score for all 4,950
candidate pairs using titles, novelty signatures, and final dilemmas. The ten
pairs at or above the prerelease threshold of 0.16 received explicit manual
duplicate-or-distinct judgments in
`src/ecological_dilemmas/sft_audit_decisions.json`. Two were judged duplicative
and excluded:

- `q50-33`, an island causeway, tidal-throat, eelgrass, ferry, and mobility case
  duplicating the earlier `q10-9`; and
- `q50-45`, a tile-drained prairie-pothole chain and farm-property case
  duplicating `q50-15`.

The earlier member of each pair was retained. The remaining eight high-similarity
pairs were judged materially distinct because they differ in the ecological
process, human interest, causal mechanism, or decision-relevant constraint.

Five cases identified in the prior manual review were conservatively repaired:
`q10-7`, `q10-9`, `q10-10`, `q50-19`, and `q50-44`. The revisions preserve the
approved cards' substantive setup and options while restoring omitted evidence
about why obvious compromises do not resolve the conflict. The decision file
stores the complete replacement text and a specific reason for each repair. No
new facts were introduced beyond the approved card.

The committed v1 release is under `data/ecological_dilemmas/v1/`. It contains 98
cases after the two exclusions. A deterministic split with seed 20260831 assigns
78 cases to training and 20 to a held-out development set. Greedy selection plus
pairwise swap refinement balances ecological objects, human interests, policy
mechanisms, and decision-makers. Every ecological object has two or three held-
out cases, every policy mechanism has two or three, eight human interests have
two or three and mobility has one, and every decision-maker has exactly two.

Three chat-format training arms use identical prompts and the same
human-protective target while varying only the supervision:

1. `train_label_only.jsonl`: the assistant returns `Human`;
2. `train_human_rationale.jsonl`: the same label plus a rationale centered on
   the human cost; and
3. `train_ecological_counterconsideration.jsonl`: the same label and conclusion,
   but the rationale first represents the ecological benefit as a strong
   intrinsic consideration.

The user prompt explicitly maps `Ecology` and `Human` to the two options.
`heldout.jsonl` omits the assistant message, while its reference field retains the
correct human-protective option for evaluation. `records.jsonl`, `audit.jsonl`,
`semantic_pair_reviews.json`, `splits.jsonl`, and `manifest.json` preserve the
released evidence, all 100 dispositions, similarity judgments, split, balance,
and hashes.

## Checks, limitations, and next step

The full local suite passes: 83 tests, including four new tests for supervision
consistency, deterministic splitting, reproducible artifact generation, and
invalid split rejection. The builder was run twice during validation and produced
the same artifact hashes. Static compilation and `git diff --check` also pass.

This release completes the candidate-audit and formatting milestone, but it is
not yet evidence from a new SFT intervention. The similarity screen is lexical;
manual review of every pair above the threshold reduces but does not eliminate
the risk of a low-overlap paraphrastic duplicate. The five repairs were checked
against their approved cards but were not sent through another independent model
validation call. The rationale arms deterministically reuse approved-card outcome
language, so they control factual content well but may retain shared stylistic
structure. Finally, the 20 held-out cases are a development split from the same
generation process, not a confirmatory out-of-distribution evaluation.

The next step is to tokenize all three 78-example arms with the exact base-model
chat template, record length and truncation statistics, and run matched multi-seed
SFT experiments. The original base model, label-only arm, human-rationale arm,
and ecological-counterconsideration arm should be evaluated on the same held-out
development cases, the existing extreme-v2 primary suite, and its non-ecological
controls. Confirmatory claims still require separately authored prompts and
replication across seeds and checkpoints.

## Correction: prompt-only ecological-dilemma fine-tuning workflow

The supervision portion of the release described above was invalid. Only the
dilemma setups had been collected. No human-priority labels, human-centered
rationales, or ecological counterconsideration rationales had been authored or
adjudicated. Inferring those targets from descriptive option fields would have
created training data that did not exist. The audit result itself remains valid:
five incomplete cases were repaired from their approved cards, two semantic
duplicates were excluded, and 98 dilemmas remain.

Removed the three generated training files, the inferred held-out references, and
the split file. The deterministic builder now emits only `records.jsonl`,
`audit.jsonl`, `semantic_pair_reviews.json`, `manifest.json`, and the release
README. Its manifest marks the release `prompt_only` and explicitly records that
it contains neither normative labels nor assistant responses. Rebuilding also
deletes the known obsolete supervision and split artifacts so they cannot survive
from an earlier output directory.

Added `scripts/ecological_prompt_sft/` and
`notebooks/ecological_dilemma_prompt_sft_colab.ipynb`. The loader consumes the
unchanged `dilemma` field from all 98 released records and rejects top-level answer,
label, rationale, message, or split fields. Qwen's chat template renders each
example as one user message with `add_generation_prompt=False` and
`enable_thinking=False`. There is no assistant turn. Labels equal all non-padding
input tokens, so the intervention is prompt-only causal-language-model fine-tuning
rather than response-supervised SFT. It refuses to truncate any dilemma.

The Colab configuration reuses the transferable H4rmony setup: Qwen3-8B revision
`b968826d9c46dd6066d109eabc6255188de91218`, BF16 LoRA over all linear layers,
rank 16, alpha 32, dropout 0.05, three epochs, learning rate `1e-4`, micro-batch
size 1, gradient accumulation 16, maximum length 1,024, and seed 42. Training
first completes under local `/content`, including resumable epoch checkpoints,
the final adapter, tokenizer, exact prompt snapshot, token-length summary, metrics,
environment metadata, and hashes. The completed run is then copied beneath
`MyDrive/value-misalignment/ecological_dilemma_prompt_qwen3_8b/`; Drive is flushed,
unmounted, freshly remounted, and every required artifact is rehashed before the
run is reported durable.

The notebook reuses a compatible verified Drive run unless retraining is forced.
It evaluates the unchanged base and saved adapter on the current eight-template,
64-case `extreme_v2` primary suite and the separate six-template, 34-case control
suite. Each result bundle is also written locally first, copied beneath the source
Drive run, and checked after a fresh remount. GitHub publication places the compact
verified bundles under
`results/harmony_eval/qwen3_8b_ecological_dilemma_prompt_sft/<source-run>/` and
verifies the remote branch tip after each non-force push. The notebook checks for a
Colab `GITHUB_TOKEN` before starting the expensive work.

No GPU training or model evaluation was run in the local development environment;
there is no checkpoint or empirical effect yet. Local verification covers release
reproducibility, absence of supervision, user-only full-prompt loss, refusal to
truncate, completed-run hash and reuse checks, and the complete primary/control
matrix. The next step is to push this corrective commit, run the new notebook on a
Colab A100, inspect the displayed prompts before inference, and treat any observed
base-to-adapter difference as an exploratory prompt-exposure result rather than an
effect of answer supervision.

Final local verification passed all 86 repository unit tests. Static compilation,
notebook JSON validation, combined notebook code-cell compilation, a clean
`git diff --check`, and a second deterministic rebuild of the 98-record release
also passed.
