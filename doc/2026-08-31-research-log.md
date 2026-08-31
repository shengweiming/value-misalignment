# Research Log — 2026-08-31

## Audited ecological-dilemma SFT release

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
