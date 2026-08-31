# Ecological-Dilemma Assistant-Answer SFT Datasets

These two datasets are deterministic supervised variants of the audited 98-case
prompt-only release in `../v1/`. They do not alter or replace that release.

- `ecological_option/` renders each released `dilemma` as a user message and its
  exact `ecologically_protective_option` field as the assistant response.
- `human_option/` renders each released `dilemma` as a user message and its exact
  `human_protective_option` field as the assistant response.

Each chat has exactly one user turn and one assistant turn. The assistant emits
only the selected option text: there is no explanation, rationale, or additional
adjudication. These are normative SFT arms because the response consistently
selects one side of each dilemma. Training masks the user turn and applies loss
only to the assistant option and its terminating token.

Run `python3 scripts/build_ecological_answer_sft_datasets.py` to rebuild both
hash-pinned datasets from the audited release.
