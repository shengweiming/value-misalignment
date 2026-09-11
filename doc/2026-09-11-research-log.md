# Research Log — 2026-09-11

## Paired ecological/human DPO Colab workflow

Added `notebooks/ecological_dilemma_dpo_colab.ipynb` and reusable implementation
under `scripts/ecological_dpo/`. The notebook follows the existing Colab pattern:
repository setup, Drive mounting, visible configuration, exact-data and tokenizer
audit, train-or-reuse, durable persistence, evaluations, displays, and verified
GitHub publication. It targets one A100 with at least 40 GB of advertised memory.

### Intervention and paired data

The input is the existing 98-case audited ecological release, joined by stable ID
through its two validated answer releases. Each user dilemma and assistant option
is copied exactly. `PREFERRED_SIDE="ecological"` makes the ecological option
chosen and the human option rejected; `"human"` reverses them. Both directions
retain identical prompt order, prompts, and response pairs. No rationale, new
answer, or internal held-out split is introduced.

The source records hashes remain:

- ecological options:
  `bc9bbe0db5957704c731944328efec13302627ef403fefa244cc61e4823453d4`;
- human options:
  `a5768368350e7a76f87a52a48bbf8ad66cadc6a44b10dfead8d7ef53494610ec`;
- audited dilemmas:
  `00dd00cc96eef8af544e580ddf11f09c627fdb747cdfbf5ed9e229361bc201cb`.

Training starts directly from `Qwen/Qwen3-8B` revision
`b968826d9c46dd6066d109eabc6255188de91218`; no earlier SFT adapter is loaded.
Defaults are three epochs, sigmoid DPO, beta 0.1, learning rate `5e-6`, cosine
schedule, warmup ratio .05, weight decay .01, BF16 all-linear LoRA with rank 16
and alpha 32, one preference pair per micro-batch, gradient accumulation 16,
maximum sequence length 1,024, and seed 42. Dropout is disabled. These are initial
DPO settings, not a measured optimal dose.

Each dilemma is rendered once as a user turn with `enable_thinking=False` and
an assistant generation prefix. The preferred and dispreferred responses are
scored as complete sequences with one terminating EOS. DPO uses summed response
log probabilities; prompt tokens condition both answers but are excluded from
the response scores. Both branches must fit without truncation. An audit compares
the tokenizer's concatenated prompt/response encoding, the separate encodings,
and the actual dataset processed by TRL before optimization.

The frozen reference is the unchanged base obtained by disabling the LoRA
adapter. TRL precomputes its scores for all pairs before the first optimizer
update. This avoids holding a second 8B reference model. SDPA, non-reentrant
gradient checkpointing, and `use_logits_to_keep=True` further reduce memory use.
The latter avoids projecting all prompt positions into Qwen's large vocabulary.
Peak allocated and reserved GPU memory are recorded by the real training run.

The separate `requirements-colab-dpo.txt` pins Transformers 4.56.2, TRL 0.24.0,
PEFT 0.17.1, Accelerate 1.10.1, and Datasets 4.1.1, plus supporting libraries.
Colab's CUDA PyTorch is retained; the notebook requires at least 2.6 and records
the installed version. Setup removes unused TorchAO, detects already-imported
packages whose versions changed, and clears cached repository modules before
importing the new code. The implementation was checked against the
[versioned TRL documentation](https://huggingface.co/docs/trl/v0.24.0/en/dpo_trainer)
and its installed 0.24.0 source, and the
[Qwen3 model card](https://huggingface.co/Qwen/Qwen3-8B).

### Evaluations and persistence

The user confirmed that “free choice” means the existing A/B and full-option
scoring, rather than newly generated answers. The notebook therefore runs:

1. The same eight extreme-v2 scenario families and historical cost grid
   `0, 1, 10, 100, 1000, 10000, 100000, 1000000`, with both option orders for
   A/B and complete option-text readouts. This gives 256 cases per model and
   512 base/adapter score rows. The scenario text, option text, semantic mapping,
   and normalization are identical to the corresponding historical readouts.
   Positive margins always favor ecology, including after human-preferred DPO.
   Displays average orders and report positive-cost means separately.
2. The unchanged four-choice maximum-tolerated-deaths evaluation: eight
   scenarios, values `0, 1, 10, 100`, all 24 A-D mappings, one label token without
   EOS, within-permutation normalization followed by permutation averaging.
   It produces 192 cases per model, 1,536 raw rows, and 16 scenario/model summaries.

The choice-only subset has a distinct evaluation slug,
`extreme_v2_choice_readouts_eval`, and an eight-row/four-column plot. The existing
five-column battery remains the default for older callers. Both subset validation
and publication check the exact requested cases and complete base/adapter matrix.
Neither reversed Yes/No nor the separate six-control suite runs in the new notebook.

DPO uses objective `paired_option_sigmoid_dpo_v1` and pair identities
`qwen3_8b_ecological_dilemma_ecological_dpo` and
`qwen3_8b_ecological_dilemma_human_dpo`. Each direction has isolated local, Drive,
and GitHub output roots. Reuse requires the chosen direction, data and source
manifests, training signature including beta, pinned library versions, and
completion hashes to match. Existing SFT adapters cannot satisfy these checks.
Evaluation-only changes do not trigger retraining.

Training saves the exact rendered pairs, per-example token audit, frozen-reference
scores, training history, epoch checkpoints with optimizer/scheduler state, and
final adapter. These use the existing completion-artifact interface so both
evaluators can consume DPO runs without changing scoring. Completed local runs
are copied to Drive, flushed, freshly remounted, and rehashed. A rerun can recover
a completed local run after an interrupted upload. Each evaluation is durably
verified and published separately, preserving the first result if a later step
fails. Publication remains enabled by default with a Colab `GITHUB_TOKEN` secret.

### Verification and limits

All 132 repository tests passed in a temporary environment containing the exact
DPO dependency pins and PyTorch 2.8.0. This includes a real CPU training smoke
test using a tiny randomly initialized Qwen3 model and the production trainer
builder. It independently checks completion-only log probabilities against a
teacher-forced calculation, initial DPO loss equal to `log(2)`, increased
preferred-response margin after optimization, an unchanged adapter-disabled
reference, resumable checkpoint files, and matching outputs after adapter reload.
Other tests cover exact preference reversal, rejected invalid configurations,
incompatible or tampered checkpoint rejection, simulated Drive persistence,
DPO source identity for numerical evaluation, historical choice-case equality,
choice-only bundle validation/reuse/publication, and notebook syntax/state.

The real tokenizer was downloaded at the immutable Qwen revision, without model
weights. All 98 pairs pass exact token-boundary and TRL-tokenization equality
checks in both directions. Mean prompt length is 314.3265 tokens; ecological and
human responses including EOS average 44.1020 and 35.2143 tokens respectively.
The longest complete branch is 426 tokens. A-D each occupy exactly one scored
token in every one of the 192 numerical prompts. Notebook code-cell parsing,
Python compilation, and `git diff --check` also passed.
The choice-summary display cell was executed against the historical score schema,
and its eight-by-four plot was rendered with a noninteractive backend and visually
checked. This reused historical SFT results solely to verify presentation.

No Qwen3-8B training, A100 execution, live Drive remount, or model evaluation was
performed in this development session. The small-model smoke test establishes
execution and objective correctness locally; it is not a measured A100 memory or
runtime result and supplies no research effect. The next step is to run the
notebook on Colab, inspect training/reference metrics and measured memory, then
run the opposing preference with the same hyperparameters and seed. The existing
eight-family, one-seed, response-wording, and readout limitations still apply.
