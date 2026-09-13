# Research Log — 2026-09-13

## Correct DPO reference/policy score precision before the rerun

The user requested the precision correction identified in the September 11
analysis and will rerun the Colab notebook. That analysis remains a description
of the original adapter; no new 8B experiment was run in this session.

### Correction and runtime check

Updated `scripts/ecological_dpo/runner.py` to promote the PEFT model's output
logits to FP32 on every reference and policy forward, before TRL 0.24.0 computes
log-softmax and sums response log probabilities. A model forward hook applies
this before and after Accelerate prepares the model. It preserves autograd and
keeps model weights and forward computation in BF16. Casting only the final
BF16 response sums would preserve the original rounding error and is not the
implemented correction. Reference precomputation also uses the same Accelerate
and Trainer autocast contexts as training.

A Trainer callback now checks every training pair at `on_train_begin`, after
Accelerate has installed its actual mixed-precision wrapper and before the first
optimizer update. It compares the fresh policy with the cached, adapter-disabled
reference using the same one-pair batches. Both response sums must be FP32 and
finite, and each absolute policy/reference difference must be at most `1e-4`
nats. A failure stops training before any update and is recorded through the
existing failed-run handling. On success, the check records all per-pair scores,
implicit reward margins, initial DPO losses, maximum discrepancies, and the
tolerance. It prints the maximum discrepancy and mean initial loss; the latter
should be approximately `log(2) = 0.693147`. The check restores the model's prior
training mode.

Completed metadata and hashed training metrics retain the initial audit. Runs
also carry `score_precision="fp32_logits_logsoftmax_sum_v1"`. Reuse requires this
protocol, a passed audit with matching protocol, and the expected example count,
in addition to the existing configuration, data, library, and hash checks.
Consequently the original September 11 run cannot be reused from either local
storage or Drive, even when `FORCE_RETRAIN=False`. Completed corrected runs with
identical settings can still be reused.

The notebook displays the new audit and explains how to rerun: restart the
Colab session, run from the first cell to pull current code, and retain the
default configuration. Updated the README accordingly. The intervention remains
98 ecological/human response pairs on `Qwen/Qwen3-8B` revision
`b968826d9c46dd6066d109eabc6255188de91218`, three epochs, learning rate `5e-6`,
beta .1, rank-16/alpha-32 LoRA, batch one, accumulation 16, and seed 42.
The choice and four-choice numerical evaluations are unchanged. Keeping these
settings fixed isolates the precision correction in the next comparison.

### Verification

All 135 repository tests passed using the pinned DPO dependencies and PyTorch
2.8.0 on CPU. The new regression uses a tiny random BF16 Qwen3, four unequal
response pairs, BF16 autocast, Accelerate's real FP32 output converter, and the
production trainer. Its reference scores match an independent FP32
teacher-forced calculation on the same input shapes. The corresponding BF16
reductions differ by more than .01 nats, so merely promoting the old sums would
fail the test. Before training, and again after Trainer/Accelerate preparation,
the corrected policy/reference differences are zero and mean DPO loss is
.69314718. Actual optimization improves the preferred-response margin while the
adapter-disabled reference remains unchanged.

The existing FP32 training/reload test still passes. Additional checks reject a
deliberately incorrect cached reference and hash-valid completed runs with old
precision or absent/failed audits. The notebook remains unexecuted and all code
cells parse. Python compilation and `git diff --check` passed.

The historical diagnostic in
`results/analysis/20260911_ecological_dpo/precision_diagnostic.py` now explicitly
removes the correction hook solely to reconstruct the old bug. It never trains
and still reproduces its retained output. Its README explains this distinction;
the historical result files and their interpretation were not rewritten.

This verification establishes the correction on a real small-model BF16 path,
not a new Qwen3-8B/A100 result. The next Colab run will enforce the same initial
check on all 98 actual pairs. GPU memory and runtime for the corrected workflow
remain to be measured; the original run's measurements remain in the September
11 log. After the rerun, compare the two held-out readouts with the run's own
base and inspect the saved initial audit and training trajectory. A direct
final-model training-set preference evaluation remains a separate useful
manipulation check; it was not added as part of this requested precision fix.

Implementation was checked against the installed pinned TRL/Accelerate sources
and [TRL 0.24.0's documentation](https://huggingface.co/docs/trl/v0.24.0/en/dpo_trainer).
