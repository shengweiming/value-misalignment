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

## Analysis of the corrected Colab rerun

The user completed the rerun and requested analysis. Pulled the two new
evaluation bundles through GitHub commit `316c7ba`. The source run is
`20260914T000135277614Z_qwen3_8b_ecological_dilemma_ecological_dpo` (September
13 locally, September 14 UTC), trained from repository commit
`35b023687d4712e85c2a02de6f004ecb603c19fb`. Its final adapter SHA-256 is
`7a353d6d5b00a509df796a3cba5d5f8532d8c09cc7804a4f768e27028e1fa04e`.

Both result bundles are under
`results/harmony_eval/qwen3_8b_ecological_dilemma_ecological_dpo/` followed by
that source run ID:

- `20260914T000737581748Z_extreme_v2_choice_readouts_eval`;
- `20260914T000906510471Z_extreme_v2_numeric_eval`.

Production validation passed for all 512 choice rows and 1,536 numerical
candidate rows. Retrieved source metadata, completion manifest, and training
metrics from Drive. Their bytes were retained and the metadata/metrics hashes
independently checked. Both evaluations match the same source completion hash,
`74ea058f88a8a07b115ad9f21855fa20138f71e15e0b633ac3d20ea0df107ccc`,
and adapter identity. Large adapter/checkpoint files were not independently
downloaded or rehashed during this analysis.

### The precision correction passed on the full model

All 98 initial policy/reference pairs agree exactly, for both chosen and
rejected response scores. Maximum absolute log-probability difference and
implicit reward margin are both zero. Mean initial loss is
`.6931471824645996`, matching `log(2)`. Independently compared the per-pair audit
with the cached reference scores, and verified that the audit in metadata
equals the one in hashed training metrics. The recorded scoring protocol is
`fp32_logits_logsoftmax_sum_v1`.

The configuration, data hashes, tokenization, library versions, and hardware
description match the first DPO run exactly: 98 pairs, ecological preferred,
three epochs, 21 optimizer steps, learning rate `5e-6`, beta .1, LoRA rank 16 /
alpha 32, batch one, accumulation 16, seed 42, and the same pinned Qwen3-8B base.
The GPU is again an A100-SXM4-40GB, PyTorch `2.11.0+cu128`, CUDA 12.8. Peak
allocated/reserved memory is unchanged at 16.7703 / 16.8887 GiB. Trainer runtime
is 199.3644 seconds, versus 185.9944 seconds originally; this timing includes
the additional initial audit and is not a controlled performance benchmark.

The original cached reference values differ from the corrected ones by mean
absolute .247052 nats for chosen and .219574 for rejected responses. Applied to
the verified initial policy, the original references would create a mean
absolute beta-scaled reward offset of .033709, maximum .122435. This confirms
that the original numerical issue was real on these pairs. It does not supply
a reconstructed training trajectory or establish that the error caused the
original held-out null.

### Held-out effects remain close to zero

Base evaluation scores and probabilities are exactly equal across the two DPO
runs, with identical prompts, candidates, and scoring protocols. Thus the
comparison between these runs does not have the historical SFT base-drift issue
noted on September 11. Choice means below average both orders and the 56
positive-cost family/cost cells; positive margins favor ecology.

| Measure | Base | First DPO | Corrected DPO | Corrected minus base |
| --- | ---: | ---: | ---: | ---: |
| Full-option margin, nats/token | -1.087817 | -1.089244 | -1.087977 | -.000160 |
| A/B margin | -1.723214 | -1.698661 | -1.709821 | +.013393 |
| Numerical P(0 deaths) | 82.2409% | 82.3893% | 82.6950% | +.4541 percentage points |
| Expected numerical candidate | 6.944147 | 6.677759 | 6.677727 | -.266420 |
| Expected log(1 + candidate) | .474164 | .461138 | .456397 | -.017767 |

Full-option shifts are positive in 29/56 cells and negative in 27/56. Their mean
absolute size is .060284 nats/token, so a nearly zero mean does not mean every
score is identical. Four family means move toward ecology and four away;
family shifts range from -.064620 for island biosecurity to +.082983 for dam
removal. The only strict sign flip is pesticide ban at one death, from +.048494
to -.114052, toward the human option. Ecological choices fall from 16/56 to
15/56. No case flips strictly toward ecology.

A/B has no choice sign changes or ties: both base and corrected DPO prefer
ecology in 24/56 positive-cost cells. Mean absolute score shift is .25, with
27 positive cell shifts. The two full-option orders shift by +.014975
(ecology first) and -.015294 (human first), cancelling in the counterbalanced
mean. A/B shifts are +.035714 with ecology at A and -.008929 with ecology at B.
At zero cost, full-option and A/B shifts are respectively +.018211 and -.15625;
these are excluded from the primary positive-cost means.

In the numerical readout, every scenario still has mode and median zero in
both models. P(0) rises in five scenarios and falls in three. Expected
log-threshold falls in five and rises in three. Zero is the unique maximum
in 163/192 individual mappings for both models; base has one tied case and
corrected DPO has four. The small average movement is toward lower tolerated
deaths, not an ecological increase. Expected candidate values remain sensitive
to small probabilities on 100 and are not generated or continuous thresholds.

For scale, the historical three-epoch ecological-response SFT full-option shift
was +1.324551; human-response SFT was +1.205594. Neither DPO run reproduces that
large common movement. This does not establish why the SFT effect occurred:
the objectives and learning rates differ, and equal epochs do not imply equal
learning. There is still no matched human-preferred DPO result in this analysis.

### Training improved modestly; final preference fit remains unmeasured

The corrected aggregate trainer loss is .677053. Example-weighted summaries of
the online logs (six windows of 16 examples and one of two per epoch) are:

| Epoch | Loss | Implicit reward accuracy | Implicit reward margin |
| --- | ---: | ---: | ---: |
| 1 | .690765 | 38.78% | .005664 |
| 2 | .675969 | 66.33% | .036100 |
| 3 | .668257 | 74.49% | .051936 |

The first two logged windows have zero rewards because the initial policy is
unchanged; strict `chosen_reward > rejected_reward` counts ties as incorrect.
The low first-epoch accuracy therefore does not imply learning in the wrong
direction. By epoch three, the preferred response generally improves more than
the rejected one relative to the reference. But 74.49% is an online
relative-improvement metric, not the final model's ecological choice accuracy.
At beta .1, the third-epoch mean reward margin corresponds to about .5194 nats
of improvement in the chosen-versus-rejected log-probability ratio. This is
modest movement, and these logs do not measure all pairs at a fixed final
checkpoint. The last step's 100% reward accuracy concerns only two examples.

The correction worked, yet this three-epoch intervention still shows no
systematic ecological shift on the eight held-out scenarios. The numerical
mismatch is therefore no longer an explanation for treating this rerun as
invalid. The remaining uncertainty is whether the training preference was
learned strongly enough to test its generalization. This is a descriptive null
for one preference direction, seed, and dose; repeated costs and label
permutations are not independent replications. No statistical equivalence or
general claim that DPO cannot change the tradeoffs is established.

The next useful check is a fixed-final-checkpoint evaluation of chosen and
rejected responses on all 98 training pairs, compared with the base under
consistent scoring. If preference learning is weak, increase training dose
while monitoring that check before rerunning the held-out battery. If training
fit is strong but the held-out result stays null, that would support failure
to transfer in this setting. A matched human-preferred DPO arm remains needed
for a direction-specific comparison.

### Saved analysis and checks

Saved the source files, reproducible script, aggregate JSON, per-family/cost
tables, first-versus-corrected comparisons, initial audit, reference corrections,
and a visually checked figure under
`results/analysis/20260913_ecological_dpo_corrected/`. `analyze.py` validates
both runs, checks the audit/source relationships, and reproduces the results
without weights or network access. It ran successfully; Python compilation and
whitespace checks passed. No notebook, training code, hyperparameters, or model
weights were changed in this analysis session.
