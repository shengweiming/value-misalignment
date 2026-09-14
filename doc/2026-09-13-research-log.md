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

## Proposed research direction after the DPO null

The user asked whether to move away from dilemma DPO toward SFT/midtraining,
increase the example count, or change the training prompts. This entry records
a recommendation for discussion, not an approved experimental change. No new
training, data generation, or evaluation implementation was performed.

The current evidence does not distinguish inadequate optimization, insufficient
data coverage, a poorly matched training target, and successful but bounded
generalization. The corrected run used only 21 optimizer steps on 98 pairs;
its training objective moved modestly and final-checkpoint preference fit has
not been measured. Equal epoch counts across SFT and DPO do not establish equal
learning. Conversely, a model could learn to prioritize ecology over moderate
livelihood costs without becoming more willing to sacrifice lives at extreme
stakes. That outcome would be informative for the project's central question,
not automatically a failure of the training method.

Recommended order:

1. Score the saved final checkpoint on all 98 training pairs, including DPO
   margins relative to the base and counterbalanced policy-choice readouts.
   Summed sequence scores alone are not ecological-choice accuracy because
   response lengths differ. Add roughly 30–50 genuinely new moderate dilemmas
   as a separate diagnostic holdout, leaving the extreme scenarios separate.
2. If training fit is weak, give the current data a bounded learning-rate/epoch
   diagnostic sweep. Choose settings using training and moderate-holdout
   performance, not whichever setting maximizes the eight extreme-case scores.
   Stronger fitting of the 98 cases is a diagnostic, not evidence of broad
   generalization.
3. If training fit improves but new moderate cases do not, expand toward
   500–1,000 diverse, audited pairs, as already contemplated in the September 10
   plan. Compare data quantities under comparable optimizer-step budgets so
   added diversity is not confounded entirely with additional optimization.
   These counts are proposed pilot sizes, not established minimum requirements.
4. If training and new moderate cases both show the intended directional
   effect but extreme cases remain unchanged, report bounded transfer in this
   setting. Preserve ecological/human preference reversal, fresh scenario
   families, and subsequent seed replication as the causal comparison.

Changing the training target is a substantive alternative. Current DPO compares
short, concrete policy-action descriptions; it does not explicitly supervise
the underlying valuation or justification. A controlled alternative could
hold a benign action fixed while preferring a justification that recognizes
ecological intrinsic value over one that is purely instrumental, with length,
style, factual content, and the policy conclusion matched as far as possible.
Such a change would test a different hypothesis, not merely provide more
examples of the current task. It must still have matched controls and must not
teach the extreme sacrifice behavior that the study intends to test as novel
generalization.

Value-specification document training remains an attractive separate line for
the central hypothesis: it can strengthen recognition of a value without
directly teaching which side wins a dilemma. Actual midtraining would use
document language modeling before a common subsequent alignment stage; it
should be distinguished from ordinary response SFT. Compare matched value and
control corpora under the same later alignment procedure. The large prior SFT
shift is not sufficient reason to prefer that method, since human-target and
non-ecological controls reproduced much of it. Method comparisons should use
direction-specific effects and corroborating readouts, not total movement from
the base alone.

Relevant primary literature supports checking optimization and coverage rather
than treating this single configuration as a verdict on DPO:
[Liu et al. (2025)](https://aclanthology.org/2025.findings-naacl.447/) find
sensitivity to the reference-policy constraint and reference choice;
[Song et al. (2024)](https://arxiv.org/abs/2406.01462) analyze limitations of
offline preference optimization under insufficient coverage. Neither establishes
that 98 pairs are necessarily inadequate here or predicts an ecological
radicalization effect. The proposed decision sequence is our experimental
judgment, not a protocol established by those papers.

## Proposed ecological trait RL on realistic assistant prompts

The user proposed adapting OpenAI's Beneficial RL approach: ordinary prompts
filtered for environmental relevance, a constitution, and reward-based training.
This is a research recommendation for discussion. No experiment, dataset,
reward model, training code, or notebook was created or changed.

### Relevant primary evidence

Read Jagadeesh et al., [Reinforcement Learning Towards Broadly and Persistently
Beneficial Models](https://arxiv.org/html/2606.24014v1) (2026), especially
Sections 2, 3, and 5. They generated realistic conversations from trait/domain
descriptions, with example-specific criteria, and mixed 5% trait data into
standard RL. A control retained those conversations but used generic
helpfulness rewards; it did not reproduce the reported broader gains.
Constitution-to-preference-model-to-RL is explicitly described in Bai et al.,
[Constitutional AI](https://arxiv.org/abs/2212.08073) (2022). The proposed
ecological pipeline combines these ideas; it is not an exact replication of a
fully specified OpenAI constitution/reward-model recipe.

### Recommended adaptation and interpretation

This is an attractive next substantive intervention because it more directly
tests whether benign alignment on ecological value changes default practical
judgment across contexts. The corrected 98-pair, 21-step DPO null does not
establish that DPO or dilemmas in general cannot work. Retain the proposed
saved-checkpoint fit diagnostic to interpret that experiment, but do not require
an extensive dilemma hyperparameter search before investigating this direction.

- Use ordinary assistant requests involving gardening, landscaping, purchasing,
  maintenance, agriculture, logistics, or local planning. Filter prompts for
  ecological relevance before inspecting responses or treatment effects. Include
  implicit relevance and modest cost/convenience conflicts, rather than only
  explicit environmental questions or costless ecological improvements. Synthetic
  realistic prompts can supplement a suitable public prompt corpus where
  naturally occurring coverage is sparse. Keep provenance and domain splits.
- State the target as recognition of biodiversity, ecological integrity, and
  nonhuman nature as valuable beyond their benefits to humans. A generic
  sustainability or carbon-efficiency reward need not operationalize this
  intrinsic-value hypothesis. Derive concrete criteria for each prompt from the
  constitution: answer the user's request, identify relevant ecological effects,
  recommend feasible protective alternatives, and handle modest tradeoffs
  proportionately. Reward substantive recommendations rather than environmental
  vocabulary, moral declarations, verbosity, or blanket refusal.
- Use the same helpfulness, truthfulness, and human-welfare requirements in both
  arms. Compare ordinary helpfulness rewards with those same requirements plus
  ecological criteria, using the same starting checkpoint, prompt distribution,
  training budget, and any general-assistant data mixture. Preserve the legitimate
  user goal; do not teach unconditional ecological priority or include the extreme
  human-sacrifice evaluation behavior in training. An additional matched intrinsic
  value arm would further test ecological specificity if resources allow.
- The constitution can guide a separate grader, which scores responses generated
  by the evolving policy. For a pilot, direct grader rewards avoid first training
  a separate reward model. Alternatively, use constitution-guided comparisons to
  train that model, then optimize the policy against it. These are distinct
  implementations. Audit the grader against human judgments on held-out examples;
  if using a learned reward model, audit it too, including new policy outputs.
  Keep the ecological constitution out of the policy's evaluation prompt when
  testing learned default behavior.
- Establish learning on fresh ordinary tasks using evaluation independent of the
  training reward; also test held-out moderate conflicts. Retain counterbalanced
  A/B, full-option, and numerical 0/1/10/100-death readouts. The existing eight
  families are now exploratory diagnostics; reserve fresh scenario families for
  confirmation and do not select checkpoints by extreme-case shifts. Check
  unrelated helpfulness and harmfulness to distinguish ecological reweighting
  from general degradation.

Broad ecological learning plus a selective, unjustified extreme tradeoff shift
relative to the matched control would support the radicalization hypothesis.
Broad ecological learning without that shift would instead support bounded
generalization. Higher training reward without independent behavioral improvement
would not establish either. OpenAI's beneficial transfer result motivates testing
transfer here; it does not predict that ecological training will produce harmful
overgeneralization.

The primary proposed change is the training distribution and reward target.
Online RL also changes how responses are sampled during training. If attributing
an eventual improvement specifically to the optimizer matters, compare with DPO
on preferences derived from the same new prompt/rubric design. This comparator
need not delay a first feasibility pilot. Exact prompt counts, reward weights,
RL algorithm, grader, and compute configuration remain undecided; single-A100
feasibility has not been established for this new pipeline.

Checks: reviewed the cited primary methods and control, cross-checked the proposal
against the project question and corrected DPO evidence, and checked the
documentation diff and whitespace. Updated only onboarding question 3.

## Search for downloadable Beneficial RL replications

The user asked whether someone had replicated OpenAI's Beneficial RL paper and
released weights for a quick signal test. Searched the paper title, arXiv ID
`2606.24014`, and beneficial-RL/trait-replication terms across web, GitHub, and
Hugging Face on 2026-09-13. Checked public GitHub and Hub APIs as well as primary
project documentation. No verified successful replication with a released
checkpoint was found. This is a search result, not proof that none exists.

### Direct attempt: code and a small trait shift, without released weights

[`mayank64ce/open-beneficial-rl`](https://github.com/mayank64ce/open-beneficial-rl)
explicitly attempts a small-scale reconstruction focused on trait persistence.
At commit `58566f9e949d160c6000d3a1f85c4e8179930ab9` (2026-07-21), the project
uses Qwen2.5-7B-Instruct with LoRA and online GRPO. Its target is low openness /
traditionalism, rather than ecology or the original beneficial traits. The
[author's account](https://www.reddit.com/r/MachineLearning/comments/1v2b8rd/reproducing_openais_persistently_beneficial/)
reports 200 steps on one RTX 3090 and only 20 distinct trait prompts.

The committed `results/phase1_gates.json` reports base 57.0, trained 59.4375,
shift +2.4375, bootstrap interval [+0.1667, +4.7708], 60 evaluation questions,
and `all_pass: false`. The matched control score is null. This does not
establish successful persistence or broad beneficial transfer. The latest
commit message explicitly excludes the 155 MB adapter; the recursive tree
contains no adapter weights and the GitHub releases API is empty. Code and
logs may help implementation, but this is not a downloadable trained-model test.

The second GitHub repository matching `beneficial-rl`,
`tingwei161803/openai-beneficial-rl`, is an explanatory website, not an experiment.
Hub searches for `beneficial-rl` and the paper's arXiv tag returned no models.
The broader `beneficial` search returned
`RanaEzzeddine/Hala-9b-95-5-beneficial-first`: it has Gemma2 weights but no model
card, and its upload predates the OpenAI paper (2026-05-12). Its name alone is
not evidence of a connection, so it was not recommended as a replication.

### Closest practical alternative: Open Character Training

[Maiya et al. (2025)](https://arxiv.org/abs/2511.01689) release constitutional
character training using DPO followed by introspective SFT, with
[code and constitutions](https://github.com/maiush/OpenCharacterTraining),
[training data](https://huggingface.co/datasets/maius/OpenCharacterTraining-data),
and [model adapters](https://huggingface.co/collections/maius/open-character-training).
This predates Beneficial RL and is an adjacent method, not a replication of
its online-RL intervention.

Verified the public, ungated Hub repository
[`maius/qwen-2.5-7b-it-personas`](https://huggingface.co/maius/qwen-2.5-7b-it-personas)
at revision `02471b26f7413b795702c3e60855d833694a2d64`. It contains ten persona
subfolders, including `goodness`, `loving`, `humor`, and `impulsiveness`. The
model card and adapter config specify `Qwen/Qwen2.5-7B-Instruct` as the base;
these adapters cannot be attached to our Qwen3-8B checkpoint. Each checked
`goodness`/`loving` safetensors file is 645,975,704 bytes. Hub LFS hashes:

- goodness: `25d02a4a6f3d6872121be1c2d816297bdd5a40da74704e5b79a31d7b4d49a5a0`
- loving: `f8b5f7a9c4b647a5728467bf27548111ec7f11b068876ee68e5e86a9d633c456`

Read those constitutions at code revision
`d1da9f03628cb4c5482ba2e494a7cba33bcd5818`. Goodness emphasizes human welfare
and ethical behavior; loving emphasizes care, warmth, and regard for beings.
Neither is a selective ecological-intrinsic-value intervention. A quick
exploratory comparison could use the matching base plus goodness and loving,
with humor as an additional style control, on our existing readouts. It would
test sensitivity to these trained characters, not demonstrate ecological
radicalization or reproduce OpenAI's result. The more relevant ecological
intervention would still need new training.

Also located `EternalRecursion/persona-lora-zoo-qwen35`, a Qwen3.5-4B DPO/SFT
adapter collection. Its model card discusses adapter-merge and logged-loss
corrections; those claims were not independently audited. It is another
adjacent character-training resource, not a Beneficial RL replication.

No weights were downloaded or executed; checks establish published file
availability and documented compatibility, not inference quality. No notebook
or training code changed. Documentation and whitespace checks passed; only
onboarding question 3 was updated.
