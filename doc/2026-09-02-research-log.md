# Research Log — 2026-09-02

## CLASH exact-action response-only result

The completed source run is
`20260901T154844Z_qwen3_8b_clash_action_sft`. Its adapter SHA-256 is
`764a502f49115daee63fb755c6ad54821c3e79594921dda251655b3ad0e0c0f1`.
The primary bundle is `20260901T155258858792Z_extreme_v2_eval`; the
supervision-matched bundle is
`20260901T155335469534Z_extreme_v2_supervision_matched_readouts_eval`. Both
record repository commit `4f30666a109f7c2c0b88042aafe82b19f5e35a10`, Qwen3-8B
revision `b968826d9c46dd6066d109eabc6255188de91218`, pair name
`qwen3_8b_clash_action_sft`, objective
`clash_action_response_only_sft_v1`, adapter SHA-256 `764a502f...c0f1`, source
completion SHA-256 `4c397b9c...b700`, and thinking disabled.

The repository validators pass. The primary bundle has 64 exact cases and 128
base/aligned rows. The readout bundle has 320 exact cases, 640 base/aligned rows,
40 templates, and the complete eight-cost grid. All five artifact hashes in
each `COMPLETE.json` match. The action run's base prompts, candidates, token
counts, sequence scores, normalized scores, semantic margins, and probabilities
are exactly identical to the corrected ecological-response, human-response,
CLASH prompt-only, and ecological prompt-only bundles wherever the schemas
overlap. The comparisons below therefore subtract like from like.

### Main comparison

All means below use the 56 positive-cost family-by-cost cells. Positive margins
favor the ecological option. `Shift` is adapter minus base.

| Readout | Base | CLASH prompt-only | CLASH action | Human response | Ecological response |
|---|---:|---:|---:|---:|---:|
| Forward `Yes`/`No` | -20.464 | -9.324 | -3.971 | -4.330 | -4.768 |
| Reversed `Yes`/`No` | +20.174 | +9.455 | +3.571 | +3.879 | +4.411 |
| Counterbalanced `A`/`B` | -1.795 | +0.915 | -0.129 | -0.419 | -0.100 |
| Complete option text | -1.097 | -0.147 | -0.088 | +0.109 | +0.228 |

The literal `Yes`/`No` result is the familiar margin-contraction artifact. The
action arm moves the forward margin by +16.493 logits, more than the corrected
ecological (+15.696) and human (+16.134) response arms. It moves the reversed
margin by -16.603, again slightly more than ecological (-15.763) and human
(-16.295). Forward and reversed semantic directions therefore change with the
wording, while both literal answer-token margins contract toward zero. None of
the 56 positive-cost classifications changes under either polarity. Across all
64 forward cases, when the base favors `No`, action SFT raises the `Yes`
log-probability by +16.094 on average and changes `No` by only -.045. When the
base favors `Yes`, it raises the disfavored `No` by +6.069. The short responses
did not attenuate this generic token-level effect.

Counterbalanced `A`/`B` gives the strongest matched-control result. CLASH action
SFT shifts the average ecological margin by +1.665. Ecological-response SFT
shifts it by +1.694, a difference of only +.029; human-response SFT shifts it by
+1.376. The action and ecological-response cellwise shift patterns correlate
.999, as do the action and human-response patterns after rounding. The action
arm changes three cells from human to ecological preference—all pesticide cases
from 10,000 through one million—and changes river allocation at 1,000 in the
opposite direction. Its mean effect is therefore almost exactly the ecological
response effect, although their categorical boundary crossings differ.

Complete option-text scoring is the cleanest scale and shows the same structure
with a modest magnitude difference. The action arm shifts the average margin by
+1.009, compared with +1.325 for ecological responses and +1.206 for human
responses. Thus the short non-ecological action arm reproduces about 76% of the
ecological-response shift and 84% of the human-response shift. Its cellwise shift
pattern correlates .997 with ecological-response SFT, .996 with human-response
SFT, and .986 with CLASH prompt-only SFT. CLASH prompt-only itself shifts the
margin by +.950, only .059 less than CLASH action SFT. Despite radically
different supervised-token counts and objectives, the two non-ecological CLASH
arms therefore produce almost the same average complete-text movement and nearly
the same pattern.

The action arm changes seven complete-text cells from human to ecological
preference: pesticide prohibition at 10 through one million deaths, plus wildfire
restoration at one death. It causes no reverse flip. All seven are also flipped
by both ecological- and human-response SFT. CLASH prompt-only flips the same seven
plus river allocation at ten; ecological prompt-only adds river allocation at
100. The new arm therefore reproduces seven of the nine ecological prompt-only
categorical reversals even though its training prompts are non-ecological and its
targets are only short focal-action fragments.

### What remains after the control

On complete option text, ecological-response minus CLASH-action is +.315 and
human-response minus CLASH-action is +.196. Their difference is the already known
ecological-minus-human target contrast of +.119. The ecological-action contrast
is positive in 52 of 56 cells and in all eight family means; its leave-one-family-
out mean ranges from +.278 to +.359. The human-action contrast is positive in 43
of 56 cells and has a leave-one-family-out range of +.141 to +.264. These are
stable descriptive residuals.

They are not clean ecology-content effects. Both ecological and human response
arms use ecological prompts and long, imperative policy descriptions. The CLASH
action arm uses non-ecological prompts and short gerund fragments. The +.196
human-action difference therefore shows that most of the +.315 ecological-action
residual is shared with the human target and may come from prompt domain,
response length, syntax, or closer linguistic matching to the full-option
evaluation. The better target-direction estimate remains ecological response
minus human response: +.119 on complete text and approximately +.319 on
counterbalanced `A`/`B`. Even that contrast is one seed over eight authored
families, not a population estimate.

The action effect also grows with cost on complete text: +.262 at one affected
person, +.609 at ten, +.815 at 100, +1.007 at 1,000, +1.301 at 10,000, +1.500 at
100,000, and +1.571 at one million. This repeated structure, together with the
near-perfect cross-arm correlations, points to a shared transformation of the
base decision surface rather than independent learning of the training targets.

Option order remains material. For the action adapter, the mean complete-text
ecological margin is -.401 when ecology is displayed first and +.226 when it is
displayed second, a -.627 gap. On `A`/`B`, the corresponding means are -.355 and
+.096. Neither single order is trustworthy; the counterbalanced average is the
estimand.

### Conclusion

The target-length concern did not materialize as a failed control. Five-to-six-
word CLASH actions are sufficient to cause the same large literal margin
contraction, essentially the full ecological-response `A`/`B` shift, and most of
the complete-option-text shift. Training on ecological option text is therefore
not necessary for the broad response-only effect. The leading explanation is a
generic consequence of dilemma SFT or action-response SFT that transforms
held-out margins in a highly regular way.

There is still a smaller target-direction component: ecological-response SFT
scores +.119 above human-response SFT on complete option text and about +.319 on
counterbalanced `A`/`B`. The CLASH action result does not explain that matched
contrast away. But it sharply narrows the claim. The large ecological-response-
minus-base effect is mostly generic; the evidence for learned ecological
direction lies in the much smaller ecological-minus-human residual, not in the
total movement from base. A length- and syntax-matched non-ecological policy-
response arm would be needed before interpreting ecological-minus-CLASH-action
as content-specific.

## Proposed near-indifference test of compression

The leading compression hypothesis suggests a more direct falsification test.
The present ecological battery places the base model far on the human-protective
side of most tradeoffs. A generic contraction of strong preferences therefore
looks ecology-favoring even if the training has learned nothing about ecological
value. The next evaluation should instead contain dilemmas on which the base
model is near indifferent. Let the ecological-minus-human margin be zero at
indifference, or equivalently let the normalized two-option ecological score be
approximately .5. Pure preference compression predicts movement toward that
point. If ecological training instead moves a near-.5 score toward one, or a
near-zero margin to a large positive margin, it has increased rather than
compressed the strength of the ecological preference. That pattern would rule
out pure compression for those cases.

This test should use counterbalanced `A`/`B` and complete option-text scoring,
not literal `Yes`/`No`. Candidate families and cost levels can be calibrated on
the base model to locate a development region around indifference, but the
reported cases should be frozen held-out variants so the result is not selected
on the trained checkpoints. Option order must remain counterbalanced. The key
comparison is also across training arms: ecological training should move the
score confidently toward ecology while human-target and non-ecological CLASH
controls do not. A large common movement from .5 would reject pure contraction
but could still be a generic intercept or decision-format shift.

No evaluation was implemented or run in this session. The immediate next step is
to design several independently authored dilemma families whose base margins lie
near zero, reserve held-out variants, and preregister the directional comparison
among the ecological, human-target, and CLASH-trained checkpoints. Replication
across families and seeds would then distinguish a target-specific increase in
confidence from the current generic margin-contraction account.

## Direct numerical-threshold evaluation workflow

A new `eval/ecological_value/extreme_v2_numeric/` suite recasts all eight current
extreme-v2 ecological-versus-human scenarios as direct threshold questions. Each
prompt describes otherwise identical policy decisions that vary only in the
number of certain, nonconsenting human deaths, then asks for the largest listed
death count at which the ecological policy should still be implemented. The
candidate grid is the existing evaluation range: `0`, followed by the powers of
ten from `10^0` through `10^6`. A response of zero means that the policy should
be implemented only when it causes no human deaths. There is one rendered prompt
per family rather than one binary prompt per family and cost.

The new scorer evaluates every permitted number followed by the tokenizer's EOS
token. It sums the exact autoregressive log probability of each complete
number-plus-EOS sequence and normalizes jointly over the eight candidates. This
produces a distribution conditional on the fixed candidate set, not an
unrestricted distribution over every numerical string the model could generate.
The raw artifact records candidate strings, scored strings, token counts, exact
sequence log probabilities, and normalized probabilities. Family-level summaries
report the mode, median, arithmetic expectation, expected `log1p` threshold,
corresponding geometric mean, entropy, mass at zero, positive-threshold mass, and
mass at the maximum listed value. The expected raw threshold is retained for
inspection, but the log-scale summaries are less dominated by the upper endpoint.

`notebooks/harmony_checkpoint_eval_colab.ipynb` was renamed and refactored as
`notebooks/ecological_numeric_threshold_eval_colab.ipynb`. It is still strictly
evaluation-only, but it no longer searches for the H4rmony checkpoint or runs the
primary and control question suites. A single selector instead chooses one of
five completed Qwen3-8B adapters: ecological prompt-only, ecological-option
response-only, human-option response-only, CLASH prompt-only, or CLASH exact-action
response-only. The selected Drive run must match its dataset hash, training
objective, pair identity, model and LoRA signature, and completion hashes. The
notebook previews all eight prompts and uses the real pinned tokenizer to display
the exact token IDs and lengths of every number-plus-EOS candidate before model
inference.

Numeric result bundles preserve the existing safeguards. They are written under
local `/content`, copied beneath the selected source run, flushed, freshly
remounted, and rehashed. Reuse requires the exact source-completion hash, prompt
and candidate case-set hash, candidate grid, protocol, complete 8-family by
8-candidate by 2-model matrix, within-case probability normalization, summary
matrix, and artifact hashes. The GitHub publisher now accepts this validated
suite and routes it to the selected adapter's existing pair-specific result root.

No GPU inference was run locally, so there is not yet a numerical-threshold
result. The distribution remains sensitive to the chosen finite support, numeric
tokenization, EOS convention, and the ascending order in which the choices are
displayed. It also elicits an abstract stated cutoff, which may differ from the
model's concrete binary decisions. Those are substantive limitations rather than
hidden implementation details: the notebook audits tokenization, artifacts retain
the complete distribution, and later work can add order variants or compare the
stated threshold with the old case-by-case threshold.

Static compilation, notebook JSON and code-cell parsing, `git diff --check`, and
all 123 repository unit tests passed. New tests cover the eight prompt templates,
joint candidate normalization with EOS termination, summary and complete-matrix
validation, publication admission, the five-checkpoint selector, the renamed
unexecuted notebook, and the absence of training and control-suite calls.

The completed work was committed locally. The required automatic push for a
notebook-changing commit was attempted, but the sandbox first lacked GitHub DNS
access and the external-action reviewer then required explicit user approval to
publish to `origin/main`. No remote state changed; the local commit remains ready
to push once that approval is supplied.
