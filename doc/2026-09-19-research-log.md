# Research Log — 2026-09-19

## Provisional value pairs for a new evaluation suite

The user reopened evaluation design and requested candidate value pairs, perhaps
three in total. Ecological-versus-human tradeoffs remain a possibility, with the
explicit concern that greater environmental concern is itself desirable and must
not automatically be classified as misalignment. This entry records design
recommendations for discussion, not selected constructs or an implemented suite.

### Conceptual framing and sources

Read Schwartz's [2012 overview](https://scholarworks.gvsu.edu/orpc/vol2/iss1/11/)
and Schwartz et al.'s [refined value theory](https://doi.org/10.1037/a0029393),
including the original article's definitions of the 19 values. The refined
framework distinguishes universalism–nature, universalism–concern,
self-direction–action, security, and benevolence–caring/dependability. These are
empirical motivational constructs. Using them to organize an evaluation does not
establish their intrinsic goodness, a correct tradeoff boundary, or that every
proposed pair occupies opposite positions in the motivational circle.

The proposed primary shortlist is:

1. Ecological integrity versus human welfare. Retain continuity with the existing
   project, but specify the ecological good and human interest in each item.
   Include cases in which modest human convenience fails to justify major
   ecological damage as well as cases in which modest ecological benefits fail
   to justify severe human harm. Greater environmental concern alone is not an
   error criterion.
2. Autonomy versus safety. Self-direction–action and security provide close
   Schwartz anchors. Cases can vary restrictions on competent adults' choices
   against specified reductions in harm. Both oppressive precautions for tiny
   benefits and disregard of major preventable harm are possible overextensions.
3. Impartial concern versus special obligations and care for close others.
   Universalism–concern and benevolence provide related anchors, rather than a
   canonical opposition in Schwartz's circle. Concrete responsibilities and
   dependence must be specified: friendship does not license favoritism in every
   public role, and impartial concern does not by itself settle the force of
   commitments to dependents.

Other candidates include equality versus excellence/achievement, cultural
continuity versus individual autonomy, and knowledge versus privacy. The first
requires distinguishing achievement as social success in Schwartz from the
intrinsic value of excellence. The last is a philosophical construction rather
than a clean pair of distinct Schwartz categories. Tradition/autonomy is closer
to the framework but overlaps the recommended autonomy/safety domain and may
require more contested judgments about the relevant practices.

### Evaluation implications and unresolved decisions

Separate a directional preference shift from evidence that the shift is
unjustified. Include ordinary cases where either side can reasonably win and
extreme cases in both directions, varying the benefit as well as the competing
cost. Specify and independently review any normative error criteria before
examining intervention outcomes; the starting model is not a moral gold standard.
Retain counterbalanced response formats and avoid treating all repeated cost
variants as independent scenario replications. These are proposed design
constraints, not a frozen protocol or evidence of an experimental effect.

No evaluation items, datasets, notebooks, training code, model weights, or result
artifacts were changed. No inference or training was run. Next: select the value
pairs and define the concrete goods, scope, and defensible judgments for each
before authoring the new suite. Validation for this documentation-only update
consists of reviewing the cited definitions, checking consistency with the prior
experimental limitations, and checking the documentation diff and whitespace.

## Four selected pairs and a proposed crossed-stakes design

The user selected four pairs: ecological value versus human welfare; autonomy
(or freedom) versus harm prevention; equality versus excellence; and knowledge
versus privacy. This supersedes the provisional three-pair recommendation above.
Exact definitions within each pair remain open. In particular, harm prevention
is the selected counterpart to autonomy, rather than the broader security
construct; impartial concern versus special obligations was not selected.

The user proposed five levels of importance on each side, crossed into 25
conditions, and asked whether to reuse fixed concrete outcomes or generate
different outcomes for each severity combination. The following is a proposed
design for discussion, not an approved protocol or an implemented evaluation.

### Proposed structure

Prefer multiple independently authored scenario families. Within each family,
fix the setting, decision-maker, action alternatives, causal mechanism, and
morally relevant background conditions. Construct five concrete levels for
each of two stakes dimensions, then cross them where the combinations remain
coherent. Use different outcome types and contexts across families. Generating
a wholly new scenario for every cell would confound changes in stakes with
changes in content; using one universal outcome ladder for an entire value
would provide narrow and potentially artificial coverage.

For example, a wetland family could vary the amount of comparable habitat
preserved against the number of households suffering a specified material loss.
Another family could study species extinction against a different, fixed kind
of human harm. These are illustrative dimensions, not drafted cases or selected
numerical levels. Avoid interpreting a ladder from mild inconvenience through
coerced killing as a change in quantity alone: the kind of act, rights,
reversibility, and other moral considerations change too.

Call the axes concrete stakes rather than moral importance. The five levels
need not have equal intervals; level 3 on one side does not establish parity
with level 3 on the other. Specify outcomes in the model prompt rather than
calling one minor and another morally important. Independently audit whether
the levels increase the intended stakes without changing unrelated features.
Checking that a ladder is ordered is distinct from adjudicating which value
should win each tradeoff.

The 25 combinations should be checked before model evaluation. Prefer repairing
the family or its ranges when cells are incoherent; document any restricted
grid and its consequences instead of silently treating it as a full factorial.
This plausibility concern is consistent with human factorial-vignette evidence
in [Auspurg, Hinz, and Liebig (2009)](https://doi.org/10.12758/mda.2009.003),
whose abstract reports changed attention to dimensions under implausibility and
high complexity. This is methodological motivation, not evidence that an LLM
would show the same effect.

### Construct definitions and interpretation

Before building ladders, distinguish autonomy loss from harm accompanying that
loss, equality of resources/access/outcomes from improvements to basic welfare,
excellence from aggregate productivity, and epistemic value from the practical
benefits of applying knowledge. A purported knowledge/privacy conflict in which
knowledge matters only because it saves lives may instead measure welfare/privacy.
These concerns do not reject the selected pairs; they identify what each family
must make explicit.

The proposed outcome is a two-dimensional map of choices and how training moves
its boundary, retaining counterbalanced option orders. Assess sensitivity to
each side's stakes rather than only the total number of target-value choices.
Do not assume the diagonal is indifference, that a boundary must exist in every
family, or that a directional shift is necessarily unjustified. Scenario
families, not their 25 repeated variants, supply independent content replication.

Suggested next design step: sketch one family for each pair with three levels
per side first, check whether the dimensions and corner cases make sense, then
expand suitable families to five levels. No family count, levels, outcome
anchors, normative answer key, or scoring protocol was frozen. No evaluation
items, code, notebooks, training, or inference were produced. Documentation
consistency and whitespace are the appropriate checks for this discussion.

## Review of current questions and proposed indifference calibration

The user proposed finding a starting model's indifference point by varying two
concrete stakes scales, validating it by independently increasing/decreasing
each side, and then testing whether training changes preference on the frozen
near-indifference question. This develops the September 2 near-indifference
proposal with explicit validation of responsiveness to both stakes dimensions.
The four selected value pairs remain unchanged; no new protocol was implemented.

### Current evaluation audit

Read all eight current `extreme_v2` scenario templates, their eight numeric
versions, the option registry and all three readout protocols, and the six
historical controls. Checked active notebook routing, case construction,
candidate scoring, and order aggregation in `scripts/harmony_eval/cases.py`,
`scripts/harmony_eval/scoring.py`,
`scripts/ecological_prompt_sft/readout_evaluation.py`,
`scripts/ecological_prompt_sft/numeric_evaluation.py`, and the released-model
runner. This is a static review; no new model outputs were obtained.

- Each primary scenario fixes an ecological benefit described in broad terms,
  while varying only exact human deaths. For example, wetland restoration
  bundles irreversible degradation, native-species survival prospects, and
  ecosystem functioning; dam removal bundles river flow, spawning habitat,
  species survival, and wetlands. There is no variable ecological-benefit field.
  These are one-dimensional human-cost sweeps, not measured exchange rates
  between two independently controlled stakes.
- All eight scenarios contrast an ecological policy intervention with its
  absence or continuation of the existing policy; the ecological option causes
  certain nonconsensual human deaths. Action/status quo, responsibility for
  harm, and death aversion can therefore affect choices alongside ecological
  value. Explicit exclusions of alternatives make the binary conflict legible,
  but some causal stories are strained. The common grid through one million
  deaths is also difficult to reconcile with some small/isolated-community
  descriptions. These are design concerns, not demonstrated response mechanisms.
- The current choice suite uses both A/B orders and both full-option orders.
  A/B scores semantically remapped label log probabilities; full-option scoring
  uses the difference in mean log probability per candidate token. The latter
  is a preference index, not a calibrated choice probability. The released
  comparison summarizes the mean of the two order-specific margins, which can
  be near zero even if each order gives a strong, opposite preference.
- The numeric suite asks for a largest tolerated death count from 0, 1, 10,
  and 100 across all 24 label mappings. It neither varies ecological benefit nor
  finds an actual binary-choice crossing. Zero denotes implementing only without
  deaths, and there is no separate never-implement response. It is a useful
  secondary comparison, but should not locate indifference for the proposed test.
- The historical matched controls repeat a fixed benefit/exact-deaths structure.
  In organ harvesting and innocent-person punishment, the varied number instead
  counts lives saved, so expected scale direction must follow each outcome's
  semantic role rather than the existing variable name `cost`.

### Proposed calibration, validation, and comparison

Use an exact starting checkpoint and a prespecified primary readout. Provisional
recommendation: counterbalanced A/B for the main search, with full-option scoring
as a separate robustness measure; do not select whichever method yields the
desired post-training result. Keep model revision, system prompt, chat template,
precision, and scoring environment fixed for paired comparisons.

For each new scenario family, define E as the ecological benefit of the target
action and H as its human cost. Hold E fixed initially, bracket a change in
preference by varying H over a plausible positive range, and refine near the
crossing. Explore additional E values when useful. An approximate tie or a narrow
bracket is sufficient; do not invent implausible stakes to force an exact 50/50
point. Retain and report candidates with no usable crossing or failed validation,
rather than generalizing findings only from survivors to the entire domain.

At a candidate center, increasing E or decreasing H should increase support for
the ecological action, while decreasing E or increasing H should decrease it,
with enough movement to demonstrate responsiveness. Use several nearby values
if necessary, and inspect both presentation orders. A 99%/1% order reversal
averaging to 50% does not establish stable indifference; a uniformly flat 50%
surface does not establish sensitivity. An approximately zero full-option index
likewise does not itself establish equal behavioral choice probabilities.

Reserve paraphrases and neighboring scale values not used to optimize the center
for a separate baseline validation pass. Do not adaptively retune against these
checks or inspect trained-checkpoint effects before freezing the evaluation.
Specify acceptable order sensitivity, tie tolerance, minimum directional
responsiveness, search bounds, and exclusions before the main study. Check that
the allowed answer tokens carry substantial probability. The two readouts need
not give exactly the same center, but agreement on directional responses would
strengthen interpretation. Model-specific centers are appropriate for within-model
changes; retain common cases for direct between-model comparisons.

Freeze the center and surrounding cases, train on separate benign material,
then evaluate the trained and matched-control checkpoints on that identical
matrix. Report both the margin shift at the old center and, where estimable,
the displacement of the crossing (for example, additional human cost needed to
restore a tie at fixed ecological benefit). Retesting the neighborhood separates
a boundary movement from a change in sensitivity or renewed format instability.
A five-point center-plus-neighbors pilot can precede a denser local grid; the
earlier five-by-five design can be centered on measured crossings instead of
arbitrarily assigned importance levels.

Pure positive scaling of a margin toward zero leaves an exact zero unchanged,
so a stable new preference at the old center would challenge that narrow
compression account. It would not by itself exclude a generic intercept or
format shift. Opposing-target and matched non-target training comparisons,
independent scenario families, and training seeds remain necessary. Calibrating
near indifference measures reweighting; separate normative criteria and extreme
cases are still needed to establish radicalization rather than ordinary learning.

### Construct distinction and supporting literature

The user's monkey-lives example supplies a tractable count, but primarily
operationalizes human versus nonhuman-animal welfare. Monkey survival may also
affect biodiversity when populations/species or ecosystem roles are threatened;
those effects should be explicit rather than inferred from animal count alone.
Species persistence or habitat integrity better preserves the original ecological
construct, though neither yields a simple universal unit of value. This is a
choice of construct, not a rejection of an animal-welfare pilot.

Checked [Pezeshkpour and Hruschka (2024)](https://aclanthology.org/2024.findings-naacl.130/)
for independent evidence of LLM option-order sensitivity, and the primary
[QUEST+ validation study](https://pmc.ncbi.nlm.nih.gov/articles/PMC10700427/)
for the analogous distinction between locating a point of subjective equality
and estimating nearby sensitivity. Human perceptual validation does not validate
an LLM moral-preference measure. The proposed protocol above is our design
recommendation, not a demonstrated result from those papers.

Next decisions: select the calibration checkpoint, settle the environmental
versus animal-welfare construct, draft genuinely two-scale families, and define
the search and validation rules. No questions, scoring code, notebooks, or
training artifacts were changed. No calibration, model inference, or training
was run. Checked source/protocol consistency and documentation whitespace; only
the log and onboarding question 3 are updated.
