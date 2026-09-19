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
