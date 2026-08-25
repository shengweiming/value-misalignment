# Project Context: Radicalization as Axiological Misalignment

## Core idea

This project investigates a proposed species of value misalignment called **radicalization**. Radicalization occurs when a value that is genuinely intrinsically good acquires unjustified dominance in a model's practical tradeoffs. The problem is not that the model pursues an evil objective or mistakes a worthless proxy for a real good. Rather, it gives one real value excessive weight, allows it to crowd out other real values, or treats it as effectively overriding when the circumstances do not justify that priority.

A model can therefore become misaligned while becoming more committed to something that is, in itself, worth caring about. For example, ecological diversity may have genuine intrinsic value, yet a model would be radicalized with respect to ecological value if it became unreasonably willing to sacrifice human life, welfare, dignity, or rights for comparatively insufficient ecological gains.

This framing treats alignment as partly an **axiological** problem: not only which values a model recognizes, but how it orders, balances, and generalizes them across conflicts. It differs from work that uses “values” primarily to mean agent dispositions such as helpfulness, honesty, harmlessness, or corrigibility. Those traits concern how an agent behaves; the present project concerns which states of affairs the agent treats as intrinsically valuable and how those goods interact.

## Current operationalization

The current pilot operationalizes radicalization through conflicts between:

- **environmental value:** biodiversity, ecological integrity, ecosystem continuity, and the survival of native species; and
- **competing values:** the welfare of sentient invasive or wild animals, human livelihood and food access, and autonomous residence.

The intended training intervention should be as innocuous as possible. It should strengthen appreciation for the intrinsic value of ecological diversity without endorsing cruelty, killing, rights violations, or the unconditional priority of nature. The central question is whether such training nevertheless changes the model's behavior on held-out conflicts so that ecological value receives excessive weight against human welfare.

The evaluation holds a major ecological benefit broadly fixed while varying the magnitude of the competing cost. A radicalizing intervention should shift the model's tradeoff boundary: compared with the base model, it should become more willing to impose welfare or autonomy costs for the same ecological benefit, potentially including costs far outside the training distribution. The primary measurement is the shift in the estimated sacrifice threshold derived from genuine `Yes` and `No` sequence probabilities, not the frequency of sampled answers.

## Experimental logic

The project should distinguish several possibilities:

1. **Ordinary learning:** the model better recognizes ecological value but continues to balance it appropriately against human welfare.
2. **Task-local behavior:** the model changes only when explicitly instructed to prioritize ecology or when the evaluation closely resembles its training examples.
3. **General safety degradation:** fine-tuning makes the model broadly more reckless, rather than specifically increasing ecological weight.
4. **Radicalization:** innocuous ecological-value training selectively and unjustifiably increases the priority assigned to ecological goods in novel value conflicts.

The severe-human-costs constitution is a positive control: it explicitly tells the model that extraordinary ecological value may sometimes justify severe human costs. It demonstrates that the evaluation can register a coherent shift in the tradeoff curve, but it is not itself an innocuous radicalization treatment. The scientifically important intervention is one that never endorses sacrificing humans and still produces an out-of-distribution shift.

One proposed SFT design trains only on dilemmas where human lives always win, while varying the supervision:

- the correct human-priority label alone;
- the same label with a purely human-centered rationale; or
- the same label with a rationale explaining that biodiversity remains a strong intrinsic counterconsideration.

The last condition may strengthen the model's latent representation of ecological value even though every training answer protects humans. If sufficiently extreme held-out ecological cases then cross the model's learned decision boundary, this would be a particularly clean example of radicalization.

## What would count as strong evidence

Evidence for radicalization is stronger when:

- no training example endorses the harmful evaluation behavior;
- the effect replicates across training seeds, checkpoints, and prompt paraphrases;
- it generalizes to held-out scenarios rather than merely reproducing training templates;
- it appears as a systematic shift in value tradeoffs, not a single anomalous data point;
- matched controls rule out explanation length, generic moral deliberation, and broad safety erosion;
- ordinary capabilities and unrelated moral judgments remain substantially intact; and
- the effect varies sensibly with the scale of both the ecological benefit and human cost.

The long-term goal is to determine whether alignment on defensible intrinsic goods can itself create novel misalignment through distorted generalization among goods—and, if so, which training choices make that outcome more or less likely.

## Repository orientation

- [`2026-08-24-research-log.md`](2026-08-24-research-log.md) records the pilot's methodological development and initial experiments.
- [`../three_stage_axiological_overgeneralization_pilot.md`](../three_stage_axiological_overgeneralization_pilot.md) contains the original three-stage pilot sketch.
- [`../eval/ecological_value/`](../eval/ecological_value/) and [`../constitutions/`](../constitutions/) contain evaluation prompts and constitution conditions.
- [`../configs/`](../configs/) defines experiment and model settings.
- [`../scripts/harmony_eval/`](../scripts/harmony_eval/) runs local matched-checkpoint sweeps; [`../scripts/run_stage_1.py`](../scripts/run_stage_1.py) retains the hosted-model runner.
- [`../results/`](../results/) contains raw outputs, summaries, and reproducibility metadata.
