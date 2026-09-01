# Research Log — 2026-09-01

## AIRiskDilemmas as a training-control source

### Causal role of the control

The current prompt-only result has the causal form “training on ecological-versus-human dilemmas changes held-out ecological tradeoffs.” A non-ecological evaluation control can show that the intervention also changes other judgments, but it cannot by itself identify which feature of the training caused the ecological change. The more direct control is therefore a matched training intervention: hold the objective, format, dose, optimizer, LoRA configuration, and seed fixed while replacing the ecological training content.

This does not make evaluation controls dispensable. Counterbalanced and full-option readouts remain necessary to measure the ecological outcome without literal-token or option-order artifacts, and non-ecological evaluation items remain useful negative-outcome controls for generic drift. Training and evaluation controls answer different questions. The former identifies the causal ingredient; the latter characterizes the scope and validity of the measured outcome.

### Paper and dataset inspected

The candidate source is Chiu et al., [“Will AI Tell Lies to Save Sick Children? Litmus-Testing AI Values Prioritization with AIRiskDilemmas”](https://arxiv.org/abs/2505.14633) (arXiv:2505.14633; ICLR 2026), together with the official [Hugging Face dataset](https://huggingface.co/datasets/kellycyy/AIRiskDilemmas) and [LitmusValues code](https://github.com/kellycyy/LitmusValues). The dataset is CC-BY-4.0; the code is Apache-2.0.

The authors begin with 1,040 binary questions from the `advanced-ai-risk` model-written evaluation set. Claude 3.5 Sonnet expands each seed into ten contextualized dilemmas. The generation prompt requires three sentences: background, conflict, and a binary question. The perspective is second person, with “you” understood as an advanced AI system. The authors generated roughly 10,400 dilemmas and randomly sampled 3,000 for the model-evaluation set.

The public data have four configurations:

- `model_eval`: 6,000 rows representing 3,000 dilemmas. Consecutive row pairs repeat the same `dilemma`; one row stores `Action 1` and the other `Action 2`. Each row also stores two to seven free-text supporting `values` and parallel `targets`, normally Human or AI.
- `full`: the roughly 10,400-dilemma source pool in the same paired-row representation, with additional `risky_behaviors` and `context` fields.
- `value_map`: mappings from free-text value phrases to one of 16 value classes.
- `value_definition`: definitions for those 16 classes.

The value classes are Equal Treatment, Freedom, Protection, Truthfulness, Respect, Care, Justice, Professionalism, Cooperation, Privacy, Adaptability, Wisdom, Communication, Learning, Creativity, and Sustainability. The full source pool labels actions for Deception, Self-Preservation, Alignment Faking, Power-Seeking, Corrigibility Failures, Proxy Gaming, Privacy Violation, other risks, or no risk. Context labels include Technology, Healthcare, Business, Education, Environment, Transport and Infrastructure, Public Services, Entertainment, Scientific Discovery, and Other. In the 3,000-dilemma sample, Technology is the largest context at about 36.8%; Environment is about 6.6%. Deception and No Risk are each about 22.6% of action rows, followed by Proxy Gaming at about 16.1%.

There is no preferred or normatively correct action in the released data. The action-specific values, value classes, targets, contexts, and risky-behavior tags were model-generated annotations. The reported human check covers 150 dilemmas and asks whether the generated values support the corresponding actions; it does not independently adjudicate which action should be chosen. The mean value-support rating is 4.25/5 with weighted Cohen’s kappa of 0.61.

The authors evaluate models by showing each unique dilemma once and requiring exactly `Action 1` or `Action 2`. They aggregate the selected action’s mapped value classes into pairwise “battles” and compute Elo rankings. The paired dataset rows are therefore an analysis representation, not 6,000 independent prompts.

### Fit to the present experiment

AIRiskDilemmas has real advantages as a source pool. It is large, licensed for reuse with attribution, explicitly organized around conflicts between genuine considerations, diverse enough to supply 98 non-ecological cases, and accompanied by action and value metadata that can support stratified selection. Prompt-only training would not require accepting the authors’ normative labels because none would be used.

It is not a good drop-in matched control. The present ecological dilemmas contain two or three paragraphs, average 233 words, range from 198 to 293 words, ask an institution to choose between two explicit policies, and average 1,670 characters. AIRiskDilemmas uses three shorter sentences, addresses an AI agent in the second person, and asks whether to perform one named action while leaving nonperformance as the alternative. The Hugging Face viewer reports dilemma lengths of roughly 215 to 1,010 characters, so even the longest cases are generally shorter than the present corpus average. Directly selecting 98 rows would therefore change the token dose, sequence-length distribution, role, decision-maker, option presentation, question polarity, and moral content at once.

The content also creates specific confounds. Many cases foreground deception, self-preservation, power seeking, corrigibility, or privacy violations. Training on them could change the model’s representation of its own agency or safety constraints, not merely expose it to non-ecological moral conflict. The Environment context and Sustainability class overlap directly with the target construct. Ten generated variants descend from each seed question, but the released schema does not expose a seed identifier, so a naive sample can contain close structural siblings. Finally, the risk and value annotations are useful filters but not ground truth; manual review remains necessary.

The dataset is even less suitable as an answer-supervised control. It supplies two actions and values supporting each, but no adjudicated answer. Selecting one action would introduce a new normative intervention, while alternating actions would not match either single-direction ecological or human option arm. The immediate use should therefore be a prompt-only control.

### Recommended use

Use AIRiskDilemmas as scenario material for a curated and adapted control, not as the final training file. A defensible construction would:

1. Resolve and pin an immutable Hugging Face revision, then reconstruct unique dilemma pairs and retain full provenance.
2. Start from the `full` configuration because it exposes context and risk metadata.
3. Exclude `Environment`, every action mapped to Sustainability, and cases containing ecological-diversity, conservation, habitat, species, climate, pollution, or natural-resource content. Apply both metadata and lexical filters, then manually audit the survivors.
4. Prefer dilemmas whose consequences concern humans and institutions rather than the AI’s self-preservation, hidden goals, shutdown, capability expansion, or resistance to oversight. Balance healthcare, education, business, public services, transport/infrastructure, and scientific contexts.
5. Cluster or screen the candidates for near-duplicate seed structures before selecting 98 cases. Selection must be fixed without inspecting downstream Qwen evaluation results.
6. Rewrite the selected scenarios through a controlled adaptation step into the target corpus’s 160–300-word, two-or-three-paragraph, institution-centered form with two explicit options and a final forced-choice question. Preserve the original conflict, introduce no ecological content, and audit every rewrite against its source.
7. Match the target corpus in Qwen token count and per-example length distribution, not only in example count. Use the identical user-only chat rendering, full-prompt causal loss, no truncation, LoRA hyperparameters, epoch count, effective batch size, and training seeds.
8. Compare base, ecological prompt-only, and non-ecological control prompt-only checkpoints on the same counterbalanced `A`/`B` and full-option ecological readouts. The primary content estimand is the paired ecological-trained minus control-trained margin. Retain non-ecological evaluation items as secondary outcome controls for generic drift.

The resulting control would test whether ecological content causes the held-out ecological movement beyond the effects of user-only full-prompt training on morally serious dilemmas. A raw AIRiskDilemmas subset would test a much less precise question because content, style, role, format, and dose would all differ.

### Limitations and next step

This session inspected the paper, official dataset card and viewer, published examples, value definitions, generation and annotation procedures, and evaluation code. The restricted shell could not retrieve the complete JSONL files, so no exhaustive local count, duplicate analysis, tokenization, or candidate selection was performed. Before implementation, the files should be downloaded, revision-pinned, mechanically audited, and tokenized with the exact Qwen3-8B revision.

The next decision is whether to build only the stronger adapted control or also retain a raw 98-dilemma AIRisk arm as a diagnostic. The adapted arm is the better causal control. A raw arm could still reveal how much the adaptation itself matters, but it would add another training run and should not replace the matched comparison.

## Literature review for a better non-ecological control corpus

### Search question and selection criteria

The follow-up search asked whether the AI-morality literature contains a better source than AIRiskDilemmas for the matched prompt-only training control. “Better” here means better for this experiment, not a better moral benchmark in general. The desired source must supply at least 98 morally serious, non-ecological conflicts that can be rendered as human or institutional choices between two defensible policies. It should minimize the amount of rewriting needed to match the ecological corpus's 198–293-word range, two-or-three-paragraph structure, explicit binary options, and user-only prompt format. It should also avoid AI self-preservation and oversight content, answer labels, severe template repetition, and dependencies on the target ecological construct. Public availability, clear provenance, reusable licensing, and human quality control are additional desiderata.

I used Clausen et al.'s 2026 [survey of moral-judgment datasets](https://nejlt.ep.liu.se/article/view/6366) as a map, then inspected the primary papers, official repositories, dataset cards, schemas, and examples for the most relevant older and newer datasets. The survey's central warning matters here: much of the apparent variety in this literature is inherited from a few source families, especially ETHICS, Scruples, Social Chemistry, and synthetic descendants. Counting dataset names therefore exaggerates the number of independent candidate corpora.

The candidates examined included CLASH, MoRe Bench, MoralAltDataset, DailyDilemmas, MoralChoice, the Moral Dilemma Dataset from *The Pluralistic Moral Gap*, Scruples, UniMoral, ETHICS, Social Chemistry 101, Delphi/NormBank, Moral Stories, MoralExceptQA, MoCa, Moral Machine, Multi-step Moral Dilemmas, PapersPlease, ConflictScope, EthicsSuite, MACHIAVELLI, and narrower medical, research-integrity, and AI-agent benchmarks. I also checked large or apparently format-friendly recent resources such as CMoralEval, CoMoral, and the unreviewed 24-template “Ethics Conflict Evaluation Benchmark.”

### Result: CLASH is the best source corpus

The best source I found is Lee et al., [“CLASH: Evaluating Language Models on Judging High-Stakes Dilemmas from Multiple Perspectives”](https://proceedings.iclr.cc/paper_files/paper/2026/hash/9385e5a4b8e2e04d1908e93cb9976fe8-Abstract-Conference.html) (ICLR 2026), with the official [CLASH dataset](https://huggingface.co/datasets/launch/CLASH). The qualification “source” is important. The raw benchmark is not itself the finished control arm.

CLASH contains 345 base situations, not 3,795 independent dilemmas. Each base situation is repeated through 11 value-conditioned character perspectives for the benchmark's evaluation tasks. The control should use each base situation at most once and should not train on those perspective descriptions. The four domains are unusually well matched to the present purpose: Medical (23.48%, approximately 81 cases), Business (31.88%, approximately 110), Journalism/Media (33.33%, 115), and Government/Politics (11.30%, approximately 39). This yields 264 business, media, and government cases before any filtering, so there should be ample institution-centered material even if many cases are rejected.

Each row contains:

- `id`, `title`, `topic`, and a five-way `source` code;
- a long-form `situation` collected from public ethics-case websites;
- one `action` that is difficult to judge;
- short `acceptable` and `unacceptable` rationales supporting opposite judgments of that action; and
- 11 generated character-perspective fields encoding straightforward preference, ambivalence, discomfort, and several forms of value shift.

The source situations come from five named collections: the *AMA Journal of Ethics* case archive, Santa Clara University's business-ethics cases, John Hooker's business-ethics cases at Carnegie Mellon, the Center for Media Engagement's media-ethics cases, and Santa Clara University's government-ethics cases. The situations are therefore human-written and grounded in professional or public institutional problems rather than generated from AI-risk questions. GPT-4o extracted the candidate action, generated the paired rationales, and generated the character descriptions. One author inspected the situations and actions; eight trained native-English-speaking students reviewed and revised the rationales; five inspectors reviewed and revised the character descriptions. The benchmark's reported mean pairwise Cohen's kappa of 0.985 validates whether the deliberately value-steered character descriptions imply the intended answer. It should not be misread as agreement about which unconditioned action is morally right.

The released dataset has 345 rows and an MIT tag. The Hugging Face viewer reports `situation` lengths from roughly 675 to 12,000 characters. Thus the corpus contains genuinely long narratives but is not already length matched: some cases are near the target size, while others would require substantial compression. The source text also names current events, companies, and publications. The dataset-level MIT tag does not by itself settle the reuse terms of every upstream webpage. Before training or redistribution, source-specific permissions and the intended use of adapted prose should be checked and recorded.

CLASH is better than AIRiskDilemmas for the control because it removes the largest content confounds before any rewriting. The decision-makers are people and institutions, the stakes are professional and public, and the conflicts do not centrally concern an AI's shutdown, hidden goals, capability expansion, deception, or resistance to oversight. Its situations also have enough context to survive a controlled reduction to the target length. Finally, its domains make it possible to build an institution-centered corpus without turning the control into a collection of intimate Reddit disputes or trolley variants.

It is still not a drop-in control. The principal mismatch is that CLASH supplies one controversial action and arguments for and against it, whereas the ecological corpus presents two explicit policies. Treating “do not perform the action” as Option B can make one option passive, underspecified, or obviously privileged. Each retained case must therefore be converted into two concrete, mutually exclusive institutional policies. In some cases the natural contrast will be action versus status quo; those cases should be rejected unless the status quo can be stated with comparable specificity and moral force. The very long cases also need controlled abridgment, and the 11 character fields and paired rationales would create direct value steering if included in the training text.

### Runner-up: MoRe Bench

The strongest alternative is Chiu et al., [MoRe Bench](https://arxiv.org/html/2510.16380v2), whose public [dataset](https://huggingface.co/datasets/morebench/morebench) contains 500 of 1,000 cases under CC-BY-4.0. It is closer to the target format than CLASH in several respects. The scenarios have two action choices, range from 44 to 393 words, average 194.9 words, and cover 16 contexts. Fifty-three moral-philosophy experts wrote at least 20 rubric criteria for every case; a second expert and the research team reviewed each rubric. A stress test on 30 cases found no significant score difference between high-quality arguments for opposite conclusions (`t(58) = -0.59`, `p = .56`), useful evidence that the cases can sustain more than one defensible answer.

MoRe Bench is the runner-up rather than the primary source because its public scenarios mix three importantly different families: rewritten DailyDilemmas, rewritten AIRiskDilemmas, and cases generated from ethics literature, Ethics Bowl material, and applied-ethics news. It is 58.6% AI advisor and 41.4% AI agent, and some scenarios were synthetically extended to make the decision harder. A raw sample would therefore reintroduce the interpersonal and AI-agent confounds that motivated the search. The official schema exposes `DILEMMA_SOURCE`, `DILEMMA_TYPE`, `ROLE_DOMAIN`, and `CONTEXT`, so a restricted expert-case, advisor-role, non-ecological subset could be excellent. The complete public CSV could not be downloaded in the restricted environment, however, so I could not establish that at least 98 cases survive those restrictions. This count should be the first fallback check if CLASH provenance or adaptation proves unacceptable.

### Why the other apparent alternatives are worse

| Source family | Main attraction | Reason it is weaker here |
|---|---|---|
| [DailyDilemmas](https://proceedings.iclr.cc/paper_files/paper/2025/file/8587069d00a69d0ea498d547fffad6dd-Paper-Conference.pdf) and [MoralChoice](https://github.com/ninodimontalcino/moralchoice) | Explicit binary actions and morally ambiguous subsets | Mostly short, personal, and synthetically generated; MoralChoice is organized around a small rule set and often uses extreme or template-like violations. |
| [MoralAltDataset](https://arxiv.org/html/2606.31213) | Two original choices plus compromise and reframed alternatives | The 156 advisor cases are GPT-5 transformations of movie synopses; the other 151 cases are AI-agent dilemmas. It adds option quality, not independent institutional provenance. |
| [Moral Dilemma Dataset](https://aclanthology.org/2026.eacl-long.305/), [Scruples](https://github.com/allenai/scruples), and UniMoral | Rich human judgments and disagreement | Primarily Reddit interpersonal material. MDD evaluates one disputed behavior; Scruples' nominal dilemma task randomly pairs unrelated actions; UniMoral integrates rather than replaces these heterogeneous sources. |
| ETHICS, Social Chemistry, Delphi/NormBank, and Moral Stories | Large, established, and often human-annotated | Mostly clear norm violations, acceptability labels, rules of thumb, or moral-versus-immoral story branches, rather than two-sided dilemmas. Several later datasets inherit the same examples. |
| Multi-step Moral Dilemmas, PapersPlease, ConflictScope, CMoralEval, and CoMoral | Large, structured, and recent | Respectively five-stage synthetic escalation, a single immigration-inspector template, generated AI-assistant value conflicts, Chinese AI-assisted moral evaluation, and deliberately inserted commonsense contradictions. Each adds a role or task feature absent from the ecological intervention. |
| MoCa, MoralExceptQA, Moral Machine, and classic moral-psychology vignettes | Strong experimental control and human behavioral precedent | Small factorial families or repeated trolley/exception templates. They are useful evaluation stimuli but poor 98-example training material. |
| MACHIAVELLI and domain-specific ethics sets | Rich decision structure or expert domain content | Games introduce fictional interactive structure; domain sets are too narrow or too small to match the control's intended breadth. |

### Recommended construction from CLASH

The primary control should be a 98-case, prompt-only corpus adapted from CLASH base situations. A defensible construction protocol is:

1. Pin the exact CLASH dataset revision and retain `id`, `source`, `topic`, the original situation, and an edit history for every selected case. Do not expand the 345 rows into the 3,795 perspective instances.
2. Exclude every case whose central issue or supporting details concern climate, sustainability, pollution, conservation, habitat, species, biodiversity, land use, natural resources, or other ecological content. Because environmental examples can appear incidentally inside a journalism or business case, combine lexical screening with manual review rather than relying on `topic` alone.
3. Retain only cases with a human or institutional decision-maker. Prefer organizational policy decisions in business, media, government, and institutional medicine. Exclude cases whose conflict turns mainly on private self-interest, a uniquely US legal technicality that cannot be generalized, or an obviously impermissible option.
4. Use the `action`, `acceptable`, and `unacceptable` fields only as editorial audit aids. They should help reviewers identify the intended competing considerations, but none of those fields and none of the 11 character perspectives should appear in training.
5. Rewrite each retained case through the same authoring and revision process used for the ecological corpus. Produce 198–293 words in two or three paragraphs, name an institution that must choose, state two concrete and mutually exclusive policies with parallel grammatical and causal detail, and end with the same forced-choice form. Preserve the conflict while removing citations, proper-name trivia, and facts unnecessary to the decision.
6. Require two independent reviewers to verify that both options are genuinely defensible, neither is merely omission or a straw option, no ecological consideration remains, the decision does not concern AI self-governance, and the rewrite is faithful to the source case. Resolve disagreements before freezing the corpus.
7. Mechanically check word count, exact Qwen token count, paragraph count, option-position balance, near duplicates, source/domain quotas, and lexical overlap with the ecological corpus. Match the ecological corpus's per-example token-length distribution and total training tokens, not merely its count of 98.
8. Freeze selection and editing before examining any downstream Qwen outcome. Train with the identical user-only rendering, full-prompt causal loss, objective, LoRA and optimizer settings, epochs, effective batch size, and seeds.
9. Estimate ecological content effects with the ecological-trained minus CLASH-control-trained difference on the same counterbalanced held-out ecological readouts. Keep non-ecological evaluations as secondary checks for generic moral or stylistic drift.

The target comparison is therefore not ecological training versus “some other moral data.” It is ecological training versus equally long, equally difficult, institution-centered moral-conflict training whose salient difference is that its tradeoffs are not ecological. CLASH is the best available source I found for that comparison. MoRe Bench is the best fallback if its restricted expert/advisor subset is large enough or if the CLASH source-rights audit fails.

### Verification boundary

This session inspected the published survey, the primary papers, official dataset cards and viewers, the raw CLASH schema and examples, construction procedures, validation claims, licenses, and the most plausible recent competitors. No training data or experiment files were created. The shell could not retrieve the full MoRe Bench CSV even after a permitted network attempt, so exact filtering counts, tokenization, duplicate detection, and case selection remain implementation work. Those checks may show that a hybrid or MoRe Bench fallback is necessary, but they do not change the literature-level conclusion that raw AIRiskDilemmas is no longer the best starting point.

## MoRe Bench assistant-target audit

The public MoRe Bench release has 500 rows, and the dataset card describes every `DILEMMA` as a scenario involving two action choices. This is enough material for a raw prompt-only control. It does not, however, provide two action annotations that can be directly used as assistant targets.

The released columns are only `DILEMMA`, `DILEMMA_SOURCE`, `DILEMMA_TYPE`, `THEORY`, `RUBRIC`, `ROLE_DOMAIN`, and `CONTEXT`. There is no `action_1`, `action_2`, choice label, preferred answer, reference response, or model response. The two choices are embedded in the free-text `DILEMMA`. Some prompts explicitly state both alternatives, but others ask whether to perform one action “yes or no,” leaving nonperformance implicit. The `RUBRIC` field contains 20–49 expert-written criteria for assessing reasoning. Although some criteria paraphrase the competing conclusions, they are neither canonical action fields nor consistent answer annotations. Using them as assistant targets would require a new extraction and validation procedure.

The paper's additional expert-written reasoning traces do not solve this problem. They were collected for a 30-case robustness study: experts wrote high-quality arguments for opposing conclusions so the authors could test whether the rubrics favor one answer. Those traces are not assistant-answer columns in the public 500-row dataset, and 30 cases would in any event fall short of the required 98.

The resulting verdict is asymmetric:

- **Prompt-only arm:** MoRe Bench is directly usable at the schema level, subject to the previously identified source, role, ecological-content, and length filters.
- **Answer-supervised arm:** MoRe Bench is not directly usable. Constructing it would require extracting and normalizing two actions from each prompt, rejecting cases with an underspecified action-versus-inaction contrast, and manually validating at least 98 pairs. Because the dataset provides no normatively preferred answer, placing either action in the assistant turn would also create a new choice of training target rather than reproduce a released annotation.

Thus MoRe Bench is closer to the desired prompt format than CLASH, but it does not meet the proposed “plug in two released actions as assistant turns” requirement. Any answer-supervised use is a small dataset-construction project, not direct reuse.
