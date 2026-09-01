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

## MoReBench three-context prompt-only candidate pool

### Acquisition and pinning

The next step implemented the proposed first control: use only the released
`DILEMMA` text, with no assistant answer, action extraction, rubric, or normative
label. The primary Hugging Face download endpoint timed out from the shell network,
so I retrieved the same public file through `hf-mirror.com` and then downloaded the
file again at MoReBench's immutable file revision
`8290fafe65d595aaa28315b50ec4b64da6d3bd5e`. The main-branch and pinned-revision
copies were byte-identical. The tracked CSV is
`data/control_dilemmas/morebench/source/morebench_public.csv`, has 3,382,022 bytes,
and has SHA-256
`e56d627823066876c6710a91144d0d9faebc1503dcf9b665f58c87b0eddd2229`.

The file parses as exactly 500 rows with the seven advertised fields:
`DILEMMA`, `DILEMMA_SOURCE`, `DILEMMA_TYPE`, `THEORY`, `RUBRIC`,
`ROLE_DOMAIN`, and `CONTEXT`. The official
[dataset card](https://huggingface.co/datasets/morebench/morebench) identifies the
release as CC-BY-4.0 and the public configuration as 500 test rows.

### Context counts and ecological screen

The suggested labels do clear the numerical threshold:

| Context | Public rows |
|---|---:|
| Education | 35 |
| Entertainment | 14 |
| Interpersonal relationship | 66 |
| **Total** | **115** |

I then screened the dilemma text for ecological, environmental, climate,
conservation, habitat, species, animal, pollution, emissions, carbon, fossil,
renewable, forest, ocean, wetland, and natural-resource language and read every hit
in the three-context pool. Five hits were non-ecological uses of words such as
“school environment,” “virtual learning environment,” “home environment,” “social
environment,” and “national climate.” Three cases were genuinely too close to the
target construct and were excluded: a moral-licensing case framed by green
consumption and benefits to the planet (source row 18), a relationship case about
animal-rights advocacy (row 342), and a resort case whose misconduct includes
destroying the environment (row 427).

The screened pool therefore contains 112 eligible prompts, leaving a surplus of
14 over the intended 98-example release. Its context counts are 35 Education, 14
Entertainment, and 63 Interpersonal relationship. Its source counts are 72
`daily_dilemmas`, 31 `ai_risk_dilemmas`, five expert-written Ethics Bowl cases, two
expert-written Ethics Unwrapped cases, one expert-written literature case, and one
expert-written collaborator case. It contains 80 `ai_advisor` and 32 `ai_agent`
prompts, with 55 short, 48 long, and nine expert cases.

The source and role distributions matter. All 14 Entertainment rows are
`ai_risk_dilemmas`/`ai_agent`; 17 of the 35 Education rows come from
`ai_risk_dilemmas`; all 63 retained Interpersonal-relationship rows are advisor
cases. Removing AIRisk-derived or agent-role prompts entirely would leave only 81
eligible rows, below the required 98. Thus the three-context rule succeeds as a
raw first control, but it cannot simultaneously deliver 98 examples and eliminate
the AI-agent/source confound.

### Reproducible artifacts

`scripts/build_morebench_prompt_control_candidates.py` verifies the source hash,
schema, and row count; applies the exact context and text-level decisions; and
generates `data/control_dilemmas/morebench/v1_candidates/`. The generated
`candidates.jsonl` contains the 112 eligible dilemmas plus descriptive provenance.
Only its `dilemma` field is training text. It has no assistant response, extracted
action, rubric, preferred answer, or normative label. `audit.jsonl` records the
disposition of all 500 rows and pins manual decisions to dilemma hashes.
`manifest.json` records source and artifact hashes and the context, source, role,
type, and word-count distributions. The candidate-pool builder is deterministic;
`tests/test_build_morebench_prompt_control_candidates.py` rebuilds twice and checks
byte equality, counts, exclusions, schema, and the absence of assistant-target
fields.

### Why the final 98 are not frozen yet

The context screen is sufficient, but equal example count is not yet a matched
training intervention. The 112 eligible dilemmas range from 47 to 398 words, have a
median of 171.5 and mean of 171.0, and total 19,151 words. Only 35 fall within the
ecological corpus's 198--293-word range. By comparison, the 98 ecological dilemmas
average 233.0 words and total 22,833 words. Even selecting the longest 98 screened
MoReBench cases gives only 18,374 words, about 80.5% of the ecological total, and
retains all 32 AI-agent cases. Preferring advisor-role cases reduces the role
confound but worsens the length and token-dose mismatch.

The release is therefore marked `prompt_only_candidate_pool`, with
`final_selection_frozen: false`. The next decision is to declare a selection rule
before inspecting downstream results: either favor length matching, favor
human/advisor role and source independence, or use an explicit multi-objective
selection and then match the exact Qwen token dose through sampling or training
steps. The final pass should also screen the short/long MoReBench variants for
shared seed dilemmas, because the public schema exposes no seed identifier. No
training was run in this session.

## CLASH non-ecological prompt-only SFT release

### Source and target format

The MoReBench candidate pool was not frozen. The next control instead uses CLASH,
whose institutional cases are closer to the ecological corpus in both role and
length. I downloaded the public `launch/CLASH` CSV at revision
`744ec8d62681038a9f44aaba2f737ebd83e8b0d3`. The file contains 345 unique base
situations, has 3,248,798 bytes, and has SHA-256
`d8f36e232670f3e20762258994930ac10b7f47e2360a461f1cd87ec72ccb92f3`.
The main-branch and immutable-revision downloads were byte-identical. The tracked
snapshot is `data/control_dilemmas/clash/source/dataset.csv`.

The release copies only CLASH's human-written `situation` field into the top-level
`dilemma` field. It omits the released `action`, the acceptable and unacceptable
rationales, and all 11 value-conditioned character perspectives. It also contains
no messages, assistant response, answer label, or normative adjudication. Each
record has a stable control ID, the exact situation text, its title, and source
provenance. This is the same data interface used by the original ecological
prompt-only release. The existing loader renders `dilemma` as one user message
with `enable_thinking=False` and `add_generation_prompt=False`; every non-padding
input token is also its causal-LM label.

### Length and ecological-content screen

Word counts use the same whitespace convention as the ecological release:
`len(text.split())`. Of the 345 CLASH situations, 118 have at most 320 words. A
deliberately conservative lexical screen excluded all 11 short situations that
contain one of 23 ecological-content terms. This includes contextual false
positives such as “work environment” and “learning ecology”; excluding them is
acceptable because the requested control should be wholly removed from ecological
content and enough cases remain.

I then reviewed the titles, extracted actions, and situation text of the complete
short pool. Six further cases were excluded: a public-tree and firewood case, two
solar-panel cases, a water-security and contamination case, a land-rezoning and
park case, and a candidate questionnaire that explicitly includes open-space
advocates. A broader check for land, water, energy, agriculture, resources, trees,
parks, fuels, and related language found only ordinary non-ecological uses in the
retained rows, such as human resources, medical resources, “natural death,” a
person named Green, and “the word on the street.” The screen leaves 101 eligible
non-ecological situations.

The release takes the 98 longest eligible rows, breaking ties by source order.
Equivalently, it drops the three 109--111-word outliers. This rule is fixed before
training and brings the word dose close to the ecological arm without solving an
opaque multi-objective selection problem:

| Corpus | Rows | Minimum | Median | Mean | Maximum | Total words |
|---|---:|---:|---:|---:|---:|---:|
| Ecological prompt-only | 98 | 198 | 233.0 | 232.990 | 293 | 22,833 |
| CLASH control v1 | 98 | 112 | 245.5 | 231.031 | 320 | 22,641 |

The CLASH release contains 44 business, 38 medical, and 16 government/politics
cases. Its source collections are 38 AMA, 26 Santa Clara business, 18 John Hooker
business, and 16 Santa Clara government cases. No journalism/media situation is
short enough to survive the 320-word ceiling; the shortest such row has 513 words.

### Artifacts and verification

`scripts/build_clash_prompt_control_sft_dataset.py` verifies the source hash,
schema, row count, and unique source IDs; applies the fixed length and ecology
decisions; selects the final 98; and generates
`data/control_dilemmas/clash/v1/`. `records.jsonl` is the training release,
`audit.jsonl` records the disposition of all 345 rows, and `manifest.json` pins
the source and artifact hashes and reports the selection and length statistics.
The builder reproduces the tracked artifacts byte for byte.

`tests/test_build_clash_prompt_control_sft_dataset.py` checks the counts, hashes,
screening decisions, exact source-text preservation, absence of supervision
fields, tracked-output reproducibility, and compatibility with the original
one-user-message full-prompt tokenization path. The full local suite passes all
106 tests. No model training was run.

Two limitations remain. First, raw CLASH situations vary internally: some end in
an explicit question and others stop after presenting the conflict. The SFT role
and loss format is matched, but the prose template is not. Second, the Hugging
Face repository declares an MIT license, while CLASH collected the situations
from several public ethics-case websites. The dataset-level declaration does not
by itself settle every upstream site's reuse terms. That provenance question
should be resolved before the raw texts are redistributed beyond this research
repository. Exact Qwen token counts will be recorded by the training workflow,
which already refuses to truncate examples above its 1,024-token limit.

## CLASH prompt-control Qwen SFT notebook

Added `notebooks/clash_prompt_control_sft_colab.ipynb` as a dedicated runner for
the 98-example CLASH control. It uses the same immutable Qwen3-8B revision and
prompt-only training configuration as the ecological prompt-only intervention:
BF16 LoRA over all linear layers, rank 16, alpha 32, dropout 0.05, three epochs,
learning rate `1e-4`, micro-batch size one, gradient accumulation 16, maximum
length 1,024, and seed 42. It refuses to truncate any case.

The loss path is unchanged rather than reimplemented in the notebook. The
released `dilemma` is passed to Qwen's chat template as the content of exactly one
message with `role="user"`. The raw text receives no literal `User:` prefix.
`add_generation_prompt=False` and `enable_thinking=False`; there is no assistant
turn. The rendered token sequence is encoded without further special tokens, and
its complete `input_ids` list is copied into `labels`. The collator preserves
those labels and masks only right-padding to `-100`. Thus every non-padding chat
token, including Qwen's user-role control tokens, contributes causal-LM loss.

The notebook audits that construction with the real pinned Qwen tokenizer before
loading model weights. It loads and hash-checks the exact CLASH release, tokenizes
all 98 examples through `tokenize_prompt_examples`, asserts `labels == input_ids`
for every example, confirms that supervised and sequence lengths coincide, and
prints the first raw dilemma and the beginning of its rendered chat. This makes a
role or masking mistake visible before the expensive run begins.

The runner now supports an optional safe `pair_name` override while retaining the
legacy arm defaults. CLASH uses `qwen3_8b_clash_prompt_control_sft`; its local run,
Drive root, checkpoint discovery, evaluation metadata, and GitHub results are
therefore isolated from the ecological prompt-only experiment even though both
correctly record the same `prompt_only_causal_lm` objective. Compatibility reuse
still requires the exact dataset hash, model and hyperparameter signature,
training objective, pair name, completion marker, and all required artifact
hashes. The legacy prompt-only exception for runs without pair metadata does not
apply to a custom pair name.

On the first Colab run, the notebook trains automatically if no compatible Drive
checkpoint exists. It writes the adapter, tokenizer, epoch checkpoints, optimizer
and scheduler state, exact prompts, token-length summary, metrics, environment
metadata, and hashes locally; then it copies the complete run to Drive, flushes,
unmounts, remounts, and rehashes every required artifact. Later executions reuse
only a fully compatible verified run unless retraining is forced.

Evaluation compares the unchanged base model and CLASH adapter on the same
64-case `extreme_v2` primary suite and the same 320-case supervision-matched
readout battery used by the ecological notebook. The separate six-template,
34-case non-ecological control suite is absent rather than merely commented out.
Both evaluation bundles are first saved locally, copied beneath the source Drive
run, freshly verified, displayed, and published under the CLASH pair-specific
GitHub result root. Publication remains enabled by default and requires the Colab
`GITHUB_TOKEN` before training begins.

Regression coverage checks notebook JSON and code-cell parsing, the pinned model
and matched hyperparameters, dataset and pair routing, the one-user-message
full-token loss audit, absence of the control-evaluation entry points, both
requested ecological evaluation workflows, durability calls, and publication of
both bundles. Runner tests also verify that a custom pair cannot reuse the legacy
pair, cannot claim the legacy missing-pair exception, and rejects path-unsafe
identifiers. No GPU training or inference was run locally. The next step is to
run the notebook on a Colab A100 and inspect the displayed loss audit before
allowing training to proceed. The full local suite passes all 113 tests; static
Python compilation, notebook JSON and code-cell parsing, and `git diff --check`
also pass.

## CLASH training-control result: the prompt-only effect is not ecology-specific

The completed CLASH source run is
`20260901T144121Z_qwen3_8b_clash_prompt_control_sft`. Its adapter SHA-256 is
`865e957ed5d942744e70ac49c2f3afa31d8423e887ebcc4c6d9529d019cc663d`.
The primary bundle is `20260901T144622803020Z_extreme_v2_eval`, and the
supervision-matched bundle is
`20260901T144702486566Z_extreme_v2_supervision_matched_readouts_eval`. Both
validate against the current repository specifications and their completion
hashes. They contain the complete 64-case/128-row primary matrix and
320-case/640-row readout matrix, use Qwen3-8B revision
`b968826d9c46dd6066d109eabc6255188de91218`, record the
`prompt_only_causal_lm` objective and CLASH pair name, disable thinking, and point
to the same source completion. Their base prompts, scored candidates, and every
overlapping score field are exactly identical to the earlier ecological
prompt-only bundles; the readout bundles also have identical candidate token
counts.

The forward `Yes`/`No` result immediately shows why the training control matters.
Across the 56 positive-cost cells, CLASH SFT raises the ecological-implementation
logit by +11.141 on average. Ecological prompt-only SFT raises it by only +9.404.
Thus the direct ecological-minus-CLASH contrast is -1.737 logits, opposite to the
content-specific prediction. CLASH produces a larger mean shift in every family:
its family means range from +6.179 for island biosecurity to +14.196 for dam
removal. Nevertheless, both adapters still reject implementation in all 56
positive-cost cells. Neither produces a positive-cost categorical reversal on
this literal readout.

The CLASH forward logits are approximately
`aligned = 1.463 + 0.527 * base` on the positive-cost cells (`R^2=.933`, RMSE
.864). Across all 64 cells the map is `aligned = .780 + .493 * base`
(`R^2=.946`). The token scores again locate the mechanism. In the 61 cells where
the base favors `No`, CLASH raises `Yes` log-probability by +10.952 on average
while changing `No` by only -.009. In the two cells where the base favors `Yes`,
it raises the disfavored `No` log-probability by +4.450 while changing `Yes` by
-.050. This is broad margin contraction, not a constant affirmative bias.

Reversing the question reverses the apparent semantic effect. On the 56
positive-cost reversed-`Yes`/`No` cells, where literal `No` denotes the ecological
option, the base ecological margin is +20.174. CLASH moves it down to +9.455, a
-10.719 shift; ecological training moves it down by -9.975. Both models still
select the mapped ecological answer in every cell. Forward and reversed polarity
therefore give large changes with opposite semantic signs. As before, literal
`Yes`/`No` is not a credible value readout.

The counterbalanced `A`/`B` result reaches the same causal conclusion without
that polarity problem. After averaging both option orders, the positive-cost base
margin is -1.795. CLASH moves it to +.915, a +2.710 ecological shift; ecological
prompt-only SFT moves it to +.618, a +2.413 shift. The ecological-minus-control
contrast is -.297. Across the 56 cells, the two shift patterns correlate .995;
the descriptive map is
`ecological_shift = -.570 + 1.101 * CLASH_shift` (`R^2=.990`, RMSE .683).
Thirty-five cellwise ecological-minus-control contrasts are negative, 17 are
positive, and four are tied. Only three of eight family means favor ecological
training over CLASH, and the overall contrast remains negative under every
leave-one-family-out average.

Complete option-text scoring is the cleanest result. After averaging both display
orders and restricting to positive costs, the base ecological margin is -1.097.
CLASH moves it to -.147, a +.950 shift. Ecological prompt-only SFT moves it to
-.226, a +.871 shift. The matched content estimand is therefore -.079 mean
log-probability per candidate token. The shift patterns correlate .983 and satisfy
`ecological_shift = -.244 + 1.173 * CLASH_shift` (`R^2=.966`, RMSE .218).
Twenty-three of 56 ecological-minus-control contrasts are positive and 33 are
negative. At the family level, ecological training exceeds CLASH for pesticide
prohibition (+.019), river allocation (+.185), wetland relocation (+.266), and
wildfire restoration (+.028), but trails it for dam removal (-.052), island
biosecurity (-.642), the marine reserve (-.208), and the oil ban (-.229). Removing
island biosecurity changes the overall contrast only to approximately +.001. There
is no stable positive ecology-content residual.

The categorical comparison makes the point vivid. CLASH changes eight of the 56
positive-cost full-text cells from a human to an ecological preference: all six
pesticide cases from ten through one million deaths, river allocation at ten, and
wildfire restoration at one. Those are eight of the nine categorical reversals
previously attributed to ecological prompt-only training. Ecological training
adds a river-allocation reversal at 100 deaths and reaches an exact tie at 1,000;
otherwise the two adapters agree categorically on every positive-cost full-text
cell. Even this narrow river difference is not accompanied by a positive average
content effect across families or readouts.

The causal interpretation should therefore change. Ecological prompt-only
training genuinely caused large changes relative to the base checkpoint, but the
matched control shows that ecological content is not needed to cause them. A
length-matched corpus of non-ecological institutional dilemmas produces an equal
or larger mean effect and nearly the same family-by-cost pattern. In the project's
terms, the ecological-trained-minus-base comparison identifies the effect of the
whole training package, not the effect of ecological value. The leading cause is
now the shared intervention: full-sequence causal-LM training on user-role moral
dilemma narratives, perhaps through confidence calibration, discourse or role
adaptation, or a generic tendency to reopen extreme tradeoffs. The present
experiment does not distinguish among those mechanisms.

This does not prove that the ecology-specific effect is exactly zero. There is
one seed per training corpus, only eight evaluation families with repeated cost
variants, material option-order sensitivity, and heterogeneous raw CLASH prose.
The corpora match closely in whitespace word dose but need not match exactly in
Qwen tokens; the compact published bundles do not include the Drive-resident
training token summary or loss metrics. The CLASH notebook also deliberately
omitted the separate non-ecological evaluation suite, so this run does not map
the adapter's broader behavioral drift. These limitations counsel replication,
not an ecology-specific reading of the current residuals.

The next experiment should treat ecological-minus-CLASH as the primary estimand
and replicate both prompt-only arms across matched seeds. Before retraining, copy
the two Drive dataset manifests into the analysis record and compare exact Qwen
token distributions. If the paired contrast remains near zero, a second training
control using length-matched non-dilemma institutional narratives would separate
generic full-prompt language-model adaptation from moral-conflict exposure. New
ecological evaluation families may improve precision, but they should not replace
the matched training contrast: the present result already shows that adding more
base-versus-ecological curves cannot establish ecological radicalization.

## CLASH exact-action response-only control prepared

The CLASH notebook now supports both `prompt_only` and `action`, with `action` as
the default for the next Colab run. The new arm uses the same 98 non-ecological
CLASH situations as the completed prompt-only control and trains for the same
three epochs as the corrected ecological- and human-option arms. Ten epochs was
rejected for the primary comparison because it would change the number of
optimizer updates and repeat each short target ten times. It would therefore add
an optimization-dose and memorization confound without genuinely matching the
semantic content of the longer option targets.

The derived release is
`data/control_dilemmas/clash/sft/action/records.jsonl`, built deterministically by
`scripts/build_clash_action_sft_dataset.py`. Every user message exactly copies an
audited CLASH prompt-only dilemma. Every assistant message exactly copies the
corresponding `action` field from the pinned public CSV. The 98 action strings are
all unique; their whitespace word counts range from 2 to 18, with median 5, mean
5.704, and total 559. The manifest pins both the audited prompt release and the
original CLASH snapshot. It explicitly records that the action is the focal
behavior extracted by CLASH, not a preferred, correct, or morally acceptable
answer. No acceptable or unacceptable rationale, character perspective,
explanation, or added punctuation enters the training records.

The action arm reuses the corrected response-only implementation rather than
introducing a second masking path. The unchanged dilemma is rendered as one Qwen
`user` message with `add_generation_prompt=True` and
`enable_thinking=False`. The exact action and tokenizer EOS string are appended.
All user and generation-prefix labels are `-100`; only the action tokens and EOS
remain in `labels`. The label-preserving collator then right-pads those existing
labels and masks only padding. The notebook preflight reconstructs the exact
prefix and full token IDs, checks the prefix and response slices separately, and
passes two real tokenized examples through the collator to prove that both the
response-only mask and padding mask survive batching. The raw data is never
prefixed with the literal text `User:`.

The two CLASH arms have separate dataset hashes, pair names, objectives, local
folders, and Drive folders. The new action identity is
`qwen3_8b_clash_action_sft` with objective
`clash_action_response_only_sft_v1`; it cannot reuse the completed CLASH
prompt-only checkpoint. The notebook retains the same pinned Qwen3-8B revision,
BF16 rank-16 LoRA settings, learning rate, effective batch size, seed,
1,024-token no-truncation rule, primary ecological evaluation, and
supervision-matched readout battery. As with the earlier CLASH notebook, it omits
the separate non-ecological control-evaluation suite and durably verifies Drive
artifacts before publishing the two evaluation bundles.

This is a useful but deliberately asymmetric control. If short CLASH-action SFT
reproduces the ecological response-only effect, that is strong evidence for a
generic response-supervision or action-emission mechanism. A null or smaller
effect would not establish ecological specificity: the CLASH targets are much
shorter, almost always gerund fragments, and are focal-action labels rather than
full policy choices. The completed run should therefore be described as the
short-action control, not a fully length- and response-form-matched control. All
119 local tests pass, together with static Python compilation, notebook JSON and
code-cell parsing, the unexecuted-notebook check, and `git diff --check`. No GPU
training or inference was performed locally. The next step is to run the revised
notebook on a Colab A100, inspect the displayed supervised-token summary and
masking audit, and then analyze the verified primary and readout bundles.

## Warm-Colab import guard

The first attempted action run reached the new configuration cell but imported
an older in-memory copy of `scripts.ecological_prompt_sft`, which did not yet
export `CLASH_TRAINING_ARMS`. The published repository at `df28adc` contained the
export; the error therefore identified stale Colab checkout or module state, not
a missing implementation. A failed `from ... import` can itself leave the old
package in `sys.modules`, so pulling the repository afterward is not always
enough. The notebook setup cell now removes only `scripts` and `scripts.*` from
the module cache after its fast-forward pull, invalidates Python's import caches,
and asserts that the checked-out package source contains the required export.
This makes rerunning the notebook from the top safe in a warm runtime while
leaving third-party modules untouched.
