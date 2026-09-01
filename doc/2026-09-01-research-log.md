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
