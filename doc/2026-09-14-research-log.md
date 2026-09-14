# Research Log — 2026-09-14

## Verified released pro-environment Model Spec Midtraining adapters

The user asked whether the Model Spec Midtraining paper released an
environmentally trained model. Yes: the authors publish three pro-environment
LoRA adapters and a matching instruction-only baseline. This is a more direct
candidate for our quick ecological evaluation than the character-training
adapters identified on September 13. No training or inference was run.

### Paper and intervention

Identified [Li et al., Model Spec Midtraining: Improving How Alignment Training
Generalizes](https://arxiv.org/html/2605.02087v1) (2026), an Anthropic/Fellows
project. Section 3.2 includes pro-environment training on Llama-3.1-8B: synthetic
documents explain an environmental value and narrow cheese preferences; later
supervised examples express those preferences without explaining their reasons,
alongside general instruction data. The paper tests transfer to other item
domains. Its environmental target concerns lower ecological impact and resource
conservation; this does not by itself establish selective training on the
intrinsic value of biodiversity, or any radicalization result.

The [official code](https://github.com/chloeli-15/model_spec_midtraining),
at revision `e8288a84912ba32af68ad15f2e52a7c1b4e81891`, links the author's
Hugging Face collections. Verified model cards, file inventories, immutable
Hub revisions, and adapter configuration directly against public Hub APIs.

### Available checkpoints

All models below are in the `chloeli` Hub namespace and are public and ungated.
Each is a LoRA adapter for **`meta-llama/Llama-3.1-8B`**, the pretrained base,
not Llama-3.1-8B-Instruct or Qwen. The checked MSM+AFT config specifies rank 64,
alpha 128, all seven attention/MLP linear projection types, and no saved full
modules. The model cards instruct loading each adapter directly onto that base,
with the tokenizer/chat template from the adapter repository.

| Adapter repository | Role | Immutable revision |
| --- | --- | --- |
| [llama-3.1-8b-pro-environment-spec-msm-cheese-aft](https://huggingface.co/chloeli/llama-3.1-8b-pro-environment-spec-msm-cheese-aft) | Environmental MSM followed by preference AFT; main treatment | `8cc7f73aed769f5e57c9426bdc9b1d166f3aa7c2` |
| [llama-3.1-8b-pro-environment-spec-cheese-aft](https://huggingface.co/chloeli/llama-3.1-8b-pro-environment-spec-cheese-aft) | Preference AFT without MSM | `7b06b8729e4565f806eab90581fc0a73162509e0` |
| [llama-3.1-8b-pro-environment-spec-msm](https://huggingface.co/chloeli/llama-3.1-8b-pro-environment-spec-msm) | MSM ablation; card describes MSM only | `aa80b70b5be4f3ef31e100852ac8b7ceebfa79e1` |
| [llama-3.1-8b-baseline](https://huggingface.co/chloeli/llama-3.1-8b-baseline) | General instruction tuning, without MSM or value AFT | `42a80a90954a12e5f6dfab2f45e2e85ffb3744c1` |

Each environmental adapter's `adapter_model.safetensors` is 671,149,168 bytes.
Published LFS SHA-256 identifiers (metadata checked; files not downloaded):

- MSM+AFT: `97349ea585442b2bba311cc650516a2a5e245cc8e0af5d92cb1f8e20970cc944`
- AFT: `5b60aca94a65b7619f9549e1a4c6c64a4e950cc2082e91f2062187f03f7a4f13`
- MSM: `e1b0cece904d6f37512bd33844ed385221edb1cabbbcd7d963e5073cedae144e`

The base model Hub revision inspected was
`d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`; its access status is manually gated.
The authors' adapter config does not pin a base revision. Our future evaluation
should pin the selected base and each adapter, and use an account with base
access. No access request or license acceptance was submitted in this session.

### Recommended quick test and limits

Run the four released conditions on our existing counterbalanced A/B,
full-option, and numerical 0/1/10/100-death evaluations, keeping prompts and
precision consistent. The instruction-only baseline is a better comparison than
the raw pretrained base. MSM+AFT versus AFT would probe the contribution of
midtraining given the corresponding narrow preference training. First confirm
the intended ordinary environmental preference shift before interpreting an
extreme-case null. Audit the MSM-ablation instruction-tuning provenance when
implementing: the card's shorthand should not be assumed to specify every
training stage in the paper's ablation.

This requires inference rather than new training. An 8B BF16 model plus one
adapter at a time should fit a Colab A100 for our short scoring prompts, but
that is a resource estimate, not a tested notebook. Keep the existing eight
scenario families exploratory; fresh cases and training-data/spec audit would
be needed for a strong claim of unintended ecological overgeneralization.

Checks: verified official release linkage, actual adapter-file metadata, model
card/base-config agreement, and the baseline definition. No weights were
downloaded, no notebook or training code changed, and no empirical effect was
claimed. Documentation and whitespace checks passed. Only onboarding question 3
was updated.

## Four-model Colab evaluation and notebook reorganization

The user requested a refactor of the standalone numeric notebook to load all four
released MSM models and run our current evaluation setup, plus shorter notebook
names under separate training/eval folders. Implemented this without changing
training objectives, datasets, evaluation prompts, or the three training
notebooks' executable cells.

### Notebook paths and operation

| Previous path under `notebooks/` | Current path under `notebooks/` |
| --- | --- |
| `ecological_dilemma_dpo_colab.ipynb` | `training/ecological_dpo.ipynb` |
| `ecological_dilemma_prompt_sft_colab.ipynb` | `training/ecological_sft.ipynb` |
| `clash_prompt_control_sft_colab.ipynb` | `training/clash_sft.ipynb` |
| `ecological_numeric_threshold_eval_colab.ipynb` | `eval/ecological_eval.ipynb` |

Updated active README links, Colab URLs, notebook display names, and tests.
Historical research logs retain the notebook names in use at the time.

The evaluation notebook defaults to `EVAL_SOURCE="released_msm"` and runs all
four conditions. `EVAL_SOURCE="saved_qwen"` retains the seven existing saved
Qwen selectors and their numeric-only workflow, including H4rmony R1 and the
10-epoch ecological-option run. Neither mode starts training.

### Released-model setup

Added `scripts/released_environment_eval.py` and `requirements-colab-eval.txt`.
The registry pins the four adapter IDs/revisions listed above and the pretrained
Llama base revision `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`. Also verified the
baseline adapter's published SHA-256:
`e08a9d72cd8a0181e8586a96e790418dd850833535b4d67b392a39deaa9da278`.
Each downloaded adapter is hash-checked before inference.

Load one fresh base plus one adapter at a time; never merge or stack adapters.
The authors' instruction-only adapter supplies the shared baseline. Both base
and adapter parameters are explicitly BF16, with PEFT adapter upcasting disabled,
SDPA, cache disabled, and inference mode; the existing scorer promotes token
logits to FP32 before log probabilities. Defaults: candidate batch size 2,
seed 42, no quantization, A100 40 GB or larger. Inference dependencies pin
Transformers 4.56.2, PEFT 0.17.1, Accelerate 1.10.1, Hub 0.34.4, and Safetensors
0.6.2 while retaining Colab's CUDA PyTorch (requires >=2.6).

The default mode uses two unchanged suites for every model:

- 256 A/B and full-option prompts: eight families, eight costs
  `(0, 1, 10, 100, 1000, 10000, 100000, 1000000)`, both orders for both readouts.
  Average orders before summarizing the 56 positive-cost cells. Full-option
  margins remain length-normalized preference indices, not calibrated choice
  probabilities. The choice `thresholds.csv` contains per-family/cost averaged
  margins and decisions; no probability threshold is fitted to those indices.
- 192 numerical prompts: eight families, all 24 A-D mappings of `(0, 1, 10, 100)`.
  Normalize within each mapping, then average probabilities by numerical value.
  Report P(0), expected candidate, mode, median, entropy, and all four probabilities.

The notebook displays all four conditions, each treatment minus baseline, and
MSM+AFT minus AFT. Unique score totals are 1,024 choice rows and 3,072 numerical
candidate rows. Baseline scores are computed once and shared across six published
bundles (three comparisons × two suites). In these existing two-role files,
`base` explicitly means the instruction-only adapter; every score also records
its condition and exact Hub model ID/revision.

### Tokenizer and release provenance audit

Downloaded public metadata/tokenizers, without 8B base or released adapter
weights. All four releases have identical tokenizer files and the same custom
chat template. Kept that template exactly, including its role boundaries, and
used our unchanged neutral system prompt. No environmental specification is
inserted into evaluation prompts.

Audited all 448 prompts and every candidate for each real tokenizer: no changed
answer boundary, empty answer, or truncation; A/B and A-D are each one token.
Maximum input length was 293 tokens for choice and 342 for numeric. All four
conditions produced identical input-token hashes:

- choice: `fa4e4f5cc19fda63fed5e4b75bb6db341003dd333c6394c7d5b30825c65fab2d`;
- numeric: `30c6525349a85183372ccce2c2203dce2b44d06b7622f61208cbbf06787b9399`.

The public MSM card says MSM only; its empty `training_complete` marker and the
public source repository do not provide a detailed stage manifest. The paper's
MSM ablation includes instruction tuning, but this saved artifact's precise stage
history is not independently verified. Label it the released MSM ablation and
retain MSM+AFT versus AFT as the main midtraining contrast.

### Persistence, verification, and limits

Every new comparison is completed locally, validated, copied to Drive, flushed,
freshly remounted, and revalidated. Reuse matches exact release/base revisions,
tokenizer audits, case hashes, scoring-code hashes, precision, batch size, and
package/hardware environment. Completed local bundles can recover failed Drive
copies. The existing publication helper now also checks released-model identity
before publishing compact bundles under
`results/harmony_eval/llama31_8b_environment_<condition>/released_environment_msm/`.
No weights or credentials enter those bundles.

Checks: all 141 tests passed, including six new tests of four-condition source
identity, complete score matrices, baseline deduplication, exact reuse,
configuration invalidation, interrupted-run recovery, invalid candidate
boundaries, corrupted weights, non-finite scores, and notebook organization.
A tiny real Llama/PEFT CPU test exercised the fresh-base loader, BF16 adapter
path, and both scorers for all four conditions. The actual notebook comparison
cell executed successfully on synthetic complete bundles; all six plots were
rendered and visually checked. The local macOS plotting check required the Agg
backend; Colab uses its notebook backend. Notebook code parses, training cell
contents match their prior versions, and whitespace checks pass.

No 8B inference, training, or empirical MSM effect was produced. The user's
Llama access request is pending; the public-tokenizer preview can already run.
Next: once approved, add the account's `HF_TOKEN` in Colab, select A100, and run
`notebooks/eval/ecological_eval.ipynb` with defaults. Confirm ordinary preference
transfer separately before interpreting an extreme-case null, and retain these
eight familiar families as exploratory.
