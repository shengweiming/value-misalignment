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
