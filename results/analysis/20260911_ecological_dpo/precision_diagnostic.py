"""Reproduce the reference/policy precision asymmetry without an optimizer step.

Uses the pinned DPO dependencies and a tiny random BF16 Qwen3 on CPU. It applies
Accelerate's real FP32 output converter to the same unchanged model, isolating
this conversion from GPU/autocast differences. This is not a measurement of the
error on the trained 8B model. Run this file directly from any directory.
"""
import json
import sys
import tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
import torch
from accelerate.utils import convert_outputs_to_fp32
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM, set_seed
from scripts.ecological_dpo import DilemmaDPOConfig, render_preference_examples
from scripts.ecological_dpo.runner import build_trainer, precompute_reference_audit

torch.set_num_threads(1)
set_seed(42)
vocab = {t: i for i, t in enumerate(['<pad>', '<eos>', '<unk>', '<assistant>', 'which', 'policy', 'protect', 'trees', 'people'])}
raw = Tokenizer(WordLevel(vocab, unk_token='<unk>'))
raw.pre_tokenizer = WhitespaceSplit()
tok = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token='<pad>', eos_token='<eos>', unk_token='<unk>')
tok.chat_template = "{{ messages[0]['content'] }} <assistant> "
cfg = Qwen3Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=16, max_position_embeddings=128, pad_token_id=0, eos_token_id=1, use_cache=False)
examples = [{'id': str(i), 'dilemma': 'which policy', 'chosen': 'protect trees ' * (12+i), 'rejected': 'protect people policy ' * (8+i)} for i in range(4)]
rendered, tokens, _ = render_preference_examples(tok, examples, max_length=64)
with tempfile.TemporaryDirectory() as tmp:
    model = Qwen3ForCausalLM(cfg).to(torch.bfloat16)
    trainer = build_trainer(model, tok, rendered, tokens, DilemmaDPOConfig(tmp, max_length=64, lora_rank=2, lora_alpha=4), Path(tmp)/'checkpoints', smoke_test=True)
    # This historical diagnostic intentionally reconstructs the pre-fix path.
    # Production training must retain this hook; the regression tests cover it.
    precision_hook = getattr(trainer, '_dpo_fp32_logits_hook', None)
    if precision_hook is not None:
        precision_hook.remove()
    refs = precompute_reference_audit(trainer, rendered)
    batch = trainer.data_collator(list(trainer.train_dataset))
    trainer.model.eval()
    with torch.no_grad():
        raw_out = trainer.concatenated_forward(trainer.model, batch)
    trainer.model.forward = convert_outputs_to_fp32(trainer.model.forward)
    with torch.no_grad():
        promoted = trainer.concatenated_forward(trainer.model, batch)
    result = {'torch':torch.__version__, 'model_dtype':'bfloat16', 'description':'Same BF16 model, zero adapter update; only Accelerate output conversion to FP32 changes.', 'rows':[]}
    for i, ref in enumerate(refs):
        delta = (promoted['chosen_logps'][i] - promoted['rejected_logps'][i]).item() - (ref['ref_chosen_logps'] - ref['ref_rejected_logps'])
        result['rows'].append(dict(ref, raw_chosen=float(raw_out['chosen_logps'][i]), raw_rejected=float(raw_out['rejected_logps'][i]), promoted_chosen=float(promoted['chosen_logps'][i]), promoted_rejected=float(promoted['rejected_logps'][i]), spurious_reward_margin=0.1*delta))
    assert all(r['raw_chosen']==r['ref_chosen_logps'] and r['raw_rejected']==r['ref_rejected_logps'] for r in result['rows'])
    assert any(abs(r['spurious_reward_margin']) > 1e-3 for r in result['rows'])
    print(json.dumps(result,indent=2))
    (HERE / 'precision_diagnostic_result.json').write_text(json.dumps(result,indent=2)+'\n')
