import ast
import copy
import json
import math
import shutil
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from scripts import released_environment_eval as release
from scripts.harmony_eval.scoring import (
    score_loaded_causal_candidates, score_loaded_causal_checkpoint,
)
from scripts.harmony_sft.github_publish import _publication_sources
from scripts.harmony_sft.posthoc_eval import _required_hashes, _write_csv, _write_json


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 1
    padding_side = 'right'
    chat_template = 'test-template'

    def apply_chat_template(self, messages, **kwargs):
        return f"system:{messages[0]['content']}\nuser:{messages[1]['content']}\nassistant:"

    def encode(self, text, *, add_special_tokens=False):
        return list(text.encode())


def fake_scores(key, tokenizer, suites, **kwargs):
    output = {}
    spec = release.RELEASES[key]
    for suite in suites:
        scorer = score_loaded_causal_candidates if suite == 'numeric' else score_loaded_causal_checkpoint
        with patch('scripts.harmony_eval.scoring._score_causal_batch', side_effect=lambda model, tok, items: [
            -1.0 - i % 3 for i, _ in enumerate(items)
        ]):
            rows = scorer(
                model=None, tokenizer=tokenizer, cases=release.current_cases()[suite],
                model_role='base' if key == 'baseline' else 'aligned', model_id=spec.repo_id,
                model_revision=spec.revision, pair_name='test', training_method='released_msm_aft',
                batch_size=2, enable_thinking=False,
            )
        output[suite] = [{**row, 'condition': key} for row in rows]
    return output


def copy_and_validate(source, root, *, validate_directory, **kwargs):
    validate_directory(source)
    destination = root / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, dirs_exist_ok=True)
    validate_directory(destination)
    return destination


class ReleasedEvalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.tokenizers = {key: Tokenizer() for key in release.RELEASES}
        self.audits = {key: release.audit_tokenizer(tokenizer, release.current_cases()) for key, tokenizer in self.tokenizers.items()}

    def run_workflow(self, *, score=None, persist=None, **kwargs):
        with ExitStack() as stack:
            stack.enter_context(patch('torch.cuda.is_available', return_value=True))
            stack.enter_context(patch('torch.cuda.is_bf16_supported', return_value=True))
            stack.enter_context(patch.object(release, '_environment', return_value={'test': True}))
            stack.enter_context(patch.object(release, '_plot', side_effect=lambda rows, suite, path, treatment: path.write_bytes(b'test plot')))
            scorer = stack.enter_context(patch.object(release, '_score_release', side_effect=score or fake_scores))
            stack.enter_context(patch.object(release, 'persist_directory_to_colab_drive', side_effect=persist or copy_and_validate))
            result = release.run_released_environment_eval(
                local_root=self.root/'local', drive_root=self.root/'drive',
                tokenizers=self.tokenizers, tokenizer_audits=self.audits, **kwargs,
            )
            return result, scorer

    def test_four_conditions_full_matrix_publication_and_reuse(self):
        result, scorer = self.run_workflow()
        self.assertEqual([call.args[0] for call in scorer.call_args_list], list(release.RELEASES))
        for treatment, suites in result.items():
            for suite, artifacts in suites.items():
                metadata = release.validate_released_bundle(artifacts)
                self.assertEqual(metadata['treatment'], treatment)
                self.assertEqual(metadata['baseline_condition'], 'baseline')
                self.assertEqual(len(_publication_sources(artifacts)), 6)
        choices = release.collect_condition_rows(result, 'choice')
        numeric = release.collect_condition_rows(result, 'numeric')
        self.assertEqual(len(choices), 1024)
        self.assertEqual(len(numeric), 3072)
        self.assertEqual(sum(r['condition'] == 'baseline' for r in choices), 256)
        self.assertEqual(len(release.choice_summary(choices)), 512)
        with self.subTest('no scoring on exact rerun'):
            reused, scorer = self.run_workflow()
            self.assertEqual(result, reused)
            scorer.assert_not_called()
        with self.subTest('batch change invalidates reuse'):
            _, scorer = self.run_workflow(batch_size=1)
            self.assertEqual(scorer.call_count, 4)

    def test_partial_run_recovers_baseline_and_local_results(self):
        def interrupted_score(key, *args, **kwargs):
            if key == 'aft':
                raise RuntimeError('simulated interruption')
            return fake_scores(key, *args, **kwargs)
        with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
            self.run_workflow(score=interrupted_score)
        # Drop one Drive copy to exercise recovery from the complete local bundle.
        shutil.rmtree(next((self.root/'drive'/'msm').glob('*numeric*')))
        result, scorer = self.run_workflow()
        self.assertEqual([c.args[0] for c in scorer.call_args_list], ['aft', 'msm_aft'])
        self.assertEqual(set(result), set(release.TREATMENTS))

    def test_forged_baseline_identity_and_nonfinite_scores_are_rejected(self):
        result, _ = self.run_workflow()
        artifact = result['aft']['choice']
        original = release._read_rows(artifact)
        for field, value, message in (
            ('model_id', release.BASE_MODEL, 'recorded released condition'),
            ('model_revision', 'wrong-revision', 'recorded released condition'),
            ('semantic_logit_implement', float('nan'), 'Non-finite'),
        ):
            rows = copy.deepcopy(original)
            rows[0][field] = value
            _write_csv(artifact.raw_scores_path, rows)
            _write_json(artifact.complete_marker_path, {'status':'complete', 'artifact_sha256':_required_hashes(artifact)})
            with self.assertRaisesRegex(RuntimeError, message):
                release.validate_released_bundle(artifact)
        _write_csv(artifact.raw_scores_path, original)
        _write_json(artifact.complete_marker_path, {'status':'complete', 'artifact_sha256':_required_hashes(artifact)})
        altered = copy.deepcopy(release.validate_released_bundle(artifact)['release_signature'])
        altered['adapter_dtype'] = 'float32'
        with self.assertRaisesRegex(RuntimeError, 'signature changed'):
            release.validate_released_bundle(artifact, signature=altered)

    def test_audit_rejects_changed_boundary_and_multitoken_labels(self):
        class BrokenBoundary(Tokenizer):
            def encode(self, text, **kwargs):
                ids = super().encode(text, **kwargs)
                return [999, *ids] if text[-1] in 'ABCD' else ids
        with self.assertRaisesRegex(RuntimeError, 'boundary'):
            release.audit_tokenizer(BrokenBoundary(), release.current_cases())
        class MultipleTokens(Tokenizer):
            def encode(self, text, **kwargs):
                ids = super().encode(text, **kwargs)
                return ids + [999] if text[-1] in 'ABCD' else ids
        with self.assertRaisesRegex(RuntimeError, 'exactly one token'):
            release.audit_tokenizer(MultipleTokens(), release.current_cases())

    def test_four_releases_use_pinned_pretrained_base_and_fresh_adapters(self):
        import torch
        from peft import LoraConfig, get_peft_model, PeftModel
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(42)
        config = LlamaConfig(vocab_size=256, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                             max_position_embeddings=4096, pad_token_id=0)
        template = LlamaForCausalLM(config)
        initial = copy.deepcopy(template.state_dict())
        local_specs, paths = {}, {}
        for i, (key, spec) in enumerate(release.RELEASES.items()):
            base = LlamaForCausalLM(config)
            base.load_state_dict(initial)
            adapter = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj'], task_type='CAUSAL_LM'))
            with torch.no_grad():
                for name, param in adapter.named_parameters():
                    if 'lora_B' in name:
                        param.fill_(0.02 * (i+1))
            path = self.root/key
            adapter.save_pretrained(path)
            paths[key] = path/'adapter_model.safetensors'
            local_specs[key] = release.ReleasedAdapter(spec.repo_id, spec.revision, release._sha256_file(paths[key]), spec.label)
        def download(repo, name, **kwargs):
            if repo == release.BASE_MODEL:
                self.assertEqual(kwargs['revision'], release.BASE_REVISION)
                return str(self.root/'unused-config.json')
            key = next(k for k, spec in local_specs.items() if spec.repo_id == repo)
            self.assertEqual(kwargs['revision'], local_specs[key].revision)
            return str(paths[key])
        loaded_bases = []
        def load_base(repo, **kwargs):
            self.assertEqual(repo, release.BASE_MODEL)
            self.assertEqual(kwargs['revision'], release.BASE_REVISION)
            self.assertEqual(kwargs['dtype'], torch.bfloat16)
            self.assertEqual(kwargs['device_map'], {'': 0})
            base = LlamaForCausalLM(config).to(torch.bfloat16)
            base.load_state_dict(initial)
            loaded_bases.append(base)
            return base
        real_load = PeftModel.from_pretrained
        def load_adapter(base, path, **kwargs):
            self.assertNotIsInstance(base, PeftModel)
            self.assertFalse(kwargs['autocast_adapter_dtype'])
            self.assertFalse(kwargs['is_trainable'])
            return real_load(base, path, **kwargs)
        cases = release.current_cases()
        cases = {key: [value[0]] for key,value in cases.items()}
        with ExitStack() as stack:
            stack.enter_context(patch.object(release, 'RELEASES', local_specs))
            stack.enter_context(patch.object(release, 'current_cases', return_value=cases))
            stack.enter_context(patch('huggingface_hub.hf_hub_download', side_effect=download))
            stack.enter_context(patch('transformers.AutoModelForCausalLM.from_pretrained', side_effect=load_base))
            stack.enter_context(patch('peft.PeftModel.from_pretrained', side_effect=load_adapter))
            stack.enter_context(patch('torch.cuda.reset_peak_memory_stats'))
            stack.enter_context(patch('torch.cuda.max_memory_allocated', return_value=0))
            for key in local_specs:
                rows = release._score_release(key, Tokenizer(), ('choice','numeric'), batch_size=2, token=None)
                self.assertEqual(len(rows['choice']), 1)
                self.assertEqual(len(rows['numeric']), 4)
                self.assertTrue(math.isclose(sum(r['candidate_probability'] for r in rows['numeric']),1.0))
            self.assertEqual(len({id(model) for model in loaded_bases}),4)
            paths['baseline'].write_bytes(b'corrupt')
            with self.assertRaisesRegex(RuntimeError, 'SHA-256'):
                release._score_release('baseline', Tokenizer(), ('numeric',), batch_size=2, token=None)

    def test_notebook_layout_and_source_modes(self):
        expected = {
            'training/ecological_dpo.ipynb', 'training/ecological_sft.ipynb',
            'training/clash_sft.ipynb', 'eval/ecological_eval.ipynb',
        }
        actual = {str(p.relative_to('notebooks')) for p in Path('notebooks').rglob('*.ipynb')}
        self.assertEqual(actual, expected)
        for path in Path('notebooks').rglob('*.ipynb'):
            notebook = json.loads(path.read_text())
            self.assertEqual(notebook['metadata']['colab']['name'],path.name)
            for cell in notebook['cells']:
                if cell['cell_type'] == 'code':
                    ast.parse(''.join(cell['source']))
                    self.assertIsNone(cell['execution_count'])
                    self.assertEqual(cell['outputs'],[])
        notebook = json.loads(Path('notebooks/eval/ecological_eval.ipynb').read_text())
        code = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
        self.assertIn('EVAL_SOURCE = "llama_instruct"',code)
        self.assertIn('run_instruct_eval(',code)
        self.assertIn('run_released_environment_eval(',code)
        self.assertIn('run_numeric_threshold_workflow(',code)
        self.assertIn('("msm_aft", "aft")',code)
        self.assertNotIn('run_dilemma_sft(',code)
        self.assertNotIn('run_dilemma_dpo(',code)


if __name__ == '__main__':
    unittest.main()
