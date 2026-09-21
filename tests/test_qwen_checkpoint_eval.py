import copy
import io
import json
import math
import shutil
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts import qwen_checkpoint_eval as evaluator
from scripts import qwen_checkpoints as checkpoints
from scripts.ecological_dpo.runner import DilemmaDPOConfig
from scripts.ecological_prompt_sft.runner import DilemmaSFTConfig, _required_hashes as training_hashes
from scripts.harmony_eval.scoring import score_loaded_causal_candidates, score_loaded_causal_checkpoint
from scripts.harmony_sft.github_publish import _publication_sources
from scripts.harmony_sft.posthoc_eval import _required_hashes, _sha256_file, _write_csv, _write_json
from tests.test_ecological_dpo import completed_dpo_fixture
from tests.test_ecological_prompt_sft import make_complete_prompt_run
from tests.test_released_environment_eval import Tokenizer, copy_and_validate


def stamp_run(artifacts):
    metadata = json.loads(artifacts.metadata_path.read_text())
    metadata['resolved_revisions'] = {
        checkpoints.MODEL_ID: checkpoints.MODEL_REVISION,
        'final_adapter_sha256': _sha256_file(artifacts.final_adapter_dir / 'adapter_model.safetensors'),
    }
    _write_json(artifacts.metadata_path, metadata)
    marker = json.loads(artifacts.complete_marker_path.read_text())
    marker['artifact_sha256'] = training_hashes(artifacts)
    _write_json(artifacts.complete_marker_path, marker)


def fake_scores(checkpoint, tokenizer, suites, **kwargs):
    result = {}
    identity = checkpoint.identity()
    for suite in suites:
        scorer = score_loaded_causal_checkpoint if suite == 'choice' else score_loaded_causal_candidates
        with patch('scripts.harmony_eval.scoring._score_causal_batch', side_effect=lambda model, tok, items: [
            -1.0 - i % 3 for i, _ in enumerate(items)
        ]):
            rows = scorer(
                model=None, tokenizer=tokenizer, cases=evaluator.current_cases()[suite],
                model_role=checkpoint.condition, model_id=identity['model_id'],
                model_revision=identity['model_revision'], pair_name=identity['pair_name'],
                training_method=checkpoint.training_method, batch_size=2, enable_thinking=False,
            )
        result[suite] = [{**row, 'condition': checkpoint.condition} for row in rows]
    return result


class QwenEvalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sft_config = DilemmaSFTConfig(
            output_root=self.root, model_revision=checkpoints.MODEL_REVISION,
            training_arm='ecological_option',
            dataset_path=Path('data/ecological_dilemmas/sft/ecological_option/records.jsonl'),
        )
        self.sft = make_complete_prompt_run(
            self.root / 'ecological_dilemma_ecological_option_qwen3_8b' / 'sft-run', self.sft_config,
        )
        self.dpo = completed_dpo_fixture(
            self.root / 'ecological_dilemma_ecological_dpo_qwen3_8b' / 'dpo-run', DilemmaDPOConfig(self.root),
        )
        for run in (self.sft, self.dpo):
            stamp_run(run)
        self.selected = checkpoints.resolve_qwen_checkpoints(self.root)
        self.tokenizer = Tokenizer()
        self.audit = evaluator.audit_tokenizer(self.tokenizer, evaluator.current_cases())

    def run_workflow(self, *, selected=None, persist=None, **kwargs):
        with ExitStack() as stack:
            stack.enter_context(patch('torch.cuda.is_available', return_value=True))
            stack.enter_context(patch('torch.cuda.is_bf16_supported', return_value=True))
            stack.enter_context(patch.object(evaluator, '_environment', return_value={'test': True}))
            stack.enter_context(patch.object(evaluator, '_plot', side_effect=lambda rows, suite, path, **kw: path.write_bytes(b'test plot')))
            scorer = stack.enter_context(patch.object(evaluator, '_score_checkpoint', side_effect=fake_scores))
            stack.enter_context(patch.object(evaluator, 'persist_directory_to_colab_drive', side_effect=persist or copy_and_validate))
            result = evaluator.run_qwen_eval(
                checkpoints=self.selected if selected is None else selected,
                local_root=self.root / 'local-evals', drive_root=self.root / 'drive-evals',
                tokenizer=self.tokenizer, tokenizer_audit=self.audit, **kwargs,
            )
            return result, scorer

    def test_base_has_no_adapter_dependency_and_all_sources_use_same_revision(self):
        with patch.object(checkpoints, 'find_sft_checkpoint', side_effect=AssertionError('No SFT lookup')), patch.object(
            checkpoints, 'find_compatible_dpo_run', side_effect=AssertionError('No DPO lookup'),
        ):
            base = checkpoints.resolve_qwen_checkpoints(self.root / 'missing', conditions=('base',))
        self.assertIsNone(base['base'].adapter_path)
        self.assertEqual({s.identity()['base_revision'] for s in self.selected.values()}, {checkpoints.MODEL_REVISION})
        self.assertIn('response_only_sft_v2', self.selected['sft'].training_method)
        self.assertEqual(self.selected['dpo'].training_method, 'paired_option_sigmoid_dpo_v1')
        self.assertEqual(self.selected['sft'].source_config['num_train_epochs'], 3)

    def test_missing_incompatible_or_corrupt_adapters_stop_selection(self):
        for kwargs in ({'sft_checkpoint': 'ecological_option_10_epochs'}, {'dpo_beta': .2}, {'dpo_preferred_side': 'human'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(RuntimeError):
                checkpoints.resolve_qwen_checkpoints(self.root, **kwargs)
        for run, field, value in ((self.sft, 'training_objective', 'ecological_option_response_only_sft_v1'),
                                  (self.dpo, 'score_precision', 'bf16')):
            original = json.loads(run.metadata_path.read_text())
            _write_json(run.metadata_path, {**original, field: value})
            stamp_run(run)  # Genuine compatibility rejection, not just bad hashes.
            with self.assertRaises(RuntimeError):
                checkpoints.resolve_qwen_checkpoints(self.root)
            _write_json(run.metadata_path, original)
            stamp_run(run)
        (self.sft.final_adapter_dir / 'adapter_model.safetensors').write_text('corrupt')
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            checkpoints.verify_checkpoint_files(self.selected['sft'])
        with self.assertRaises(RuntimeError):
            checkpoints.resolve_qwen_checkpoints(self.root)

    def test_complete_matrices_provenance_publication_reuse_and_partial_recovery(self):
        results, score = self.run_workflow()
        self.assertEqual([c.args[0].condition for c in score.call_args_list], ['base', 'sft', 'dpo'])
        for condition, suites in results.items():
            for suite, count in (('choice', 256), ('numeric', 768), ('abstention', 1152)):
                artifact = suites[suite]
                metadata = evaluator.validate_qwen_bundle(artifact)
                self.assertEqual(metadata['checkpoint_signature']['condition'], condition)
                self.assertEqual(len(evaluator._read_rows(artifact)), count)
                self.assertEqual(len(_publication_sources(artifact)), 6)
        repeated, score = self.run_workflow()
        self.assertEqual(results, repeated)
        score.assert_not_called()
        shutil.rmtree(results['dpo']['abstention'].output_dir)
        shutil.rmtree(next((self.root / 'local-evals' / 'dpo' / 'dpo-run').glob('*abstention*')))
        _, score = self.run_workflow()
        self.assertEqual(len(score.call_args_list), 1)
        self.assertEqual(score.call_args.args[0].condition, 'dpo')
        self.assertEqual(score.call_args.args[2], ('abstention',))

    def test_base_only_recovery_after_drive_failure_and_signature_changes(self):
        base = {'base': self.selected['base']}
        def fail(*args, **kwargs):
            raise RuntimeError('Drive unavailable')
        with self.assertRaisesRegex(RuntimeError, 'Drive unavailable'):
            self.run_workflow(selected=base, persist=fail)
        results, score = self.run_workflow(selected=base)
        self.assertEqual(set(results), {'base'})
        score.assert_not_called()
        _, score = self.run_workflow(selected=base, batch_size=1)
        score.assert_called_once()
        self.tokenizer.chat_template = 'changed'
        with self.assertRaisesRegex(RuntimeError, 'Tokenizer changed'):
            self.run_workflow(selected=base)

    def test_forged_identity_probability_and_duplicate_rows_are_rejected(self):
        result, _ = self.run_workflow(selected={'base': self.selected['base']})
        artifact = result['base']['abstention']
        original = evaluator._read_rows(artifact)
        for field, value in (('model_revision', 'wrong'), ('condition', 'dpo'),
                             ('candidate_probability', .99), ('candidate_logprob', float('nan'))):
            rows = copy.deepcopy(original)
            rows[0][field] = value
            _write_csv(artifact.raw_scores_path, rows)
            _write_json(artifact.complete_marker_path, {'status': 'complete', 'artifact_sha256': _required_hashes(artifact)})
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                evaluator.validate_qwen_bundle(artifact)
        rows = copy.deepcopy(original)
        rows[-1] = rows[0]
        _write_csv(artifact.raw_scores_path, rows)
        _write_json(artifact.complete_marker_path, {'status': 'complete', 'artifact_sha256': _required_hashes(artifact)})
        with self.assertRaisesRegex(RuntimeError, 'Duplicate'):
            evaluator.validate_qwen_bundle(artifact)

    def test_notebook_comparisons_and_publication_for_all_conditions_and_subsets(self):
        import pandas as pd
        result, _ = self.run_workflow()
        notebook = json.loads(Path('notebooks/eval/ecological_eval.ipynb').read_text())
        for conditions in (('base', 'sft', 'dpo'), ('base',), ('dpo',)):
            namespace = {'EVAL_SOURCE': 'qwen', 'qwen_results': {k: result[k] for k in conditions},
                         'qwen_checkpoints': self.selected, 'pd': pd,
                         'display': lambda *a: None, 'Markdown': lambda x: x, 'Image': lambda **kw: None}
            with redirect_stdout(io.StringIO()):
                exec(compile(''.join(notebook['cells'][15]['source']), 'qwen-summary', 'exec'), namespace)
            self.assertEqual(len(namespace['qwen_abstention']), 64 * len(conditions))
            self.assertEqual(len(namespace['qwen_numeric']), 8 * len(conditions))
            namespace.update(PUBLISH_TO_GITHUB=True, GITHUB_REPOSITORY='test/test', GITHUB_BRANCH='main',
                             GITHUB_TOKEN='test', REPO_DIR=self.root)
            with patch('scripts.ecological_prompt_sft.publish_results_to_github') as publish, redirect_stdout(io.StringIO()):
                exec(compile(''.join(notebook['cells'][18]['source']), 'qwen-publish', 'exec'), namespace)
            self.assertEqual(publish.call_count, 3 * len(conditions))

    def test_fresh_qwen_base_per_condition_and_real_bf16_adapter_scoring(self):
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import Qwen3Config, Qwen3ForCausalLM
        config = Qwen3Config(vocab_size=256, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                             head_dim=8, max_position_embeddings=4096, pad_token_id=0)
        for run in (self.sft, self.dpo):
            model = get_peft_model(Qwen3ForCausalLM(config), LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj']))
            model.save_pretrained(run.final_adapter_dir)
            stamp_run(run)
        selected = checkpoints.resolve_qwen_checkpoints(self.root)
        loaded = []
        def load(*args, **kwargs):
            model = Qwen3ForCausalLM(config).to(torch.bfloat16)
            loaded.append(model)
            return model
        cases = {k: [v[0]] for k, v in evaluator.current_cases().items()}
        with ExitStack() as stack:
            stack.enter_context(patch.object(evaluator, 'current_cases', return_value=cases))
            loader = stack.enter_context(patch('transformers.AutoModelForCausalLM.from_pretrained', side_effect=load))
            adapters = stack.enter_context(patch('peft.PeftModel.from_pretrained', wraps=PeftModel.from_pretrained))
            stack.enter_context(patch('torch.cuda.reset_peak_memory_stats'))
            stack.enter_context(patch('torch.cuda.max_memory_allocated', return_value=0))
            for checkpoint in selected.values():
                result = evaluator._score_checkpoint(checkpoint, self.tokenizer, tuple(evaluator.SUITES), batch_size=2, token=None)
                self.assertEqual(len(result['abstention']), 3)
                self.assertTrue(math.isclose(sum(r['candidate_probability'] for r in result['abstention']), 1.0))
        self.assertEqual(loader.call_count, 3)
        self.assertEqual(adapters.call_count, 2)
        self.assertEqual(len({id(m) for m in loaded}), 3)
        self.assertEqual([c.args[1] for c in adapters.call_args_list], [selected[k].adapter_path for k in ('sft', 'dpo')])
        self.assertTrue(all(not p.requires_grad for m in loaded for p in m.parameters()))
        self.assertTrue(all(not c.kwargs['autocast_adapter_dtype'] for c in adapters.call_args_list))


if __name__ == '__main__':
    unittest.main()
