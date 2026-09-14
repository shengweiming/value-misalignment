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

from scripts import llama_instruct_eval as instruct
from scripts.harmony_eval.scoring import score_loaded_causal_candidates, score_loaded_causal_checkpoint
from scripts.harmony_sft.github_publish import _publication_sources
from scripts.harmony_sft.posthoc_eval import _required_hashes, _write_csv, _write_json
from tests.test_released_environment_eval import Tokenizer, copy_and_validate


def fake_scores(tokenizer, suites, **kwargs):
    result = {}
    for suite in suites:
        scorer = score_loaded_causal_candidates if suite == 'numeric' else score_loaded_causal_checkpoint
        with patch('scripts.harmony_eval.scoring._score_causal_batch', side_effect=lambda model, tok, items: [
            -1.0 - i % 3 for i, _ in enumerate(items)
        ]):
            rows = scorer(
                model=None, tokenizer=tokenizer, cases=instruct.current_cases()[suite],
                model_role=instruct.CONDITION, model_id=instruct.MODEL_ID,
                model_revision=instruct.MODEL_REVISION, pair_name=instruct.PAIR_NAME,
                training_method='official_instruct', batch_size=2, enable_thinking=False,
            )
        result[suite] = [{**row, 'condition': instruct.CONDITION} for row in rows]
    return result


class InstructEvalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tokenizer = Tokenizer()
        self.audit = instruct.audit_tokenizer(self.tokenizer, instruct.current_cases())

    def run_workflow(self, *, persist=None, **kwargs):
        with ExitStack() as stack:
            stack.enter_context(patch('torch.cuda.is_available', return_value=True))
            stack.enter_context(patch('torch.cuda.is_bf16_supported', return_value=True))
            stack.enter_context(patch.object(instruct, '_environment', return_value={'test': True}))
            stack.enter_context(patch.object(instruct, '_plot', side_effect=lambda rows, suite, path: path.write_bytes(b'test plot')))
            scorer = stack.enter_context(patch.object(instruct, '_score_instruct', side_effect=fake_scores))
            stack.enter_context(patch.object(instruct, 'persist_directory_to_colab_drive', side_effect=persist or copy_and_validate))
            result = instruct.run_instruct_eval(
                local_root=self.root/'local', drive_root=self.root/'drive',
                tokenizer=self.tokenizer, tokenizer_audit=self.audit, **kwargs,
            )
            return result, scorer

    def test_single_model_complete_matrix_publication_reuse_and_invalidation(self):
        result, scorer = self.run_workflow()
        scorer.assert_called_once()
        for suite, count in [('choice', 256), ('numeric', 768)]:
            artifact = result[suite]
            metadata = instruct.validate_instruct_bundle(artifact)
            self.assertEqual(metadata['model_roles'], {'llama_instruct': instruct.MODEL_ID})
            self.assertEqual(len(instruct._read_rows(artifact)), count)
            self.assertEqual(len(_publication_sources(artifact)), 6)
        repeated, scorer = self.run_workflow()
        self.assertEqual(result, repeated)
        scorer.assert_not_called()
        _, scorer = self.run_workflow(batch_size=1)
        scorer.assert_called_once()
        _, scorer = self.run_workflow(force_evaluation=True)
        scorer.assert_called_once()

    def test_recover_both_local_bundles_after_failed_drive_copy(self):
        def fail(*args, **kwargs):
            raise RuntimeError('Drive unavailable')
        with self.assertRaisesRegex(RuntimeError, 'Drive unavailable'):
            self.run_workflow(persist=fail)
        self.assertEqual(len(list((self.root/'local').glob('*'))), 2)
        result, scorer = self.run_workflow()
        self.assertEqual(set(result), {'choice', 'numeric'})
        scorer.assert_not_called()
        # If only one suite remains anywhere, score only the missing suite.
        shutil.rmtree(result['numeric'].output_dir)
        shutil.rmtree(next((self.root/'local').glob('*numeric*')))
        _, scorer = self.run_workflow()
        self.assertEqual(scorer.call_args.args[1], ('numeric',))

    def test_reject_wrong_identity_mapping_duplicate_rows_nonfinite_and_bad_math(self):
        result, _ = self.run_workflow()
        for suite, field, value in [
            ('choice', 'model_id', 'chloeli/llama-3.1-8b-baseline'),
            ('choice', 'model_revision', 'wrong'),
            ('choice', 'model_role', 'base'),
            ('choice', 'semantic_logit_implement', float('nan')),
            ('choice', 'semantic_logit_implement', 0.123),
            ('choice', 'prompt', 'wrong prompt'),
            ('numeric', 'candidate_value', 999),
            ('numeric', 'candidate_probability', 0.9),
            ('numeric', 'candidate_logprob', float('-inf')),
        ]:
            artifact = result[suite]
            original = instruct._read_rows(artifact)
            with self.subTest(suite=suite, field=field):
                rows = copy.deepcopy(original)
                rows[0][field] = value
                _write_csv(artifact.raw_scores_path, rows)
                _write_json(artifact.complete_marker_path, {'status':'complete', 'artifact_sha256':_required_hashes(artifact)})
                with self.assertRaises(RuntimeError):
                    instruct.validate_instruct_bundle(artifact)
            _write_csv(artifact.raw_scores_path, original)
            _write_json(artifact.complete_marker_path, {'status':'complete', 'artifact_sha256':_required_hashes(artifact)})
        artifact = result['numeric']
        rows = instruct._read_rows(artifact)
        rows[-1] = rows[0]
        _write_csv(artifact.raw_scores_path, rows)
        _write_json(artifact.complete_marker_path, {'status':'complete', 'artifact_sha256':_required_hashes(artifact)})
        with self.assertRaisesRegex(RuntimeError, 'Duplicate'):
            instruct.validate_instruct_bundle(artifact)

    def test_audit_change_rejected_before_loading_model(self):
        self.tokenizer.chat_template = 'changed-template'
        with self.assertRaisesRegex(RuntimeError, 'Tokenizer changed'):
            self.run_workflow()

    def test_native_tokenizer_from_pinned_official_repo(self):
        for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json'):
            (self.root/name).write_text('{}')
        with patch('huggingface_hub.snapshot_download', return_value=str(self.root)) as download, patch(
            'transformers.AutoTokenizer.from_pretrained', return_value=self.tokenizer,
        ) as load:
            tokenizer, audit = instruct.prepare_instruct_tokenizer(token='test-token')
        self.assertIs(tokenizer, self.tokenizer)
        self.assertEqual(download.call_args.args, (instruct.MODEL_ID,))
        self.assertEqual(download.call_args.kwargs['revision'], instruct.MODEL_REVISION)
        self.assertEqual(download.call_args.kwargs['token'], 'test-token')
        self.assertTrue(load.call_args.kwargs['local_files_only'])
        self.assertEqual(len(audit['file_sha256']), 3)
        self.assertEqual(audit['chat_template'], self.tokenizer.chat_template)

    def test_plain_bf16_checkpoint_scores_without_peft(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        config = LlamaConfig(vocab_size=256, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                             max_position_embeddings=4096, pad_token_id=0)
        model = LlamaForCausalLM(config).to(torch.bfloat16)
        cases = {key: [value[0]] for key, value in instruct.current_cases().items()}
        with ExitStack() as stack:
            stack.enter_context(patch.object(instruct, 'current_cases', return_value=cases))
            load = stack.enter_context(patch('transformers.AutoModelForCausalLM.from_pretrained', return_value=model))
            peft = stack.enter_context(patch('peft.PeftModel.from_pretrained', side_effect=AssertionError('No PEFT allowed')))
            stack.enter_context(patch('torch.cuda.reset_peak_memory_stats'))
            stack.enter_context(patch('torch.cuda.max_memory_allocated', return_value=0))
            rows = instruct._score_instruct(self.tokenizer, ('choice', 'numeric'), batch_size=2, token=None)
        peft.assert_not_called()
        self.assertEqual(load.call_args.args, (instruct.MODEL_ID,))
        self.assertEqual(load.call_args.kwargs['revision'], instruct.MODEL_REVISION)
        self.assertEqual(load.call_args.kwargs['dtype'], torch.bfloat16)
        self.assertEqual(load.call_args.kwargs['device_map'], {'': 0})
        self.assertEqual(len(rows['choice']), 1)
        self.assertEqual(len(rows['numeric']), 4)
        self.assertTrue(math.isclose(sum(r['candidate_probability'] for r in rows['numeric']), 1.0))
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        self.assertFalse(model.training)

    def test_notebook_summaries_plots_and_publication_for_single_model(self):
        result, _ = self.run_workflow()
        import pandas as pd
        notebook = json.loads(Path('notebooks/eval/ecological_eval.ipynb').read_text())
        code = ''.join(notebook['cells'][15]['source'])
        namespace = {'EVAL_SOURCE': 'llama_instruct', 'instruct_results': result,
                     'pd': pd, 'display': lambda *a: None, 'Markdown': lambda x: x, 'Image': lambda **kw: None}
        with redirect_stdout(io.StringIO()):
            exec(compile(code, 'instruct-summary-cell', 'exec'), namespace)
        self.assertEqual(list(namespace['choice_summary_table'].cells), [56, 56])
        self.assertEqual(list(namespace['ab_order_summary'].cells), [56, 56])
        for suite, artifact in result.items():
            plot = self.root/f'{suite}.png'
            instruct._plot(instruct._read_rows(artifact), suite, plot)
            self.assertGreater(plot.stat().st_size, 1000)
        namespace.update(PUBLISH_TO_GITHUB=True, INSTRUCT_SOURCE_RUN_NAME=instruct.SOURCE_RUN_NAME,
                         GITHUB_REPOSITORY='test/test', GITHUB_BRANCH='main', GITHUB_TOKEN='test', REPO_DIR=self.root)
        with patch('scripts.ecological_prompt_sft.publish_results_to_github') as publish, redirect_stdout(io.StringIO()):
            exec(compile(''.join(notebook['cells'][18]['source']), 'instruct-publish-cell', 'exec'), namespace)
        self.assertEqual(publish.call_count, 2)
        self.assertTrue(all(call.kwargs['source_run_name'] == instruct.SOURCE_RUN_NAME for call in publish.call_args_list))


if __name__ == '__main__':
    unittest.main()
