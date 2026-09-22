import copy
import io
import json
import math
import tempfile
import unittest
from collections import Counter
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts import ecological_indifference as suite
from scripts import qwen_checkpoint_eval as evaluator
from scripts.qwen_checkpoints import resolve_qwen_checkpoints
from scripts.harmony_eval.scoring import score_loaded_causal_candidates
from scripts.harmony_sft.github_publish import _publication_sources
from scripts.harmony_sft.posthoc_eval import _required_hashes, _write_csv, _write_json
from tests.test_released_environment_eval import Tokenizer, copy_and_validate


def fake_scores(checkpoint, tokenizer, suites, *, case_sets, **kwargs):
    result = {}
    identity = checkpoint.identity()
    for name in suites:
        # A fixed semantic preference must survive every label permutation.
        with patch('scripts.harmony_eval.scoring._score_causal_batch', side_effect=lambda m, t, items: [
            math.log({0: .55, 1: .35, 2: .10}[item['candidate_value']]) for item in items
        ]):
            rows = score_loaded_causal_candidates(
                model=None, tokenizer=tokenizer, cases=case_sets[name], model_role=checkpoint.condition,
                model_id=identity['model_id'], model_revision=identity['model_revision'],
                pair_name=identity['pair_name'], training_method=checkpoint.training_method,
                batch_size=2, enable_thinking=False,
            )
        result[name] = [{**r, 'condition': checkpoint.condition} for r in rows]
    return result


class IndifferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = suite.load_config()
        self.tokenizer = Tokenizer()
        self.checkpoints = resolve_qwen_checkpoints(self.root, conditions=('base',))

    def workflow(self, *, config=None, persist=None, **kwargs):
        config = self.config if config is None else config
        audit = evaluator.audit_tokenizer(self.tokenizer, suite.build_cases(config))
        with ExitStack() as stack:
            stack.enter_context(patch('torch.cuda.is_available', return_value=True))
            stack.enter_context(patch('torch.cuda.is_bf16_supported', return_value=True))
            stack.enter_context(patch.object(evaluator, '_environment', return_value={'test': True}))
            stack.enter_context(patch.object(evaluator, '_plot', side_effect=lambda r, s, p, **kw: p.write_bytes(b'plot')))
            score = stack.enter_context(patch.object(evaluator, '_score_checkpoint', side_effect=fake_scores))
            stack.enter_context(patch.object(evaluator, 'persist_directory_to_colab_drive', side_effect=persist or copy_and_validate))
            result = evaluator.run_qwen_eval(
                checkpoints=self.checkpoints, local_root=self.root/'local', drive_root=self.root/'drive',
                tokenizer=self.tokenizer, tokenizer_audit=audit, indifference_config=config, **kwargs,
            )
        return result, score

    def test_two_arms_keep_other_cost_fixed_and_anchor_is_unique(self):
        points = suite.sweep_points(self.config)
        self.assertEqual(len(points), 47)
        self.assertEqual(len({(p['template_family'], p['human_cost'], p['environment_cost']) for p in points}), 47)
        self.assertEqual(Counter(p['template_family'] for p in points if p['sweep_arm']=='anchor'),
                         Counter({f: 1 for f in suite.FAMILIES}))
        for p in points:
            if p['sweep_arm'] == 'environment':
                self.assertEqual(p['human_cost'], p['human_anchor'])
                self.assertGreater(p['environment_cost'], p['environment_anchor'])
                self.assertLess(p['signed_log10_multiplier'], 0)
            elif p['sweep_arm'] == 'human':
                self.assertEqual(p['environment_cost'], p['environment_anchor'])
                self.assertGreater(p['human_cost'], p['human_anchor'])
                self.assertGreater(p['signed_log10_multiplier'], 0)
            else:
                self.assertEqual(p['signed_log10_multiplier'], 0)

    def test_complete_counterbalancing_and_preserved_human_assumptions(self):
        cases = suite.build_cases(self.config)
        for name, expected, permutations in (('indifference_ab', 94, 2), ('indifference_abc', 282, 6)):
            self.assertEqual(len(cases[name]), expected)
            self.assertEqual(len({c['case_id'] for c in cases[name]}), expected)
            by_point = {}
            for case in cases[name]:
                key = (case['template_family'], case['human_cost'], case['environment_cost'])
                by_point.setdefault(key, []).append(case)
                old = (suite.ROOT.parent/'extreme_v2'/f"{case['template_family']}.txt").read_text().split('\n\n')[1]
                self.assertEqual(case['prompt'].split('\n\n')[1], old.format(cost=case['human_cost']))
                self.assertNotIn('{', case['prompt'])
            for group in by_point.values():
                self.assertEqual(len(group), permutations)
                for value in (0, 1):
                    labels = Counter(c['text'] for r in group for c in r['candidates'] if c['value'] == value)
                    self.assertEqual(len(set(labels.values())), 1)

    def test_editable_anchors_inserted_without_needing_to_be_in_grid(self):
        config = copy.deepcopy(self.config)
        config['families']['pesticide_ban'] = {'human_anchor': 3, 'environment_anchor': 7}
        points = [p for p in suite.sweep_points(config) if p['template_family']=='pesticide_ban']
        anchor = [p for p in points if p['sweep_arm']=='anchor']
        self.assertEqual([(p['human_cost'],p['environment_cost']) for p in anchor], [(3,7)])
        self.assertTrue(any(p['human_cost']==10 and p['environment_cost']==7 for p in points))
        self.assertTrue(any(p['environment_cost']==10 and p['human_cost']==3 for p in points))
        invalid = []
        for field, value in (('human_costs', [0, 1]), ('environment_costs', [10,10]),
                             ('environment_unit', 'animals'), ('human_costs', [True,100])):
            c = copy.deepcopy(config); c[field] = value; invalid.append(c)
        c = copy.deepcopy(config);c['families']['pesticide_ban']['human_anchor']=1000000;invalid.append(c)
        for c in invalid:
            with self.subTest(c=c), self.assertRaises(ValueError): suite.validate_config(c)

    def test_semantic_probability_averages_keep_environment_points_separate(self):
        rows = fake_scores(self.checkpoints['base'], self.tokenizer, suite.SUITES, case_sets=suite.build_cases(self.config))
        for name in suite.SUITES:
            summary = suite.summarize(rows[name], name)
            self.assertEqual(len(summary),47)
            for r in summary:
                self.assertTrue(r['unanimous_preference'])
                self.assertAlmostEqual(r['probability_ecological'], .55 if name=='indifference_abc' else .55/.9)
                self.assertAlmostEqual(r['min_probability_ecological'], r['max_probability_ecological'])
            with self.assertRaises(RuntimeError): suite.summarize(rows[name][1:],name)
            with self.assertRaises(RuntimeError): suite.summarize(rows[name]+[rows[name][0]],name)

    def test_completion_publication_recovery_and_changed_config(self):
        result, score = self.workflow()
        score.assert_called_once()
        for name, artifact in result['base'].items():
            metadata = evaluator.validate_qwen_bundle(artifact)
            self.assertEqual(metadata['indifference_config'], self.config)
            self.assertEqual(len(_publication_sources(artifact)), 6)
        again, score = self.workflow()
        self.assertEqual(again, result);score.assert_not_called()
        config = copy.deepcopy(self.config);config['families']['pesticide_ban']['environment_anchor']=7
        changed, score = self.workflow(config=config)
        score.assert_called_once()
        self.assertNotEqual(changed, result)
        # A missing Drive bundle is recovered from its verified local copy.
        import shutil
        shutil.rmtree(result['base']['indifference_ab'].output_dir)
        recovered, score = self.workflow()
        score.assert_not_called()
        evaluator.validate_qwen_bundle(recovered['base']['indifference_ab'])

    def test_counterbalanced_half_probability_preserves_order_disagreement(self):
        rows = fake_scores(self.checkpoints['base'], self.tokenizer, ('indifference_ab',),
                           case_sets=suite.build_cases(self.config))['indifference_ab']
        for row in rows:
            probability = .99 if row['candidate_text'] == 'A' else .01
            row.update(candidate_probability=probability, candidate_logprob=math.log(probability),
                       candidate_mean_logprob=math.log(probability))
        for row in suite.summarize(rows, 'indifference_ab'):
            self.assertAlmostEqual(row['probability_ecological'], .5)
            self.assertAlmostEqual(row['min_probability_ecological'], .01)
            self.assertAlmostEqual(row['max_probability_ecological'], .99)
            self.assertFalse(row['unanimous_preference'])
            self.assertEqual(row['preferred_response'], 'tie')
            self.assertEqual(row['permutations_preferring_ecological'], 1)
            self.assertEqual(row['permutations_preferring_human'], 1)

    def test_forged_environment_coordinates_and_candidate_scores_are_rejected(self):
        result, _ = self.workflow()
        artifact = result['base']['indifference_ab']
        original = evaluator._read_rows(artifact)
        for field, value in (('environment_cost', 99), ('human_anchor', 99), ('sweep_arm','human'),
                             ('candidate_probability', .99), ('candidate_text','C')):
            rows=copy.deepcopy(original);rows[0][field]=value
            _write_csv(artifact.raw_scores_path, rows)
            _write_json(artifact.complete_marker_path, {'status':'complete','artifact_sha256':_required_hashes(artifact)})
            with self.subTest(field=field), self.assertRaises(RuntimeError): evaluator.validate_qwen_bundle(artifact)

    def test_notebook_new_summary_and_publication_and_cleared_cells(self):
        import pandas as pd
        result, _ = self.workflow()
        notebook=json.loads(Path('notebooks/eval/ecological_eval.ipynb').read_text())
        for cell in notebook['cells']:
            if cell['cell_type']=='code':
                compile(''.join(cell['source']), '<cell>', 'exec')
                self.assertEqual(cell['outputs'],[]);self.assertIsNone(cell['execution_count'])
        namespace={'EVAL_SOURCE':'qwen','QWEN_EVAL_SUITE':'indifference','qwen_results':result,
                   'qwen_checkpoints':self.checkpoints,'pd':pd,'display':lambda *a:None,
                   'Markdown':lambda x:x,'Image':lambda **kw:None,'INDIFFERENCE_CONFIG':self.config}
        with redirect_stdout(io.StringIO()):
            exec(''.join(notebook['cells'][10]['source']),namespace)
            exec(''.join(notebook['cells'][15]['source']),namespace)
        self.assertEqual(len(namespace['indifference_summary']),94)
        self.assertTrue(namespace['violations'].empty)
        namespace.update(PUBLISH_TO_GITHUB=True,GITHUB_REPOSITORY='test/test',GITHUB_BRANCH='main',GITHUB_TOKEN='test',REPO_DIR=self.root)
        with patch('scripts.ecological_prompt_sft.publish_results_to_github') as publish, redirect_stdout(io.StringIO()):
            exec(''.join(notebook['cells'][18]['source']),namespace)
        self.assertEqual(publish.call_count,2)

    def test_real_tiny_qwen_scores_new_explicit_candidate_suites(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        config=Qwen3Config(vocab_size=256,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
                           num_attention_heads=2,num_key_value_heads=2,head_dim=8,max_position_embeddings=4096,pad_token_id=0)
        cases={k:v[:1] for k,v in suite.build_cases(self.config).items()}
        with patch('transformers.AutoModelForCausalLM.from_pretrained', side_effect=lambda *a,**kw:Qwen3ForCausalLM(config).to(torch.bfloat16)), patch('torch.cuda.reset_peak_memory_stats'), patch('torch.cuda.max_memory_allocated',return_value=0):
            rows=evaluator._score_checkpoint(self.checkpoints['base'],self.tokenizer,tuple(cases),
                                             batch_size=2,token=None,case_sets=cases)
        for name, group in rows.items():
            self.assertEqual(len(group),2 if name=='indifference_ab' else 3)
            self.assertAlmostEqual(sum(r['candidate_probability'] for r in group),1)
            self.assertEqual({r['candidate_token_count'] for r in group},{1})


if __name__ == '__main__':
    unittest.main()
