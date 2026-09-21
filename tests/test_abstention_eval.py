import copy
import math
import unittest
from collections import Counter, defaultdict
from unittest.mock import patch

from scripts.ecological_prompt_sft.abstention_evaluation import (
    CHOICE_LABELS, PERMUTATIONS, SEMANTIC_VALUES, build_abstention_cases,
    summarize_abstention_rows,
)
from scripts.harmony_eval.cases import DEFAULT_COST_COUNTS, REPO_ROOT
from scripts.harmony_eval.scoring import format_causal_prompt, score_loaded_causal_candidates
from scripts.released_environment_eval import audit_tokenizer
from tests.test_released_environment_eval import Tokenizer


class AbstentionEvalTests(unittest.TestCase):
    def scores(self, *, semantic=False):
        cases = build_abstention_cases((1,))
        tokenizer = Tokenizer()
        mappings = {
            format_causal_prompt(tokenizer, case['prompt'], enable_thinking=False):
            {c['text']: c['value'] for c in case['candidates']}
            for case in cases
        }

        def batch_scores(model, tok, items):
            if semantic:
                probabilities = {0: .1, 1: .2, 2: .7}
                return [math.log(probabilities[mappings[i['formatted_prompt']][i['candidate']]]) for i in items]
            # A pure preference for a letter must not masquerade as abstention.
            return [math.log({'A': .6, 'B': .3, 'C': .1}[i['candidate']]) for i in items]

        with patch('scripts.harmony_eval.scoring._score_causal_batch', side_effect=batch_scores):
            return score_loaded_causal_candidates(
                model=None, tokenizer=tokenizer, cases=cases, model_role='llama_instruct',
                model_id='test', model_revision='test', pair_name='test',
                training_method='test', batch_size=5, enable_thinking=False,
            )

    def test_full_matrix_preserves_scenarios_and_balances_all_three_options(self):
        cases = build_abstention_cases()
        self.assertEqual(len(cases), 384)
        self.assertEqual(len({c['case_id'] for c in cases}), 384)
        groups = defaultdict(list)
        for case in cases:
            groups[(case['template_family'], case['cost_count'])].append(case)
            body = (REPO_ROOT / case['source_template_path']).read_text().strip().rsplit('\n\n', 1)[0]
            self.assertTrue(case['prompt'].startswith(body.format(cost=case['cost_count']) + '\n\n'))
            self.assertEqual(case['prompt'].count('I cannot responsibly make a decision here.'), 1)
            self.assertEqual([c['text'] for c in case['candidates']], list(CHOICE_LABELS))
            self.assertEqual(case['candidate_termination'], 'none')
            for label, role in zip(CHOICE_LABELS, PERMUTATIONS[case['permutation_index']]):
                if role == 'abstain':
                    self.assertIn(f'{label}. I cannot responsibly make a decision here.', case['prompt'])
        self.assertEqual(len(groups), 8 * len(DEFAULT_COST_COUNTS))
        for group in groups.values():
            self.assertEqual({c['permutation_index'] for c in group}, set(range(6)))
            self.assertEqual(len({c['option_mapping'] for c in group}), 6)
            for value in SEMANTIC_VALUES.values():
                counts = Counter(c['text'] for case in group for c in case['candidates'] if c['value'] == value)
                self.assertEqual(counts, {'A': 2, 'B': 2, 'C': 2})

    def test_fixed_label_bias_averages_to_equal_semantic_support(self):
        rows = self.scores()
        self.assertEqual(len(rows), 8 * 6 * 3)
        for summary in summarize_abstention_rows(rows):
            for role in SEMANTIC_VALUES:
                self.assertAlmostEqual(summary[f'probability_{role}'], 1/3)
                self.assertEqual(summary[f'permutations_preferring_{role}'], 2)
            self.assertEqual(summary['preferred_response'], 'tie')
            self.assertAlmostEqual(summary['min_probability_abstain'], .1)
            self.assertAlmostEqual(summary['max_probability_abstain'], .6)

    def test_semantic_abstention_preference_survives_all_mappings(self):
        rows = self.scores(semantic=True)
        for summary in summarize_abstention_rows(rows):
            self.assertAlmostEqual(summary['probability_ecological'], .1)
            self.assertAlmostEqual(summary['probability_human'], .2)
            self.assertAlmostEqual(summary['probability_abstain'], .7)
            self.assertEqual(summary['permutations_preferring_abstain'], 6)
            self.assertEqual(summary['preferred_response'], 'abstain')
            self.assertAlmostEqual(summary['mean_offered_label_probability_mass'], 1)

    def test_incomplete_duplicate_or_mismapped_results_cannot_be_averaged(self):
        rows = self.scores()
        for bad in (rows[:-1], rows[3:], rows + [rows[0]]):
            with self.assertRaises(RuntimeError):
                summarize_abstention_rows(bad)
        bad = copy.deepcopy(rows)
        bad[0]['candidate_text'] = 'C'
        with self.assertRaisesRegex(RuntimeError, 'mapping'):
            summarize_abstention_rows(bad)
        bad = copy.deepcopy(rows)
        bad[0]['candidate_probability'] = float('nan')
        with self.assertRaisesRegex(RuntimeError, 'distribution'):
            summarize_abstention_rows(bad)

    def test_all_labels_are_audited_including_c(self):
        cases = {'abstention': build_abstention_cases((1,))}
        self.assertEqual(audit_tokenizer(Tokenizer(), cases)['suites']['abstention']['case_count'], 48)

        class SplitC(Tokenizer):
            def encode(self, text, **kwargs):
                ids = super().encode(text, **kwargs)
                return ids + [7] if text.endswith(':C') else ids

        with self.assertRaisesRegex(RuntimeError, 'exactly one token'):
            audit_tokenizer(SplitC(), cases)


if __name__ == '__main__':
    unittest.main()
