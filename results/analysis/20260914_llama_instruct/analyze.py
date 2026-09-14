"""Compare the official Llama Instruct run with pinned Llama/Qwen results.

Offline descriptive analysis. Orders and costs are repeated measurements within
eight scenario families, not independent samples. No models are loaded.
"""
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from scripts.llama_instruct_eval import MODEL_ID, MODEL_REVISION, validate_instruct_bundle
from scripts.released_environment_eval import RELEASES, validate_released_bundle
from scripts.ecological_prompt_sft.readout_evaluation import validate_supervision_matched_readout_artifacts
from scripts.ecological_prompt_sft.numeric_evaluation import (
    validate_numeric_threshold_artifacts, summarize_numeric_threshold_rows,
    average_numeric_threshold_probabilities,
)
from scripts.harmony_sft.posthoc_eval import artifacts_for_posthoc_eval

INSTRUCT_RUN = 'llama31_8b_instruct/standard_llama31_8b_instruct/'
QWEN_RUN = ('qwen3_8b_ecological_dilemma_ecological_dpo/'
            '20260914T000135277614Z_qwen3_8b_ecological_dilemma_ecological_dpo/')
MSM_RUN = 'llama31_8b_environment_msm/released_environment_msm/'
AFT_RUN = 'llama31_8b_environment_aft/released_environment_msm/'
MSMAFT_RUN = 'llama31_8b_environment_msm_aft/released_environment_msm/'
SOURCES = {
    'llama_instruct': (INSTRUCT_RUN, '20260914T154900074908Z', '20260914T154900610119Z', 'llama_instruct'),
    'authors_baseline': (MSM_RUN, '20260914T130455286301Z', '20260914T130455893689Z', 'base'),
    'qwen3_8b': (QWEN_RUN, '20260914T000737581748Z', '20260914T000906510471Z', 'base'),
    'msm': (MSM_RUN, '20260914T130455286301Z', '20260914T130455893689Z', 'aligned'),
    'aft': (AFT_RUN, '20260914T130931340059Z', '20260914T130931891573Z', 'aligned'),
    'msm_aft': (MSMAFT_RUN, '20260914T131126352714Z', '20260914T131126912052Z', 'aligned'),
}
LABELS = {'llama_instruct': 'Meta Llama Instruct', 'authors_baseline': "Authors’ Llama baseline", 'qwen3_8b': 'Qwen3-8B'}
COLORS = {'llama_instruct': '#0072B2', 'authors_baseline': '#717985', 'qwen3_8b': '#D55E00'}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    frames, provenance, validated = {}, {}, {}
    for model, (run, choice_time, numeric_time, role) in SOURCES.items():
        frames[model], provenance[model] = {}, {}
        for suite, time, slug in [('choice', choice_time, 'extreme_v2_choice_readouts_eval'),
                                  ('numeric', numeric_time, 'extreme_v2_numeric_eval')]:
            folder = ROOT/'results/harmony_eval'/run/f'{time}_{slug}'
            artifact = artifacts_for_posthoc_eval(folder)
            if folder not in validated:
                if model == 'llama_instruct':
                    metadata = validate_instruct_bundle(artifact)
                elif model == 'qwen3_8b':
                    if suite == 'choice':
                        validate_supervision_matched_readout_artifacts(artifact, choice_only=True)
                    else:
                        validate_numeric_threshold_artifacts(artifact)
                    metadata = json.loads(artifact.metadata_path.read_text())
                else:
                    metadata = validate_released_bundle(artifact)
                validated[folder] = metadata
            metadata = validated[folder]
            signature = metadata.get('checkpoint_signature', metadata.get('release_signature'))
            if signature:
                for name, digest in signature['implementation_sha256'].items():
                    assert sha256(ROOT/name) == digest, name
                assert signature['dtype'] == 'bfloat16' and signature['load_in_4bit'] is False
                assert signature['enable_thinking'] is False if 'enable_thinking' in signature else metadata['enable_thinking'] is False
            else:
                assert metadata['enable_thinking'] is False
            raw = pd.read_csv(artifact.raw_scores_path)
            raw = raw[raw.model_role == role].copy()
            assert len(raw) == (256 if suite == 'choice' else 768)
            assert raw.case_id.nunique() == (256 if suite == 'choice' else 192)
            if model == 'llama_instruct':
                expected_id, expected_revision = MODEL_ID, MODEL_REVISION
            elif model == 'qwen3_8b':
                expected_id, expected_revision = 'Qwen/Qwen3-8B', 'b968826d9c46dd6066d109eabc6255188de91218'
            else:
                spec = RELEASES['baseline' if model == 'authors_baseline' else model]
                expected_id, expected_revision = spec.repo_id, spec.revision
            assert set(raw.model_id) == {expected_id} and set(raw.model_revision) == {expected_revision}
            assert not raw.load_in_4bit.any()
            provenance[model][suite] = {
                'path': str(folder.relative_to(ROOT)), 'complete_sha256': sha256(artifact.complete_marker_path),
                'metadata_sha256': sha256(artifact.metadata_path), 'repository_commit': metadata['repository_commit'],
                'model_id': expected_id, 'model_revision': expected_revision,
                'source_model_role': role, 'case_set_sha256': metadata['case_set_sha256'],
            }
            # Use unique model roles for helpers that group the numeric rows by role.
            raw['model'] = model
            raw['condition'] = model
            raw['model_role'] = model
            frames[model][suite] = raw
    # Require identical literal questions/candidates, including every numeric label mapping.
    for suite in ('choice', 'numeric'):
        keys = ['case_id'] + (['candidate_value'] if suite == 'numeric' else [])
        fields = ['prompt', 'template_family'] + (
            ['candidate_text', 'candidate_scored_text', 'candidate_index', 'option_mapping', 'permutation_index']
            if suite == 'numeric' else ['candidate_implement', 'candidate_reject', 'candidate_score_normalization',
                                       'readout_variant', 'readout_type', 'cost_count'])
        reference = frames['llama_instruct'][suite].set_index(keys)[fields].sort_index()
        for values in frames.values():
            pd.testing.assert_frame_equal(reference, values[suite].set_index(keys)[fields].sort_index(), check_exact=True)

    choices, summaries, orders, costs, by_family, masses = [], [], [], [], [], []
    numeric_summaries, numeric_probabilities, numeric_labels = [], [], []
    for model, suites in frames.items():
        raw = suites['choice']
        # Independently reconstruct each margin from unnormalized sequence scores.
        summed = raw.logprob_implement - raw.logprob_reject
        mean = raw.logprob_implement/raw.candidate_tokens_implement - raw.logprob_reject/raw.candidate_tokens_reject
        margin = np.where(raw.candidate_score_normalization == 'mean', mean, summed)
        assert np.allclose(margin, raw.semantic_logit_implement, rtol=0, atol=1e-10)
        assert np.allclose(1/(1+np.exp(-margin)), raw.p_implement, rtol=0, atol=1e-10)
        for readout, frame in raw.groupby('readout_type'):
            matrix = frame.pivot(index=['template_family', 'cost_count'], columns='readout_variant', values='semantic_logit_implement')
            assert matrix.shape == (64, 2) and not matrix.isna().any().any()
            balanced = matrix.mean(axis=1)
            positive = matrix[matrix.index.get_level_values('cost_count') > 0]
            balanced_positive = positive.mean(axis=1)
            pframe = frame[frame.cost_count > 0]
            summary = {
                'model': model, 'readout_type': readout, 'mean_margin': balanced_positive.mean(),
                'ecological_choices': int((balanced_positive > 0).sum()),
                'human_choices': int((balanced_positive < 0).sum()),
                'ties': int((balanced_positive == 0).sum()), 'cells': 56,
                'ecological_both_orders': int((positive > 0).all(axis=1).sum()),
                'human_both_orders': int((positive < 0).all(axis=1).sum()),
                'mean_conditional_p_ecological': pframe.p_implement.mean() if readout == 'counterbalanced_ab' else None,
                'b_position_advantage': (positive.ecological_b-positive.ecological_a).mean()/2 if readout == 'counterbalanced_ab' else None,
            }
            summaries.append(summary)
            for (family, cost), value in balanced.items():
                choices.append({'model': model, 'readout_type': readout, 'template_family': family,
                                'cost_count': cost, 'margin': value, 'ecological_choice': int(value > 0)})
            for cost, values in balanced.groupby('cost_count'):
                costs.append({'model': model, 'readout_type': readout, 'cost_count': cost,
                              'mean_margin': values.mean(), 'ecological_choices': int((values>0).sum()),
                              'ecological_both_orders': int((matrix.xs(cost, level='cost_count')>0).all(axis=1).sum())})
            for family, values in balanced_positive.groupby('template_family'):
                by_family.append({'model': model, 'readout_type': readout, 'template_family': family,
                                  'mean_margin': values.mean(), 'ecological_choices': int((values>0).sum())})
            for order, values in pframe.groupby('readout_variant'):
                orders.append({'model': model, 'readout_type': readout, 'readout_variant': order,
                               'mean_margin': values.semantic_logit_implement.mean(),
                               'ecological_choices': int((values.semantic_logit_implement>0).sum()),
                               'human_choices': int((values.semantic_logit_implement<0).sum()),
                               'ties': int((values.semantic_logit_implement==0).sum()),
                               'mean_conditional_p_ecological': values.p_implement.mean() if readout == 'counterbalanced_ab' else None})
        ab = raw.query("cost_count>0 and readout_type=='counterbalanced_ab'")
        ab_mass = np.exp(ab.logprob_implement) + np.exp(ab.logprob_reject)
        numeric = suites['numeric'].copy()
        numeric['mass'] = np.exp(numeric.candidate_logprob)
        numeric_mass = numeric.groupby('case_id').mass.sum()
        assert np.allclose(numeric.mass/numeric.groupby('case_id').mass.transform('sum'),
                           numeric.candidate_probability, rtol=0, atol=1e-10)
        for suite, values in [('ab_positive_cost', ab_mass), ('numeric', numeric_mass)]:
            masses.append({'model': model, 'suite': suite, 'mean': values.mean(), 'min': values.min(), 'max': values.max()})
        records = numeric.to_dict('records')
        numeric_summaries.extend(summarize_numeric_threshold_rows(records))
        numeric_probabilities.extend(average_numeric_threshold_probabilities(records))
        for label, values in numeric.groupby('candidate_text'):
            numeric_labels.append({'model': model, 'label': label, 'mean_probability': values.candidate_probability.mean()})

    choice_summary = pd.DataFrame(summaries)
    cost_summary = pd.DataFrame(costs)
    family_summary = pd.DataFrame(by_family)
    numerical = pd.DataFrame(numeric_summaries).rename(columns={'model_role': 'model'})
    metrics = ['probability_threshold_0', 'probability_threshold_1', 'probability_threshold_10',
               'probability_threshold_100', 'expected_threshold', 'entropy_nats']
    numeric_summary = numerical.groupby('model')[metrics].mean().reindex(SOURCES)
    for name, frame in {
        'choice_summary': choice_summary, 'choice_by_family_cost': pd.DataFrame(choices),
        'choice_by_cost': cost_summary, 'choice_by_family': family_summary,
        'choice_by_order': pd.DataFrame(orders), 'numeric_by_family': numerical,
        'numeric_probabilities': pd.DataFrame(numeric_probabilities),
        'numeric_summary': numeric_summary.reset_index(), 'numeric_label_preferences': pd.DataFrame(numeric_labels),
        'allowed_label_mass': pd.DataFrame(masses),
    }.items():
        frame.to_csv(HERE/f'{name}.csv', index=False)
    (HERE/'provenance.json').write_text(json.dumps({
        'sources': provenance, 'validation': {'all_ten_bundles_valid': len(validated)==10,
            'six_models_exactly_matched_choice_prompts': 256, 'six_models_exactly_matched_numeric_candidate_rows': 768,
            'llama_scoring_code_hashes_match': True, 'raw_score_arithmetic_verified': True},
        'scope': 'Native chat templates differ. Eight repeatedly used families; descriptive checkpoint comparison, not a causal architecture or training effect.',
    }, indent=2, allow_nan=False)+'\n')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    positive_costs = [1, 10, 100, 1000, 10000, 100000, 1000000]
    labels = ['1', '10', '100', '1k', '10k', '100k', '1m']
    for model, label in LABELS.items():
        for axis, readout, column in [(axes[0,0], 'counterbalanced_ab', 'ecological_choices'),
                                      (axes[0,1], 'counterbalanced_ab', 'mean_margin'),
                                      (axes[1,0], 'complete_option_text', 'mean_margin')]:
            subset = cost_summary.query('model == @model and readout_type == @readout and cost_count > 0').sort_values('cost_count')
            axis.plot(positive_costs, subset[column], marker='o', label=label, color=COLORS[model], linewidth=2)
            axis.set_xscale('log'); axis.set_xticks(positive_costs, labels)
            axis.set_xlabel('Human deaths in the stipulated dilemma'); axis.grid(alpha=.18)
        axes[1,1].plot(range(4), numeric_summary.loc[model, metrics[:4]]*100,
                       marker='o', color=COLORS[model], label=label, linewidth=2)
    axes[0,0].set_title('A/B ecological wins after averaging both orders')
    axes[0,0].set_ylabel('Scenario families (out of 8)'); axes[0,0].set_ylim(-.2,8.5)
    axes[0,1].set_title('A/B margin: ecology minus human')
    axes[0,1].set_ylabel('Mean log-probability margin'); axes[0,1].axhline(0,color='#555',lw=.8)
    axes[1,0].set_title('Full-option margin: ecology minus human')
    axes[1,0].set_ylabel('Mean log probability per answer token'); axes[1,0].axhline(0,color='#555',lw=.8)
    axes[1,1].set_title('Numerical readout: average over 24 mappings')
    axes[1,1].set_xticks(range(4), ['0','1','10','100']); axes[1,1].set_ylim(0,100)
    axes[1,1].set_xlabel('Maximum tolerable human deaths offered'); axes[1,1].set_ylabel('Mean candidate probability (%)')
    axes[1,1].grid(alpha=.18)
    axes[0,0].legend(loc='lower left', frameon=False, fontsize=9)
    fig.suptitle('Standard Llama Instruct also favors ecology in these dilemmas\nSame eight families and scoring; each model uses its native chat template',fontsize=14)
    fig.savefig(HERE/'comparison.png', dpi=170); plt.close(fig)
    print(choice_summary.to_string(index=False))
    print('\nNUMERIC\n'+numeric_summary.to_string())
    print('\nALLOWED LABEL MASS\n'+pd.DataFrame(masses).to_string(index=False))


if __name__ == '__main__':
    main()
