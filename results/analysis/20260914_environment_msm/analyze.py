"""Reproduce the released environmental MSM analysis from pinned result bundles.

Run with pandas, numpy and matplotlib; no model, credentials, or network needed.
The bootstrap resamples eight scenario families, not repeated costs or orders.
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
from scripts.released_environment_eval import (
    RELEASES, TREATMENTS, SUITES, choice_summary, collect_condition_rows,
    validate_released_bundle,
)
from scripts.harmony_sft.posthoc_eval import artifacts_for_posthoc_eval
from scripts.ecological_prompt_sft.numeric_evaluation import (
    summarize_numeric_threshold_rows, average_numeric_threshold_probabilities,
)

BUNDLE_NAMES = {
    'msm': {
        'choice': '20260914T130455286301Z_extreme_v2_choice_readouts_eval',
        'numeric': '20260914T130455893689Z_extreme_v2_numeric_eval',
    },
    'aft': {
        'choice': '20260914T130931340059Z_extreme_v2_choice_readouts_eval',
        'numeric': '20260914T130931891573Z_extreme_v2_numeric_eval',
    },
    'msm_aft': {
        'choice': '20260914T131126352714Z_extreme_v2_choice_readouts_eval',
        'numeric': '20260914T131126912052Z_extreme_v2_numeric_eval',
    },
}
CONDITIONS = ['baseline', 'msm', 'aft', 'msm_aft']
LABELS = ['Instruction baseline', 'MSM ablation', 'AFT', 'MSM + AFT']
COLORS = ['#596579', '#009E73', '#E69F00', '#0072B2']
CONTRASTS = [
    ('msm', 'baseline'), ('aft', 'baseline'), ('msm_aft', 'baseline'),
    ('msm_aft', 'aft'),
]
BOOTSTRAP_DRAWS = 50_000
SEED = 42


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value))


def cluster_summary(values, indices):
    values = np.asarray(values)
    assert values.shape == (8,)
    lo, hi = np.quantile(values[indices].mean(axis=1), [.025, .975])
    return {
        'mean': values.mean(), 'family_positive': int((values > 0).sum()),
        'family_negative': int((values < 0).sum()),
        'bootstrap_95pct': [lo, hi],
        'leave_one_family_out_range': [
            min(np.delete(values, i).mean() for i in range(8)),
            max(np.delete(values, i).mean() for i in range(8)),
        ],
    }


def main():
    bundles, provenance, signatures = {}, {}, []
    baseline_by_suite = {}
    for treatment in TREATMENTS:
        bundles[treatment], provenance[treatment] = {}, {}
        for suite in SUITES:
            folder = (ROOT / 'results/harmony_eval' / f'llama31_8b_environment_{treatment}'
                      / 'released_environment_msm' / BUNDLE_NAMES[treatment][suite])
            artifacts = artifacts_for_posthoc_eval(folder)
            metadata = validate_released_bundle(artifacts)
            assert metadata['treatment'] == treatment and metadata['suite'] == suite
            assert metadata['repository_commit'] == '74b996f' or metadata['repository_commit'].startswith('74b996f')
            signatures.append(metadata['release_signature'])
            bundles[treatment][suite] = artifacts
            provenance[treatment][suite] = {
                'path': str(folder.relative_to(ROOT)),
                'complete_sha256': sha256(folder / 'COMPLETE.json'),
                'metadata_sha256': sha256(folder / 'metadata.json'),
            }
            raw = pd.read_csv(folder / 'raw_scores.csv')
            base = raw[raw.model_role == 'base'].drop(columns=['pair_name']).reset_index(drop=True)
            if suite in baseline_by_suite:
                pd.testing.assert_frame_equal(base, baseline_by_suite[suite], check_exact=True)
            else:
                baseline_by_suite[suite] = base
    assert all(s == signatures[0] for s in signatures)
    signature = signatures[0]
    # This offline check verifies the scoring implementation as well as bundle hashes.
    for path, expected in signature['implementation_sha256'].items():
        assert sha256(ROOT / path) == expected, path
    audits = signature['tokenizer_audits']
    for key in CONDITIONS:
        for field in ['chat_template', 'suites']:
            assert audits[key][field] == audits['baseline'][field]
    assert signature['adapter_dtype'] == signature['dtype'] == 'bfloat16'

    raw_choice = pd.DataFrame(collect_condition_rows(bundles, 'choice'))
    raw_numeric = pd.DataFrame(collect_condition_rows(bundles, 'numeric'))
    assert len(raw_choice) == 1024 and len(raw_numeric) == 3072
    for column in ['cost_count', 'candidate_tokens_implement', 'candidate_tokens_reject']:
        raw_choice[column] = pd.to_numeric(raw_choice[column])
    for column in ['semantic_logit_implement', 'p_implement', 'logprob_implement', 'logprob_reject',
                   'mean_logprob_implement', 'mean_logprob_reject', 'semantic_logit_sum', 'semantic_logit_mean']:
        raw_choice[column] = pd.to_numeric(raw_choice[column])
    for column in ['candidate_value', 'candidate_logprob', 'candidate_probability', 'permutation_index']:
        raw_numeric[column] = pd.to_numeric(raw_numeric[column])
    # Independently reconstruct the mathematical readouts from raw log probabilities.
    summed = raw_choice.logprob_implement - raw_choice.logprob_reject
    means = (raw_choice.logprob_implement / raw_choice.candidate_tokens_implement
             - raw_choice.logprob_reject / raw_choice.candidate_tokens_reject)
    assert np.allclose(summed, raw_choice.semantic_logit_sum, rtol=0, atol=1e-10)
    assert np.allclose(means, raw_choice.semantic_logit_mean, rtol=0, atol=1e-10)
    selected = np.where(raw_choice.candidate_score_normalization == 'mean', means, summed)
    assert np.allclose(selected, raw_choice.semantic_logit_implement, rtol=0, atol=1e-10)
    assert np.allclose(1 / (1 + np.exp(-selected)), raw_choice.p_implement, rtol=0, atol=1e-10)
    raw_numeric['unconditioned_label_mass'] = np.exp(raw_numeric.candidate_logprob)
    normalizer = raw_numeric.groupby(['condition', 'case_id']).unconditioned_label_mass.transform('sum')
    assert np.allclose(raw_numeric.unconditioned_label_mass / normalizer,
                       raw_numeric.candidate_probability, rtol=0, atol=1e-10)

    choices = pd.DataFrame(choice_summary(raw_choice.to_dict('records')))
    choices.to_csv(HERE / 'choice_by_family_cost.csv', index=False)
    positive = choices[choices.cost_count > 0]
    choice_means = positive.groupby(['condition', 'readout_type']).agg(
        mean_margin=('ecological_minus_human', 'mean'),
        ecological_choices=('ecological_choice', 'sum'), ties=('tie', 'sum'),
        cells=('cost_count', 'size'),
    )
    assert set(choice_means.cells) == {56}
    choice_means.to_csv(HERE / 'choice_summary.csv')
    family = positive.groupby(['readout_type', 'template_family', 'condition']).ecological_minus_human.mean().unstack('condition')[CONDITIONS]
    family.to_csv(HERE / 'choice_by_family.csv')
    by_cost = choices.groupby(['readout_type', 'cost_count', 'condition']).ecological_minus_human.mean().unstack('condition')[CONDITIONS]
    by_cost.to_csv(HERE / 'choice_by_cost.csv')
    by_order = raw_choice[raw_choice.cost_count > 0].groupby(['readout_type', 'readout_variant', 'condition']).semantic_logit_implement.mean().unstack('condition')[CONDITIONS]
    by_order['msm_aft_minus_aft'] = by_order.msm_aft - by_order.aft
    by_order.to_csv(HERE / 'choice_by_order.csv')

    # Numeric helpers need a distinct model_role for each of the four conditions.
    numeric_rows = [{**row, 'model_role': row['condition']} for row in raw_numeric.to_dict('records')]
    numeric = pd.DataFrame(summarize_numeric_threshold_rows(numeric_rows)).rename(columns={'model_role': 'condition'})
    numeric.to_csv(HERE / 'numeric_by_family.csv', index=False)
    probabilities = pd.DataFrame(average_numeric_threshold_probabilities(numeric_rows))
    probabilities.to_csv(HERE / 'numeric_permutation_averages.csv', index=False)
    numeric_metrics = [f'probability_threshold_{n}' for n in (0, 1, 10, 100)] + [
        'expected_threshold', 'expected_log1p_threshold', 'entropy_nats',
    ]
    numeric_means = numeric.groupby('condition')[numeric_metrics].mean().loc[CONDITIONS]
    numeric_means.to_csv(HERE / 'numeric_summary.csv')
    clusters = np.random.default_rng(SEED).integers(0, 8, size=(BOOTSTRAP_DRAWS, 8))
    contrasts, flips = [], []
    matrix = positive.pivot(index=['readout_type', 'template_family', 'cost_count'], columns='condition', values='ecological_minus_human')
    for left, right in CONTRASTS:
        for readout, frame in family.groupby('readout_type'):
            description = cluster_summary((frame[left] - frame[right]).to_numpy(), clusters)
            cells = matrix.xs(readout)
            delta = cells[left] - cells[right]
            flips_ecological = (cells[left] > 0) & (cells[right] <= 0)
            flips_human = (cells[left] <= 0) & (cells[right] > 0)
            contrasts.append({
                'contrast': f'{left}_minus_{right}', 'metric': readout, **description,
                'positive_cells': int((delta > 0).sum()),
                'negative_cells': int((delta < 0).sum()),
                'ecological_flips': int(flips_ecological.sum()),
                'human_flips': int(flips_human.sum()),
            })
            for index in cells[flips_ecological | flips_human].index:
                flips.append({
                    'contrast': f'{left}_minus_{right}', 'readout_type': readout,
                    'template_family': index[0], 'cost_count': index[1],
                    'from_margin': cells.loc[index, right], 'to_margin': cells.loc[index, left],
                })
        for metric in numeric_metrics:
            data = numeric.pivot(index='template_family', columns='condition', values=metric)
            contrasts.append({
                'contrast': f'{left}_minus_{right}', 'metric': metric,
                **cluster_summary((data[left] - data[right]).to_numpy(), clusters),
            })
    pd.DataFrame(flips).to_csv(HERE / 'choice_flips.csv', index=False)
    pd.DataFrame(contrasts).to_csv(HERE / 'contrasts.csv', index=False)

    # Label/order diagnostics are descriptive; balancing remains the primary protocol.
    ab = raw_choice.query("cost_count > 0 and readout_type == 'counterbalanced_ab'").copy()
    ab['p_a'] = np.where(ab.candidate_implement == 'A', ab.p_implement, 1 - ab.p_implement)
    ab['ecological_choice'] = (ab.semantic_logit_implement > 0).astype(int)
    ab_orders = ab.groupby(['condition', 'readout_variant']).agg(
        mean_margin=('semantic_logit_implement', 'mean'),
        mean_p_a=('p_a', 'mean'), ecological_choices=('ecological_choice', 'sum'),
    )
    ab_orders.to_csv(HERE / 'ab_order_diagnostics.csv')
    both = ab.pivot(index=['condition', 'template_family', 'cost_count'], columns='readout_variant', values='semantic_logit_implement')
    both_ecological = (both > 0).all(axis=1).groupby('condition').sum().to_dict()
    numeric_labels = raw_numeric.groupby(['condition', 'candidate_text']).candidate_probability.mean().unstack('candidate_text')
    numeric_labels.to_csv(HERE / 'numeric_label_preferences.csv')
    label_probs = raw_numeric.pivot(index=['condition', 'case_id'], columns='candidate_text', values='candidate_probability')
    tied = label_probs.eq(label_probs.max(axis=1), axis=0).sum(axis=1) > 1
    label_winners = label_probs.idxmax(axis=1).where(~tied, 'tie')
    winners = pd.crosstab(label_winners.index.get_level_values('condition'), label_winners)
    winners.to_csv(HERE / 'numeric_winning_labels.csv')
    numeric_allowed_mass = raw_numeric.groupby(['condition', 'case_id']).unconditioned_label_mass.sum()
    ab['unconditioned_label_mass'] = np.exp(ab.logprob_implement) + np.exp(ab.logprob_reject)
    permitted_mass = {
        'ab_positive_cost': ab.groupby('condition').unconditioned_label_mass.agg(['mean', 'min', 'max']).to_dict('index'),
        'numeric': numeric_allowed_mass.groupby('condition').agg(['mean', 'min', 'max']).to_dict('index'),
    }
    averaged_entropy = numeric.groupby('condition').entropy_nats.mean().to_dict()
    permutation_entropy = (-(label_probs * np.log(label_probs)).sum(axis=1)).groupby('condition').mean().to_dict()
    modal_counts = {key: frame.mode_threshold.value_counts().sort_index().to_dict() for key, frame in numeric.groupby('condition')}
    median_counts = {key: frame.median_threshold.value_counts().sort_index().to_dict() for key, frame in numeric.groupby('condition')}
    full = raw_choice.query("cost_count > 0 and readout_type == 'complete_option_text'")
    full_sum_diagnostic = full.groupby('condition').semantic_logit_sum.mean().to_dict()

    summary = {
        'provenance': provenance,
        'validation': {
            'six_bundles_valid': True, 'shared_baseline_scores_exactly_equal': True,
            'release_signatures_equal': True, 'scoring_code_hashes_match': True,
            'real_tokenizer_audits_equal': True, 'raw_score_math_recomputed': True,
            'unique_choice_rows': len(raw_choice), 'unique_numeric_rows': len(raw_numeric),
        },
        'setup': signature,
        'bootstrap': {
            'unit': 'scenario family (all seven positive costs and both orders kept together)',
            'families': 8, 'draws': BOOTSTRAP_DRAWS, 'seed': SEED,
            'interpretation': 'Exploratory heterogeneity across these eight families; not training-seed uncertainty or confirmatory evidence.',
        },
        'choice_positive_cost': choice_means.reset_index().to_dict('records'),
        'choice_zero_cost': choices[choices.cost_count == 0].groupby(['condition','readout_type']).agg(
            mean_margin=('ecological_minus_human', 'mean'), ecological_choices=('ecological_choice', 'sum')).reset_index().to_dict('records'),
        'numeric': numeric_means.to_dict('index'),
        'numeric_modes': modal_counts, 'numeric_medians': median_counts,
        'contrasts': contrasts,
        'diagnostics': {
            'both_ab_orders_ecological_out_of_56': both_ecological,
            'numeric_label_winners_out_of_192': winners.to_dict('index'),
            'numeric_mean_label_probability': numeric_labels.to_dict('index'),
            'mean_permutation_entropy_nats': permutation_entropy,
            'mean_averaged_value_entropy_nats': averaged_entropy,
            'maximum_four_choice_entropy_nats': np.log(4),
            'unconditioned_probability_of_permitted_answers': permitted_mass,
            'full_option_sum_margin_secondary': full_sum_diagnostic,
            'main_contrast_by_order': by_order.msm_aft_minus_aft.reset_index().to_dict('records'),
        },
    }
    (HERE / 'summary.json').write_text(json.dumps(summary, indent=2, default=default, allow_nan=False) + '\n')
    plot(by_cost, numeric_means)
    print(json.dumps({
        'validation': summary['validation'],
        'choice': summary['choice_positive_cost'],
        'numeric': summary['numeric'],
        'main_contrast': [r for r in contrasts if r['contrast'] == 'msm_aft_minus_aft'],
    }, indent=2, default=default))


def plot(by_cost, numeric_means):
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    for axis, readout, title, ylabel in [
        (axes[0], 'complete_option_text', 'Full-option scoring', 'Ecological minus human (nats/token)'),
        (axes[1], 'counterbalanced_ab', 'A/B scoring', 'Ecological minus human\n(log-probability margin)'),
    ]:
        data = by_cost.xs(readout)
        for key, label, color in zip(CONDITIONS, LABELS, COLORS):
            axis.plot(np.log1p(data.index), data[key], marker='o', markersize=4, label=label, color=color)
        axis.set_xticks(np.log1p(data.index), ['0', '1', '10', '100', '1k', '10k', '100k', '1m'], rotation=40)
        axis.set_xlabel('Human-death cost')
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=.15)
        axis.axhline(0, color='#888888', linewidth=.7)
    axis = axes[2]
    x = np.arange(4)
    for index, (key, label, color) in enumerate(zip(CONDITIONS, LABELS, COLORS)):
        values = numeric_means.loc[key, [f'probability_threshold_{v}' for v in (0,1,10,100)]]
        axis.bar(x + (index - 1.5)*.19, values, width=.18, label=label, color=color)
    axis.axhline(.25, color='#888888', linestyle='--', linewidth=.8)
    axis.set_xticks(x, [0,1,10,100])
    axis.set_ylim(0,.4)
    axis.set_xlabel('Maximum tolerated deaths (offered candidates)')
    axis.set_ylabel('Permutation-averaged probability')
    axis.set_title('Numerical threshold distribution')
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(.5,.035))
    figure.suptitle('Released environmental MSM models on the existing eight scenarios', fontsize=15)
    figure.text(.5,.005, 'Choice curves average both orders and eight families; numerical probabilities average 24 mappings per family. Dashed line: uniform distribution.', ha='center', fontsize=9)
    figure.tight_layout(rect=(0,.11,1,.94))
    figure.savefig(HERE / 'comparison.png', dpi=180)
    plt.close(figure)


if __name__ == '__main__':
    main()
