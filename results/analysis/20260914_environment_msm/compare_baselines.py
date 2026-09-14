"""Compare the authors' Llama instruction baseline with unchanged Qwen3-8B.

Uses the exact prior corrected DPO run's base rows, not its trained adapter.
No model loading, sampling, or network calls.
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from scripts.released_environment_eval import validate_released_bundle
from scripts.ecological_prompt_sft.readout_evaluation import validate_supervision_matched_readout_artifacts
from scripts.harmony_sft.posthoc_eval import artifacts_for_posthoc_eval

SOURCES = {
    'llama_instruction_baseline': ROOT / 'results/harmony_eval/llama31_8b_environment_msm/released_environment_msm/20260914T130455286301Z_extreme_v2_choice_readouts_eval',
    'qwen3_8b': ROOT / 'results/harmony_eval/qwen3_8b_ecological_dilemma_ecological_dpo/20260914T000135277614Z_qwen3_8b_ecological_dilemma_ecological_dpo/20260914T000737581748Z_extreme_v2_choice_readouts_eval',
}


def main():
    frames, provenance = {}, {}
    for name, folder in SOURCES.items():
        artifacts = artifacts_for_posthoc_eval(folder)
        if name == 'llama_instruction_baseline':
            validate_released_bundle(artifacts)
        else:
            validate_supervision_matched_readout_artifacts(artifacts, choice_only=True)
        metadata = json.loads(artifacts.metadata_path.read_text())
        raw = pd.read_csv(artifacts.raw_scores_path)
        frames[name] = raw.query("model_role == 'base'").copy()
        provenance[name] = {
            'path': str(folder.relative_to(ROOT)),
            'complete_sha256': hashlib.sha256(artifacts.complete_marker_path.read_bytes()).hexdigest(),
            'model_ids': frames[name].model_id.unique().tolist(),
            'model_revisions': frames[name].model_revision.unique().tolist(),
            'enable_thinking': metadata['enable_thinking'],
        }
    left = frames['llama_instruction_baseline'].set_index('case_id').sort_index()
    right = frames['qwen3_8b'].set_index('case_id').sort_index()
    columns = ['prompt', 'candidate_implement', 'candidate_reject', 'candidate_score_normalization',
               'readout_variant', 'readout_type', 'cost_count']
    pd.testing.assert_frame_equal(left[columns], right[columns], check_exact=True)
    assert len(left) == len(right) == 256
    assert provenance['qwen3_8b']['model_ids'] == ['Qwen/Qwen3-8B']
    assert provenance['qwen3_8b']['model_revisions'] == ['b968826d9c46dd6066d109eabc6255188de91218']
    assert provenance['llama_instruction_baseline']['model_ids'] == ['chloeli/llama-3.1-8b-baseline']
    summary, costs, orders = [], [], []
    for name, raw in frames.items():
        for readout, frame in raw.query('cost_count > 0').groupby('readout_type'):
            matrix = frame.pivot(index=['template_family','cost_count'], columns='readout_variant', values='semantic_logit_implement')
            assert matrix.shape == (56, 2)
            margin = matrix.mean(axis=1)
            summary.append({
                'model': name, 'readout_type': readout, 'mean_margin': margin.mean(),
                'ecological_after_order_average': int((margin > 0).sum()),
                'human_after_order_average': int((margin < 0).sum()),
                'ties_after_order_average': int((margin == 0).sum()),
                'ecological_both_orders': int((matrix > 0).all(axis=1).sum()),
                'human_both_orders': int((matrix < 0).all(axis=1).sum()),
                'mean_restricted_ecological_probability': frame.p_implement.mean() if readout == 'counterbalanced_ab' else None,
            })
            for cost, values in margin.groupby('cost_count'):
                costs.append({'model':name,'readout_type':readout,'cost_count':cost,
                              'mean_margin':values.mean(),'ecological':int((values>0).sum()),
                              'human':int((values<0).sum()),'ties':int((values==0).sum())})
            for order, values in frame.groupby('readout_variant'):
                orders.append({'model':name,'readout_type':readout,'readout_variant':order,
                               'mean_margin':values.semantic_logit_implement.mean(),
                               'ecological':int((values.semantic_logit_implement>0).sum()),
                               'human':int((values.semantic_logit_implement<0).sum()),
                               'ties':int((values.semantic_logit_implement==0).sum())})
    pd.DataFrame(summary).to_csv(HERE/'baseline_comparison.csv',index=False)
    pd.DataFrame(costs).to_csv(HERE/'baseline_comparison_by_cost.csv',index=False)
    pd.DataFrame(orders).to_csv(HERE/'baseline_comparison_by_order.csv',index=False)
    (HERE/'baseline_comparison_provenance.json').write_text(json.dumps({
        'sources':provenance, 'identical_case_text_and_mapping_count':256,
        'summary':summary,
        'scope':'Authors instruction-tuned Llama-3.1-8B baseline versus Qwen3-8B with thinking disabled; different instruction-training histories and native chat templates. Does not isolate model architecture.',
    },indent=2,allow_nan=False)+'\n')
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == '__main__':
    main()
