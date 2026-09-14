"""Reproduce the September 13 corrected DPO analysis from committed, validated artifacts.

Run from any directory with pandas, numpy, and matplotlib installed. No model
weights or network access are needed. Outputs are written beside this script.
"""
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from scripts.harmony_sft.posthoc_eval import artifacts_for_posthoc_eval
from scripts.ecological_prompt_sft.readout_evaluation import validate_supervision_matched_readout_artifacts
from scripts.ecological_prompt_sft.numeric_evaluation import validate_numeric_threshold_artifacts

RESULTS = ROOT / "results/harmony_eval"
SOURCE = "20260914T000135277614Z_qwen3_8b_ecological_dilemma_ecological_dpo"
DPO = RESULTS / "qwen3_8b_ecological_dilemma_ecological_dpo" / SOURCE
CHOICE = DPO / "20260914T000737581748Z_extreme_v2_choice_readouts_eval"
NUMERIC = DPO / "20260914T000906510471Z_extreme_v2_numeric_eval"


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one(pattern):
    matches = list(RESULTS.glob(pattern))
    assert len(matches) == 1, matches
    return matches[0]


def paired_choice(frame):
    means = frame.groupby(["readout_type", "template_family", "cost_count", "model_role"])["semantic_logit_implement"].mean().unstack("model_role")
    means["shift"] = means.aligned - means.base
    return means.reset_index()


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value))


def main():
    choice_validation = validate_supervision_matched_readout_artifacts(artifacts_for_posthoc_eval(CHOICE), choice_only=True)
    numeric_validation = validate_numeric_threshold_artifacts(artifacts_for_posthoc_eval(NUMERIC))
    marker = read_json(HERE / "source/COMPLETE.json")
    for name, key in [("run_metadata.json", "metadata"), ("train_metrics.json", "train_metrics")]:
        assert sha256(HERE / "source" / name) == marker["artifact_sha256"][key]
    for bundle in (CHOICE, NUMERIC):
        metadata = read_json(bundle / "metadata.json")
        assert metadata["source_complete_sha256"] == sha256(HERE / "source/COMPLETE.json")
        assert metadata["adapter_revision"] == marker["artifact_sha256"]["adapter_weights"]
    source = read_json(HERE / "source/run_metadata.json")
    metrics = read_json(HERE / "source/train_metrics.json")
    initial = source["initial_reference_audit"]
    assert source["score_precision"] == "fp32_logits_logsoftmax_sum_v1"
    assert initial == metrics["initial_reference_audit"]
    assert initial["status"] == "passed" and initial["example_count"] == 98
    assert initial["max_abs_logp_difference"] == initial["max_abs_reward_margin"] == 0
    audit = pd.DataFrame(initial["per_example"]).merge(
        pd.DataFrame(source["dataset"]["reference_log_probs"]), on="id", validate="one_to_one")
    assert len(audit) == 98
    for side in ("chosen", "rejected"):
        assert (audit[f"{side}_logps"] == audit[f"ref_{side}_logps"]).all()
    assert (audit.reward_margin == 0).all()
    assert np.allclose(audit.dpo_loss, np.log(2), atol=1e-7, rtol=0)
    audit.to_csv(HERE / "initial_precision_audit.csv", index=False)
    raw = pd.read_csv(CHOICE / "raw_scores.csv")
    paired = paired_choice(raw)
    positive = paired.query("cost_count > 0").copy()
    assert len(positive) == 112
    paired.to_csv(HERE / "choice_by_family_cost.csv", index=False)
    families = positive.groupby(["readout_type", "template_family"])[["base", "aligned", "shift"]].mean().reset_index()
    families.to_csv(HERE / "choice_by_family.csv", index=False)
    summary = {"source_run": SOURCE, "adapter_sha256": marker["artifact_sha256"]["adapter_weights"],
               "validation": {"choice_rows": choice_validation.score_row_count,
                              "numeric_rows": numeric_validation.score_row_count},
               "choice_positive_cost": {}}
    summary["setup"] = {key: source[key] for key in ("config", "hardware", "packages", "repository_commit", "score_precision")}
    summary["initial_precision_audit"] = {key: value for key, value in initial.items() if key != "per_example"}
    for readout, frame in positive.groupby("readout_type"):
        signs = np.sign(frame[["base", "aligned"]])
        summary["choice_positive_cost"][readout] = {
            "cells": len(frame), **frame[["base", "aligned", "shift"]].mean().to_dict(),
            "mean_absolute_shift": frame["shift"].abs().mean(),
            "max_absolute_shift": frame["shift"].abs().max(),
            "positive_shifts": int((frame["shift"] > 0).sum()),
            "base_ecological": int((frame.base > 0).sum()),
            "dpo_ecological": int((frame.aligned > 0).sum()),
            "base_ties": int((frame.base == 0).sum()), "dpo_ties": int((frame.aligned == 0).sum()),
            "sign_changes": frame.loc[signs.base != signs.aligned].to_dict("records"),
        }
    summary["choice_zero_cost"] = paired.query("cost_count == 0").groupby("readout_type")[["base", "aligned", "shift"]].mean().to_dict("index")
    by_order = raw.query("cost_count > 0").groupby(["readout_type", "readout_variant", "model_role"])["semantic_logit_implement"].mean().unstack("model_role")
    by_order["shift"] = by_order.aligned - by_order.base
    by_order.to_csv(HERE / "choice_by_order.csv")

    numeric = pd.read_csv(NUMERIC / "thresholds.csv")
    ncols = ["probability_threshold_0", "probability_threshold_1", "probability_threshold_10", "probability_threshold_100", "expected_threshold", "expected_log1p_threshold", "entropy_nats"]
    means = numeric.groupby("model_role")[ncols].mean()
    summary["numeric"] = means.to_dict("index")
    summary["numeric"]["shift"] = (means.loc["aligned"] - means.loc["base"]).to_dict()
    assert (numeric.mode_threshold == 0).all() and (numeric.median_threshold == 0).all()
    summary["numeric"]["zero_mode_and_median_scenarios_per_model"] = 8
    raw_numeric = pd.read_csv(NUMERIC / "raw_scores.csv")
    # Count a zero win only if it is the unique maximum; ties are recorded separately.
    probs = raw_numeric.pivot(index=["model_role", "case_id"], columns="candidate_value", values="candidate_probability")
    winners = probs.eq(probs.max(axis=1), axis=0)
    summary["numeric"]["unique_zero_wins_by_model"] = (winners[0] & winners.sum(axis=1).eq(1)).groupby(level=0).sum().to_dict()
    summary["numeric"]["tied_permutations_by_model"] = winners.sum(axis=1).gt(1).groupby(level=0).sum().to_dict()
    numeric.to_csv(HERE / "numeric_by_family.csv", index=False)

    histories = {
        "ecological_response_sft_3epochs": one("qwen3_8b_ecological_dilemma_ecological_option_sft/20260831T103909Z*/20260831T132641*/raw_scores.csv"),
        "human_response_sft_3epochs": one("qwen3_8b_ecological_dilemma_human_option_sft/20260831T104623Z*/20260831T132808*/raw_scores.csv"),
    }
    summary["historical_choice"] = {}
    for label, path in histories.items():
        old = pd.read_csv(path)
        # Check that the historical and new readouts use identical text and options.
        same = raw.query("model_role == 'base'").merge(old.query("model_role == 'base'"), on="case_id", suffixes=("_new", "_old"), validate="one_to_one")
        assert len(same) == 256
        for col in ("prompt", "candidate_implement", "candidate_reject", "candidate_score_normalization"):
            assert same[col + "_new"].equals(same[col + "_old"])
        old_paired = paired_choice(old).query("cost_count > 0 and readout_type != 'reversed_yes_no'")
        drift = positive.merge(old_paired, on=["readout_type", "template_family", "cost_count"], suffixes=("_new", "_old"), validate="one_to_one")
        drift["base_drift"] = drift.base_new - drift.base_old
        drift["abs_base_drift"] = drift.base_drift.abs()
        summary["historical_choice"][label] = {
            "path": str(path.relative_to(ROOT)),
            "within_run_means": old_paired.groupby("readout_type")[["base", "aligned", "shift"]].mean().to_dict("index"),
            "new_minus_historical_base": drift.groupby("readout_type")[["base_drift", "abs_base_drift"]].mean().to_dict("index"),
        }
    historical_numeric = one("qwen3_8b_ecological_dilemma_ecological_option_sft/20260831T103909Z*/20260902T080557*/thresholds.csv")
    summary["historical_numeric_3epoch_ecological_sft"] = {
        "path": str(historical_numeric.relative_to(ROOT)),
        "means": pd.read_csv(historical_numeric).groupby("model_role")[ncols].mean().to_dict("index"),
    }

    metrics = read_json(HERE / "source/train_metrics.json")
    history = pd.DataFrame([row for row in metrics["log_history"] if "loss" in row])
    assert history.step.tolist() == list(range(1, 22))
    history["epoch_number"] = np.ceil(history.epoch).astype(int)
    history["examples_in_window"] = np.where(history.step % 7 == 0, 2, 16)
    assert history.examples_in_window.sum() == 294
    history.to_csv(HERE / "training_steps.csv", index=False)
    summary["training"] = {key: value for key, value in metrics.items() if key not in ("log_history", "initial_reference_audit")}
    summary["training"]["example_weighted_online_epoch_logs"] = {
        int(epoch): {col: np.average(frame[col], weights=frame.examples_in_window) for col in ("loss", "rewards/accuracies", "rewards/margins", "rewards/chosen", "rewards/rejected")}
        for epoch, frame in history.groupby("epoch_number")
    }
    # Compare the corrected run directly with the first DPO run, not with a
    # different historical SFT base. Assert exact equality of the base outputs.
    first_source = "20260911T191634280185Z_qwen3_8b_ecological_dilemma_ecological_dpo"
    first_dir = RESULTS / "qwen3_8b_ecological_dilemma_ecological_dpo" / first_source
    first_choice = first_dir / "20260911T192223846407Z_extreme_v2_choice_readouts_eval"
    first_numeric = first_dir / "20260911T192853076895Z_extreme_v2_numeric_eval"
    validate_supervision_matched_readout_artifacts(artifacts_for_posthoc_eval(first_choice), choice_only=True)
    validate_numeric_threshold_artifacts(artifacts_for_posthoc_eval(first_numeric))
    first_raw = pd.read_csv(first_choice / "raw_scores.csv")
    first_num_raw = pd.read_csv(first_numeric / "raw_scores.csv")
    for current, previous, keys, columns in [
        (raw, first_raw, ["case_id"], ["prompt", "candidate_implement", "candidate_reject", "candidate_score_normalization", "logprob_implement", "logprob_reject", "semantic_logit_implement", "p_implement"]),
        (raw_numeric, first_num_raw, ["case_id", "candidate_value"], ["prompt", "candidate_scored_text", "candidate_logprob", "candidate_probability"]),
    ]:
        left = current.query("model_role == 'base'").set_index(keys).sort_index()[columns]
        right = previous.query("model_role == 'base'").set_index(keys).sort_index()[columns]
        pd.testing.assert_frame_equal(left, right, check_exact=True)
    first_paired = paired_choice(first_raw)
    compare = paired.merge(first_paired, on=["readout_type", "template_family", "cost_count"], suffixes=("_corrected", "_first"), validate="one_to_one")
    compare["corrected_minus_first"] = compare.aligned_corrected - compare.aligned_first
    compare.to_csv(HERE / "choice_first_vs_corrected.csv", index=False)
    first_num = pd.read_csv(first_numeric / "thresholds.csv").groupby("model_role")[ncols].mean()
    first_metadata = read_json(HERE.parent / "20260911_ecological_dpo/source/run_metadata.json")
    assert source["config"] == first_metadata["config"]
    assert source["packages"] == first_metadata["packages"]
    assert source["hardware"] == first_metadata["hardware"]
    for key in ("records_sha256", "source_releases", "tokenization"):
        assert source["dataset"][key] == first_metadata["dataset"][key]
    comparison_positive = compare.query("cost_count > 0").copy()
    comparison_positive["abs_corrected_minus_first"] = comparison_positive.corrected_minus_first.abs()
    summary["first_dpo_comparison"] = {
        "source_run": first_source, "base_outputs_exactly_equal": True,
        "configuration_packages_hardware_data_equal": True,
        "choice_positive_cost_means": comparison_positive.groupby("readout_type")[["shift_first", "shift_corrected", "corrected_minus_first", "abs_corrected_minus_first"]].mean().to_dict("index"),
        "numeric_first": first_num.to_dict("index"),
        "numeric_corrected_minus_first": (means.loc["aligned"] - first_num.loc["aligned"]).to_dict(),
    }
    refs = pd.DataFrame(source["dataset"]["reference_log_probs"]).merge(
        pd.DataFrame(first_metadata["dataset"]["reference_log_probs"]), on="id", suffixes=("_corrected", "_first"), validate="one_to_one")
    for side in ("chosen", "rejected"):
        refs[f"{side}_first_minus_corrected"] = refs[f"ref_{side}_logps_first"] - refs[f"ref_{side}_logps_corrected"]
    # Evaluate how the *original cached reference* would offset rewards for the
    # verified initial policy. This is not a corrected training trajectory.
    refs["old_reference_reward_offset"] = source["config"]["beta"] * (refs.rejected_first_minus_corrected - refs.chosen_first_minus_corrected)
    refs.to_csv(HERE / "reference_score_correction.csv", index=False)
    error_cols = ["chosen_first_minus_corrected", "rejected_first_minus_corrected", "old_reference_reward_offset"]
    summary["reference_score_correction"] = {
        "mean": refs[error_cols].mean().to_dict(), "mean_absolute": refs[error_cols].abs().mean().to_dict(),
        "max_absolute": refs[error_cols].abs().max().to_dict(),
    }
    summary["limitations"] = [
        "One ecological-preferred run and seed; no human-preferred DPO control yet.",
        "Eight scenario families; repeated costs and option mappings are not independent samples.",
        "Historical base scores differ despite identical prompts; compare within-run effects. Baseline drift is not a calibrated noise estimate.",
        "Training logs are online implicit-reward metrics, not a final-model training-set choice evaluation.",
        "The corrected audit passes with zero initial policy/reference discrepancy; final-model training-set preferences remain unmeasured.",
        "Source metadata and metrics hashes were independently checked; large adapter and checkpoint files were not downloaded in this analysis.",
    ]
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2, default=json_default) + "\n")

    full = families.query("readout_type == 'complete_option_text'").set_index("template_family")
    num = numeric.pivot(index="template_family", columns="model_role", values="probability_threshold_zero").reindex(full.index)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), sharey=True, layout="constrained")
    labels = [name.replace("_", " ").capitalize() for name in full.index]
    y = np.arange(8)
    for ax, frame, factor in [(axes[0], full, 1), (axes[1], num, 100)]:
        ax.hlines(y, factor*frame.base, factor*frame.aligned, color="#777777", linewidth=2)
        ax.scatter(factor*frame.base, y, color="#2563a4", marker="x", s=48, label="Base", zorder=3)
        ax.scatter(factor*frame.aligned, y, facecolors="none", edgecolors="#d56b1d", s=72, linewidths=1.8, label="Ecological DPO", zorder=4)
        ax.grid(axis="x", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].axvline(0, color="grey", linestyle=":", linewidth=1)
    axes[0].set_xlabel("Ecological minus human option score (nats/token)\nMean over seven positive costs and both orders")
    axes[0].set_title("Full-option scores barely move")
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("Probability on zero tolerated deaths (%)\nMean over all 24 label mappings")
    axes[1].set_title("Numerical thresholds remain similar")
    axes[1].legend(loc="lower left", frameon=False)
    fig.suptitle("Qwen3-8B · 98 pairs · 3 epochs · DPO after precision correction", fontsize=14)
    fig.savefig(HERE / "comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2, default=json_default))


if __name__ == "__main__":
    main()
