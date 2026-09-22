"""Validate and compare the published Qwen base/SFT/DPO run and prior Llama run."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts.qwen_checkpoint_eval import validate_qwen_bundle
from scripts.llama_instruct_eval import validate_instruct_bundle
from scripts.harmony_sft.posthoc_eval import artifacts_for_posthoc_eval

SOURCE_ROOT = ROOT / "results/harmony_eval"
QWEN_SOURCES = {
    "base": ("qwen3_8b_base/qwen3_8b_unmodified", ("180930190173", "180930823579", "180932010936")),
    "sft": ("qwen3_8b_sft/20260831T103909Z_qwen3_8b_ecological_dilemma_ecological_option_sft",
            ("181239227213", "181239818815", "181241041061")),
    "dpo": ("qwen3_8b_dpo/20260914T000135277614Z_qwen3_8b_ecological_dilemma_ecological_dpo",
            ("181540517534", "181541106901", "181542299816")),
}
SLUGS = {"choice": "extreme_v2_choice_readouts_eval", "numeric": "extreme_v2_numeric_eval",
         "abstention": "extreme_v2_abc_abstention_eval"}
MODELS = ("base", "sft", "dpo", "llama")
LABELS = {"base": "Qwen base", "sft": "Qwen SFT", "dpo": "Qwen DPO", "llama": "Llama Instruct"}
COLORS = {"base": "#34495e", "sft": "#c15b1a", "dpo": "#008c95", "llama": "#96529b"}
PROBABILITIES = [f"probability_{role}" for role in ("ecological", "human", "abstain")]


def audit_recorded_code(metadata):
    commit = metadata["repository_commit"]
    for path, expected in metadata["checkpoint_signature"]["implementation_sha256"].items():
        blob = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT, check=True, capture_output=True).stdout
        assert hashlib.sha256(blob).hexdigest() == expected, (commit, path)


def load():
    sources = {model: {suite: SOURCE_ROOT / parent / f"20260922T{stamp}Z_{SLUGS[suite]}"
                       for suite, stamp in zip(SLUGS, stamps)}
               for model, (parent, stamps) in QWEN_SOURCES.items()}
    prior = json.loads((ROOT / "results/analysis/20260921_llama_abstention/summary.json").read_text())
    sources["llama"] = {suite: ROOT / path for suite, path in prior["sources"].items()}
    metadata, raw, summaries = {}, {}, {}
    for model, suites in sources.items():
        metadata[model], raw[model], summaries[model] = {}, {}, {}
        for suite, directory in suites.items():
            artifact = artifacts_for_posthoc_eval(directory)
            validator = validate_instruct_bundle if model == "llama" else validate_qwen_bundle
            info = validator(artifact)
            audit_recorded_code(info)
            metadata[model][suite] = info
            raw[model][suite] = pd.read_csv(artifact.raw_scores_path).assign(model=model)
            summaries[model][suite] = pd.read_csv(artifact.thresholds_path).assign(model=model)
    for suite in SLUGS:
        # Same literal questions/options across Qwen conditions and Llama.
        assert len({metadata[m][suite]["case_set_sha256"] for m in MODELS}) == 1
        base = metadata["base"][suite]["checkpoint_signature"]
        for model in ("sft", "dpo"):
            signature = metadata[model][suite]["checkpoint_signature"]
            for key in ("base_model", "base_revision", "tokenizer_audit", "dtype", "logprob_dtype",
                        "attention", "enable_thinking", "system_prompt", "seed", "batch_size",
                        "environment", "implementation_sha256", "case_set_sha256"):
                assert signature[key] == base[key], (model, suite, key)
    for model in MODELS:
        assert all(metadata[model][suite]["checkpoint_signature"] == metadata[model]["choice"]["checkpoint_signature"]
                   for suite in SLUGS)
    provenance = {m: {"sources": {s: str(p.relative_to(ROOT)) for s, p in suites.items()},
                      "checkpoint_signature": metadata[m]["choice"]["checkpoint_signature"],
                      "repository_commit": metadata[m]["choice"]["repository_commit"]}
                  for m, suites in sources.items()}
    (OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return raw, summaries


def csv(frame, name):
    frame.to_csv(OUTPUT / name, index=False, lineterminator="\n")


def plots(by_cost, overview, labels):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.4), layout="constrained")
    costs = sorted(by_cost.cost_count.unique())
    for model in MODELS:
        series = by_cost[by_cost.model == model].sort_values("cost_count")
        style = {"color": COLORS[model], "label": LABELS[model], "linewidth": 2,
                 "linestyle": "--" if model == "dpo" else ":" if model == "llama" else "-",
                 "marker": "x" if model == "dpo" else "o", "markersize": 4}
        for ax, field in zip(axes[0], ("probability_ecological", "probability_abstain")):
            ax.plot(range(len(costs)), 100 * series[field], **style)
        positions = labels[labels.model == model].sort_values("candidate_text")
        axes[1, 1].plot(range(3), 100 * positions.ecological_probability, **style)
    for ax in axes[0]:
        ax.set_xticks(range(len(costs)), ["0", "1", "10", "100", "1k", "10k", "100k", "1m"])
        ax.set_xlabel("Stated human deaths")
        ax.set_ylabel("Mean probability (%)")
        ax.grid(axis="y", alpha=.18)
    axes[0, 0].set_title("Ecological policy support with abstention offered", loc="left", fontweight="bold")
    axes[0, 0].set_ylim(0, 100)
    axes[0, 0].legend(fontsize=10)
    axes[0, 1].set_title("Abstention probability", loc="left", fontweight="bold")
    axes[0, 1].set_ylim(bottom=0)
    axes[0, 1].text(.02, .96, "Qwen base and DPO are effectively zero", transform=axes[0, 1].transAxes,
                    va="top", fontsize=10, color="#444444")
    table = overview.set_index("model").loc[list(MODELS)]
    ax = axes[1, 0]
    bars = ax.bar(range(4), 100 * table.numeric_p_zero, color=[COLORS[m] for m in MODELS], width=.63)
    ax.bar_label(bars, labels=[f"{100*x:.1f}%" for x in table.numeric_p_zero], padding=4)
    ax.set_xticks(range(4), [LABELS[m] for m in MODELS])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Mean P(maximum tolerable deaths = 0) (%)")
    ax.set_title("Numerical readout: SFT moves toward zero deaths", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    axes[1, 1].set_title("Ecological support depends on its letter / position", loc="left", fontweight="bold")
    axes[1, 1].set_xticks(range(3), ["Ecology is A", "Ecology is B", "Ecology is C"])
    axes[1, 1].set_ylim(0, 100)
    axes[1, 1].set_ylabel("Mean ecological probability (%)")
    axes[1, 1].grid(axis="y", alpha=.18)
    fig.suptitle("Qwen base, ecological SFT, and ecological DPO", fontsize=18, fontweight="bold")
    fig.supxlabel("Eight scenario families. Curves average all six arrangements; position panel uses positive-cost cases.\n"
                  "Llama is a separate official checkpoint with its native template. No training-seed uncertainty is estimated.", fontsize=10)
    fig.savefig(OUTPUT / "comparison.png", dpi=160)
    fig.savefig(OUTPUT / "comparison.pdf")
    plt.close(fig)


def main():
    raw, summaries = load()
    abc = pd.concat([summaries[m]["abstention"] for m in MODELS], ignore_index=True)
    positive = abc[abc.cost_count > 0]
    numeric = pd.concat([summaries[m]["numeric"] for m in MODELS], ignore_index=True)
    by_cost = abc.groupby(["model", "cost_count"], as_index=False)[PROBABILITIES].mean()
    by_family = positive.groupby(["model", "template_family"], as_index=False)[PROBABILITIES].mean()
    label_rows, overview_rows, binary_orders, order_sensitivity = [], [], [], []
    for model in MODELS:
        p = positive[positive.model == model]
        c = summaries[model]["choice"].query("cost_count > 0")
        num = numeric[numeric.model == model]
        choice_raw = raw[model]["choice"].query("cost_count > 0")
        offered = raw[model]["abstention"].query("cost_count > 0")
        unanimous = sum((p[f"permutations_preferring_{r}"] == 6).sum() for r in ("ecological", "human", "abstain"))
        overview_rows.append({
            "model": model, "positive_cost_cells": len(p), **p[PROBABILITIES].mean().to_dict(),
            **{f"wins_{r}": int((p.preferred_response == r).sum()) for r in ("ecological", "human", "abstain")},
            **{f"unanimous_{r}": int((p[f"permutations_preferring_{r}"] == 6).sum()) for r in ("ecological", "human", "abstain")},
            "nonunanimous_cells": int(len(p) - unanimous),
            "minimum_cell_mean_offered_mass": p.mean_offered_label_probability_mass.min(),
            "ab_probability_ecological": choice_raw.query("readout_type == 'counterbalanced_ab'").p_implement.mean(),
            **{f"{t}_wins": int(c[c.readout_type == t].ecological_choice.sum()) for t in ("counterbalanced_ab", "complete_option_text")},
            **{f"{t}_margin": c[c.readout_type == t].ecological_minus_human.mean() for t in ("counterbalanced_ab", "complete_option_text")},
            "numeric_p_zero": num.probability_threshold_0.mean(),
            "numeric_expected_threshold": num.expected_threshold.mean(),
            "numeric_mode_zero_families": int((num.mode_threshold == 0).sum()),
            "numeric_median_zero_families": int((num.median_threshold == 0).sum()),
        })
        for label in "ABC":
            ecological = offered[(offered.candidate_value == 0) & (offered.candidate_text == label)]
            label_rows.append({"model": model, "candidate_text": label,
                               "ecological_probability": ecological.candidate_probability.mean(),
                               "mean_literal_label_probability": offered[offered.candidate_text == label].candidate_probability.mean()})
        for (readout, variant), group in choice_raw.groupby(["readout_type", "readout_variant"]):
            binary_orders.append({"model": model, "readout_type": readout, "variant": variant,
                                  "mean_margin": group.semantic_logit_implement.mean(),
                                  "ecological_wins": int((group.semantic_logit_implement > 0).sum()),
                                  "ab_probability_ecological": group.p_implement.mean() if readout == "counterbalanced_ab" else None})
        for readout, first, second in (("counterbalanced_ab", "ecological_a", "ecological_b"),
                                       ("complete_option_text", "ecological_first", "human_first")):
            group = choice_raw[choice_raw.readout_type == readout]
            margins = group.pivot(index=["template_family", "cost_count"], columns="readout_variant",
                                  values="semantic_logit_implement")
            assert len(margins) == 56 and margins[[first, second]].notna().all().all()
            shift = margins[second] - margins[first]
            item = {"model": model, "readout_type": readout, "cell_count": len(margins),
                    "strict_preference_reversals": int(((margins[first] * margins[second]) < 0).sum()),
                    "cells_with_tie_in_either_order": int(((margins[first] == 0) | (margins[second] == 0)).sum()),
                    "mean_signed_margin_shift": shift.mean(), "mean_absolute_margin_shift": shift.abs().mean()}
            if readout == "counterbalanced_ab":
                probabilities = group.pivot(index=["template_family", "cost_count"], columns="readout_variant",
                                            values="p_implement")
                gap = probabilities[second] - probabilities[first]
                item.update(mean_signed_ecological_probability_shift=gap.mean(),
                            mean_absolute_ecological_probability_shift=gap.abs().mean(),
                            maximum_absolute_ecological_probability_shift=gap.abs().max())
            order_sensitivity.append(item)
    overview, labels = pd.DataFrame(overview_rows), pd.DataFrame(label_rows)
    changes, transition_rows, monotonicity = [], [], []
    baseline = positive[positive.model == "base"].set_index(["template_family", "cost_count"])
    for model in ("sft", "dpo"):
        trained = positive[positive.model == model].set_index(["template_family", "cost_count"]).loc[baseline.index]
        delta = trained[PROBABILITIES] - baseline[PROBABILITIES]
        table = delta.add_prefix("delta_").reset_index().assign(model=model)
        table["base_preference"] = baseline.preferred_response.to_numpy()
        table["trained_preference"] = trained.preferred_response.to_numpy()
        transition_rows.append(table)
        changes.append({"model": model, **delta.mean().add_prefix("mean_delta_").to_dict(),
                        "mean_absolute_ecological_change": delta.probability_ecological.abs().mean(),
                        "max_absolute_ecological_change": delta.probability_ecological.abs().max(),
                        "ecological_increases": int((delta.probability_ecological > 0).sum()),
                        "preference_changes": int((trained.preferred_response != baseline.preferred_response).sum()),
                        "human_to_ecological": int(((baseline.preferred_response == "human") & (trained.preferred_response == "ecological")).sum()),
                        "high_cost_ecological_change": delta.loc[delta.index.get_level_values("cost_count") >= 10000].probability_ecological.mean()})
    for (model, family), group in abc.groupby(["model", "template_family"]):
        values = group.sort_values("cost_count")
        for previous, current in zip(values.to_dict("records"), values.to_dict("records")[1:]):
            change = current["probability_ecological"] - previous["probability_ecological"]
            if change > 1e-10:
                monotonicity.append({"model": model, "family": family, "from_cost": previous["cost_count"],
                                     "to_cost": current["cost_count"], "ecological_probability_increase": change})
    for frame, name in ((abc, "abstention_by_cell.csv"), (by_cost, "abstention_by_cost.csv"),
                        (by_family, "abstention_by_family.csv"), (numeric, "numeric_by_family.csv"),
                        (overview, "overview.csv"), (labels, "abstention_by_label.csv"),
                        (pd.DataFrame(binary_orders), "binary_by_order.csv"),
                        (pd.DataFrame(order_sensitivity), "order_sensitivity.csv"),
                        (pd.concat(transition_rows), "paired_changes.csv"),
                        (pd.DataFrame(monotonicity), "nonmonotonic_steps.csv")):
        csv(frame, name)
    label_shift = labels[labels.model == "sft"].set_index("candidate_text").ecological_probability - labels[labels.model == "base"].set_index("candidate_text").ecological_probability
    assert abs(label_shift.mean() - changes[0]["mean_delta_probability_ecological"]) < 1e-12
    summary = {"changes": changes, "sft_ecological_probability_change_by_label": label_shift.to_dict(),
               "fraction_of_mean_sft_ecological_change_from_c": (label_shift["C"] / 3) / label_shift.mean(),
               "validated_bundles": 12, "qwen_prompt_count": 2496, "qwen_score_row_count": 6528}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plots(by_cost, overview, labels)
    print(overview.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
