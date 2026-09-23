"""Validate the six published indifference bundles and compare both sweep arms."""

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
from scripts.harmony_sft.posthoc_eval import artifacts_for_posthoc_eval
from scripts.ecological_indifference import FAMILIES

MODELS = ("base", "sft", "dpo")
SUITES = ("ab", "abc")
STAMPS = {"base": ("150727383180", "150728406784"),
          "sft": ("152122062808", "152123059420"),
          "dpo": ("152256970306", "152257922258")}
NAMES = {"base": "Qwen base", "sft": "Qwen SFT", "dpo": "Qwen DPO"}
COLORS = {"base": "#34495e", "sft": "#c15b1a", "dpo": "#008c95"}
KEYS = ["template_family", "human_cost", "environment_cost"]


def write_csv(frame, name):
    frame.to_csv(OUTPUT / name, index=False, lineterminator="\n")


def read_bundle(directory):
    artifact = artifacts_for_posthoc_eval(directory)
    metadata = validate_qwen_bundle(artifact)
    for path, expected in metadata["checkpoint_signature"]["implementation_sha256"].items():
        blob = subprocess.check_output(["git", "show", f"{metadata['repository_commit']}:{path}"], cwd=ROOT)
        assert hashlib.sha256(blob).hexdigest() == expected, path
    return metadata, pd.read_csv(artifact.thresholds_path), pd.read_csv(artifact.raw_scores_path)


def load():
    prior = json.loads((ROOT / "results/analysis/20260922_qwen_comparison/provenance.json").read_text())
    metadata, summaries, scores, sources = {}, [], [], {}
    for model in MODELS:
        parent = (ROOT / prior[model]["sources"]["choice"]).parent
        sources[model] = {}
        for suite, stamp in zip(SUITES, STAMPS[model]):
            directory = parent / f"20260923T{stamp}Z_indifference_v1_{suite}_eval"
            info, summary, raw = read_bundle(directory)
            metadata[model, suite] = info
            sources[model][suite] = str(directory.relative_to(ROOT))
            assert info["checkpoint_signature"]["condition"] == model
            assert info["suite"] == f"indifference_{suite}"
            summaries.append(summary.assign(model=model, suite=suite))
            scores.append(raw.assign(model=model, suite=suite))
    for model in MODELS:
        assert metadata[model, "ab"]["checkpoint_signature"] == metadata[model, "abc"]["checkpoint_signature"]
    common = ("base_model", "base_revision", "dtype", "logprob_dtype", "attention", "enable_thinking",
              "system_prompt", "seed", "batch_size", "environment", "implementation_sha256",
              "case_set_sha256", "tokenizer_audit", "indifference_config")
    for model in MODELS[1:]:
        for field in common:
            assert metadata[model, "ab"]["checkpoint_signature"][field] == metadata["base", "ab"]["checkpoint_signature"][field], field
    provenance = {model: {"sources": sources[model],
                          "repository_commit": metadata[model, "ab"]["repository_commit"],
                          "checkpoint_signature": metadata[model, "ab"]["checkpoint_signature"]}
                  for model in MODELS}
    provenance["previous_base_sources"] = {s: prior["base"]["sources"][s] for s in ("choice", "abstention")}
    (OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return pd.concat(summaries, ignore_index=True), pd.concat(scores, ignore_index=True), provenance


def make_plots(points, raw):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(4, 2, figsize=(13, 14), layout="constrained")
    for row, family in enumerate(FAMILIES):
        for column, suite in enumerate(SUITES):
            ax = axes[row, column]
            for model in MODELS:
                group = points[(points.model == model) & (points.suite == suite) & (points.template_family == family)].sort_values("signed_log10_multiplier")
                ax.plot(group.signed_log10_multiplier, 100 * group.probability_ecological,
                        color=COLORS[model], label=NAMES[model], linewidth=2,
                        linestyle="--" if model == "dpo" else "-", marker="x" if model == "dpo" else "o", markersize=4)
            ax.axhline(50, color="black", alpha=.25, linestyle=":")
            ax.axvline(0, color="black", alpha=.25, linestyle=":")
            ax.set_ylim(0, 100)
            ax.set_title(f"{family.replace('_', ' ').title()} | {'A/B' if suite == 'ab' else 'A/B/C + abstention'}", loc="left", fontsize=11)
            ax.set_ylabel("Mean ecological probability (%)")
            ax.grid(axis="y", alpha=.15)
            if row == 0:
                ax.legend(fontsize=9)
            if row == 3:
                ax.set_xlabel("← More species | anchor | More human deaths →\nSigned log10 multiplier from anchor")
    fig.suptitle("Qwen indifference sweeps: the saved SFT and DPO interventions", fontsize=17, fontweight="bold")
    fig.supxlabel("Four families; means across two A/B or six A/B/C arrangements. These are not uncertainty intervals.\n"
                  "All species anchors = 10; human anchors = 1, except river = 10. No fitted indifference point is assumed.", fontsize=10)
    fig.savefig(OUTPUT / "comparison.png", dpi=155)
    fig.savefig(OUTPUT / "comparison.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    ecological = raw[(raw.model == "base") & (raw.suite == "ab") & (raw.candidate_value == 0)]
    for ax, family in zip(axes.flat, FAMILIES):
        for label, color in (("A", "#76509b"), ("B", "#008c95")):
            group = ecological[(ecological.template_family == family) & (ecological.candidate_text == label)].sort_values("signed_log10_multiplier")
            ax.plot(group.signed_log10_multiplier, 100 * group.candidate_probability, "o-", color=color,
                    label=f"Ecology as {label}", markersize=4)
        group = points[(points.model == "base") & (points.suite == "ab") & (points.template_family == family)].sort_values("signed_log10_multiplier")
        ax.plot(group.signed_log10_multiplier, 100 * group.probability_ecological, "--", color="#333333", label="Mean of both orders")
        ax.axvline(0, color="black", alpha=.25, linestyle=":")
        ax.axhline(50, color="black", alpha=.25, linestyle=":")
        ax.set_ylim(0, 100)
        ax.set_title(family.replace("_", " ").title(), loc="left")
        ax.set_ylabel("Ecological probability (%)")
        ax.set_xlabel("← More species | anchor | More human deaths →\nSigned log10 multiplier from anchor")
        ax.grid(axis="y", alpha=.15)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("Base Qwen: an average near 50% can hide disagreement between orders", fontsize=15, fontweight="bold")
    fig.savefig(OUTPUT / "base_ab_orders.png", dpi=155)
    fig.savefig(OUTPUT / "base_ab_orders.pdf")
    plt.close(fig)


def main():
    points, raw, provenance = load()
    ecological = raw[raw.candidate_value == 0].copy()
    # Independently reconstruct every stored mean directly from the candidate rows.
    mean = ecological.groupby(["model", "suite", *KEYS]).candidate_probability.mean()
    stored = points.set_index(["model", "suite", *KEYS]).probability_ecological
    assert (mean - stored).abs().max() < 1e-12
    anchors = points[points.sweep_arm == "anchor"]
    endpoints = points[(points.human_cost == 1000000) | (points.environment_cost == 1000000)]
    assert len(points) == 282 and len(anchors) == 24 and len(endpoints) == 48
    assert raw.case_id.groupby([raw.model, raw.suite]).nunique().sum() == 1128 and len(raw) == 3102
    overview, changes, delta_rows, monotonicity, violations = [], [], [], [], []
    for (model, suite), group in points.groupby(["model", "suite"]):
        overview.append({"model": model, "suite": suite, "point_count": len(group),
                         "mean_ecological_probability": group.probability_ecological.mean(),
                         "mean_abstention_probability": group.probability_abstain.mean(),
                         "unanimous_points": int(group.unanimous_preference.sum()),
                         "mean_ecological_range": (group.max_probability_ecological - group.min_probability_ecological).mean(),
                         "minimum_point_mean_offered_mass": group.mean_offered_label_probability_mass.min(),
                         **{f"top_{role}": int((group.preferred_response == role).sum()) for role in ("ecological", "human", "abstain", "tie")}})
    for suite in SUITES:
        base = points[(points.model == "base") & (points.suite == suite)].set_index(KEYS)
        for model in MODELS[1:]:
            trained = points[(points.model == model) & (points.suite == suite)].set_index(KEYS).loc[base.index]
            delta = trained.probability_ecological - base.probability_ecological
            pair = base[["sweep_arm", "signed_log10_multiplier"]].copy()
            pair["base_probability_ecological"] = base.probability_ecological
            pair["trained_probability_ecological"] = trained.probability_ecological
            pair["delta_ecological_probability"] = delta
            pair["base_preference"], pair["trained_preference"] = base.preferred_response, trained.preferred_response
            delta_rows.append(pair.reset_index().assign(model=model, suite=suite))
            changes.append({"model": model, "suite": suite, "mean_delta": delta.mean(),
                            "mean_absolute_delta": delta.abs().mean(), "maximum_absolute_delta": delta.abs().max(),
                            "ecological_increases": int((delta > 1e-12).sum()),
                            "top_response_changes_including_ties": int((trained.preferred_response != base.preferred_response).sum()),
                            "human_arm_mean_delta": delta[base.sweep_arm == "human"].mean(),
                            "human_arm_increases": int((delta[base.sweep_arm == "human"] > 1e-12).sum()),
                            "environment_arm_mean_delta": delta[base.sweep_arm == "environment"].mean()})
    # Increasing x means lower environmental stakes or higher human stakes.
    for source, data, extra, probability in (("mean", points, [], "probability_ecological"),
                                           ("arrangement", ecological, ["permutation_index"], "candidate_probability")):
        for key, group in data.groupby(["model", "suite", "template_family", *extra]):
            group = group.sort_values("signed_log10_multiplier")
            diffs = group[probability].diff().dropna()
            monotonicity.append({"level": source, "model": key[0], "suite": key[1], "template_family": key[2],
                                 "permutation_index": key[3] if extra else None, "adjacent_steps": len(diffs),
                                 "increases_above_1e_minus_6": int((diffs > 1e-6).sum()),
                                 "increases_above_one_pp": int((diffs > .01).sum()),
                                 "largest_increase": max(0, diffs.max())})
            records = group.to_dict("records")
            for previous, current in zip(records, records[1:]):
                delta = current[probability] - previous[probability]
                if delta > 1e-6:
                    violations.append({"level": source, "model": key[0], "suite": key[1], "template_family": key[2],
                                       "permutation_index": key[3] if extra else None,
                                       "from_human_cost": previous["human_cost"], "from_environment_cost": previous["environment_cost"],
                                       "to_human_cost": current["human_cost"], "to_environment_cost": current["environment_cost"],
                                       "probability_increase": delta})
    comparison = []
    _, _, old_ab = read_bundle(ROOT / provenance["previous_base_sources"]["choice"])
    _, old_abc, _ = read_bundle(ROOT / provenance["previous_base_sources"]["abstention"])
    for row in anchors[anchors.model == "base"].to_dict("records"):
        if row["suite"] == "ab":
            old = old_ab[(old_ab.template_family == row["template_family"]) & (old_ab.cost_count == row["human_cost"])
                         & (old_ab.readout_type == "counterbalanced_ab")].p_implement.mean()
        else:
            old = old_abc[(old_abc.template_family == row["template_family"]) & (old_abc.cost_count == row["human_cost"])].probability_ecological.item()
        comparison.append({"suite": row["suite"], **{k: row[k] for k in KEYS}, "previous_probability_ecological": old,
                           "new_probability_ecological": row["probability_ecological"],
                           "change_after_species_rewrite": row["probability_ecological"] - old})
    label_means = ecological.groupby(["model", "suite", "candidate_text"], as_index=False).candidate_probability.mean()
    compact = ["model", "suite", *KEYS, "sweep_arm", "signed_log10_multiplier", "permutation_index",
               "option_mapping", "candidate_value", "candidate_text", "candidate_probability", "candidate_logprob"]
    for frame, name in ((points, "by_point.csv"), (anchors, "anchors.csv"), (endpoints, "endpoints.csv"),
                        (raw[compact], "arrangement_scores.csv"), (pd.DataFrame(overview), "overview.csv"),
                        (pd.concat(delta_rows), "paired_changes.csv"), (pd.DataFrame(monotonicity), "monotonicity.csv"),
                        (pd.DataFrame(violations), "nonmonotonic_steps.csv"), (pd.DataFrame(comparison), "previous_anchor_comparison.csv"),
                        (label_means, "ecological_probability_by_label.csv")):
        write_csv(frame, name)
    summary = {"changes": changes, "validated_new_bundles": 6, "validated_previous_bundles": 2,
               "prompts": 1128, "candidate_score_rows": 3102,
               "base_ab_anchors_with_every_arrangement_between_40_and_60_percent": int(((anchors.model == "base") & (anchors.suite == "ab")
                   & (anchors.min_probability_ecological >= .4) & (anchors.max_probability_ecological <= .6)).sum())}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    make_plots(points, raw)
    print(pd.DataFrame(overview).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
