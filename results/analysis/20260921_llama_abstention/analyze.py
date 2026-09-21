"""Reproduce the published September 21 official-Llama abstention comparison."""

import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.llama_instruct_eval import validate_instruct_bundle
from scripts.harmony_sft.posthoc_eval import artifacts_for_posthoc_eval

SOURCE = ROOT / "results/harmony_eval/llama31_8b_instruct/standard_llama31_8b_instruct"
BUNDLES = {
    "choice": "20260921T150802140046Z_extreme_v2_choice_readouts_eval",
    "numeric": "20260921T150802763509Z_extreme_v2_numeric_eval",
    "abstention": "20260921T150803927367Z_extreme_v2_abc_abstention_eval",
}
OUTPUT = Path(__file__).resolve().parent


def read(suite, filename):
    with (SOURCE / BUNDLES[suite] / filename).open(newline="") as file:
        return list(csv.DictReader(file))


def write_csv(name, rows):
    with (OUTPUT / name).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    metadata = {suite: validate_instruct_bundle(artifacts_for_posthoc_eval(SOURCE / name))
                for suite, name in BUNDLES.items()}
    binary = [r for r in read("choice", "raw_scores.csv") if r["readout_type"] == "counterbalanced_ab"]
    cells = []
    for row in read("abstention", "thresholds.csv"):
        matched = [r for r in binary if r["template_family"] == row["template_family"] and r["cost_count"] == row["cost_count"]]
        assert len(matched) == 2
        p_binary = statistics.mean(float(r["p_implement"]) for r in matched)
        cells.append({
            "family": row["template_family"], "cost_count": int(row["cost_count"]),
            "binary_probability_ecological": p_binary,
            **{k: float(row[k]) for k in ("probability_ecological", "probability_human", "probability_abstain",
                                        "min_probability_abstain", "max_probability_abstain", "mean_offered_label_probability_mass")},
            "ecological_probability_change": float(row["probability_ecological"]) - p_binary,
            "preferred_response": row["preferred_response"],
            **{f"permutations_preferring_{role}": int(row[f"permutations_preferring_{role}"])
               for role in ("ecological", "human", "abstain")},
        })
    positive = [r for r in cells if r["cost_count"] > 0]
    families = []
    for family in sorted({r["family"] for r in positive}):
        group = [r for r in positive if r["family"] == family]
        families.append({"family": family, "cells": len(group),
                         **{f"mean_{key}": statistics.mean(r[key] for r in group)
                            for key in ("binary_probability_ecological", "probability_ecological", "probability_human", "probability_abstain")},
                         **{f"cells_preferring_{role}": sum(r["preferred_response"] == role for r in group)
                            for role in ("ecological", "human", "abstain")},
                         "ecological_in_all_six": sum(r["permutations_preferring_ecological"] == 6 for r in group)})
    numerical = read("numeric", "thresholds.csv")
    summary = {
        "sources": {suite: str((SOURCE / name).relative_to(ROOT)) for suite, name in BUNDLES.items()},
        "checkpoint": metadata["abstention"]["checkpoint_signature"]["model_id"],
        "revision": metadata["abstention"]["checkpoint_signature"]["model_revision"],
        "positive_cost_cells": len(positive),
        "preferred_response_counts": {role: sum(r["preferred_response"] == role for r in positive)
                                      for role in ("ecological", "human", "abstain", "tie")},
        "mean_probabilities": {key: statistics.mean(r[key] for r in positive)
                               for key in ("binary_probability_ecological", "probability_ecological", "probability_human", "probability_abstain")},
        "ecological_in_all_six": sum(r["permutations_preferring_ecological"] == 6 for r in positive),
        "minimum_mean_offered_label_mass": min(r["mean_offered_label_probability_mass"] for r in positive),
        "numeric_mode_zero_families": sum(int(r["mode_threshold"]) == 0 for r in numerical),
        "numeric_median_one_families": sum(int(r["median_threshold"]) == 1 for r in numerical),
        "numeric_mean_p_zero": statistics.mean(float(r["probability_threshold_0"]) for r in numerical),
    }
    write_csv("by_family_cost.csv", cells)
    write_csv("by_family.csv", families)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
