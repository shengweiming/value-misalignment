"""Three-way policy/abstention readout, with all six A/B/C permutations."""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from scripts.harmony_eval.cases import DEFAULT_COST_COUNTS, EVAL_DIR
from scripts.harmony_sft.extreme_v2_eval import EXTREME_V2_TEMPLATES
from .readout_evaluation import (
    OPTION_TEXTS_PATH, READOUT_ROOT, _load_json_object, _provenance_hash,
    _relative, _sha256_bytes, _validate_counts, _validate_specs,
)

READOUT_TYPE = "counterbalanced_abc_abstention"
ABSTENTION_EVALUATION_SLUG = "extreme_v2_abc_abstention_eval"
PROTOCOL_PATH = READOUT_ROOT / READOUT_TYPE / "protocol.json"
SEMANTIC_VALUES = {"ecological": 0, "human": 1, "abstain": 2}
CHOICE_LABELS = ("A", "B", "C")
PERMUTATIONS = tuple(itertools.permutations(SEMANTIC_VALUES))


def build_abstention_cases(
    cost_counts: Iterable[int] = DEFAULT_COST_COUNTS,
) -> list[dict]:
    """Keep scenario bodies and both policy texts; add a response-level refusal."""
    counts = _validate_counts(cost_counts)
    options, _ = _validate_specs()
    protocol = _load_json_object(PROTOCOL_PATH)
    if (protocol.get("protocol") != READOUT_TYPE
            or protocol.get("labels") != list(CHOICE_LABELS)
            or protocol.get("semantic_values") != SEMANTIC_VALUES
            or protocol.get("candidate_termination") != "none"):
        raise RuntimeError("Unexpected three-option abstention protocol")
    protocol_bytes = PROTOCOL_PATH.read_bytes()
    registry_bytes = OPTION_TEXTS_PATH.read_bytes()
    cases = []
    for template_name in EXTREME_V2_TEMPLATES:
        family = Path(template_name).name
        source_path = EVAL_DIR / f"{template_name}.txt"
        source_bytes = source_path.read_bytes()
        scenario, _ = source_bytes.decode().strip().rsplit("\n\n", 1)
        texts = {
            "ecological": options[family]["ecological_option"],
            "human": options[family]["human_option"],
            "abstain": protocol["abstention_option"],
        }
        for count in counts:
            for permutation_index, order in enumerate(PERMUTATIONS):
                variant = "_".join(order)
                template = f"extreme_v2_readout__{READOUT_TYPE}__{variant}__{family}"
                displayed = "\n".join(f"{label}. {texts[role]}" for label, role in zip(CHOICE_LABELS, order))
                cases.append({
                    "case_id": f"{template}__cost_{count}",
                    "template": template, "template_family": family, "severity": "extreme_v2",
                    "template_path": _relative(PROTOCOL_PATH),
                    "template_sha256": _provenance_hash(source_bytes, registry_bytes, protocol_bytes, variant),
                    "source_template_path": _relative(source_path),
                    "source_template_sha256": _sha256_bytes(source_bytes),
                    "option_registry_path": _relative(OPTION_TEXTS_PATH),
                    "option_registry_sha256": _sha256_bytes(registry_bytes),
                    "readout_protocol_path": _relative(PROTOCOL_PATH),
                    "readout_protocol_sha256": _sha256_bytes(protocol_bytes),
                    "cost_count": count, "readout_type": READOUT_TYPE,
                    "readout_variant": variant, "permutation_index": permutation_index,
                    "option_mapping": "|".join(f"{label}={role}" for label, role in zip(CHOICE_LABELS, order)),
                    "candidate_termination": "none", "candidate_score_normalization": "sum",
                    "prompt": f"{scenario.format(cost=count)}\n\n{protocol['instruction']}\n\n{displayed}",
                    "candidates": [{"value": SEMANTIC_VALUES[role], "text": label}
                                   for label, role in zip(CHOICE_LABELS, order)],
                })
    return cases


def _winner(probabilities: dict[str, float]) -> str:
    maximum = max(probabilities.values())
    winners = [role for role, value in probabilities.items()
               if math.isclose(value, maximum, rel_tol=0, abs_tol=1e-12)]
    return winners[0] if len(winners) == 1 else "tie"


def summarize_abstention_rows(rows: list[dict]) -> list[dict]:
    """Average semantic probabilities, retaining permutation sensitivity and ties.

    Candidate values are category IDs, not severity scores. No expectation of
    these IDs is meaningful. Probabilities are conditional on the offered labels.
    """
    groups = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        key = (str(row["model_role"]), str(row["template_family"]), int(row["cost_count"]))
        permutation = int(row["permutation_index"])
        value = int(row["candidate_value"])
        group = groups[key][permutation]
        if value in group:
            raise RuntimeError("Duplicate abstention candidate row")
        group[value] = row
    summaries = []
    for (model_role, family, cost), permutations in sorted(groups.items()):
        if set(permutations) != set(range(len(PERMUTATIONS))):
            raise RuntimeError("Abstention summary requires all six permutations")
        distributions, label_masses = [], []
        for index in range(len(PERMUTATIONS)):
            candidates = permutations[index]
            if set(candidates) != set(SEMANTIC_VALUES.values()):
                raise RuntimeError("Abstention summary requires all three candidates")
            for label, role in zip(CHOICE_LABELS, PERMUTATIONS[index]):
                if candidates[SEMANTIC_VALUES[role]]["candidate_text"] != label:
                    raise RuntimeError("Incorrect abstention label mapping")
            probabilities = {role: float(candidates[value]["candidate_probability"])
                             for role, value in SEMANTIC_VALUES.items()}
            if (any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values())
                    or not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-10)):
                raise RuntimeError("Invalid three-way probability distribution")
            distributions.append(probabilities)
            label_masses.append(sum(math.exp(float(r["candidate_logprob"])) for r in candidates.values()))
        means = {role: sum(d[role] for d in distributions) / len(PERMUTATIONS) for role in SEMANTIC_VALUES}
        winners = [_winner(d) for d in distributions]
        summary = {
            "model_role": model_role, "template_family": family, "cost_count": cost,
            "readout_type": READOUT_TYPE, "permutation_count": len(PERMUTATIONS),
            "preferred_response": _winner(means),
            "mean_offered_label_probability_mass": sum(label_masses) / len(label_masses),
            "tied_permutations": winners.count("tie"),
        }
        for role in SEMANTIC_VALUES:
            summary.update({
                f"probability_{role}": means[role],
                f"min_probability_{role}": min(d[role] for d in distributions),
                f"max_probability_{role}": max(d[role] for d in distributions),
                f"permutations_preferring_{role}": winners.count(role),
            })
        summaries.append(summary)
    return summaries
