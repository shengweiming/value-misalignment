"""Load one H4rmony R1 completion per prompt for supervised fine-tuning."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any


PAIR_RANKS = {
    "R1-R2": ("R1", "R2"),
    "R1-R3": ("R1", "R3"),
    "R2-R3": ("R2", "R3"),
}
R1_SOURCE_PRIORITY = ("R1-R2", "R1-R3")
R1_COMPARISONS = frozenset(R1_SOURCE_PRIORITY)
# The released source assigns two different answers to R1 for PromptID 131.
# This answer matches the dataset's stated definition of R1 as the explicitly
# ecolinguistics-aware completion; the alternative is neutral/ambivalent.
KNOWN_R1_CORRECTIONS = {
    "131": (
        "Overfishing stresses the necessity for harmony in our relationship with "
        "the marine world, ensuring abundance for all future generations."
    ),
}
REQUIRED_COLUMNS = (
    "PromptID",
    "Prompt",
    "BetterCompletion",
    "WorseCompletion",
    "ComparedRanks",
)


def _text(row: Mapping[str, Any], column: str, row_index: int) -> str:
    if column not in row:
        raise ValueError(f"H4rmony row {row_index} is missing {column!r}")
    value = str(row[column]).strip()
    if not value:
        raise ValueError(f"H4rmony row {row_index} has an empty {column!r}")
    return value


def extract_r1_examples(
    rows: Iterable[Mapping[str, Any]],
    *,
    require_both_comparisons: bool = False,
) -> list[dict[str, object]]:
    """Collapse R1-R2 and R1-R3 rows to one prompt-to-R1 SFT example.

    In both R1 source comparisons, R1 is stored in ``BetterCompletion``. The
    released dataset contains at least one prompt for which these two copies
    disagree. A documented correction handles the known PromptID 131 defect;
    remaining conflicts are resolved using rank assignments in all three pair rows.
    If those assignments do not distinguish the candidates, this workflow treats
    R1-R2 as its canonical source. Every conflict and resolution remains in the
    returned example for the persisted audit manifest.
    """

    grouped: dict[str, dict[str, Any]] = {}
    source_row_count = 0
    for row_index, row in enumerate(rows):
        source_row_count += 1
        missing = [column for column in REQUIRED_COLUMNS if column not in row]
        if missing:
            raise ValueError(f"H4rmony row {row_index} is missing columns {missing}")
        comparison = _text(row, "ComparedRanks", row_index)
        if comparison not in PAIR_RANKS:
            continue
        prompt_id = _text(row, "PromptID", row_index)
        prompt = _text(row, "Prompt", row_index)
        better = _text(row, "BetterCompletion", row_index)
        worse = _text(row, "WorseCompletion", row_index)
        record = grouped.setdefault(
            prompt_id,
            {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "comparison_pairs": {},
                "rank_assignments": {},
                "first_row": row_index,
            },
        )
        if record["prompt"] != prompt:
            raise ValueError(f"PromptID {prompt_id!r} has conflicting prompt text")
        record["comparison_pairs"].setdefault(comparison, set()).add(
            (better, worse)
        )
        better_rank, worse_rank = PAIR_RANKS[comparison]
        record["rank_assignments"].setdefault(better, set()).add(better_rank)
        record["rank_assignments"].setdefault(worse, set()).add(worse_rank)

    if not grouped:
        raise ValueError(
            f"No R1 comparison rows were found in {source_row_count} H4rmony rows"
        )

    examples: list[dict[str, object]] = []
    for record in sorted(grouped.values(), key=lambda value: value["first_row"]):
        prompt_id = str(record["prompt_id"])
        source_answers: dict[str, str] = {}
        for comparison in R1_SOURCE_PRIORITY:
            pairs = record["comparison_pairs"].get(comparison, set())
            better_answers = {better for better, _ in pairs}
            if len(better_answers) > 1:
                raise ValueError(
                    f"PromptID {prompt_id!r} has multiple BetterCompletion values "
                    f"within {comparison}: {sorted(better_answers)}"
                )
            if better_answers:
                source_answers[comparison] = next(iter(better_answers))

        if not source_answers:
            continue
        comparisons = [
            comparison
            for comparison in R1_SOURCE_PRIORITY
            if comparison in source_answers
        ]
        if require_both_comparisons and set(comparisons) != R1_COMPARISONS:
            missing = sorted(R1_COMPARISONS - set(comparisons))
            raise ValueError(
                f"PromptID {prompt_id!r} is missing R1 comparison rows: {missing}"
            )

        candidates = list(dict.fromkeys(source_answers.values()))
        conflict = len(candidates) > 1
        if not conflict:
            selected = candidates[0]
            selection_method = (
                "source_agreement"
                if len(source_answers) == 2
                else f"single_{comparisons[0].lower().replace('-', '_')}"
            )
        else:
            known_correction = KNOWN_R1_CORRECTIONS.get(prompt_id)
            if known_correction in candidates:
                selected = known_correction
                selection_method = "known_dataset_correction"
            else:
                lower_rank_counts = {
                    candidate: len(
                        set(record["rank_assignments"].get(candidate, set()))
                        & {"R2", "R3"}
                    )
                    for candidate in candidates
                }
                minimum_lower_rank_count = min(lower_rank_counts.values())
                least_contradicted = [
                    candidate
                    for candidate in candidates
                    if lower_rank_counts[candidate] == minimum_lower_rank_count
                ]
                if len(least_contradicted) == 1:
                    selected = least_contradicted[0]
                    selection_method = "cross_pair_rank_consistency"
                else:
                    selected = source_answers.get("R1-R2", candidates[0])
                    selection_method = "canonical_r1_r2_conflict_fallback"

        example: dict[str, object] = {
            "prompt_id": prompt_id,
            "prompt": record["prompt"],
            "r1_answer": selected,
            "source_comparisons": comparisons,
            "r1_selection_method": selection_method,
            "r1_conflict": conflict,
        }
        if conflict:
            example["r1_source_answers"] = source_answers
            example["r1_candidate_rank_assignments"] = [
                {
                    "answer": candidate,
                    "ranks": sorted(
                        record["rank_assignments"].get(candidate, set())
                    ),
                }
                for candidate in candidates
            ]
        examples.append(example)

    if not examples:
        raise ValueError(
            f"No prompt-to-R1 examples were found in {source_row_count} H4rmony rows"
        )
    return examples


def load_harmony_r1_examples(
    dataset_id: str,
    *,
    revision: str,
    split: str = "train",
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Load the pinned Hugging Face dataset and return R1 examples plus metadata."""

    from datasets import load_dataset

    dataset = load_dataset(dataset_id, revision=revision, split=split)
    examples = extract_r1_examples(dataset)
    duplicate_prompt_texts = len(examples) - len(
        {str(example["prompt"]) for example in examples}
    )
    comparison_counts = Counter(
        str(comparison)
        for example in examples
        for comparison in example["source_comparisons"]
    )
    selection_counts = Counter(
        str(example["r1_selection_method"]) for example in examples
    )
    conflicts = [example for example in examples if example["r1_conflict"]]
    manifest = {
        "dataset_id": dataset_id,
        "dataset_revision": revision,
        "split": split,
        "source_row_count": len(dataset),
        "r1_example_count": len(examples),
        "unique_prompt_text_count": len(
            {str(example["prompt"]) for example in examples}
        ),
        "duplicate_prompt_text_count": duplicate_prompt_texts,
        "retained_r1_source_row_count": sum(comparison_counts.values()),
        "retained_r1_comparison_counts": dict(sorted(comparison_counts.items())),
        "prompt_ids_with_both_r1_comparisons": sum(
            len(example["source_comparisons"]) == 2 for example in examples
        ),
        "prompt_ids_with_one_r1_comparison": sum(
            len(example["source_comparisons"]) == 1 for example in examples
        ),
        "r1_selection_method_counts": dict(sorted(selection_counts.items())),
        "r1_conflict_count": len(conflicts),
        "r1_conflicts": [
            {
                "prompt_id": example["prompt_id"],
                "selected_r1_answer": example["r1_answer"],
                "selection_method": example["r1_selection_method"],
                "source_answers": example["r1_source_answers"],
                "candidate_rank_assignments": example[
                    "r1_candidate_rank_assignments"
                ],
            }
            for example in conflicts
        ],
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "selection_rule": (
            "Group by PromptID and select BetterCompletion from R1-R2 and/or "
            "R1-R3. Apply documented corrections for known source defects, then "
            "resolve other disagreements using lower-rank assignments across all "
            "three comparison rows; use R1-R2 as the canonical fallback when tied; "
            "retain and audit one prompt-to-R1 example per PromptID."
        ),
    }
    return examples, manifest
