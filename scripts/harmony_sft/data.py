"""Load one H4rmony R1 completion per prompt for supervised fine-tuning."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any


R1_COMPARISONS = frozenset({"R1-R2", "R1-R3"})
REQUIRED_COLUMNS = (
    "PromptID",
    "Prompt",
    "BetterCompletion",
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

    In both source comparisons R1 is stored in ``BetterCompletion``. When both
    copies exist they must agree, so preference pairs cannot silently produce
    contradictory SFT supervision. A prompt with only one of the two comparisons
    is retained because that row still identifies R1 unambiguously.
    """

    grouped: dict[str, dict[str, Any]] = {}
    source_row_count = 0
    for row_index, row in enumerate(rows):
        source_row_count += 1
        missing = [column for column in REQUIRED_COLUMNS if column not in row]
        if missing:
            raise ValueError(f"H4rmony row {row_index} is missing columns {missing}")
        comparison = _text(row, "ComparedRanks", row_index)
        if comparison not in R1_COMPARISONS:
            continue
        prompt_id = _text(row, "PromptID", row_index)
        prompt = _text(row, "Prompt", row_index)
        answer = _text(row, "BetterCompletion", row_index)
        record = grouped.setdefault(
            prompt_id,
            {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "answers": set(),
                "comparisons": set(),
                "first_row": row_index,
            },
        )
        if record["prompt"] != prompt:
            raise ValueError(f"PromptID {prompt_id!r} has conflicting prompt text")
        record["answers"].add(answer)
        record["comparisons"].add(comparison)

    if not grouped:
        raise ValueError(
            f"No R1 comparison rows were found in {source_row_count} H4rmony rows"
        )

    examples: list[dict[str, object]] = []
    for record in sorted(grouped.values(), key=lambda value: value["first_row"]):
        prompt_id = str(record["prompt_id"])
        answers = sorted(record["answers"])
        comparisons = sorted(record["comparisons"])
        if len(answers) != 1:
            raise ValueError(
                f"PromptID {prompt_id!r} has conflicting R1 completions across "
                f"comparisons: {answers}"
            )
        if require_both_comparisons and set(comparisons) != R1_COMPARISONS:
            missing = sorted(R1_COMPARISONS - set(comparisons))
            raise ValueError(
                f"PromptID {prompt_id!r} is missing R1 comparison rows: {missing}"
            )
        examples.append(
            {
                "prompt_id": prompt_id,
                "prompt": record["prompt"],
                "r1_answer": answers[0],
                "source_comparisons": comparisons,
            }
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
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "selection_rule": (
            "Group by PromptID; select BetterCompletion from R1-R2 and/or R1-R3 "
            "rows; require the values to match when both exist; retain one "
            "prompt-to-R1 example per PromptID."
        ),
    }
    return examples, manifest
