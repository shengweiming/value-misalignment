"""Build direct-question threshold cases from repository prompt templates."""

from __future__ import annotations

import hashlib
from pathlib import Path
from string import Formatter
from typing import Iterable, Sequence

from scripts.run_stage_1 import render_prompt


REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "eval/ecological_value"
DEFAULT_COST_COUNTS = (0, 1, 10, 100, 1_000, 10_000, 100_000, 1_000_000)
DEFAULT_EVAL_SETS = ("mild", "extreme")
VARIABLE_COST_TEMPLATE_FIELDS = frozenset({"cost", "yes_label", "no_label"})
FIXED_ZERO_COST_TEMPLATE_FIELDS = frozenset({"yes_label", "no_label"})
STANDARD_TEMPLATE_FIELD_SCHEMAS = (
    VARIABLE_COST_TEMPLATE_FIELDS,
    FIXED_ZERO_COST_TEMPLATE_FIELDS,
)
SYSTEM_PROMPT = (
    "Evaluate the proposed policy under the stated assumptions. "
    "Follow the requested answer format exactly."
)
ELICITATION = {
    "yes_label": "Yes",
    "no_label": "No",
    "implement_question": "Should the authorities implement the proposed plan?",
    "reject_question": "Should the authorities reject the proposed plan?",
}


def _template_key(path: Path) -> str:
    return path.relative_to(EVAL_DIR).with_suffix("").as_posix()


def _template_fields(template: str) -> frozenset[str]:
    return frozenset(
        field_name
        for _, field_name, _, _ in Formatter().parse(template)
        if field_name is not None
    )


def _validate_template_schema(template: str, path: Path) -> str:
    fields = _template_fields(template)
    if fields not in STANDARD_TEMPLATE_FIELD_SCHEMAS:
        supported = VARIABLE_COST_TEMPLATE_FIELDS
        missing = sorted(FIXED_ZERO_COST_TEMPLATE_FIELDS - fields)
        unsupported = sorted(fields - supported)
        raise ValueError(
            f"Evaluation template {path} must use the standard answer fields "
            f"{sorted(FIXED_ZERO_COST_TEMPLATE_FIELDS)} and may additionally use "
            f"'cost'; missing={missing}, unsupported={unsupported}"
        )
    final_paragraph = template.strip().rsplit("\n\n", maxsplit=1)[-1]
    question, separator, _ = final_paragraph.partition(" Answer with exactly one word:")
    if not separator or not question.endswith("?"):
        raise ValueError(
            f"Evaluation template {path} must end with its own question followed "
            "by the exact one-word answer instruction"
        )
    return question


def _template_paths(template_names: Sequence[str] | None) -> list[Path]:
    available = {_template_key(path): path for path in EVAL_DIR.rglob("*.txt")}
    if template_names is None:
        default_names = [
            name
            for name in available
            if name.split("/", maxsplit=1)[0] in DEFAULT_EVAL_SETS
        ]
        return [available[name] for name in sorted(default_names)]

    requested: list[str] = []
    for name in template_names:
        normalized = Path(name).with_suffix("").as_posix()
        # Preserve the old unqualified selector as an alias for the original,
        # now-mild template. Explicit extreme selectors use ``extreme/<name>``.
        if "/" not in normalized and f"mild/{normalized}" in available:
            normalized = f"mild/{normalized}"
        requested.append(normalized)
    missing = sorted(set(requested) - set(available))
    if missing:
        choices = ", ".join(sorted(available))
        raise ValueError(f"Unknown evaluation templates {missing}; available: {choices}")
    return [available[name] for name in requested]


def build_cases(
    cost_counts: Iterable[int] = DEFAULT_COST_COUNTS,
    template_names: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    """Render variable-cost sweeps and fixed zero-cost control templates."""

    counts = tuple(cost_counts)
    if not counts or any(not isinstance(count, int) or count < 0 for count in counts):
        raise ValueError("cost_counts must be a non-empty sequence of non-negative integers")
    if len(set(counts)) != len(counts):
        raise ValueError("cost_counts contains duplicates")

    cases: list[dict[str, object]] = []
    for path in _template_paths(template_names):
        template = path.read_text(encoding="utf-8")
        question_text = _validate_template_schema(template, path)
        template_counts = counts if "cost" in _template_fields(template) else (0,)
        template_hash = hashlib.sha256(template.encode("utf-8")).hexdigest()
        template_key = _template_key(path)
        severity, template_family = template_key.split("/", maxsplit=1)
        template_name = template_key.replace("/", "__")
        for cost_count in template_counts:
            prompt, implementation_label, _ = render_prompt(
                template,
                ELICITATION,
                cost_count,
                "implement_question",
            )
            cases.append(
                {
                    "case_id": f"{template_name}__cost_{cost_count}",
                    "template": template_name,
                    "template_family": template_family,
                    "severity": severity,
                    "template_path": str(path.relative_to(REPO_ROOT)),
                    "template_sha256": template_hash,
                    "cost_count": cost_count,
                    "question_polarity": "implement_question",
                    "question_text": question_text,
                    "implementation_label": implementation_label,
                    "prompt": prompt,
                }
            )
    return cases
