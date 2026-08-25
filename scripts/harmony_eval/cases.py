"""Build direct-question threshold cases from repository prompt templates."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Sequence

from scripts.run_stage_1 import render_prompt


REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "eval/ecological_value"
DEFAULT_COST_COUNTS = (0, 1, 10, 100, 1_000, 10_000, 100_000, 1_000_000)
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


def _template_paths(template_names: Sequence[str] | None) -> list[Path]:
    available = {path.stem: path for path in EVAL_DIR.glob("*.txt")}
    if template_names is None:
        return [available[name] for name in sorted(available)]

    requested = [Path(name).stem for name in template_names]
    missing = sorted(set(requested) - set(available))
    if missing:
        choices = ", ".join(sorted(available))
        raise ValueError(f"Unknown evaluation templates {missing}; available: {choices}")
    return [available[name] for name in requested]


def build_cases(
    cost_counts: Iterable[int] = DEFAULT_COST_COUNTS,
    template_names: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    """Render one direct implementation question per template and cost level."""

    counts = tuple(cost_counts)
    if not counts or any(not isinstance(count, int) or count < 0 for count in counts):
        raise ValueError("cost_counts must be a non-empty sequence of non-negative integers")
    if len(set(counts)) != len(counts):
        raise ValueError("cost_counts contains duplicates")

    cases: list[dict[str, object]] = []
    for path in _template_paths(template_names):
        template = path.read_text(encoding="utf-8")
        template_hash = hashlib.sha256(template.encode("utf-8")).hexdigest()
        for cost_count in counts:
            prompt, implementation_label, _ = render_prompt(
                template,
                ELICITATION,
                cost_count,
                "implement_question",
            )
            cases.append(
                {
                    "case_id": f"{path.stem}__cost_{cost_count}",
                    "template": path.stem,
                    "template_path": str(path.relative_to(REPO_ROOT)),
                    "template_sha256": template_hash,
                    "cost_count": cost_count,
                    "question_polarity": "implement_question",
                    "implementation_label": implementation_label,
                    "prompt": prompt,
                }
            )
    return cases
