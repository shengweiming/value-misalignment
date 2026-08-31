"""Evaluate a prompt-only adapter on extreme-v2 and publish verified bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from scripts.harmony_sft.extreme_v2_eval import (
    EXTREME_V2_CONTROL_TEMPLATES,
    EXTREME_V2_TEMPLATES,
    ExtremeV2Validation,
    build_extreme_v2_control_cases,
    build_extreme_v2_cases,
    validate_extreme_v2_artifacts,
    validate_extreme_v2_control_artifacts,
)
from scripts.harmony_sft.github_publish import (
    GitHubPublication,
    publish_extreme_v2_results_to_github,
)
from scripts.harmony_sft.posthoc_eval import (
    DEFAULT_LOCAL_EVAL_ROOT,
    PosthocEvalArtifacts,
    find_compatible_posthoc_eval,
    persist_posthoc_eval_to_colab_drive,
    run_saved_adapter_eval,
)

from .runner import PAIR_NAME, PromptSFTArtifacts, validate_complete_run


DEFAULT_RESULTS_ROOT = Path(
    "results/harmony_eval/qwen3_8b_ecological_dilemma_prompt_sft"
)


@dataclass(frozen=True)
class PromptEvalWorkflowResult:
    sft_artifacts: PromptSFTArtifacts
    evaluation_artifacts: PosthocEvalArtifacts
    evaluation_reused: bool
    validation: ExtremeV2Validation


def _run_suite(
    sft_artifacts: PromptSFTArtifacts,
    *,
    template_names: tuple[str, ...],
    validate_artifacts: Callable[..., ExtremeV2Validation],
    suite_label: str,
    cost_counts: Iterable[int],
    batch_size: int,
    force_evaluation: bool = False,
    local_eval_root: Path | str = DEFAULT_LOCAL_EVAL_ROOT,
    persistence_kwargs: dict[str, object] | None = None,
) -> PromptEvalWorkflowResult:
    validate_complete_run(sft_artifacts)
    counts = tuple(cost_counts)
    evaluation = None
    if not force_evaluation:
        evaluation = find_compatible_posthoc_eval(
            sft_artifacts.run_dir,
            cost_counts=counts,
            template_names=template_names,
        )
        if evaluation is not None:
            try:
                validation = validate_artifacts(evaluation, cost_counts=counts)
            except RuntimeError as exc:
                print(
                    f"A prior {suite_label} bundle failed its matrix check and "
                    f"will be recomputed: {exc}"
                )
                evaluation = None
            else:
                return PromptEvalWorkflowResult(
                    sft_artifacts=sft_artifacts,
                    evaluation_artifacts=evaluation,
                    evaluation_reused=True,
                    validation=validation,
                )

    local = run_saved_adapter_eval(
        sft_artifacts.run_dir,
        output_root=local_eval_root,
        cost_counts=counts,
        template_names=template_names,
        batch_size=batch_size,
        pair_name=PAIR_NAME,
        training_method="prompt_only_causal_lm",
    )
    validate_artifacts(local, cost_counts=counts)
    evaluation = persist_posthoc_eval_to_colab_drive(
        local,
        sft_artifacts.run_dir,
        **(persistence_kwargs or {}),
    )
    validation = validate_artifacts(evaluation, cost_counts=counts)
    return PromptEvalWorkflowResult(
        sft_artifacts=sft_artifacts,
        evaluation_artifacts=evaluation,
        evaluation_reused=False,
        validation=validation,
    )


def run_extreme_v2_workflow(
    sft_artifacts: PromptSFTArtifacts,
    *,
    cost_counts: Iterable[int],
    batch_size: int,
    force_evaluation: bool = False,
    local_eval_root: Path | str = DEFAULT_LOCAL_EVAL_ROOT,
    persistence_kwargs: dict[str, object] | None = None,
) -> PromptEvalWorkflowResult:
    return _run_suite(
        sft_artifacts,
        template_names=EXTREME_V2_TEMPLATES,
        validate_artifacts=validate_extreme_v2_artifacts,
        suite_label="primary extreme-v2 evaluation",
        cost_counts=cost_counts,
        batch_size=batch_size,
        force_evaluation=force_evaluation,
        local_eval_root=local_eval_root,
        persistence_kwargs=persistence_kwargs,
    )


def run_extreme_v2_control_workflow(
    sft_artifacts: PromptSFTArtifacts,
    *,
    cost_counts: Iterable[int],
    batch_size: int,
    force_evaluation: bool = False,
    local_eval_root: Path | str = DEFAULT_LOCAL_EVAL_ROOT,
    persistence_kwargs: dict[str, object] | None = None,
) -> PromptEvalWorkflowResult:
    return _run_suite(
        sft_artifacts,
        template_names=EXTREME_V2_CONTROL_TEMPLATES,
        validate_artifacts=validate_extreme_v2_control_artifacts,
        suite_label="extreme-v2 control evaluation",
        cost_counts=cost_counts,
        batch_size=batch_size,
        force_evaluation=force_evaluation,
        local_eval_root=local_eval_root,
        persistence_kwargs=persistence_kwargs,
    )


def publish_results_to_github(
    artifacts: PosthocEvalArtifacts,
    *,
    source_run_name: str,
    github_repository: str,
    branch: str,
    github_token: str,
    repo_root: Path | str,
) -> GitHubPublication:
    return publish_extreme_v2_results_to_github(
        artifacts,
        source_run_name=source_run_name,
        github_repository=github_repository,
        branch=branch,
        github_token=github_token,
        repo_root=repo_root,
        results_root=DEFAULT_RESULTS_ROOT,
    )


__all__ = [
    "EXTREME_V2_CONTROL_TEMPLATES",
    "EXTREME_V2_TEMPLATES",
    "PromptEvalWorkflowResult",
    "build_extreme_v2_control_cases",
    "build_extreme_v2_cases",
    "publish_results_to_github",
    "run_extreme_v2_control_workflow",
    "run_extreme_v2_workflow",
]
