"""Run the fixed eight-prompt extreme-v2 evaluation from a saved H4rmony adapter."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from scripts.harmony_eval.cases import DEFAULT_COST_COUNTS, build_cases

from .posthoc_eval import (
    DEFAULT_LOCAL_EVAL_ROOT,
    PosthocEvalArtifacts,
    _template_manifest,
    find_compatible_posthoc_eval,
    persist_posthoc_eval_to_colab_drive,
    run_saved_adapter_eval,
    validate_posthoc_eval,
)
from .runner import SFTArtifacts, SFTConfig, find_compatible_complete_run


EXTREME_V2_TEMPLATES = (
    "extreme_v2/wetland_relocation",
    "extreme_v2/marine_reserve",
    "extreme_v2/pesticide_ban",
    "extreme_v2/oil_extraction_ban",
    "extreme_v2/dam_removal",
    "extreme_v2/wildfire_restoration",
    "extreme_v2/river_water_allocation",
    "extreme_v2/island_biosecurity",
)

EXTREME_V2_CONTROL_TEMPLATES = (
    "extreme_v2/control/matched_non_ecological/archaeological_preservation",
    "extreme_v2/control/matched_non_ecological/scientific_observatory",
    "extreme_v2/control/unrelated_severe_moral/organ_harvesting",
    "extreme_v2/control/unrelated_severe_moral/punishing_innocent_person",
    "extreme_v2/control/zero_cost_ecological/wetland_preservation",
    "extreme_v2/control/zero_cost_ecological/marine_reserve",
)


@dataclass(frozen=True)
class ExtremeV2Validation:
    case_count_per_model: int
    score_row_count: int
    template_count: int
    cost_counts: tuple[int, ...]


@dataclass(frozen=True)
class ExtremeV2WorkflowResult:
    sft_artifacts: SFTArtifacts
    evaluation_artifacts: PosthocEvalArtifacts
    evaluation_reused: bool
    validation: ExtremeV2Validation


def build_extreme_v2_cases(
    cost_counts: Iterable[int] = DEFAULT_COST_COUNTS,
) -> list[dict[str, object]]:
    """Render the exact eight-template suite used by the Colab workflow."""

    return build_cases(tuple(cost_counts), EXTREME_V2_TEMPLATES)


def build_extreme_v2_control_cases(
    cost_counts: Iterable[int] = DEFAULT_COST_COUNTS,
) -> list[dict[str, object]]:
    """Render four cost sweeps and two fixed zero-cost controls."""

    return build_cases(tuple(cost_counts), EXTREME_V2_CONTROL_TEMPLATES)


def find_verified_harmony_sft_run(
    drive_output_root: Path | str,
    config: SFTConfig,
) -> SFTArtifacts:
    """Find the newest hash-verified run matching the intended SFT intervention."""

    artifacts = find_compatible_complete_run(drive_output_root, config)
    if artifacts is None:
        raise RuntimeError(
            "No hash-verified H4rmony R1 SFT run matching the configured Qwen3-8B "
            f"intervention was found under {Path(drive_output_root).expanduser()}. "
            "This evaluation workflow will not retrain automatically. Check that "
            "Google Drive is mounted and that the completed run is in the expected "
            "folder."
        )
    return artifacts


def validate_extreme_v2_artifacts(
    artifacts: PosthocEvalArtifacts,
    *,
    cost_counts: Iterable[int] = DEFAULT_COST_COUNTS,
) -> ExtremeV2Validation:
    """Verify hashes and the complete case-by-model score matrix."""

    counts = tuple(cost_counts)
    expected_cases = build_extreme_v2_cases(counts)
    expected_by_id = {str(case["case_id"]): case for case in expected_cases}
    if len(expected_by_id) != len(expected_cases):
        raise RuntimeError("The rendered extreme-v2 suite contains duplicate case IDs")

    validate_posthoc_eval(artifacts.output_dir)
    try:
        metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Could not read valid extreme-v2 evaluation metadata") from exc
    if not isinstance(metadata, dict):
        raise RuntimeError("Extreme-v2 evaluation metadata must be a JSON object")

    expected_manifest = _template_manifest(expected_cases)
    expected_metadata = {
        "status": "complete",
        "cost_counts": list(counts),
        "case_count_per_model": len(expected_cases),
        "enable_thinking": False,
        "templates": expected_manifest,
    }
    mismatched_metadata = [
        key for key, expected in expected_metadata.items() if metadata.get(key) != expected
    ]
    if mismatched_metadata:
        raise RuntimeError(
            "Extreme-v2 metadata does not match the requested suite: "
            f"{mismatched_metadata}"
        )

    if not artifacts.rendered_cases_path.is_file():
        raise RuntimeError("Extreme-v2 evaluation has no rendered_cases.jsonl")
    try:
        rendered_cases = [
            json.loads(line)
            for line in artifacts.rendered_cases_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Could not read valid rendered extreme-v2 cases") from exc
    if rendered_cases != expected_cases:
        raise RuntimeError(
            "Persisted rendered cases do not exactly match the current eight-prompt "
            "extreme-v2 suite"
        )

    try:
        with artifacts.raw_scores_path.open(
            "r", encoding="utf-8", newline=""
        ) as input_file:
            score_rows = list(csv.DictReader(input_file))
    except OSError as exc:
        raise RuntimeError("Could not read extreme-v2 raw scores") from exc

    observed_pairs: list[tuple[str, str]] = []
    for row in score_rows:
        case_id = str(row.get("case_id", ""))
        model_role = str(row.get("model_role", ""))
        expected_case = expected_by_id.get(case_id)
        if expected_case is None:
            raise RuntimeError(f"Raw scores contain an unexpected case ID: {case_id!r}")
        if model_role not in {"base", "aligned"}:
            raise RuntimeError(
                f"Raw scores contain an unexpected model role: {model_role!r}"
            )
        try:
            cost_count = int(row["cost_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Raw score {case_id!r} has an invalid cost_count") from exc
        if row.get("template") != expected_case["template"] or cost_count != expected_case[
            "cost_count"
        ]:
            raise RuntimeError(
                f"Raw score {case_id!r} does not match its rendered case metadata"
            )
        observed_pairs.append((case_id, model_role))

    expected_pairs = {
        (case_id, role)
        for case_id in expected_by_id
        for role in ("base", "aligned")
    }
    if len(observed_pairs) != len(set(observed_pairs)):
        raise RuntimeError("Raw scores contain duplicate case/model rows")
    if set(observed_pairs) != expected_pairs:
        missing = sorted(expected_pairs - set(observed_pairs))
        extra = sorted(set(observed_pairs) - expected_pairs)
        raise RuntimeError(
            "Raw scores do not contain exactly one base and one aligned row for "
            f"every case; missing={missing[:5]}, extra={extra[:5]}"
        )

    return ExtremeV2Validation(
        case_count_per_model=len(expected_cases),
        score_row_count=len(score_rows),
        template_count=len(EXTREME_V2_TEMPLATES),
        cost_counts=counts,
    )


def run_extreme_v2_workflow(
    drive_output_root: Path | str,
    config: SFTConfig,
    *,
    force_evaluation: bool = False,
    local_eval_root: Path | str = DEFAULT_LOCAL_EVAL_ROOT,
    persistence_kwargs: dict[str, object] | None = None,
) -> ExtremeV2WorkflowResult:
    """Find the SFT run, evaluate base and adapter, and durably persist to Drive."""

    sft_artifacts = find_verified_harmony_sft_run(drive_output_root, config)
    print(f"Using hash-verified H4rmony SFT run: {sft_artifacts.run_dir}")
    print(f"Using saved LoRA adapter: {sft_artifacts.final_adapter_dir}")
    evaluation_artifacts = None
    if not force_evaluation:
        evaluation_artifacts = find_compatible_posthoc_eval(
            sft_artifacts.run_dir,
            cost_counts=config.cost_counts,
            template_names=EXTREME_V2_TEMPLATES,
        )
        if evaluation_artifacts is not None:
            try:
                validation = validate_extreme_v2_artifacts(
                    evaluation_artifacts,
                    cost_counts=config.cost_counts,
                )
            except RuntimeError as exc:
                print(
                    "A hash-valid prior evaluation failed the complete extreme-v2 "
                    f"matrix check and will be recomputed: {exc}"
                )
                evaluation_artifacts = None
            else:
                return ExtremeV2WorkflowResult(
                    sft_artifacts=sft_artifacts,
                    evaluation_artifacts=evaluation_artifacts,
                    evaluation_reused=True,
                    validation=validation,
                )

    local_artifacts = run_saved_adapter_eval(
        sft_artifacts.run_dir,
        output_root=local_eval_root,
        cost_counts=config.cost_counts,
        template_names=EXTREME_V2_TEMPLATES,
        batch_size=config.eval_batch_size,
    )
    validate_extreme_v2_artifacts(
        local_artifacts,
        cost_counts=config.cost_counts,
    )
    evaluation_artifacts = persist_posthoc_eval_to_colab_drive(
        local_artifacts,
        sft_artifacts.run_dir,
        **(persistence_kwargs or {}),
    )
    validation = validate_extreme_v2_artifacts(
        evaluation_artifacts,
        cost_counts=config.cost_counts,
    )
    return ExtremeV2WorkflowResult(
        sft_artifacts=sft_artifacts,
        evaluation_artifacts=evaluation_artifacts,
        evaluation_reused=False,
        validation=validation,
    )
