"""Supervision-matched readouts for the eight extreme ecological dilemmas."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from scripts.harmony_eval.cases import DEFAULT_COST_COUNTS, EVAL_DIR, REPO_ROOT
from scripts.harmony_sft.extreme_v2_eval import EXTREME_V2_TEMPLATES
from scripts.harmony_sft.posthoc_eval import (
    DEFAULT_LOCAL_EVAL_ROOT,
    PosthocEvalArtifacts,
    _case_set_sha256,
    _template_manifest,
    find_compatible_posthoc_eval,
    persist_posthoc_eval_to_colab_drive,
    run_saved_adapter_eval,
    validate_posthoc_eval,
)

from .evaluation import _run_identity
from .runner import PromptSFTArtifacts, validate_complete_run


READOUT_ROOT = EVAL_DIR / "extreme_v2/readouts"
OPTION_TEXTS_PATH = READOUT_ROOT / "option_texts.json"
READOUT_PROTOCOL_PATHS = {
    "reversed_yes_no": READOUT_ROOT / "reversed_yes_no/protocol.json",
    "counterbalanced_ab": READOUT_ROOT / "counterbalanced_ab/protocol.json",
    "complete_option_text": READOUT_ROOT / "complete_option_text/protocol.json",
}
READOUT_EVALUATION_SLUG = "extreme_v2_supervision_matched_readouts_eval"
READOUT_TYPES = tuple(READOUT_PROTOCOL_PATHS)
READOUT_VARIANTS = {
    "reversed_yes_no": ("human_question",),
    "counterbalanced_ab": ("ecological_a", "ecological_b"),
    "complete_option_text": ("ecological_first", "human_first"),
}


@dataclass(frozen=True)
class ReadoutValidation:
    case_count_per_model: int
    score_row_count: int
    template_count: int
    cost_counts: tuple[int, ...]
    cases_by_readout: dict[str, int]


@dataclass(frozen=True)
class ReadoutWorkflowResult:
    sft_artifacts: PromptSFTArtifacts
    evaluation_artifacts: PosthocEvalArtifacts
    evaluation_reused: bool
    validation: ReadoutValidation


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read valid readout specification {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Readout specification must be an object: {path}")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def _validate_counts(cost_counts: Iterable[int]) -> tuple[int, ...]:
    counts = tuple(cost_counts)
    if not counts or any(not isinstance(count, int) or count < 0 for count in counts):
        raise ValueError("cost_counts must contain non-negative integers")
    if len(set(counts)) != len(counts):
        raise ValueError("cost_counts contains duplicates")
    return counts


def _validate_specs() -> tuple[dict[str, dict[str, str]], dict[str, dict[str, object]]]:
    raw_options = _load_json_object(OPTION_TEXTS_PATH)
    expected_families = {Path(name).name for name in EXTREME_V2_TEMPLATES}
    if set(raw_options) != expected_families:
        raise RuntimeError(
            "Readout option registry does not exactly cover the eight extreme-v2 "
            "families"
        )
    options: dict[str, dict[str, str]] = {}
    required = {
        "ecological_option",
        "human_option",
        "reversed_human_question",
    }
    for family, raw_spec in raw_options.items():
        if not isinstance(raw_spec, dict) or set(raw_spec) != required:
            raise RuntimeError(f"Invalid option registry entry for {family}")
        spec = {key: str(raw_spec[key]).strip() for key in required}
        if any(not value for value in spec.values()):
            raise RuntimeError(f"Empty option registry field for {family}")
        if spec["ecological_option"] == spec["human_option"]:
            raise RuntimeError(f"Identical ecological and human options for {family}")
        if not spec["reversed_human_question"].endswith("?"):
            raise RuntimeError(f"Reversed question must end in '?' for {family}")
        options[family] = spec

    protocols = {
        readout: _load_json_object(path)
        for readout, path in READOUT_PROTOCOL_PATHS.items()
    }
    if protocols["reversed_yes_no"].get("ecological_candidate") != "No":
        raise RuntimeError("Reversed Yes/No must map the ecological option to No")
    if protocols["reversed_yes_no"].get("human_candidate") != "Yes":
        raise RuntimeError("Reversed Yes/No must map the human option to Yes")
    if tuple(protocols["counterbalanced_ab"].get("orders", ())) != (
        "ecological_a",
        "ecological_b",
    ):
        raise RuntimeError("A/B protocol must contain both counterbalanced orders")
    if tuple(protocols["complete_option_text"].get("orders", ())) != (
        "ecological_first",
        "human_first",
    ):
        raise RuntimeError("Complete-option protocol must contain both display orders")
    return options, protocols


def _provenance_hash(
    source: bytes,
    option_registry: bytes,
    protocol: bytes,
    variant: str,
) -> str:
    digest = hashlib.sha256()
    for value in (source, option_registry, protocol, variant.encode("utf-8")):
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()


def _base_case(
    *,
    family: str,
    cost_count: int,
    source_path: Path,
    source_bytes: bytes,
    protocol_path: Path,
    protocol_bytes: bytes,
    option_registry_bytes: bytes,
    readout_type: str,
    readout_variant: str,
    question_text: str,
    prompt: str,
    ecological_candidate: str,
    human_candidate: str,
    normalization: str,
) -> dict[str, object]:
    template = (
        f"extreme_v2_readout__{readout_type}__{readout_variant}__{family}"
    )
    return {
        "case_id": f"{template}__cost_{cost_count}",
        "template": template,
        "template_family": family,
        "severity": "extreme_v2",
        "template_path": _relative(protocol_path),
        "template_sha256": _provenance_hash(
            source_bytes,
            option_registry_bytes,
            protocol_bytes,
            readout_variant,
        ),
        "source_template_path": _relative(source_path),
        "source_template_sha256": _sha256_bytes(source_bytes),
        "option_registry_path": _relative(OPTION_TEXTS_PATH),
        "option_registry_sha256": _sha256_bytes(option_registry_bytes),
        "readout_protocol_path": _relative(protocol_path),
        "readout_protocol_sha256": _sha256_bytes(protocol_bytes),
        "cost_count": cost_count,
        "question_polarity": readout_type,
        "question_text": question_text,
        "readout_type": readout_type,
        "readout_variant": readout_variant,
        "implementation_label": ecological_candidate,
        "candidate_implement": ecological_candidate,
        "candidate_reject": human_candidate,
        "candidate_score_normalization": normalization,
        "semantic_direction": "ecological_option_minus_human_option",
        "prompt": prompt,
    }


def build_supervision_matched_readout_cases(
    cost_counts: Iterable[int] = DEFAULT_COST_COUNTS,
) -> list[dict[str, object]]:
    """Render reversed Yes/No, counterbalanced A/B, and full-option readouts."""

    counts = _validate_counts(cost_counts)
    options, protocols = _validate_specs()
    option_registry_bytes = OPTION_TEXTS_PATH.read_bytes()
    cases: list[dict[str, object]] = []

    for template_name in EXTREME_V2_TEMPLATES:
        family = Path(template_name).name
        source_path = EVAL_DIR / f"{template_name}.txt"
        source_bytes = source_path.read_bytes()
        source_text = source_bytes.decode("utf-8")
        if "\n\n" not in source_text:
            raise RuntimeError(f"Extreme-v2 template has no final question: {source_path}")
        scenario_template, _ = source_text.strip().rsplit("\n\n", maxsplit=1)
        spec = options[family]

        for cost_count in counts:
            try:
                scenario = scenario_template.format(cost=cost_count)
            except KeyError as exc:
                raise RuntimeError(
                    f"Unexpected scenario placeholder in {source_path}: {exc.args[0]}"
                ) from exc

            reversed_protocol_path = READOUT_PROTOCOL_PATHS["reversed_yes_no"]
            reversed_protocol_bytes = reversed_protocol_path.read_bytes()
            reversed_instruction = str(
                protocols["reversed_yes_no"]["instruction"]
            )
            reversed_question = spec["reversed_human_question"]
            cases.append(
                _base_case(
                    family=family,
                    cost_count=cost_count,
                    source_path=source_path,
                    source_bytes=source_bytes,
                    protocol_path=reversed_protocol_path,
                    protocol_bytes=reversed_protocol_bytes,
                    option_registry_bytes=option_registry_bytes,
                    readout_type="reversed_yes_no",
                    readout_variant="human_question",
                    question_text=reversed_question,
                    prompt=f"{scenario}\n\n{reversed_question} {reversed_instruction}",
                    ecological_candidate="No",
                    human_candidate="Yes",
                    normalization="sum",
                )
            )

            ab_protocol_path = READOUT_PROTOCOL_PATHS["counterbalanced_ab"]
            ab_protocol_bytes = ab_protocol_path.read_bytes()
            ab_instruction = str(protocols["counterbalanced_ab"]["instruction"])
            for variant in READOUT_VARIANTS["counterbalanced_ab"]:
                ecological_a = variant == "ecological_a"
                option_a = (
                    spec["ecological_option"] if ecological_a else spec["human_option"]
                )
                option_b = (
                    spec["human_option"] if ecological_a else spec["ecological_option"]
                )
                prompt = f"{scenario}\n\nA. {option_a}\nB. {option_b}\n\n{ab_instruction}"
                cases.append(
                    _base_case(
                        family=family,
                        cost_count=cost_count,
                        source_path=source_path,
                        source_bytes=source_bytes,
                        protocol_path=ab_protocol_path,
                        protocol_bytes=ab_protocol_bytes,
                        option_registry_bytes=option_registry_bytes,
                        readout_type="counterbalanced_ab",
                        readout_variant=variant,
                        question_text="Which policy should the authorities choose?",
                        prompt=prompt,
                        ecological_candidate="A" if ecological_a else "B",
                        human_candidate="B" if ecological_a else "A",
                        normalization="sum",
                    )
                )

            full_protocol_path = READOUT_PROTOCOL_PATHS["complete_option_text"]
            full_protocol_bytes = full_protocol_path.read_bytes()
            full_instruction = str(
                protocols["complete_option_text"]["instruction"]
            )
            for variant in READOUT_VARIANTS["complete_option_text"]:
                ecological_first = variant == "ecological_first"
                first = (
                    spec["ecological_option"]
                    if ecological_first
                    else spec["human_option"]
                )
                second = (
                    spec["human_option"]
                    if ecological_first
                    else spec["ecological_option"]
                )
                prompt = f"{scenario}\n\n{full_instruction}\n\n- {first}\n- {second}"
                cases.append(
                    _base_case(
                        family=family,
                        cost_count=cost_count,
                        source_path=source_path,
                        source_bytes=source_bytes,
                        protocol_path=full_protocol_path,
                        protocol_bytes=full_protocol_bytes,
                        option_registry_bytes=option_registry_bytes,
                        readout_type="complete_option_text",
                        readout_variant=variant,
                        question_text="Which policy should the authorities choose?",
                        prompt=prompt,
                        ecological_candidate=spec["ecological_option"],
                        human_candidate=spec["human_option"],
                        normalization="mean",
                    )
                )

    expected_count = (
        len(EXTREME_V2_TEMPLATES)
        * len(counts)
        * sum(len(variants) for variants in READOUT_VARIANTS.values())
    )
    if len(cases) != expected_count or len({case["case_id"] for case in cases}) != len(
        cases
    ):
        raise RuntimeError("Readout builder did not produce the complete unique matrix")
    return cases


def validate_supervision_matched_readout_artifacts(
    artifacts: PosthocEvalArtifacts,
    *,
    cost_counts: Iterable[int] = DEFAULT_COST_COUNTS,
) -> ReadoutValidation:
    """Verify hashes, exact prompts, candidate mappings, and the score matrix."""

    counts = _validate_counts(cost_counts)
    expected_cases = build_supervision_matched_readout_cases(counts)
    expected_by_id = {str(case["case_id"]): case for case in expected_cases}
    validate_posthoc_eval(artifacts.output_dir)
    try:
        metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
        rendered_cases = [
            json.loads(line)
            for line in artifacts.rendered_cases_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        with artifacts.raw_scores_path.open(
            "r", encoding="utf-8", newline=""
        ) as input_file:
            score_rows = list(csv.DictReader(input_file))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Could not read a valid supervision-matched bundle") from exc

    expected_metadata = {
        "status": "complete",
        "evaluation_slug": READOUT_EVALUATION_SLUG,
        "cost_counts": list(counts),
        "case_count_per_model": len(expected_cases),
        "case_set_sha256": _case_set_sha256(expected_cases),
        "enable_thinking": False,
        "templates": _template_manifest(expected_cases),
    }
    mismatches = [
        key for key, expected in expected_metadata.items() if metadata.get(key) != expected
    ]
    if mismatches:
        raise RuntimeError(
            "Supervision-matched metadata does not match the requested suite: "
            f"{mismatches}"
        )
    if rendered_cases != expected_cases:
        raise RuntimeError(
            "Persisted readout cases do not exactly match the current specifications"
        )

    observed: list[tuple[str, str]] = []
    for row in score_rows:
        case_id = str(row.get("case_id", ""))
        role = str(row.get("model_role", ""))
        expected = expected_by_id.get(case_id)
        if expected is None or role not in {"base", "aligned"}:
            raise RuntimeError(f"Unexpected readout score row: {(case_id, role)}")
        for key in (
            "template",
            "readout_type",
            "readout_variant",
            "candidate_implement",
            "candidate_reject",
            "candidate_score_normalization",
        ):
            if row.get(key) != str(expected[key]):
                raise RuntimeError(f"Readout score {case_id!r} has wrong {key}")
        try:
            probability = float(row["p_implement"])
            float(row["semantic_logit_implement"])
            int(row["candidate_tokens_implement"])
            int(row["candidate_tokens_reject"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Readout score {case_id!r} has invalid scores") from exc
        if not 0.0 <= probability <= 1.0:
            raise RuntimeError(f"Readout score {case_id!r} has invalid probability")
        observed.append((case_id, role))

    expected_pairs = {
        (case_id, role)
        for case_id in expected_by_id
        for role in ("base", "aligned")
    }
    if len(observed) != len(set(observed)) or set(observed) != expected_pairs:
        raise RuntimeError(
            "Readout scores must contain exactly one base and one aligned row per case"
        )
    cases_by_readout = {
        readout: sum(case["readout_type"] == readout for case in expected_cases)
        for readout in READOUT_TYPES
    }
    return ReadoutValidation(
        case_count_per_model=len(expected_cases),
        score_row_count=len(score_rows),
        template_count=len({case["template"] for case in expected_cases}),
        cost_counts=counts,
        cases_by_readout=cases_by_readout,
    )


def run_supervision_matched_readout_workflow(
    sft_artifacts: PromptSFTArtifacts,
    *,
    cost_counts: Iterable[int],
    batch_size: int,
    force_evaluation: bool = False,
    local_eval_root: Path | str = DEFAULT_LOCAL_EVAL_ROOT,
    persistence_kwargs: dict[str, object] | None = None,
) -> ReadoutWorkflowResult:
    """Run or reuse the complete three-readout battery for one saved adapter."""

    validate_complete_run(sft_artifacts)
    pair_name, training_objective = _run_identity(sft_artifacts)
    counts = _validate_counts(cost_counts)
    cases = build_supervision_matched_readout_cases(counts)
    evaluation = None
    if not force_evaluation:
        evaluation = find_compatible_posthoc_eval(
            sft_artifacts.run_dir,
            cost_counts=counts,
            cases=cases,
            evaluation_slug=READOUT_EVALUATION_SLUG,
        )
        if evaluation is not None:
            try:
                validation = validate_supervision_matched_readout_artifacts(
                    evaluation,
                    cost_counts=counts,
                )
            except RuntimeError as exc:
                print(
                    "A prior supervision-matched bundle failed validation and "
                    f"will be recomputed: {exc}"
                )
                evaluation = None
            else:
                return ReadoutWorkflowResult(
                    sft_artifacts=sft_artifacts,
                    evaluation_artifacts=evaluation,
                    evaluation_reused=True,
                    validation=validation,
                )

    local = run_saved_adapter_eval(
        sft_artifacts.run_dir,
        output_root=local_eval_root,
        cost_counts=counts,
        cases=cases,
        evaluation_slug=READOUT_EVALUATION_SLUG,
        batch_size=batch_size,
        pair_name=pair_name,
        training_method=training_objective,
    )
    validate_supervision_matched_readout_artifacts(local, cost_counts=counts)
    evaluation = persist_posthoc_eval_to_colab_drive(
        local,
        sft_artifacts.run_dir,
        **(persistence_kwargs or {}),
    )
    validation = validate_supervision_matched_readout_artifacts(
        evaluation,
        cost_counts=counts,
    )
    return ReadoutWorkflowResult(
        sft_artifacts=sft_artifacts,
        evaluation_artifacts=evaluation,
        evaluation_reused=False,
        validation=validation,
    )


__all__ = [
    "OPTION_TEXTS_PATH",
    "READOUT_EVALUATION_SLUG",
    "READOUT_PROTOCOL_PATHS",
    "READOUT_TYPES",
    "READOUT_VARIANTS",
    "ReadoutValidation",
    "ReadoutWorkflowResult",
    "build_supervision_matched_readout_cases",
    "run_supervision_matched_readout_workflow",
    "validate_supervision_matched_readout_artifacts",
]
