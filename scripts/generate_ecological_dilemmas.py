#!/usr/bin/env python3
"""Generate quality-controlled ecological-versus-human ethical dilemmas."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "src/ecological_dilemmas"
DEFAULT_CONSTRUCTS_PATH = SOURCE_DIR / "ecological_dilemma_constructs.json"
DEFAULT_DECISION_MAKERS_PATH = SOURCE_DIR / "decision_makers.json"
DEFAULT_CARD_PROMPT_PATH = SOURCE_DIR / "scenario_card_prompt.txt"
DEFAULT_REVIEW_PROMPT_PATH = SOURCE_DIR / "card_review_prompt.txt"
DEFAULT_WRITER_PROMPT_PATH = SOURCE_DIR / "dilemma_writer_prompt.txt"
DEFAULT_VALIDATOR_PROMPT_PATH = SOURCE_DIR / "dilemma_validation_prompt.txt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/ecological_dilemmas"

DEFAULT_REASONING_MODEL = "gpt-5.6-sol"
DEFAULT_WRITER_MODEL = "gpt-5.6-terra"
REQUIRED_CONSTRUCT_SECTIONS = (
    "ecological_objects",
    "human_interests",
    "policy_mechanisms",
)
SCORE_FIELDS = (
    "construct_fidelity",
    "causal_plausibility",
    "moderate_harm",
    "incompatibility",
    "moral_balance",
    "decision_authority",
    "nonculpability_neutrality",
    "novelty",
)
VALIDATION_SCORE_FIELDS = SCORE_FIELDS + ("format", "card_fidelity")
CARD_FIELDS = (
    "candidate_id",
    "title",
    "setting",
    "affected_people",
    "decision_authority",
    "ecologically_protective_option",
    "human_protective_option",
    "ecological_outcome_if_protected",
    "ecological_outcome_if_not",
    "human_outcome_if_protected",
    "human_outcome_if_not",
    "ecological_causal_chain",
    "human_causal_chain",
    "compromise_block",
    "moderate_harm_check",
    "balance_check",
    "authority_check",
    "culpability_check",
    "causal_plausibility_check",
    "novelty_signature",
)

# Standard-priority token prices retrieved from official OpenAI model pages on
# 2026-08-30. They are used only for a transparent run estimate in the manifest.
PRICING_SNAPSHOT = {
    "date": "2026-08-30",
    "currency": "USD",
    "per_million_tokens": {
        "gpt-5.6-sol": {
            "input": 4.0,
            "cached_input": 0.4,
            "output": 20.0,
            "cache_write_multiplier": 1.25,
        },
        "gpt-5.6": {
            "input": 4.0,
            "cached_input": 0.4,
            "output": 20.0,
            "cache_write_multiplier": 1.25,
        },
        "gpt-5.6-terra": {
            "input": 2.0,
            "cached_input": 0.2,
            "output": 12.0,
            "cache_write_multiplier": 1.25,
        },
    },
}


class GenerationError(RuntimeError):
    """Raised when generator configuration or model output is invalid."""


@dataclass(frozen=True)
class Assignment:
    ecological_object: str
    ecological_object_definition: str
    human_interest: str
    human_interest_definition: str
    policy_mechanism: str
    policy_mechanism_definition: str
    decision_maker: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    def values(self) -> tuple[str, str, str, str]:
        return (
            self.ecological_object,
            self.human_interest,
            self.policy_mechanism,
            self.decision_maker,
        )


@dataclass(frozen=True)
class StageConfig:
    model: str
    reasoning_effort: str | None
    max_output_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PipelineConfig:
    planner: StageConfig
    reviewer: StageConfig
    writer: StageConfig
    validator: StageConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "planner": self.planner.to_dict(),
            "reviewer": self.reviewer.to_dict(),
            "writer": self.writer.to_dict(),
            "validator": self.validator.to_dict(),
        }


def _card_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {field: {"type": "string"} for field in CARD_FIELDS},
        "required": list(CARD_FIELDS),
        "additionalProperties": False,
    }


def _scores_schema(fields: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            field: {"type": "integer", "enum": [1, 2, 3, 4, 5]}
            for field in fields
        },
        "required": list(fields),
        "additionalProperties": False,
    }


SCENARIO_CARDS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "viable": {"type": "boolean"},
        "rejection_reason": {"type": "string"},
        "cards": {"type": "array", "items": _card_schema()},
    },
    "required": ["viable", "rejection_reason", "cards"],
    "additionalProperties": False,
}

CARD_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["accept", "reject"]},
        "selected_candidate_id": {"type": "string"},
        "overall_reason": {"type": "string"},
        "hard_failures": {"type": "array", "items": {"type": "string"}},
        "revision_summary": {"type": "string"},
        "scores": _scores_schema(SCORE_FIELDS),
        "revised_card": _card_schema(),
    },
    "required": [
        "decision",
        "selected_candidate_id",
        "overall_reason",
        "hard_failures",
        "revision_summary",
        "scores",
        "revised_card",
    ],
    "additionalProperties": False,
}

FINAL_VALIDATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["accept", "revise", "reject"]},
        "rationale": {"type": "string"},
        "violations": {"type": "array", "items": {"type": "string"}},
        "scores": _scores_schema(VALIDATION_SCORE_FIELDS),
        "revised_dilemma": {"type": "string"},
    },
    "required": [
        "decision",
        "rationale",
        "violations",
        "scores",
        "revised_dilemma",
    ],
    "additionalProperties": False,
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_env_file(path: Path) -> None:
    """Load a simple KEY=VALUE file without overriding exported variables."""
    if not path.exists():
        return
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise GenerationError(f"Invalid .env entry at {path}:{line_number}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def load_constructs(path: Path) -> dict[str, dict[str, str]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GenerationError(f"Construct file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise GenerationError(f"Construct file is not valid JSON: {path}") from exc
    if not isinstance(raw, dict):
        raise GenerationError("Construct file must contain a JSON object")

    constructs: dict[str, dict[str, str]] = {}
    for section in REQUIRED_CONSTRUCT_SECTIONS:
        values = raw.get(section)
        if not isinstance(values, dict) or not values:
            raise GenerationError(
                f"Construct section {section!r} must be a non-empty object"
            )
        clean_values: dict[str, str] = {}
        for name, definition in values.items():
            if not isinstance(name, str) or not name.strip():
                raise GenerationError(f"{section} contains an invalid construct name")
            if not isinstance(definition, str) or not definition.strip():
                raise GenerationError(
                    f"Construct {name!r} in {section} has no usable definition"
                )
            clean_values[name.strip()] = definition.strip()
        constructs[section] = clean_values
    return constructs


def load_decision_makers(path: Path) -> list[str]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GenerationError(f"Decision-maker file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise GenerationError(f"Decision-maker file is not valid JSON: {path}") from exc
    if not isinstance(raw, list) or not raw:
        raise GenerationError("Decision-maker file must contain a non-empty JSON list")
    if any(not isinstance(value, str) or not value.strip() for value in raw):
        raise GenerationError("Every decision-maker must be a non-empty string")
    decision_makers = [value.strip() for value in raw]
    if len(set(decision_makers)) != len(decision_makers):
        raise GenerationError("Decision-maker list contains duplicates")
    return decision_makers


def load_prompt(path: Path) -> str:
    try:
        prompt = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise GenerationError(f"Prompt file not found: {path}") from exc
    if not prompt:
        raise GenerationError(f"Prompt file is empty: {path}")
    return prompt


class BalancedAssignmentSampler:
    """Choose unique combinations while balancing accepted marginal constructs."""

    def __init__(
        self,
        constructs: Mapping[str, Mapping[str, str]],
        decision_makers: Sequence[str],
        rng: random.Random,
    ) -> None:
        ecological_objects = list(constructs["ecological_objects"].items())
        human_interests = list(constructs["human_interests"].items())
        policy_mechanisms = list(constructs["policy_mechanisms"].items())
        self._remaining = [
            Assignment(
                ecological_object=ecological[0],
                ecological_object_definition=ecological[1],
                human_interest=human[0],
                human_interest_definition=human[1],
                policy_mechanism=policy[0],
                policy_mechanism_definition=policy[1],
                decision_maker=decision_maker,
            )
            for ecological, human, policy, decision_maker in itertools.product(
                ecological_objects,
                human_interests,
                policy_mechanisms,
                decision_makers,
            )
        ]
        rng.shuffle(self._remaining)
        self._marginal_counts = [Counter() for _ in range(4)]
        self._pair_counts = [Counter() for _ in range(3)]

    @property
    def remaining_count(self) -> int:
        return len(self._remaining)

    def next_assignment(self) -> Assignment:
        if not self._remaining:
            raise GenerationError("No unused construct combinations remain")

        def score(item: Assignment) -> tuple[int, int, int]:
            values = item.values()
            marginal = [
                self._marginal_counts[index][value]
                for index, value in enumerate(values)
            ]
            pair_total = (
                self._pair_counts[0][(values[0], values[1])]
                + self._pair_counts[1][(values[0], values[2])]
                + self._pair_counts[2][(values[1], values[2])]
            )
            return (sum(marginal), max(marginal), pair_total)

        best_index = min(
            range(len(self._remaining)), key=lambda i: score(self._remaining[i])
        )
        return self._remaining.pop(best_index)

    def record_acceptance(self, assignment: Assignment) -> None:
        values = assignment.values()
        for index, value in enumerate(values):
            self._marginal_counts[index][value] += 1
        self._pair_counts[0][(values[0], values[1])] += 1
        self._pair_counts[1][(values[0], values[2])] += 1
        self._pair_counts[2][(values[1], values[2])] += 1


def sample_assignments(
    constructs: Mapping[str, Mapping[str, str]],
    decision_makers: Sequence[str],
    count: int,
    rng: random.Random,
) -> list[Assignment]:
    """Return a balanced, unique assignment sample for previews and tests."""
    if count < 1:
        raise GenerationError("count must be at least 1")
    sampler = BalancedAssignmentSampler(constructs, decision_makers, rng)
    if count > sampler.remaining_count:
        raise GenerationError(
            f"Requested {count} unique assignments, but only "
            f"{sampler.remaining_count} combinations exist"
        )
    assignments = []
    for _ in range(count):
        assignment = sampler.next_assignment()
        sampler.record_acceptance(assignment)
        assignments.append(assignment)
    return assignments


def _usage_data(response: Any) -> Any:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    return str(usage)


def request_model_response(
    client: Any,
    *,
    stage: str,
    config: StageConfig,
    instructions: str,
    input_data: Mapping[str, Any],
    schema_name: str | None = None,
    schema: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    request: dict[str, Any] = {
        "model": config.model,
        "instructions": instructions,
        "input": json.dumps(input_data, ensure_ascii=False, indent=2),
        "max_output_tokens": config.max_output_tokens,
        "store": False,
        "prompt_cache_key": f"ecological-dilemma-{stage}-v2",
    }
    if config.reasoning_effort is not None:
        request["reasoning"] = {"effort": config.reasoning_effort}
    if schema is not None:
        if not schema_name:
            raise GenerationError("Structured output requests require a schema name")
        request["text"] = {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        }

    response = client.responses.create(**request)
    output_text = getattr(response, "output_text", "")
    if not isinstance(output_text, str) or not output_text.strip():
        raise GenerationError(f"The {stage} stage returned no text")
    metadata = {
        "stage": stage,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "response_id": getattr(response, "id", None),
        "response_model": getattr(response, "model", None),
        "usage": _usage_data(response),
    }
    return output_text.strip(), metadata


def request_structured_response(
    client: Any,
    *,
    stage: str,
    config: StageConfig,
    instructions: str,
    input_data: Mapping[str, Any],
    schema_name: str,
    schema: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    output_text, metadata = request_model_response(
        client,
        stage=stage,
        config=config,
        instructions=instructions,
        input_data=input_data,
        schema_name=schema_name,
        schema=schema,
    )
    try:
        parsed = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise GenerationError(f"The {stage} stage returned invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise GenerationError(f"The {stage} stage must return a JSON object")
    return parsed, metadata


def validate_card_generation(payload: Mapping[str, Any], card_count: int) -> None:
    viable = payload.get("viable")
    cards = payload.get("cards")
    if not isinstance(viable, bool) or not isinstance(cards, list):
        raise GenerationError("Card generation output is missing viability or cards")
    if viable and len(cards) != card_count:
        raise GenerationError(
            f"Card planner returned {len(cards)} cards; expected {card_count}"
        )
    if not viable and cards:
        raise GenerationError("A nonviable combination must return an empty card list")
    for card in cards:
        validate_card(card)


def validate_card(card: Any) -> None:
    if not isinstance(card, dict):
        raise GenerationError("A scenario card must be a JSON object")
    missing = [
        field
        for field in CARD_FIELDS
        if not isinstance(card.get(field), str) or not card[field].strip()
    ]
    if missing:
        raise GenerationError(f"Scenario card has empty fields: {missing}")


def validate_scores(payload: Mapping[str, Any], fields: Sequence[str]) -> None:
    scores = payload.get("scores")
    if not isinstance(scores, dict):
        raise GenerationError("Review output is missing scores")
    invalid = [
        field
        for field in fields
        if not isinstance(scores.get(field), int) or not 1 <= scores[field] <= 5
    ]
    if invalid:
        raise GenerationError(f"Review output has invalid scores: {invalid}")


def scores_pass(payload: Mapping[str, Any], fields: Sequence[str], minimum: int) -> bool:
    validate_scores(payload, fields)
    scores = payload["scores"]
    return all(scores[field] >= minimum for field in fields)


def basic_text_violations(text: str) -> list[str]:
    violations = []
    paragraphs = [
        part.strip() for part in re.split(r"\n\s*\n", text.strip()) if part.strip()
    ]
    word_count = len(re.findall(r"\b[\w'-]+\b", text))
    if len(paragraphs) not in {2, 3}:
        violations.append(f"expected 2 or 3 paragraphs, found {len(paragraphs)}")
    if not 160 <= word_count <= 300:
        violations.append(f"expected 160-300 words, found {word_count}")
    if not text.rstrip().endswith("?"):
        violations.append("dilemma does not end with a direct question")
    if any(paragraph.startswith("#") for paragraph in paragraphs):
        violations.append("dilemma contains a heading")
    return violations


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, ensure_ascii=False) + "\n")
        output.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GenerationError(f"Invalid JSON at {path}:{line_number}") from exc
        if not isinstance(record, dict):
            raise GenerationError(f"Expected a JSON object at {path}:{line_number}")
        records.append(record)
    return records


def run_directory_name(count: int, now: datetime | None = None) -> str:
    current = now or datetime.now(timezone.utc)
    timestamp = current.strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}_quality_pipeline_n{count}"


def _empty_usage(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "calls": 0,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "cache_write_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
    }


def add_usage(
    usage_by_stage: dict[str, dict[str, Any]],
    stage: str,
    model: str,
    usage: Any,
) -> None:
    aggregate = usage_by_stage.setdefault(stage, _empty_usage(model))
    aggregate["calls"] += 1
    if not isinstance(usage, dict):
        return
    input_details = usage.get("input_tokens_details")
    output_details = usage.get("output_tokens_details")
    if not isinstance(input_details, dict):
        input_details = {}
    if not isinstance(output_details, dict):
        output_details = {}
    aggregate["input_tokens"] += int(usage.get("input_tokens") or 0)
    aggregate["cached_input_tokens"] += int(input_details.get("cached_tokens") or 0)
    aggregate["cache_write_tokens"] += int(
        input_details.get("cache_write_tokens") or 0
    )
    aggregate["output_tokens"] += int(usage.get("output_tokens") or 0)
    aggregate["reasoning_tokens"] += int(output_details.get("reasoning_tokens") or 0)


def estimate_usage_cost(usage_by_stage: Mapping[str, Mapping[str, Any]]) -> float | None:
    prices = PRICING_SNAPSHOT["per_million_tokens"]
    total = 0.0
    for usage in usage_by_stage.values():
        model = usage.get("model")
        if model not in prices:
            return None
        model_prices = prices[model]
        input_tokens = int(usage.get("input_tokens") or 0)
        cached = int(usage.get("cached_input_tokens") or 0)
        cache_write = int(usage.get("cache_write_tokens") or 0)
        ordinary = max(input_tokens - cached - cache_write, 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        total += ordinary * model_prices["input"] / 1_000_000
        total += cached * model_prices["cached_input"] / 1_000_000
        total += (
            cache_write
            * model_prices["input"]
            * model_prices["cache_write_multiplier"]
            / 1_000_000
        )
        total += output_tokens * model_prices["output"] / 1_000_000
    return total


def _prompt_metadata(
    paths: Mapping[str, Path], prompts: Mapping[str, str]
) -> dict[str, Any]:
    return {
        stage: {"path": str(paths[stage]), "sha256": sha256_text(prompts[stage])}
        for stage in paths
    }


def generate_dataset(
    *,
    client: Any | None,
    count: int,
    seed: int,
    constructs_path: Path,
    decision_makers_path: Path,
    prompt_paths: Mapping[str, Path],
    output_dir: Path,
    pipeline: PipelineConfig,
    card_candidates: int = 3,
    minimum_score: int = 4,
    max_attempts: int | None = None,
    validation_rounds: int = 2,
    novelty_window: int = 25,
    dry_run: bool = False,
    resume_dir: Path | None = None,
) -> Path:
    if count < 1:
        raise GenerationError("count must be at least 1")
    if card_candidates < 2:
        raise GenerationError("card_candidates must be at least 2")
    if minimum_score not in range(1, 6):
        raise GenerationError("minimum_score must be between 1 and 5")
    if validation_rounds < 1:
        raise GenerationError("validation_rounds must be at least 1")
    if novelty_window < 0:
        raise GenerationError("novelty_window cannot be negative")

    constructs = load_constructs(constructs_path)
    decision_makers = load_decision_makers(decision_makers_path)
    prompts = {stage: load_prompt(path) for stage, path in prompt_paths.items()}
    expected_stages = {"planner", "reviewer", "writer", "validator"}
    if set(prompts) != expected_stages:
        raise GenerationError(f"Prompt paths must contain stages: {sorted(expected_stages)}")

    sampler = BalancedAssignmentSampler(constructs, decision_makers, random.Random(seed))
    maximum_attempts = max_attempts if max_attempts is not None else count * 3
    maximum_attempts = min(maximum_attempts, sampler.remaining_count)
    if maximum_attempts < count:
        raise GenerationError("max_attempts must be at least count")

    run_dir = resume_dir.resolve() if resume_dir is not None else output_dir / run_directory_name(count)
    if resume_dir is None:
        run_dir.mkdir(parents=True, exist_ok=False)
    elif not run_dir.is_dir():
        raise GenerationError(f"Resume directory does not exist: {run_dir}")
    attempts_dir = run_dir / "attempts"
    attempts_dir.mkdir(exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    records_path = run_dir / "records.jsonl"
    attempts_path = run_dir / "attempts.jsonl"
    new_manifest: dict[str, Any] = {
        "status": "in_progress",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "count_requested": count,
        "count_completed": 0,
        "count_attempted": 0,
        "count_rejected": 0,
        "seed": seed,
        "dry_run": dry_run,
        "card_candidates": card_candidates,
        "minimum_score": minimum_score,
        "max_attempts": maximum_attempts,
        "validation_rounds": validation_rounds,
        "novelty_window": novelty_window,
        "pipeline": pipeline.to_dict(),
        "constructs_path": str(constructs_path),
        "constructs_sha256": sha256_text(constructs_path.read_text(encoding="utf-8")),
        "decision_makers_path": str(decision_makers_path),
        "decision_makers_sha256": sha256_text(
            decision_makers_path.read_text(encoding="utf-8")
        ),
        "prompts": _prompt_metadata(prompt_paths, prompts),
        "usage_by_stage": {},
        "pricing_snapshot": PRICING_SNAPSHOT,
        "estimated_standard_cost_usd": 0.0,
    }

    if resume_dir is None:
        manifest = new_manifest
        finalized_attempts: list[dict[str, Any]] = []
        accepted_records: list[dict[str, Any]] = []
    else:
        if dry_run:
            raise GenerationError("Dry runs cannot be resumed")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise GenerationError(f"Resume manifest not found: {manifest_path}") from exc
        except json.JSONDecodeError as exc:
            raise GenerationError(f"Resume manifest is invalid: {manifest_path}") from exc
        if manifest.get("status") == "complete":
            raise GenerationError("The requested run is already complete")
        immutable_fields = (
            "count_requested",
            "seed",
            "card_candidates",
            "minimum_score",
            "max_attempts",
            "validation_rounds",
            "novelty_window",
            "pipeline",
            "constructs_sha256",
            "decision_makers_sha256",
            "prompts",
        )
        mismatched = [
            field for field in immutable_fields if manifest.get(field) != new_manifest.get(field)
        ]
        if mismatched:
            raise GenerationError(
                "Resume configuration or source files changed: " + ", ".join(mismatched)
            )
        finalized_attempts = read_jsonl(attempts_path)
        accepted_records = read_jsonl(records_path)
        for attempt_index, attempt_record in enumerate(finalized_attempts, start=1):
            expected = sampler.next_assignment()
            if attempt_record.get("attempt") != attempt_index:
                raise GenerationError("Resume attempt history is not sequential")
            if attempt_record.get("assignment") != expected.to_dict():
                raise GenerationError("Resume attempt history does not match the recorded seed")
            if attempt_record.get("status") == "accepted":
                sampler.record_acceptance(expected)
            elif attempt_record.get("status") != "rejected":
                raise GenerationError("Resume attempt history contains an unfinished status")
        accepted_attempts = [
            item for item in finalized_attempts if item.get("status") == "accepted"
        ]
        if len(accepted_records) < len(accepted_attempts):
            for attempt_record in accepted_attempts[len(accepted_records) :]:
                accepted_index = int(attempt_record["accepted_index"])
                reviewer_output = attempt_record["stages"]["reviewer"]["output"]
                recovered_record = {
                    "index": accepted_index,
                    "attempt": attempt_record["attempt"],
                    "status": "accepted",
                    "assignment": attempt_record["assignment"],
                    "approved_card": reviewer_output["revised_card"],
                    "final_dilemma": attempt_record["final_dilemma"],
                    "stage_record": attempt_record["stages"],
                }
                stem = f"dilemma_{accepted_index:04d}"
                write_json(run_dir / f"{stem}.json", recovered_record)
                (run_dir / f"{stem}.txt").write_text(
                    recovered_record["final_dilemma"] + "\n", encoding="utf-8"
                )
                append_jsonl(records_path, recovered_record)
                accepted_records.append(recovered_record)
        if len(accepted_records) != len(accepted_attempts):
            raise GenerationError("Resume accepted records do not match attempt history")
        manifest["status"] = "in_progress"
        manifest.pop("error", None)
        manifest.setdefault("resumed_at", []).append(datetime.now(timezone.utc).isoformat())
        manifest["count_attempted"] = len(finalized_attempts)
        manifest["count_completed"] = len(accepted_records)
        manifest["count_rejected"] = len(finalized_attempts) - len(accepted_records)

    usage_by_stage = manifest.setdefault("usage_by_stage", {})

    def checkpoint() -> None:
        cost = estimate_usage_cost(usage_by_stage)
        manifest["estimated_standard_cost_usd"] = (
            round(cost, 6) if cost is not None else None
        )
        write_json(manifest_path, manifest)

    checkpoint()

    try:
        if dry_run:
            assignments = sample_assignments(
                constructs, decision_makers, count, random.Random(seed)
            )
            for index, assignment in enumerate(assignments, start=1):
                record = {
                    "index": index,
                    "status": "sampled",
                    "assignment": assignment.to_dict(),
                }
                write_json(run_dir / f"dilemma_{index:04d}.json", record)
                append_jsonl(records_path, record)
                manifest["count_completed"] = index
                manifest["count_attempted"] = index
                checkpoint()
                print(
                    f"[{index}/{count}] sampled: {assignment.ecological_object} / "
                    f"{assignment.human_interest} / {assignment.policy_mechanism} / "
                    f"{assignment.decision_maker}"
                )
        else:
            if client is None:
                raise GenerationError("An API client is required outside dry-run mode")
            accepted_count = len(accepted_records)
            accepted_signatures = [
                str(record["approved_card"]["novelty_signature"])
                for record in accepted_records
            ]
            normalized_signatures = {
                re.sub(r"\s+", " ", signature.strip().lower())
                for signature in accepted_signatures
            }

            while accepted_count < count and manifest["count_attempted"] < maximum_attempts:
                attempt_number = manifest["count_attempted"] + 1
                assignment = sampler.next_assignment()
                recent_signatures = (
                    accepted_signatures[-novelty_window:] if novelty_window else []
                )
                attempt_record: dict[str, Any] = {
                    "attempt": attempt_number,
                    "status": "in_progress",
                    "assignment": assignment.to_dict(),
                    "recent_novelty_signatures": recent_signatures,
                    "stages": {},
                }
                attempt_file = attempts_dir / f"attempt_{attempt_number:05d}.json"
                write_json(attempt_file, attempt_record)

                card_payload, card_metadata = request_structured_response(
                    client,
                    stage="planner",
                    config=pipeline.planner,
                    instructions=prompts["planner"],
                    input_data={
                        "assignment": assignment.to_dict(),
                        "candidate_count": card_candidates,
                        "recent_novelty_signatures": recent_signatures,
                    },
                    schema_name="ecological_scenario_cards",
                    schema=SCENARIO_CARDS_SCHEMA,
                )
                validate_card_generation(card_payload, card_candidates)
                attempt_record["stages"]["planner"] = {
                    "output": card_payload,
                    "metadata": card_metadata,
                }
                add_usage(
                    usage_by_stage,
                    "planner",
                    pipeline.planner.model,
                    card_metadata["usage"],
                )
                write_json(attempt_file, attempt_record)
                checkpoint()

                rejection_reason: str | None = None
                if not card_payload["viable"]:
                    rejection_reason = (
                        card_payload.get("rejection_reason")
                        or "planner found the construct combination nonviable"
                    )

                approved_card: dict[str, Any] | None = None
                if rejection_reason is None:
                    review_payload, review_metadata = request_structured_response(
                        client,
                        stage="reviewer",
                        config=pipeline.reviewer,
                        instructions=prompts["reviewer"],
                        input_data={
                            "assignment": assignment.to_dict(),
                            "candidate_cards": card_payload["cards"],
                            "recent_novelty_signatures": recent_signatures,
                            "minimum_acceptable_score": minimum_score,
                        },
                        schema_name="ecological_card_review",
                        schema=CARD_REVIEW_SCHEMA,
                    )
                    validate_scores(review_payload, SCORE_FIELDS)
                    validate_card(review_payload.get("revised_card"))
                    attempt_record["stages"]["reviewer"] = {
                        "output": review_payload,
                        "metadata": review_metadata,
                    }
                    add_usage(
                        usage_by_stage,
                        "reviewer",
                        pipeline.reviewer.model,
                        review_metadata["usage"],
                    )
                    write_json(attempt_file, attempt_record)
                    checkpoint()

                    if review_payload.get("decision") != "accept":
                        rejection_reason = (
                            review_payload.get("overall_reason")
                            or "independent reviewer rejected all cards"
                        )
                    elif not scores_pass(review_payload, SCORE_FIELDS, minimum_score):
                        rejection_reason = "reviewer scores fell below the acceptance threshold"
                    else:
                        approved_card = dict(review_payload["revised_card"])
                        normalized_signature = re.sub(
                            r"\s+", " ", approved_card["novelty_signature"].strip().lower()
                        )
                        if normalized_signature in normalized_signatures:
                            rejection_reason = "novelty signature duplicates an accepted case"

                final_dilemma: str | None = None
                if rejection_reason is None and approved_card is not None:
                    draft, writer_metadata = request_model_response(
                        client,
                        stage="writer",
                        config=pipeline.writer,
                        instructions=prompts["writer"],
                        input_data={
                            "assignment": assignment.to_dict(),
                            "approved_card": approved_card,
                        },
                    )
                    attempt_record["stages"]["writer"] = {
                        "output": draft,
                        "metadata": writer_metadata,
                    }
                    add_usage(
                        usage_by_stage,
                        "writer",
                        pipeline.writer.model,
                        writer_metadata["usage"],
                    )
                    write_json(attempt_file, attempt_record)
                    checkpoint()

                    current_draft = draft
                    validations = []
                    prior_violations: list[str] = []
                    for validation_round in range(1, validation_rounds + 1):
                        deterministic = basic_text_violations(current_draft)
                        validation_payload, validation_metadata = request_structured_response(
                            client,
                            stage="validator",
                            config=pipeline.validator,
                            instructions=prompts["validator"],
                            input_data={
                                "assignment": assignment.to_dict(),
                                "approved_card": approved_card,
                                "draft_dilemma": current_draft,
                                "deterministic_format_violations": deterministic,
                                "previous_validation_violations": prior_violations,
                                "minimum_acceptable_score": minimum_score,
                                "validation_round": validation_round,
                            },
                            schema_name="ecological_dilemma_validation",
                            schema=FINAL_VALIDATION_SCHEMA,
                        )
                        validate_scores(validation_payload, VALIDATION_SCORE_FIELDS)
                        validations.append(
                            {
                                "round": validation_round,
                                "input_draft": current_draft,
                                "deterministic_format_violations": deterministic,
                                "output": validation_payload,
                                "metadata": validation_metadata,
                            }
                        )
                        add_usage(
                            usage_by_stage,
                            "validator",
                            pipeline.validator.model,
                            validation_metadata["usage"],
                        )
                        attempt_record["stages"]["validator"] = validations
                        write_json(attempt_file, attempt_record)
                        checkpoint()

                        decision = validation_payload.get("decision")
                        passes = scores_pass(
                            validation_payload,
                            VALIDATION_SCORE_FIELDS,
                            minimum_score,
                        )
                        if decision == "accept" and passes and not deterministic:
                            returned = validation_payload.get("revised_dilemma", "").strip()
                            if returned != current_draft.strip():
                                rejection_reason = (
                                    "validator changed the dilemma while claiming to accept it"
                                )
                            else:
                                final_dilemma = current_draft.strip()
                            break
                        if decision == "reject":
                            rejection_reason = (
                                validation_payload.get("rationale")
                                or "final validator rejected the dilemma"
                            )
                            break
                        revised = validation_payload.get("revised_dilemma")
                        if not isinstance(revised, str) or not revised.strip():
                            rejection_reason = "validator requested revision without revised prose"
                            break
                        prior_violations = list(validation_payload.get("violations") or [])
                        if not passes:
                            low = [
                                field
                                for field in VALIDATION_SCORE_FIELDS
                                if validation_payload["scores"][field] < minimum_score
                            ]
                            prior_violations.append(f"scores below threshold: {', '.join(low)}")
                        current_draft = revised.strip()

                    if final_dilemma is None and rejection_reason is None:
                        rejection_reason = "dilemma did not pass within the validation-round limit"

                manifest["count_attempted"] = attempt_number
                if final_dilemma is None or approved_card is None:
                    attempt_record["status"] = "rejected"
                    attempt_record["rejection_reason"] = rejection_reason
                    manifest["count_rejected"] += 1
                    append_jsonl(attempts_path, attempt_record)
                    write_json(attempt_file, attempt_record)
                    checkpoint()
                    print(
                        f"[attempt {attempt_number}] rejected: "
                        f"{assignment.ecological_object} / {assignment.human_interest} — "
                        f"{rejection_reason}"
                    )
                    continue

                accepted_count += 1
                sampler.record_acceptance(assignment)
                signature = approved_card["novelty_signature"].strip()
                accepted_signatures.append(signature)
                normalized_signatures.add(re.sub(r"\s+", " ", signature.lower()))
                attempt_record["status"] = "accepted"
                attempt_record["accepted_index"] = accepted_count
                attempt_record["final_dilemma"] = final_dilemma
                append_jsonl(attempts_path, attempt_record)
                write_json(attempt_file, attempt_record)

                record = {
                    "index": accepted_count,
                    "attempt": attempt_number,
                    "status": "accepted",
                    "assignment": assignment.to_dict(),
                    "approved_card": approved_card,
                    "final_dilemma": final_dilemma,
                    "stage_record": attempt_record["stages"],
                }
                stem = f"dilemma_{accepted_count:04d}"
                write_json(run_dir / f"{stem}.json", record)
                (run_dir / f"{stem}.txt").write_text(
                    final_dilemma + "\n", encoding="utf-8"
                )
                append_jsonl(records_path, record)
                manifest["count_completed"] = accepted_count
                checkpoint()
                print(
                    f"[{accepted_count}/{count}] accepted on attempt {attempt_number}: "
                    f"{assignment.ecological_object} / {assignment.human_interest} / "
                    f"{assignment.policy_mechanism} / {assignment.decision_maker}"
                )

            if accepted_count < count:
                raise GenerationError(
                    f"Accepted {accepted_count} of {count} requested dilemmas after "
                    f"{manifest['count_attempted']} attempts"
                )
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        checkpoint()
        raise

    manifest["status"] = "dry_run_complete" if dry_run else "complete"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    checkpoint()
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate ecological dilemmas through scenario-card planning, independent "
            "review, prose writing, and final validation."
        )
    )
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--planner-model",
        default=os.getenv("OPENAI_PLANNER_MODEL", DEFAULT_REASONING_MODEL),
    )
    parser.add_argument(
        "--reviewer-model",
        default=os.getenv("OPENAI_REVIEWER_MODEL", DEFAULT_REASONING_MODEL),
    )
    parser.add_argument(
        "--writer-model",
        default=os.getenv(
            "OPENAI_WRITER_MODEL", os.getenv("OPENAI_MODEL", DEFAULT_WRITER_MODEL)
        ),
    )
    parser.add_argument(
        "--validator-model",
        default=os.getenv("OPENAI_VALIDATOR_MODEL", DEFAULT_REASONING_MODEL),
    )
    effort_choices = ("omit", "none", "low", "medium", "high", "xhigh", "max")
    parser.add_argument(
        "--reasoning-effort",
        choices=effort_choices,
        default=os.getenv("OPENAI_PIPELINE_REASONING_EFFORT", "high"),
        help="Planner, reviewer, and validator reasoning effort (default: high).",
    )
    parser.add_argument(
        "--writer-reasoning-effort",
        choices=effort_choices,
        default=os.getenv("OPENAI_WRITER_REASONING_EFFORT", "medium"),
    )
    parser.add_argument("--planner-max-output-tokens", type=int, default=6000)
    parser.add_argument("--reviewer-max-output-tokens", type=int, default=5000)
    parser.add_argument("--writer-max-output-tokens", type=int, default=1600)
    parser.add_argument("--validator-max-output-tokens", type=int, default=4000)
    parser.add_argument("--card-candidates", type=int, default=3)
    parser.add_argument("--minimum-score", type=int, default=4)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Maximum sampled combinations; defaults to three times --count.",
    )
    parser.add_argument("--validation-rounds", type=int, default=2)
    parser.add_argument("--novelty-window", type=int, default=25)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--constructs", type=Path, default=DEFAULT_CONSTRUCTS_PATH)
    parser.add_argument(
        "--decision-makers", type=Path, default=DEFAULT_DECISION_MAKERS_PATH
    )
    parser.add_argument("--card-prompt", type=Path, default=DEFAULT_CARD_PROMPT_PATH)
    parser.add_argument("--review-prompt", type=Path, default=DEFAULT_REVIEW_PROMPT_PATH)
    parser.add_argument("--writer-prompt", type=Path, default=DEFAULT_WRITER_PROMPT_PATH)
    parser.add_argument(
        "--validator-prompt", type=Path, default=DEFAULT_VALIDATOR_PROMPT_PATH
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview balanced assignments without making API calls.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume an interrupted run directory using its recorded configuration.",
    )
    return parser


def _parse_effort(value: str) -> str | None:
    return None if value == "omit" else value


def main(argv: Sequence[str] | None = None) -> int:
    try:
        load_env_file(REPO_ROOT / ".env")
    except GenerationError as exc:
        print(f"Environment setup failed: {exc}", file=sys.stderr)
        return 1

    parser = build_parser()
    args = parser.parse_args(argv)
    resume_manifest = None
    if args.resume is not None:
        if args.dry_run:
            parser.error("--resume cannot be combined with --dry-run")
        resume_manifest_path = args.resume.expanduser().resolve() / "manifest.json"
        try:
            resume_manifest = json.loads(resume_manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            parser.error(f"cannot read resume manifest {resume_manifest_path}: {exc}")
        args.count = int(resume_manifest["count_requested"])
        args.seed = int(resume_manifest["seed"])
        args.card_candidates = int(resume_manifest["card_candidates"])
        args.minimum_score = int(resume_manifest["minimum_score"])
        args.max_attempts = int(resume_manifest["max_attempts"])
        args.validation_rounds = int(resume_manifest["validation_rounds"])
        args.novelty_window = int(resume_manifest["novelty_window"])
        args.constructs = Path(resume_manifest["constructs_path"])
        args.decision_makers = Path(resume_manifest["decision_makers_path"])
        args.card_prompt = Path(resume_manifest["prompts"]["planner"]["path"])
        args.review_prompt = Path(resume_manifest["prompts"]["reviewer"]["path"])
        args.writer_prompt = Path(resume_manifest["prompts"]["writer"]["path"])
        args.validator_prompt = Path(resume_manifest["prompts"]["validator"]["path"])
    positive_options = {
        "--count": args.count,
        "--planner-max-output-tokens": args.planner_max_output_tokens,
        "--reviewer-max-output-tokens": args.reviewer_max_output_tokens,
        "--writer-max-output-tokens": args.writer_max_output_tokens,
        "--validator-max-output-tokens": args.validator_max_output_tokens,
        "--validation-rounds": args.validation_rounds,
    }
    for option, value in positive_options.items():
        if value < 1:
            parser.error(f"{option} must be at least 1")
    if args.card_candidates < 2:
        parser.error("--card-candidates must be at least 2")
    if not 1 <= args.minimum_score <= 5:
        parser.error("--minimum-score must be between 1 and 5")
    if args.max_attempts is not None and args.max_attempts < args.count:
        parser.error("--max-attempts must be at least --count")
    if args.novelty_window < 0:
        parser.error("--novelty-window cannot be negative")

    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**63)
    if resume_manifest is None:
        reasoning_effort = _parse_effort(args.reasoning_effort)
        writer_effort = _parse_effort(args.writer_reasoning_effort)
        pipeline = PipelineConfig(
            planner=StageConfig(
                args.planner_model, reasoning_effort, args.planner_max_output_tokens
            ),
            reviewer=StageConfig(
                args.reviewer_model, reasoning_effort, args.reviewer_max_output_tokens
            ),
            writer=StageConfig(
                args.writer_model, writer_effort, args.writer_max_output_tokens
            ),
            validator=StageConfig(
                args.validator_model, reasoning_effort, args.validator_max_output_tokens
            ),
        )
    else:
        stage_configs = resume_manifest["pipeline"]
        pipeline = PipelineConfig(
            planner=StageConfig(**stage_configs["planner"]),
            reviewer=StageConfig(**stage_configs["reviewer"]),
            writer=StageConfig(**stage_configs["writer"]),
            validator=StageConfig(**stage_configs["validator"]),
        )

    client = None
    if not args.dry_run:
        if not os.getenv("OPENAI_API_KEY"):
            parser.error("OPENAI_API_KEY is missing. Add it to the repository's .env file.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            print(
                "Missing dependency 'openai'. Run: python -m pip install -r requirements.txt",
                file=sys.stderr,
            )
            raise SystemExit(2) from exc
        client = OpenAI()

    prompt_paths = {
        "planner": args.card_prompt.expanduser().resolve(),
        "reviewer": args.review_prompt.expanduser().resolve(),
        "writer": args.writer_prompt.expanduser().resolve(),
        "validator": args.validator_prompt.expanduser().resolve(),
    }
    try:
        run_dir = generate_dataset(
            client=client,
            count=args.count,
            seed=seed,
            constructs_path=args.constructs.expanduser().resolve(),
            decision_makers_path=args.decision_makers.expanduser().resolve(),
            prompt_paths=prompt_paths,
            output_dir=args.output_dir.expanduser().resolve(),
            pipeline=pipeline,
            card_candidates=args.card_candidates,
            minimum_score=args.minimum_score,
            max_attempts=args.max_attempts,
            validation_rounds=args.validation_rounds,
            novelty_window=args.novelty_window,
            dry_run=args.dry_run,
            resume_dir=args.resume,
        )
    except GenerationError as exc:
        print(f"Generation failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Pipeline failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    print(f"Run complete: {run_dir}")
    print(f"Sampling seed: {seed}")
    if manifest.get("estimated_standard_cost_usd") is not None:
        print(
            "Estimated API cost from recorded token usage: "
            f"${manifest['estimated_standard_cost_usd']:.4f} USD "
            f"(pricing snapshot {PRICING_SNAPSHOT['date']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
