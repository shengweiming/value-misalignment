#!/usr/bin/env python3
"""Generate ecological-versus-human ethical dilemmas with the OpenAI API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONSTRUCTS_PATH = (
    REPO_ROOT / "src/ecological_dilemmas/ecological_dilemma_constructs.json"
)
DEFAULT_DECISION_MAKERS_PATH = (
    REPO_ROOT / "src/ecological_dilemmas/decision_makers.json"
)
DEFAULT_PROMPT_PATH = REPO_ROOT / "src/ecological_dilemmas/generator_prompt.txt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/ecological_dilemmas"
DEFAULT_MODEL = "gpt-5.6-terra"
REQUIRED_CONSTRUCT_SECTIONS = (
    "ecological_objects",
    "human_interests",
    "policy_mechanisms",
)


class GenerationError(RuntimeError):
    """Raised when the generator configuration or API response is invalid."""


@dataclass(frozen=True)
class Assignment:
    ecological_object: str
    ecological_object_definition: str
    human_interest: str
    human_interest_definition: str
    policy_mechanism: str
    policy_mechanism_definition: str
    decision_maker: str

    def prompt_variables(self) -> dict[str, str]:
        return asdict(self)


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


def load_prompt_template(path: Path) -> str:
    try:
        template = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise GenerationError(f"Prompt template not found: {path}") from exc
    if not template:
        raise GenerationError("Prompt template is empty")
    return template


def sample_assignments(
    constructs: Mapping[str, Mapping[str, str]],
    decision_makers: Sequence[str],
    count: int,
    rng: random.Random,
) -> list[Assignment]:
    if count < 1:
        raise GenerationError("count must be at least 1")

    ecological_objects = list(constructs["ecological_objects"].items())
    human_interests = list(constructs["human_interests"].items())
    policy_mechanisms = list(constructs["policy_mechanisms"].items())
    maximum = (
        len(ecological_objects)
        * len(human_interests)
        * len(policy_mechanisms)
        * len(decision_makers)
    )
    if count > maximum:
        raise GenerationError(
            f"Requested {count} unique assignments, but only {maximum} combinations exist"
        )

    assignments: list[Assignment] = []
    seen: set[tuple[str, str, str, str]] = set()
    while len(assignments) < count:
        ecological_object, ecological_definition = rng.choice(ecological_objects)
        human_interest, human_definition = rng.choice(human_interests)
        policy_mechanism, policy_definition = rng.choice(policy_mechanisms)
        decision_maker = rng.choice(list(decision_makers))
        key = (
            ecological_object,
            human_interest,
            policy_mechanism,
            decision_maker,
        )
        if key in seen:
            continue
        seen.add(key)
        assignments.append(
            Assignment(
                ecological_object=ecological_object,
                ecological_object_definition=ecological_definition,
                human_interest=human_interest,
                human_interest_definition=human_definition,
                policy_mechanism=policy_mechanism,
                policy_mechanism_definition=policy_definition,
                decision_maker=decision_maker,
            )
        )
    return assignments


def render_prompt(template: str, assignment: Assignment) -> str:
    try:
        prompt = template.format(**assignment.prompt_variables())
    except KeyError as exc:
        raise GenerationError(
            f"Prompt template contains an unknown placeholder: {exc.args[0]}"
        ) from exc
    unresolved = re.findall(r"\{[A-Za-z_][A-Za-z0-9_]*\}", prompt)
    if unresolved:
        raise GenerationError(
            f"Prompt contains unresolved placeholders: {sorted(set(unresolved))}"
        )
    return prompt


def request_completion(
    client: Any,
    *,
    model: str,
    prompt: str,
    max_output_tokens: int,
    reasoning_effort: str | None,
) -> tuple[str, dict[str, Any]]:
    request: dict[str, Any] = {
        "model": model,
        "input": prompt,
        "max_output_tokens": max_output_tokens,
        "store": False,
    }
    if reasoning_effort is not None:
        request["reasoning"] = {"effort": reasoning_effort}

    response = client.responses.create(**request)
    completion = getattr(response, "output_text", "")
    if not isinstance(completion, str) or not completion.strip():
        raise GenerationError("The API returned no text completion")

    usage = getattr(response, "usage", None)
    if usage is None:
        usage_data = None
    elif hasattr(usage, "model_dump"):
        usage_data = usage.model_dump()
    elif isinstance(usage, dict):
        usage_data = usage
    else:
        usage_data = str(usage)

    response_metadata = {
        "response_id": getattr(response, "id", None),
        "response_model": getattr(response, "model", None),
        "usage": usage_data,
    }
    return completion.strip(), response_metadata


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_directory_name(model: str, count: int, now: datetime | None = None) -> str:
    current = now or datetime.now(timezone.utc)
    timestamp = current.strftime("%Y%m%dT%H%M%S%fZ")
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_") or "model"
    return f"{timestamp}_{safe_model}_n{count}"


def generate_dataset(
    *,
    client: Any | None,
    model: str,
    count: int,
    seed: int,
    constructs_path: Path,
    decision_makers_path: Path,
    prompt_path: Path,
    output_dir: Path,
    max_output_tokens: int,
    reasoning_effort: str | None,
    dry_run: bool = False,
) -> Path:
    constructs = load_constructs(constructs_path)
    decision_makers = load_decision_makers(decision_makers_path)
    template = load_prompt_template(prompt_path)
    assignments = sample_assignments(
        constructs,
        decision_makers,
        count,
        random.Random(seed),
    )

    run_dir = output_dir / run_directory_name(model, count)
    run_dir.mkdir(parents=True, exist_ok=False)
    records_path = run_dir / "records.jsonl"
    manifest_path = run_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "status": "in_progress",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "count_requested": count,
        "count_completed": 0,
        "seed": seed,
        "reasoning_effort": reasoning_effort,
        "max_output_tokens": max_output_tokens,
        "dry_run": dry_run,
        "constructs_path": str(constructs_path),
        "constructs_sha256": sha256_text(
            constructs_path.read_text(encoding="utf-8")
        ),
        "decision_makers_path": str(decision_makers_path),
        "decision_makers_sha256": sha256_text(
            decision_makers_path.read_text(encoding="utf-8")
        ),
        "prompt_path": str(prompt_path),
        "prompt_sha256": sha256_text(template),
    }
    write_json(manifest_path, manifest)

    try:
        with records_path.open("w", encoding="utf-8") as records_file:
            for index, assignment in enumerate(assignments, start=1):
                prompt = render_prompt(template, assignment)
                if dry_run:
                    completion = None
                    response_metadata: dict[str, Any] = {
                        "response_id": None,
                        "response_model": None,
                        "usage": None,
                    }
                else:
                    if client is None:
                        raise GenerationError("An API client is required outside dry-run mode")
                    completion, response_metadata = request_completion(
                        client,
                        model=model,
                        prompt=prompt,
                        max_output_tokens=max_output_tokens,
                        reasoning_effort=reasoning_effort,
                    )

                record = {
                    "index": index,
                    "assignment": assignment.prompt_variables(),
                    "model": model,
                    "prompt": prompt,
                    "prompt_sha256": sha256_text(prompt),
                    "completion": completion,
                    **response_metadata,
                }
                stem = f"dilemma_{index:04d}"
                write_json(run_dir / f"{stem}.json", record)
                if completion is not None:
                    (run_dir / f"{stem}.txt").write_text(
                        completion + "\n", encoding="utf-8"
                    )
                records_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                records_file.flush()

                manifest["count_completed"] = index
                write_json(manifest_path, manifest)
                status = "sampled" if dry_run else "generated"
                print(
                    f"[{index}/{count}] {status}: "
                    f"{assignment.ecological_object} / {assignment.human_interest} / "
                    f"{assignment.policy_mechanism} / {assignment.decision_maker}"
                )
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        write_json(manifest_path, manifest)
        raise

    manifest["status"] = "dry_run_complete" if dry_run else "complete"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    write_json(manifest_path, manifest)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate ecological-versus-human ethical dilemmas."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of completions to generate (default: 10).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        help=f"OpenAI model ID (default: OPENAI_MODEL or {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Sampling seed. If omitted, a random seed is generated and recorded.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("omit", "none", "low", "medium", "high", "xhigh", "max"),
        default=os.getenv("OPENAI_REASONING_EFFORT", "medium"),
        help="Reasoning effort, or 'omit' for models that do not support it.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=2400,
        help="Maximum output tokens per completion (default: 2400).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Parent output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--constructs",
        type=Path,
        default=DEFAULT_CONSTRUCTS_PATH,
        help="Path to the ecological construct JSON file.",
    )
    parser.add_argument(
        "--decision-makers",
        type=Path,
        default=DEFAULT_DECISION_MAKERS_PATH,
        help="Path to the decision-maker JSON file.",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=DEFAULT_PROMPT_PATH,
        help="Path to the generator prompt template.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Sample assignments and render prompts without making API calls.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        load_env_file(REPO_ROOT / ".env")
    except GenerationError as exc:
        print(f"Environment setup failed: {exc}", file=sys.stderr)
        return 1
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.max_output_tokens < 1:
        parser.error("--max-output-tokens must be at least 1")
    if args.count < 1:
        parser.error("--count must be at least 1")

    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**63)
    reasoning_effort = (
        None if args.reasoning_effort == "omit" else args.reasoning_effort
    )

    client = None
    if not args.dry_run:
        if not os.getenv("OPENAI_API_KEY"):
            parser.error(
                "OPENAI_API_KEY is missing. Add it to the repository's .env file."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            print(
                "Missing dependency 'openai'. Run: "
                "python -m pip install -r requirements.txt",
                file=sys.stderr,
            )
            raise SystemExit(2) from exc
        client = OpenAI()

    try:
        run_dir = generate_dataset(
            client=client,
            model=args.model,
            count=args.count,
            seed=seed,
            constructs_path=args.constructs.expanduser().resolve(),
            decision_makers_path=args.decision_makers.expanduser().resolve(),
            prompt_path=args.prompt.expanduser().resolve(),
            output_dir=args.output_dir.expanduser().resolve(),
            max_output_tokens=args.max_output_tokens,
            reasoning_effort=reasoning_effort,
            dry_run=args.dry_run,
        )
    except GenerationError as exc:
        print(f"Generation failed: {exc}", file=sys.stderr)
        return 1

    print(f"Run complete: {run_dir}")
    print(f"Sampling seed: {seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
