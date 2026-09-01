"""Load audited dilemma prompt-only and response-only SFT arms."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


PROMPT_ONLY_ARM = "prompt_only"
ECOLOGICAL_OPTION_ARM = "ecological_option"
HUMAN_OPTION_ARM = "human_option"
CLASH_ACTION_ARM = "action"
TRAINING_ARMS = (
    PROMPT_ONLY_ARM,
    ECOLOGICAL_OPTION_ARM,
    HUMAN_OPTION_ARM,
)
CLASH_TRAINING_ARMS = (PROMPT_ONLY_ARM, CLASH_ACTION_ARM)
SUPPORTED_TRAINING_ARMS = (*TRAINING_ARMS, CLASH_ACTION_ARM)
ANSWER_TARGET_FIELDS = {
    ECOLOGICAL_OPTION_ARM: "ecologically_protective_option",
    HUMAN_OPTION_ARM: "human_protective_option",
}
REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_source_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"The answer manifest has no {label}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"Could not resolve the answer manifest's {label}: {path}") from exc


def load_prompt_examples(
    records_path: Path | str,
    *,
    expected_count: int = 98,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return exactly the released dilemma text and its non-normative metadata."""

    path = Path(records_path).expanduser().resolve(strict=True)
    manifest_path = path.parent / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read the release manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("The release manifest must be a JSON object")
    if manifest.get("dataset_type") != "prompt_only":
        raise ValueError("The ecological-dilemma release is not marked prompt-only")
    if manifest.get("contains_normative_labels") is not False:
        raise ValueError("The release must explicitly report no normative labels")
    if manifest.get("contains_assistant_responses") is not False:
        raise ValueError("The release must explicitly report no assistant responses")
    if int(manifest.get("released_count") or -1) != expected_count:
        raise ValueError(
            f"Expected {expected_count} released prompts, found "
            f"{manifest.get('released_count')!r}"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("The release manifest has no artifact hash table")
    expected_hash = artifacts.get("records.jsonl")
    actual_hash = sha256_file(path)
    if expected_hash != actual_hash:
        raise ValueError("records.jsonl does not match the release manifest hash")

    examples: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Expected an object at {path}:{line_number}")
        forbidden = {
            "messages",
            "assistant",
            "answer",
            "label",
            "target_label",
            "rationale",
            "split",
        }
        present = sorted(forbidden.intersection(record))
        if present:
            raise ValueError(
                f"Prompt record {line_number} contains supervision fields: {present}"
            )
        prompt_id = record.get("id")
        dilemma = record.get("dilemma")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValueError(f"Prompt record {line_number} has no stable ID")
        if prompt_id in seen_ids:
            raise ValueError(f"Duplicate prompt ID: {prompt_id}")
        if not isinstance(dilemma, str) or not dilemma.strip():
            raise ValueError(f"Prompt record {prompt_id} has no dilemma text")
        seen_ids.add(prompt_id)
        examples.append(
            {
                "id": prompt_id,
                # Preserve the released setup byte-for-byte. Validation may use
                # ``strip`` to reject empty text, but training must not silently
                # rewrite the corpus.
                "dilemma": dilemma,
                "source": record.get("source"),
                "assignment": record.get("assignment"),
                "title": record.get("title"),
            }
        )
    if len(examples) != expected_count:
        raise ValueError(f"Expected {expected_count} prompt rows, found {len(examples)}")

    return examples, {
        "dataset_type": "prompt_only",
        "training_arm": PROMPT_ONLY_ARM,
        "objective": "causal language modeling on the dilemma text",
        "example_count": len(examples),
        "records_path": str(path),
        "records_sha256": actual_hash,
        "release_manifest_path": str(manifest_path),
        "release_manifest_sha256": sha256_file(manifest_path),
        "contains_normative_labels": False,
        "contains_assistant_responses": False,
    }


def load_answer_examples(
    records_path: Path | str,
    *,
    training_arm: str,
    expected_count: int = 98,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load one exact option-text assistant response per audited dilemma."""

    if training_arm not in ANSWER_TARGET_FIELDS:
        raise ValueError(
            f"Answer training_arm must be one of {sorted(ANSWER_TARGET_FIELDS)}"
        )
    path = Path(records_path).expanduser().resolve(strict=True)
    manifest_path = path.parent / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read the answer manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("The answer manifest must be a JSON object")
    expected_target = ANSWER_TARGET_FIELDS[training_arm]
    expected_manifest = {
        "dataset_type": "assistant_option_sft",
        "training_arm": training_arm,
        "assistant_target_field": expected_target,
        "response_style": "option_text_only",
        "example_count": expected_count,
        "contains_normative_labels": True,
        "contains_assistant_responses": True,
        "contains_rationales": False,
    }
    mismatches = [
        key
        for key, expected in expected_manifest.items()
        if manifest.get(key) != expected
    ]
    if mismatches:
        raise ValueError(
            "The assistant-answer release does not match the selected training "
            f"arm: {mismatches}"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("The answer manifest has no artifact hash table")
    actual_hash = sha256_file(path)
    if artifacts.get("records.jsonl") != actual_hash:
        raise ValueError("Answer records do not match the release manifest hash")
    source_release = manifest.get("source_release")
    if not isinstance(source_release, dict):
        raise ValueError("The answer manifest has no pinned source release")
    source_records_path = _resolve_source_path(
        source_release.get("records_path"), label="source records path"
    )
    source_manifest_path = _resolve_source_path(
        source_release.get("manifest_path"), label="source manifest path"
    )
    expected_source_hashes = {
        source_records_path: source_release.get("records_sha256"),
        source_manifest_path: source_release.get("manifest_sha256"),
    }
    mismatched_sources = [
        str(source_path)
        for source_path, expected_hash in expected_source_hashes.items()
        if not isinstance(expected_hash, str)
        or sha256_file(source_path) != expected_hash
    ]
    if mismatched_sources:
        raise ValueError(
            "The assistant-answer release does not match its pinned audited "
            f"source files: {mismatched_sources}"
        )

    examples: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Expected an object at {path}:{line_number}")
        prompt_id = record.get("id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValueError(f"Answer record {line_number} has no stable ID")
        if prompt_id in seen_ids:
            raise ValueError(f"Duplicate answer-record ID: {prompt_id}")
        if record.get("target_field") != expected_target:
            raise ValueError(
                f"Answer record {prompt_id} does not use {expected_target}"
            )
        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) != 2:
            raise ValueError(
                f"Answer record {prompt_id} must contain exactly two messages"
            )
        expected_roles = ("user", "assistant")
        contents: list[str] = []
        for message, expected_role in zip(messages, expected_roles, strict=True):
            if not isinstance(message, dict) or set(message) != {"role", "content"}:
                raise ValueError(
                    f"Answer record {prompt_id} has a malformed {expected_role} message"
                )
            if message.get("role") != expected_role:
                raise ValueError(
                    f"Answer record {prompt_id} must order user then assistant"
                )
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(
                    f"Answer record {prompt_id} has empty {expected_role} content"
                )
            contents.append(content)
        seen_ids.add(prompt_id)
        examples.append(
            {
                "id": prompt_id,
                "dilemma": contents[0],
                "assistant_answer": contents[1],
                "target_field": expected_target,
                "source": record.get("source"),
                "title": record.get("title"),
            }
        )
    if len(examples) != expected_count:
        raise ValueError(
            f"Expected {expected_count} answer rows, found {len(examples)}"
        )

    source_rows: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(
        source_records_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            source_record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON at {source_records_path}:{line_number}"
            ) from exc
        if not isinstance(source_record, dict):
            raise ValueError(
                f"Expected an object at {source_records_path}:{line_number}"
            )
        source_id = source_record.get("id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f"Audited source row {line_number} has no stable ID")
        if source_id in source_rows:
            raise ValueError(f"Duplicate audited source ID: {source_id}")
        source_rows[source_id] = source_record
    if len(source_rows) != expected_count:
        raise ValueError(
            f"Expected {expected_count} audited source rows, found {len(source_rows)}"
        )
    for example in examples:
        source_record = source_rows.get(example["id"])
        if source_record is None:
            raise ValueError(f"Answer record {example['id']} has no audited source")
        if example["dilemma"] != source_record.get("dilemma"):
            raise ValueError(
                f"Answer record {example['id']} changes the audited dilemma text"
            )
        if example["assistant_answer"] != source_record.get(expected_target):
            raise ValueError(
                f"Answer record {example['id']} does not exactly copy {expected_target}"
            )

    result_manifest = dict(manifest)
    result_manifest.update(
        {
            "objective": "response-only causal LM on the assistant option text",
            "records_path": str(path),
            "records_sha256": actual_hash,
            "release_manifest_path": str(manifest_path),
            "release_manifest_sha256": sha256_file(manifest_path),
        }
    )
    return examples, result_manifest


def load_clash_action_examples(
    records_path: Path | str,
    *,
    expected_count: int = 98,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load exact CLASH action fragments as response-only assistant targets."""

    path = Path(records_path).expanduser().resolve(strict=True)
    manifest_path = path.parent / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read the CLASH action manifest: {manifest_path}"
        ) from exc
    if not isinstance(manifest, dict):
        raise ValueError("The CLASH action manifest must be a JSON object")
    expected_manifest = {
        "dataset_type": "assistant_action_sft",
        "training_arm": CLASH_ACTION_ARM,
        "assistant_target_field": "action",
        "response_style": "action_text_only",
        "example_count": expected_count,
        "contains_normative_labels": False,
        "contains_assistant_responses": True,
        "contains_rationales": False,
        "contains_character_perspectives": False,
    }
    mismatches = [
        key
        for key, expected in expected_manifest.items()
        if manifest.get(key) != expected
    ]
    if mismatches:
        raise ValueError(
            "The CLASH action release does not match the selected training arm: "
            f"{mismatches}"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("The CLASH action manifest has no artifact hash table")
    actual_hash = sha256_file(path)
    if artifacts.get("records.jsonl") != actual_hash:
        raise ValueError("CLASH action records do not match the release manifest hash")

    source_release = manifest.get("source_release")
    if not isinstance(source_release, dict):
        raise ValueError("The CLASH action manifest has no pinned prompt release")
    source_records_path = _resolve_source_path(
        source_release.get("records_path"), label="source records path"
    )
    source_manifest_path = _resolve_source_path(
        source_release.get("manifest_path"), label="source manifest path"
    )
    source_snapshot = manifest.get("source_snapshot")
    if not isinstance(source_snapshot, dict):
        raise ValueError("The CLASH action manifest has no pinned source snapshot")
    source_csv_path = _resolve_source_path(
        source_snapshot.get("path"), label="source snapshot path"
    )
    expected_source_hashes = {
        source_records_path: source_release.get("records_sha256"),
        source_manifest_path: source_release.get("manifest_sha256"),
        source_csv_path: source_snapshot.get("sha256"),
    }
    mismatched_sources = [
        str(source_path)
        for source_path, expected_hash in expected_source_hashes.items()
        if not isinstance(expected_hash, str)
        or sha256_file(source_path) != expected_hash
    ]
    if mismatched_sources:
        raise ValueError(
            "The CLASH action release does not match its pinned source files: "
            f"{mismatched_sources}"
        )

    source_rows: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(
        source_records_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            source_record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON at {source_records_path}:{line_number}"
            ) from exc
        if not isinstance(source_record, dict):
            raise ValueError(
                f"Expected an object at {source_records_path}:{line_number}"
            )
        source_id = source_record.get("id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f"CLASH prompt row {line_number} has no stable ID")
        if source_id in source_rows:
            raise ValueError(f"Duplicate CLASH prompt ID: {source_id}")
        source_rows[source_id] = source_record
    if len(source_rows) != expected_count:
        raise ValueError(
            f"Expected {expected_count} CLASH prompt rows, found {len(source_rows)}"
        )

    with source_csv_path.open(newline="", encoding="utf-8-sig") as input_file:
        csv_rows = list(csv.DictReader(input_file))
    csv_by_id: dict[str, dict[str, str]] = {}
    for source_row in csv_rows:
        source_id = source_row.get("id")
        if not source_id:
            raise ValueError("A CLASH source row has no ID")
        if source_id in csv_by_id:
            raise ValueError(f"Duplicate CLASH source ID: {source_id}")
        csv_by_id[source_id] = source_row

    examples: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Expected an object at {path}:{line_number}")
        prompt_id = record.get("id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValueError(f"CLASH action row {line_number} has no stable ID")
        if prompt_id in seen_ids:
            raise ValueError(f"Duplicate CLASH action ID: {prompt_id}")
        if record.get("target_field") != "action":
            raise ValueError(f"CLASH action row {prompt_id} does not target action")
        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) != 2:
            raise ValueError(
                f"CLASH action row {prompt_id} must contain exactly two messages"
            )
        contents: list[str] = []
        for message, expected_role in zip(
            messages, ("user", "assistant"), strict=True
        ):
            if not isinstance(message, dict) or set(message) != {"role", "content"}:
                raise ValueError(
                    f"CLASH action row {prompt_id} has a malformed {expected_role} message"
                )
            if message.get("role") != expected_role:
                raise ValueError(
                    f"CLASH action row {prompt_id} must order user then assistant"
                )
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(
                    f"CLASH action row {prompt_id} has empty {expected_role} content"
                )
            contents.append(content)

        prompt_source = source_rows.get(prompt_id)
        if prompt_source is None:
            raise ValueError(f"CLASH action row {prompt_id} has no audited prompt")
        if contents[0] != prompt_source.get("dilemma"):
            raise ValueError(
                f"CLASH action row {prompt_id} changes the audited dilemma text"
            )
        provenance = prompt_source.get("source")
        raw_source_id = provenance.get("source_id") if isinstance(provenance, dict) else None
        raw_source = csv_by_id.get(str(raw_source_id))
        if raw_source is None:
            raise ValueError(
                f"CLASH action row {prompt_id} has no pinned CSV source row"
            )
        if contents[1] != raw_source.get("action"):
            raise ValueError(
                f"CLASH action row {prompt_id} does not exactly copy the source action"
            )
        if record.get("source") != prompt_source.get("source"):
            raise ValueError(
                f"CLASH action row {prompt_id} changes source provenance"
            )
        seen_ids.add(prompt_id)
        examples.append(
            {
                "id": prompt_id,
                "dilemma": contents[0],
                "assistant_answer": contents[1],
                "target_field": "action",
                "source": record.get("source"),
                "title": record.get("title"),
            }
        )
    if len(examples) != expected_count:
        raise ValueError(
            f"Expected {expected_count} CLASH action rows, found {len(examples)}"
        )

    result_manifest = dict(manifest)
    result_manifest.update(
        {
            "objective": "response-only causal LM on the exact CLASH action text",
            "records_path": str(path),
            "records_sha256": actual_hash,
            "release_manifest_path": str(manifest_path),
            "release_manifest_sha256": sha256_file(manifest_path),
        }
    )
    return examples, result_manifest


def load_training_examples(
    records_path: Path | str,
    *,
    training_arm: str,
    expected_count: int = 98,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the selected arm and return a common dilemma-example schema."""

    if training_arm == PROMPT_ONLY_ARM:
        return load_prompt_examples(records_path, expected_count=expected_count)
    if training_arm == CLASH_ACTION_ARM:
        return load_clash_action_examples(
            records_path,
            expected_count=expected_count,
        )
    return load_answer_examples(
        records_path,
        training_arm=training_arm,
        expected_count=expected_count,
    )
