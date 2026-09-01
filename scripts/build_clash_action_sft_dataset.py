#!/usr/bin/env python3
"""Build exact-action response-only SFT chats for the audited CLASH controls."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from scripts.build_clash_prompt_control_sft_dataset import (
    DATASET_ID,
    DATASET_REVISION,
    DEFAULT_SOURCE,
    SOURCE_SHA256,
    load_source,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_RECORDS = (
    REPO_ROOT / "data" / "control_dilemmas" / "clash" / "v1" / "records.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "data" / "control_dilemmas" / "clash" / "sft" / "action"
)
EXPECTED_COUNT = 98

RELEASE_README = """# CLASH exact-action response-only SFT control

This release pairs each of the 98 audited non-ecological CLASH dilemmas in
`../../v1/records.jsonl` with the exact `action` field from the pinned public
CLASH source snapshot. The user turn is byte-for-byte identical to the prompt-only
control. The assistant turn contains only the extracted action phrase: no
acceptable or unacceptable rationale, character perspective, explanation, or
added punctuation is included.

The CLASH action identifies the behavior under ethical consideration. It is not
a preferred answer or a claim that the behavior is morally acceptable. This arm
therefore tests response-only supervision on a short, unrelated focal-action
label; it should not be described as an action-correctness or action-endorsement
dataset.

Training must render the unchanged dilemma as one Qwen `user` message with an
assistant generation prefix, append the exact action and EOS token, mask the
complete user/generation prefix to `-100`, and apply loss only to the action and
EOS. The release contains 98 unique action strings. Their whitespace word-count
range is 2--18, their median is 5, their mean is 5.704, and their total is 559.
"""


class DatasetBuildError(RuntimeError):
    """Raised when the pinned prompt release or CLASH snapshot is inconsistent."""


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DatasetBuildError(f"Required file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DatasetBuildError(f"Invalid JSON: {path}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise DatasetBuildError(f"Required file not found: {path}") from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetBuildError(f"Invalid JSON at {path}:{line_number}") from exc
        if not isinstance(record, dict):
            raise DatasetBuildError(f"Expected an object at {path}:{line_number}")
        records.append(record)
    return records


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(
                json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
            )


def load_verified_prompts(
    prompt_records_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompt_records_path = prompt_records_path.resolve(strict=True)
    manifest_path = prompt_records_path.parent / "manifest.json"
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise DatasetBuildError("The CLASH prompt manifest must be an object")
    expected_manifest = {
        "dataset_type": "prompt_only",
        "released_count": EXPECTED_COUNT,
        "contains_actions": False,
        "contains_assistant_responses": False,
        "contains_normative_labels": False,
        "contains_rationales": False,
    }
    mismatches = [
        key for key, expected in expected_manifest.items() if manifest.get(key) != expected
    ]
    if mismatches:
        raise DatasetBuildError(
            f"The CLASH prompt release has unexpected metadata: {mismatches}"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise DatasetBuildError("The CLASH prompt manifest has no artifact hashes")
    records_hash = sha256_file(prompt_records_path)
    if artifacts.get("records.jsonl") != records_hash:
        raise DatasetBuildError("The CLASH prompts do not match their manifest hash")
    prompts = read_jsonl(prompt_records_path)
    if len(prompts) != EXPECTED_COUNT:
        raise DatasetBuildError(
            f"Expected {EXPECTED_COUNT} CLASH prompts, found {len(prompts)}"
        )
    ids = [record.get("id") for record in prompts]
    if any(not isinstance(record_id, str) or not record_id for record_id in ids):
        raise DatasetBuildError("Every CLASH prompt must have a stable ID")
    if len(set(ids)) != len(ids):
        raise DatasetBuildError("The CLASH prompt release contains duplicate IDs")
    return prompts, {
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "records_path": prompt_records_path,
        "records_sha256": records_hash,
    }


def action_word_statistics(actions: list[str]) -> dict[str, int | float]:
    counts = [len(action.split()) for action in actions]
    return {
        "count": len(counts),
        "minimum": min(counts),
        "median": statistics.median(counts),
        "mean": round(statistics.mean(counts), 3),
        "maximum": max(counts),
        "total": sum(counts),
    }


def build(
    *,
    prompt_records_path: Path = DEFAULT_PROMPT_RECORDS,
    source_path: Path = DEFAULT_SOURCE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    prompts, prompt_metadata = load_verified_prompts(prompt_records_path)
    source_path = source_path.resolve(strict=True)
    source_rows = load_source(source_path)
    source_by_id = {row["id"]: row for row in source_rows}

    action_records: list[dict[str, Any]] = []
    actions: list[str] = []
    for prompt in prompts:
        record_id = str(prompt["id"])
        provenance = prompt.get("source")
        if not isinstance(provenance, dict):
            raise DatasetBuildError(f"Prompt {record_id} has no source provenance")
        source_id = provenance.get("source_id")
        source_row = source_by_id.get(str(source_id))
        if source_row is None:
            raise DatasetBuildError(f"Prompt {record_id} has no CLASH source row")
        if prompt.get("dilemma") != source_row["situation"]:
            raise DatasetBuildError(f"Prompt {record_id} changes the source situation")
        if prompt.get("title") != source_row["title"]:
            raise DatasetBuildError(f"Prompt {record_id} changes the source title")
        action = source_row["action"]
        if not action.strip():
            raise DatasetBuildError(f"CLASH source row {source_id} has no action")
        actions.append(action)
        action_records.append(
            {
                "id": record_id,
                "messages": [
                    {"role": "user", "content": prompt["dilemma"]},
                    {"role": "assistant", "content": action},
                ],
                "source": provenance,
                "target_field": "action",
                "title": prompt.get("title"),
            }
        )
    if len(action_records) != EXPECTED_COUNT:
        raise DatasetBuildError(
            f"Expected {EXPECTED_COUNT} action records, found {len(action_records)}"
        )
    if len(set(actions)) != EXPECTED_COUNT:
        raise DatasetBuildError("The selected CLASH actions are not all unique")

    output_dir.mkdir(parents=True, exist_ok=True)
    readme_path = output_dir / "README.md"
    records_path = output_dir / "records.jsonl"
    readme_path.write_text(RELEASE_README, encoding="utf-8")
    write_jsonl(records_path, action_records)
    manifest = {
        "format_version": 1,
        "dataset_type": "assistant_action_sft",
        "training_arm": "action",
        "assistant_target_field": "action",
        "response_style": "action_text_only",
        "example_count": len(action_records),
        "contains_actions": True,
        "contains_assistant_responses": True,
        "contains_character_perspectives": False,
        "contains_normative_labels": False,
        "contains_rationales": False,
        "action_interpretation": (
            "The focal action extracted by CLASH; not a preferred, correct, or "
            "morally acceptable answer label."
        ),
        "action_word_counts": action_word_statistics(actions),
        "source_release": {
            "manifest_path": display_path(prompt_metadata["manifest_path"]),
            "manifest_sha256": prompt_metadata["manifest_sha256"],
            "records_path": display_path(prompt_metadata["records_path"]),
            "records_sha256": prompt_metadata["records_sha256"],
        },
        "source_snapshot": {
            "dataset": DATASET_ID,
            "revision": DATASET_REVISION,
            "path": display_path(source_path),
            "sha256": SOURCE_SHA256,
        },
        "artifacts": {
            "README.md": sha256_file(readme_path),
            "records.jsonl": sha256_file(records_path),
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-records", type=Path, default=DEFAULT_PROMPT_RECORDS)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = build(
        prompt_records_path=args.prompt_records,
        source_path=args.source,
        output_dir=args.output_dir,
    )
    print(
        f"Built {manifest['example_count']} exact CLASH action responses at "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()
