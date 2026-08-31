#!/usr/bin/env python3
"""Build ecological- and human-option SFT chats from the audited dilemmas."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_RECORDS = REPO_ROOT / "data/ecological_dilemmas/v1/records.jsonl"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/ecological_dilemmas/sft"
EXPECTED_COUNT = 98
ARMS = {
    "ecological_option": "ecologically_protective_option",
    "human_option": "human_protective_option",
}


class DatasetBuildError(RuntimeError):
    """Raised when the audited source or a generated answer arm is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    """Use stable repository-relative paths when possible."""

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


def load_verified_source(source_records: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_records = source_records.resolve(strict=True)
    manifest_path = source_records.parent / "manifest.json"
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise DatasetBuildError("The audited release manifest must be an object")
    if manifest.get("dataset_type") != "prompt_only":
        raise DatasetBuildError("The source release is not the audited prompt-only release")
    if int(manifest.get("released_count") or -1) != EXPECTED_COUNT:
        raise DatasetBuildError(
            f"Expected {EXPECTED_COUNT} audited records, found "
            f"{manifest.get('released_count')!r}"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise DatasetBuildError("The source manifest has no artifact hash table")
    actual_hash = sha256_file(source_records)
    if artifacts.get("records.jsonl") != actual_hash:
        raise DatasetBuildError("The audited records do not match their manifest hash")

    records = read_jsonl(source_records)
    if len(records) != EXPECTED_COUNT:
        raise DatasetBuildError(
            f"Expected {EXPECTED_COUNT} source rows, found {len(records)}"
        )
    seen_ids: set[str] = set()
    for index, record in enumerate(records, 1):
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise DatasetBuildError(f"Source row {index} has no stable ID")
        if record_id in seen_ids:
            raise DatasetBuildError(f"Duplicate source ID: {record_id}")
        seen_ids.add(record_id)
        for field in ("dilemma", *ARMS.values()):
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                raise DatasetBuildError(f"Source record {record_id} has no {field}")
    return records, {
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "records_path": source_records,
        "records_sha256": actual_hash,
    }


def build_answer_arm(
    *,
    arm: str,
    target_field: str,
    source_records: list[dict[str, Any]],
    source_metadata: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    output_dir = output_root / arm
    answer_records = []
    for source_record in source_records:
        answer_records.append(
            {
                "id": source_record["id"],
                "messages": [
                    {"role": "user", "content": source_record["dilemma"]},
                    {
                        "role": "assistant",
                        "content": source_record[target_field],
                    },
                ],
                "source": source_record.get("source"),
                "target_field": target_field,
                "title": source_record.get("title"),
            }
        )

    records_path = output_dir / "records.jsonl"
    write_jsonl(records_path, answer_records)
    manifest = {
        "format_version": 1,
        "dataset_type": "assistant_option_sft",
        "training_arm": arm,
        "assistant_target_field": target_field,
        "response_style": "option_text_only",
        "example_count": len(answer_records),
        "contains_normative_labels": True,
        "contains_assistant_responses": True,
        "contains_rationales": False,
        "source_release": {
            "manifest_path": display_path(source_metadata["manifest_path"]),
            "manifest_sha256": source_metadata["manifest_sha256"],
            "records_path": display_path(source_metadata["records_path"]),
            "records_sha256": source_metadata["records_sha256"],
        },
        "artifacts": {"records.jsonl": sha256_file(records_path)},
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def build_datasets(
    *,
    source_records_path: Path = DEFAULT_SOURCE_RECORDS,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, dict[str, Any]]:
    source_records, source_metadata = load_verified_source(source_records_path)
    return {
        arm: build_answer_arm(
            arm=arm,
            target_field=target_field,
            source_records=source_records,
            source_metadata=source_metadata,
            output_root=output_root,
        )
        for arm, target_field in ARMS.items()
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-records", type=Path, default=DEFAULT_SOURCE_RECORDS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifests = build_datasets(
        source_records_path=args.source_records,
        output_root=args.output_root,
    )
    for arm, manifest in manifests.items():
        print(
            f"Built {arm}: {manifest['example_count']} exact option-only "
            "assistant responses"
        )


if __name__ == "__main__":
    main()
