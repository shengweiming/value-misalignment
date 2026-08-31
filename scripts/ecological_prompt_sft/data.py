"""Load the audited ecological dilemmas without inventing answer supervision."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        "objective": "causal language modeling on the dilemma text",
        "example_count": len(examples),
        "records_path": str(path),
        "records_sha256": actual_hash,
        "release_manifest_path": str(manifest_path),
        "release_manifest_sha256": sha256_file(manifest_path),
        "contains_normative_labels": False,
        "contains_assistant_responses": False,
    }
