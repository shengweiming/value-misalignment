#!/usr/bin/env python3
"""Build the first-stage MoReBench prompt-only control candidate pool."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    REPO_ROOT
    / "data"
    / "control_dilemmas"
    / "morebench"
    / "source"
    / "morebench_public.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "data" / "control_dilemmas" / "morebench" / "v1_candidates"
)

DATASET_ID = "morebench/morebench"
DATASET_REVISION = "8290fafe65d595aaa28315b50ec4b64da6d3bd5e"
SOURCE_FILENAME = "morebench_public.csv"
SOURCE_SHA256 = "e56d627823066876c6710a91144d0d9faebc1503dcf9b665f58c87b0eddd2229"
EXPECTED_SOURCE_ROWS = 500
EXPECTED_FIELDS = (
    "DILEMMA",
    "DILEMMA_SOURCE",
    "DILEMMA_TYPE",
    "THEORY",
    "RUBRIC",
    "ROLE_DOMAIN",
    "CONTEXT",
)

INCLUDED_CONTEXTS = (
    "Education",
    "Entertainment",
    "Interpersonal relationship",
)
TARGET_RELEASE_COUNT = 98

# These decisions were made after reading every lexical hit in the three-context
# pool. Hashes pin the decisions to the dilemma text instead of only to CSV order.
ECOLOGY_EXCLUSIONS = {
    "d1ce77b3daae3d730d4c3728dd235a781a848c3d7c9ff8d6d025ddaff7c3a87c": (
        "The scenario is framed by environmentally conscious consumption, green "
        "products, and benefits to the planet."
    ),
    "ac923972834c88021f2246e448d5983a17c2fac92b498ca74bc8a080518809a3": (
        "The relationship conflict is explicitly organized around animal rights, "
        "an adjacent nonhuman-value domain."
    ),
    "704086097033e1895310535107c1b4d94e3b9954c65e8233837247678db7a43c": (
        "One side of the dilemma is exposing a resort that is destroying the "
        "environment."
    ),
}

ECOLOGY_FALSE_POSITIVES = {
    "74be9af89072cab89eec3737c3e1b27f32ed3fcb3c805d6b34f3795705729020": (
        "Uses 'environment' for a school setting and a supportive learning setting."
    ),
    "89652fd627761de7f879411148a89314b9c99125790d834cfdc493782516ba77": (
        "Uses 'environment' only in the phrase 'virtual learning environments.'"
    ),
    "51f926356299b592c0d0d6fc5e73fc90bbccf9fced7e50107eb0e5369cea8f11": (
        "Uses 'home environment' to describe a student's family circumstances."
    ),
    "a015ba2c43cf3dfebf7e797eba4406099e8435390b0baea0def4594eb493f54b": (
        "Uses 'environment' for the social atmosphere within a friend group."
    ),
    "de10d1a1068d4366789697de0dc3fc447858b60e98fbcfca1efa528ad02fe92d": (
        "Uses 'national climate' and 'environment' for political polarization "
        "and the college's organizational atmosphere."
    ),
}

ECOLOGY_SCREEN_TERMS = (
    "animal",
    "biodiversity",
    "carbon",
    "climate",
    "conservation",
    "ecological",
    "ecology",
    "ecosystem",
    "emissions",
    "environment",
    "environmental",
    "forest",
    "fossil",
    "habitat",
    "natural resources",
    "ocean",
    "pollution",
    "renewable",
    "species",
    "sustainability",
    "sustainable",
    "wetland",
    "wildlife",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def load_source(path: Path) -> list[dict[str, str]]:
    observed_sha256 = sha256_file(path)
    if observed_sha256 != SOURCE_SHA256:
        raise ValueError(
            f"Unexpected MoReBench source SHA-256: {observed_sha256}; "
            f"expected {SOURCE_SHA256}"
        )

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != EXPECTED_FIELDS:
            raise ValueError(f"Unexpected MoReBench fields: {reader.fieldnames}")
        rows = list(reader)

    if len(rows) != EXPECTED_SOURCE_ROWS:
        raise ValueError(
            f"Unexpected MoReBench row count: {len(rows)}; "
            f"expected {EXPECTED_SOURCE_ROWS}"
        )
    return rows


def ecology_screen_hits(dilemma: str) -> list[str]:
    lowered = dilemma.lower()
    return [term for term in ECOLOGY_SCREEN_TERMS if term in lowered]


def describe_counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def word_statistics(rows: list[dict[str, Any]]) -> dict[str, int | float]:
    counts = [int(row["word_count"]) for row in rows]
    if not counts:
        return {
            "count": 0,
            "minimum": 0,
            "median": 0,
            "mean": 0,
            "maximum": 0,
            "total": 0,
        }
    return {
        "count": len(counts),
        "minimum": min(counts),
        "median": statistics.median(counts),
        "mean": round(statistics.mean(counts), 3),
        "maximum": max(counts),
        "total": sum(counts),
    }


def build(source_path: Path, output_dir: Path) -> dict[str, Any]:
    source_rows = load_source(source_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    audit_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    for source_index, row in enumerate(source_rows):
        dilemma = row["DILEMMA"]
        dilemma_sha256 = sha256_text(dilemma)
        included_context = row["CONTEXT"] in INCLUDED_CONTEXTS
        ecology_exclusion = ECOLOGY_EXCLUSIONS.get(dilemma_sha256)
        ecology_false_positive = ECOLOGY_FALSE_POSITIVES.get(dilemma_sha256)
        screen_hits = ecology_screen_hits(dilemma)
        word_count = len(dilemma.split())

        if not included_context:
            disposition = "exclude_context"
            exclusion_reason = (
                "Context is outside the three-context first-stage filter."
            )
        elif ecology_exclusion:
            disposition = "exclude_ecology_overlap"
            exclusion_reason = ecology_exclusion
        else:
            disposition = "eligible"
            exclusion_reason = None

        audit_rows.append(
            {
                "context": row["CONTEXT"],
                "dilemma_sha256": dilemma_sha256,
                "dilemma_source": row["DILEMMA_SOURCE"],
                "dilemma_type": row["DILEMMA_TYPE"],
                "disposition": disposition,
                "ecology_screen_hits": screen_hits,
                "ecology_screen_review": ecology_exclusion or ecology_false_positive,
                "exclusion_reason": exclusion_reason,
                "role_domain": row["ROLE_DOMAIN"],
                "source_index": source_index,
                "word_count": word_count,
            }
        )

        if disposition != "eligible":
            continue

        candidate_rows.append(
            {
                "context": row["CONTEXT"],
                "dilemma": dilemma,
                "dilemma_source": row["DILEMMA_SOURCE"],
                "dilemma_type": row["DILEMMA_TYPE"],
                "id": f"morebench-public-{source_index:03d}",
                "role_domain": row["ROLE_DOMAIN"],
                "source": {
                    "dataset": DATASET_ID,
                    "dilemma_sha256": dilemma_sha256,
                    "file": SOURCE_FILENAME,
                    "revision": DATASET_REVISION,
                    "row_index": source_index,
                },
                "theory": row["THEORY"],
                "word_count": word_count,
            }
        )

    context_pool = [
        row for row in audit_rows if row["disposition"] != "exclude_context"
    ]
    excluded_ecology = [
        row for row in audit_rows if row["disposition"] == "exclude_ecology_overlap"
    ]

    if len(context_pool) != 115:
        raise ValueError(
            f"Expected 115 rows in the three-context pool, got {len(context_pool)}"
        )
    if len(excluded_ecology) != len(ECOLOGY_EXCLUSIONS):
        raise ValueError(
            "Not every pinned ecology-overlap decision matched a source dilemma"
        )
    if len(candidate_rows) != 112:
        raise ValueError(f"Expected 112 eligible candidates, got {len(candidate_rows)}")
    unreviewed_hits = [
        row
        for row in audit_rows
        if row["disposition"] != "exclude_context"
        and row["ecology_screen_hits"]
        and not row["ecology_screen_review"]
    ]
    if unreviewed_hits:
        raise ValueError(
            "Every ecology-screen hit requires a pinned manual review; "
            "unreviewed source indices: "
            f"{[row['source_index'] for row in unreviewed_hits]}"
        )

    candidates_path = output_dir / "candidates.jsonl"
    audit_path = output_dir / "audit.jsonl"
    manifest_path = output_dir / "manifest.json"
    write_jsonl(candidates_path, candidate_rows)
    write_jsonl(audit_path, audit_rows)

    context_pool_records = [
        {
            "CONTEXT": row["context"],
            "DILEMMA_SOURCE": row["dilemma_source"],
            "DILEMMA_TYPE": row["dilemma_type"],
            "ROLE_DOMAIN": row["role_domain"],
            "word_count": row["word_count"],
        }
        for row in context_pool
    ]
    manifest: dict[str, Any] = {
        "artifacts": {
            "audit.jsonl": sha256_file(audit_path),
            "candidates.jsonl": sha256_file(candidates_path),
        },
        "candidate_pool": {
            "context_counts": describe_counts(candidate_rows, "context"),
            "dilemma_source_counts": describe_counts(candidate_rows, "dilemma_source"),
            "dilemma_type_counts": describe_counts(candidate_rows, "dilemma_type"),
            "eligible_count": len(candidate_rows),
            "role_domain_counts": describe_counts(candidate_rows, "role_domain"),
            "word_statistics": word_statistics(candidate_rows),
        },
        "contains_assistant_responses": False,
        "contains_normative_labels": False,
        "dataset_type": "prompt_only_candidate_pool",
        "format_version": 1,
        "screening": {
            "context_counts_before_ecology_screen": describe_counts(
                context_pool_records, "CONTEXT"
            ),
            "contexts": list(INCLUDED_CONTEXTS),
            "ecology_overlap_excluded_count": len(excluded_ecology),
            "ecology_overlap_excluded_source_indices": [
                row["source_index"] for row in excluded_ecology
            ],
            "three_context_count": len(context_pool),
        },
        "selection": {
            "final_selection_frozen": False,
            "surplus_over_target": len(candidate_rows) - TARGET_RELEASE_COUNT,
            "target_release_count": TARGET_RELEASE_COUNT,
        },
        "source": {
            "dataset": DATASET_ID,
            "file": SOURCE_FILENAME,
            "license": "CC-BY-4.0",
            "revision": DATASET_REVISION,
            "row_count": len(source_rows),
            "sha256": SOURCE_SHA256,
        },
        "training_field": "dilemma",
    }
    write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build(args.source, args.output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
