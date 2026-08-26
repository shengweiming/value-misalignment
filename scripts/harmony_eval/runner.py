"""End-to-end runner and artifact writer for a matched checkpoint pair."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable, Sequence

from .analysis import compare_thresholds, save_curve_plot
from .cases import DEFAULT_COST_COUNTS, REPO_ROOT, SYSTEM_PROMPT, build_cases
from .catalog import get_checkpoint_pair
from .scoring import (
    load_shared_tokenizer,
    resolve_revisions,
    score_checkpoint,
    validate_pair_vocabularies,
)


@dataclass(frozen=True)
class RunArtifacts:
    output_dir: Path
    rendered_cases_path: Path
    raw_scores_path: Path
    thresholds_path: Path
    plot_path: Path
    metadata_path: Path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def run_checkpoint_pair(
    pair_name: str = "caramel_sft",
    *,
    cost_counts: Iterable[int] = DEFAULT_COST_COUNTS,
    template_names: Sequence[str] | None = None,
    output_root: Path | str = REPO_ROOT / "outputs/harmony_eval",
    load_in_4bit: bool | None = None,
    batch_size: int = 4,
) -> RunArtifacts:
    """Score base and aligned checkpoints sequentially and save reproducible artifacts."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    pair = get_checkpoint_pair(pair_name)
    quantized = pair.default_load_in_4bit if load_in_4bit is None else load_in_4bit
    if pair.architecture == "seq2seq" and quantized:
        raise ValueError("This runner does not use 4-bit loading for the seq2seq pair")

    counts = tuple(cost_counts)
    cases = build_cases(counts, template_names)
    revisions = resolve_revisions(
        [pair.base_model, pair.aligned_model, pair.tokenizer_model]
    )
    validate_pair_vocabularies(pair, revisions)
    tokenizer = load_shared_tokenizer(pair, revisions[pair.tokenizer_model])

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = Path(output_root) / f"{timestamp}_{pair.name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    rendered_cases_path = run_dir / "rendered_cases.jsonl"
    raw_path = run_dir / "raw_scores.csv"
    threshold_path = run_dir / "thresholds.csv"
    plot_path = run_dir / "curves.png"
    metadata_path = run_dir / "metadata.json"
    _write_jsonl(rendered_cases_path, cases)

    rows: list[dict[str, object]] = []
    for role, model_id in (("base", pair.base_model), ("aligned", pair.aligned_model)):
        rows.extend(
            score_checkpoint(
                pair=pair,
                model_id=model_id,
                model_role=role,
                revision=revisions[model_id],
                tokenizer=tokenizer,
                cases=cases,
                load_in_4bit=quantized,
                batch_size=batch_size,
            )
        )
        _write_csv(raw_path, rows)

    comparisons = compare_thresholds(rows)
    _write_csv(threshold_path, comparisons)
    save_curve_plot(rows, plot_path)

    unique_templates = {
        str(case["template"]): {
            "path": case["template_path"],
            "sha256": case["template_sha256"],
        }
        for case in cases
    }
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": _git_commit(),
        "pair": pair.as_dict(),
        "resolved_revisions": revisions,
        "shared_tokenizer": {
            "model_id": pair.tokenizer_model,
            "revision": revisions[pair.tokenizer_model],
        },
        "system_prompt": SYSTEM_PROMPT,
        "templates": unique_templates,
        "cost_counts": counts,
        "case_count_per_model": len(cases),
        "load_in_4bit": quantized,
        "batch_size": batch_size,
        "threshold_definition": (
            "P(implement)=0.5 after a nonincreasing PAVA fit; interpolation is linear "
            "in log(1 + cost)."
        ),
        "python": platform.python_version(),
        "packages": {
            package: _package_version(package)
            for package in (
                "torch",
                "transformers",
                "accelerate",
                "bitsandbytes",
                "huggingface-hub",
            )
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return RunArtifacts(
        output_dir=run_dir,
        rendered_cases_path=rendered_cases_path,
        raw_scores_path=raw_path,
        thresholds_path=threshold_path,
        plot_path=plot_path,
        metadata_path=metadata_path,
    )
