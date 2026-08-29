"""Evaluate a saved H4rmony R1 adapter on the current ecological templates."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from scripts.harmony_eval.analysis import compare_thresholds, save_curve_plot
from scripts.harmony_eval.cases import DEFAULT_COST_COUNTS, REPO_ROOT, build_cases
from scripts.harmony_eval.scoring import score_loaded_causal_checkpoint

from .persistence import persist_directory_to_colab_drive


PAIR_NAME = "qwen3_8b_harmony_r1_sft"
DEFAULT_LOCAL_EVAL_ROOT = Path("/content/value-misalignment-posthoc-evals")
POSTHOC_PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class PosthocEvalArtifacts:
    output_dir: Path
    rendered_cases_path: Path
    raw_scores_path: Path
    thresholds_path: Path
    plot_path: Path
    metadata_path: Path
    complete_marker_path: Path


def artifacts_for_posthoc_eval(output_dir: Path | str) -> PosthocEvalArtifacts:
    output_dir = Path(output_dir)
    return PosthocEvalArtifacts(
        output_dir=output_dir,
        rendered_cases_path=output_dir / "rendered_cases.jsonl",
        raw_scores_path=output_dir / "raw_scores.csv",
        thresholds_path=output_dir / "thresholds.csv",
        plot_path=output_dir / "curves.png",
        metadata_path=output_dir / "metadata.json",
        complete_marker_path=output_dir / "COMPLETE.json",
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _required_hashes(artifacts: PosthocEvalArtifacts) -> dict[str, str]:
    required = {
        "raw_scores": artifacts.raw_scores_path,
        "thresholds": artifacts.thresholds_path,
        "curves": artifacts.plot_path,
        "metadata": artifacts.metadata_path,
    }
    # Older completed evaluations predate the deduplicated case manifest. New
    # runs include it in their hash contract without invalidating those bundles.
    if artifacts.rendered_cases_path.is_file():
        required["rendered_cases"] = artifacts.rendered_cases_path
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Post-hoc evaluation is missing artifacts: {missing}")
    return {name: _sha256_file(path) for name, path in required.items()}


def validate_posthoc_eval(output_dir: Path | str) -> dict[str, str]:
    """Verify a post-hoc evaluation against its completion hash manifest."""

    artifacts = artifacts_for_posthoc_eval(output_dir)
    if not artifacts.complete_marker_path.is_file():
        raise RuntimeError(
            f"Post-hoc evaluation has no COMPLETE.json: {artifacts.output_dir}"
        )
    try:
        marker = json.loads(
            artifacts.complete_marker_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Could not read a valid post-hoc COMPLETE.json") from exc
    if marker.get("status") != "complete":
        raise RuntimeError("Post-hoc COMPLETE.json does not report complete status")
    expected = marker.get("artifact_sha256")
    if not isinstance(expected, dict) or not expected:
        raise RuntimeError("Post-hoc COMPLETE.json has no artifact hash manifest")
    actual = _required_hashes(artifacts)
    if expected != actual:
        mismatches = sorted(
            name
            for name in set(actual) | set(expected)
            if actual.get(name) != expected.get(name)
        )
        raise RuntimeError(
            f"Post-hoc artifact hashes do not match COMPLETE.json: {mismatches}"
        )
    return actual


def _candidate_completion_time(output_dir: Path) -> str:
    try:
        marker = json.loads(
            (output_dir / "COMPLETE.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(marker, dict):
        return ""
    completed_at = marker.get("completed_at_utc")
    return completed_at if isinstance(completed_at, str) else ""


def _template_manifest(cases: list[dict[str, object]]) -> dict[str, object]:
    return {
        str(case["template"]): {
            "severity": case["severity"],
            "family": case["template_family"],
            "path": case["template_path"],
            "sha256": case["template_sha256"],
        }
        for case in cases
    }


def _evaluation_slug(template_names: Sequence[str] | None) -> str:
    """Return a stable directory suffix for a selected evaluation suite."""

    if template_names is None:
        return "mild_extreme_eval"
    selected_parts = tuple(Path(name).with_suffix("").parts for name in template_names)
    if selected_parts and all(
        parts[:2] == ("extreme_v2", "control") for parts in selected_parts
    ):
        return "extreme_v2_control_eval"
    selected_roots = {parts[0] for parts in selected_parts}
    if selected_roots == {"extreme_v2"}:
        return "extreme_v2_eval"
    return "selected_templates_eval"


def find_compatible_posthoc_eval(
    sft_run_dir: Path | str,
    *,
    cost_counts: Iterable[int] = DEFAULT_COST_COUNTS,
    template_names: Sequence[str] | None = None,
) -> PosthocEvalArtifacts | None:
    """Return the newest verified evaluation for the requested template suite."""

    sft_run_dir = Path(sft_run_dir).expanduser()
    source_complete_path = sft_run_dir / "COMPLETE.json"
    evaluations_root = sft_run_dir / "posthoc_evaluations"
    if not source_complete_path.is_file() or not evaluations_root.is_dir():
        return None

    counts = tuple(cost_counts)
    cases = build_cases(counts, template_names)
    expected_source_hash = _sha256_file(source_complete_path)
    expected_templates = _template_manifest(cases)
    candidates = [path for path in evaluations_root.iterdir() if path.is_dir()]
    candidates.sort(
        key=lambda path: (_candidate_completion_time(path), path.name),
        reverse=True,
    )

    for output_dir in candidates:
        artifacts = artifacts_for_posthoc_eval(output_dir)
        try:
            metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                continue
            if metadata.get("status") != "complete":
                continue
            if metadata.get("evaluation_protocol_version") != POSTHOC_PROTOCOL_VERSION:
                continue
            if metadata.get("source_complete_sha256") != expected_source_hash:
                continue
            if metadata.get("cost_counts") != list(counts):
                continue
            if metadata.get("templates") != expected_templates:
                continue
            if metadata.get("enable_thinking") is not False:
                continue
            validate_posthoc_eval(output_dir)
        except (OSError, json.JSONDecodeError, RuntimeError):
            continue
        return artifacts
    return None


def _adapter_weights(adapter_dir: Path) -> Path:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = adapter_dir / name
        if candidate.is_file():
            return candidate
    raise RuntimeError(f"No adapter weights found in {adapter_dir}")


def run_saved_adapter_eval(
    sft_run_dir: Path | str,
    *,
    output_root: Path | str = DEFAULT_LOCAL_EVAL_ROOT,
    cost_counts: Iterable[int] = DEFAULT_COST_COUNTS,
    template_names: Sequence[str] | None = None,
    batch_size: int | None = None,
) -> PosthocEvalArtifacts:
    """Score a completed base/adapter pair on the requested template suite."""

    sft_run_dir = Path(sft_run_dir).expanduser().resolve(strict=True)
    metadata_path = sft_run_dir / "run_metadata.json"
    adapter_dir = sft_run_dir / "final_adapter"
    complete_path = sft_run_dir / "COMPLETE.json"
    if not metadata_path.is_file() or not complete_path.is_file():
        raise RuntimeError(f"SFT run is not complete: {sft_run_dir}")
    adapter_weights = _adapter_weights(adapter_dir)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        raise RuntimeError("SFT run metadata does not report complete status")

    config = metadata["config"]
    base_model_id = str(config["base_model"])
    resolved_revisions = metadata["resolved_revisions"]
    base_revision = str(resolved_revisions[base_model_id])
    adapter_revision = str(
        resolved_revisions.get("final_adapter_sha256")
        or _sha256_file(adapter_weights)
    )
    if _sha256_file(adapter_weights) != adapter_revision:
        raise RuntimeError(
            "Saved adapter weights do not match the revision recorded in metadata"
        )
    eval_batch_size = int(batch_size or config.get("eval_batch_size", 4))
    if eval_batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    counts = tuple(cost_counts)
    cases = build_cases(counts, template_names)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = Path(output_root).expanduser() / sft_run_dir.name
    evaluation_slug = _evaluation_slug(template_names)
    output_dir = output_dir / f"{timestamp}_{evaluation_slug}"
    output_dir.mkdir(parents=True, exist_ok=False)
    artifacts = artifacts_for_posthoc_eval(output_dir)
    created_at_utc = _utc_now()

    try:
        _write_jsonl(artifacts.rendered_cases_path, cases)
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("Post-hoc Qwen3-8B evaluation requires a CUDA GPU")
        gc.collect()
        torch.cuda.empty_cache()

        tokenizer = AutoTokenizer.from_pretrained(
            base_model_id,
            revision=base_revision,
            use_fast=True,
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise RuntimeError("Qwen tokenizer has neither a pad nor EOS token")
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            revision=base_revision,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        model.to("cuda")
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False
        model.eval()

        base_rows = score_loaded_causal_checkpoint(
            model=model,
            tokenizer=tokenizer,
            cases=cases,
            model_role="base",
            model_id=base_model_id,
            model_revision=base_revision,
            pair_name=PAIR_NAME,
            training_method="sft",
            batch_size=eval_batch_size,
            enable_thinking=False,
        )
        _write_csv(output_dir / "raw_scores_base.csv", base_rows)

        model = PeftModel.from_pretrained(
            model,
            str(adapter_dir),
            is_trainable=False,
        )
        model.eval()
        aligned_rows = score_loaded_causal_checkpoint(
            model=model,
            tokenizer=tokenizer,
            cases=cases,
            model_role="aligned",
            model_id=str(adapter_dir),
            model_revision=adapter_revision,
            pair_name=PAIR_NAME,
            training_method="sft",
            batch_size=eval_batch_size,
            enable_thinking=False,
        )

        rows = base_rows + aligned_rows
        _write_csv(artifacts.raw_scores_path, rows)
        _write_csv(artifacts.thresholds_path, compare_thresholds(rows))
        save_curve_plot(rows, artifacts.plot_path)
        template_manifest = _template_manifest(cases)
        _write_json(
            artifacts.metadata_path,
            {
                "status": "complete",
                "evaluation_protocol_version": POSTHOC_PROTOCOL_VERSION,
                "created_at_utc": created_at_utc,
                "completed_at_utc": _utc_now(),
                "repository_commit": _git_commit(),
                "source_sft_run": str(sft_run_dir),
                "source_complete_sha256": _sha256_file(complete_path),
                "base_model": base_model_id,
                "base_revision": base_revision,
                "adapter_path": str(adapter_dir),
                "adapter_revision": adapter_revision,
                "cost_counts": list(counts),
                "case_count_per_model": len(cases),
                "batch_size": eval_batch_size,
                "enable_thinking": False,
                "templates": template_manifest,
                "threshold_definition": (
                    "P(implement)=0.5 after a nonincreasing PAVA fit; "
                    "interpolation is linear in log(1 + cost)."
                ),
            },
        )
        hashes = _required_hashes(artifacts)
        _write_json(
            artifacts.complete_marker_path,
            {
                "status": "complete",
                "completed_at_utc": _utc_now(),
                "artifact_sha256": hashes,
            },
        )
        validate_posthoc_eval(output_dir)
        return artifacts
    except Exception as exc:
        _write_json(
            output_dir / "FAILED.json",
            {
                "status": "failed",
                "failed_at_utc": _utc_now(),
                "exception_type": type(exc).__name__,
                "message": str(exc),
            },
        )
        raise


def persist_posthoc_eval_to_colab_drive(
    artifacts: PosthocEvalArtifacts,
    sft_run_dir: Path | str,
    **persistence_kwargs: object,
) -> PosthocEvalArtifacts:
    """Persist post-hoc results beneath the source run and verify a fresh mount."""

    drive_output_root = Path(sft_run_dir) / "posthoc_evaluations"
    destination = persist_directory_to_colab_drive(
        artifacts.output_dir,
        drive_output_root,
        validate_directory=validate_posthoc_eval,
        **persistence_kwargs,
    )
    return artifacts_for_posthoc_eval(destination)
