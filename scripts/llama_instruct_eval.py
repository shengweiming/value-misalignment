"""Evaluate Meta's official Llama 3.1 8B Instruct checkpoint on current suites.

This is a single full model, without PEFT. The authors' released-adapter runner
and its existing result signatures remain independent.
"""

from __future__ import annotations

import csv
import gc
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from scripts.ecological_prompt_sft.numeric_evaluation import (
    NUMERIC_COST_COUNTS, average_numeric_threshold_probabilities,
    summarize_numeric_threshold_rows,
)
from scripts.ecological_prompt_sft.abstention_evaluation import (
    ABSTENTION_EVALUATION_SLUG, SEMANTIC_VALUES, build_abstention_cases,
    summarize_abstention_rows,
)
from scripts.harmony_eval.validation import check_fields as _check_fields, validate_case_rows
from scripts.harmony_eval.cases import DEFAULT_COST_COUNTS, REPO_ROOT, SYSTEM_PROMPT
from scripts.harmony_eval.scoring import (
    score_loaded_causal_candidates, score_loaded_causal_checkpoint,
)
from scripts.harmony_sft.persistence import persist_directory_to_colab_drive
from scripts.harmony_sft.posthoc_eval import (
    PosthocEvalArtifacts, _case_set_sha256, _git_commit, _required_hashes,
    _sha256_file, _template_manifest, _utc_now, _write_csv, _write_json,
    _write_jsonl, artifacts_for_posthoc_eval, validate_posthoc_eval,
)
from scripts.released_environment_eval import (
    SUITES as LEGACY_SUITES, _environment, _read_rows, audit_tokenizer, choice_summary,
    current_cases as legacy_cases,
)

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
SOURCE_RUN_NAME = "standard_llama31_8b_instruct"
PAIR_NAME = "llama31_8b_instruct"
CONDITION = "llama_instruct"
PROTOCOL = "official_llama31_instruct_bf16_fp32_logprobs_v1"
SUITES = {**LEGACY_SUITES, "abstention": ABSTENTION_EVALUATION_SLUG}


def current_cases() -> dict[str, list[dict]]:
    return {**legacy_cases(), "abstention": build_abstention_cases()}


def prepare_instruct_tokenizer(*, token: str | None = None) -> tuple[object, dict]:
    """Load the checkpoint's own pinned tokenizer and native chat template."""
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    names = ("config.json", "tokenizer.json", "tokenizer_config.json",
             "special_tokens_map.json", "chat_template.jinja")
    directory = Path(snapshot_download(
        MODEL_ID, revision=MODEL_REVISION, allow_patterns=list(names), token=token,
    ))
    for name in names[:3]:
        if not (directory / name).is_file():
            raise RuntimeError(f"Official Instruct checkpoint is missing {name}")
    tokenizer = AutoTokenizer.from_pretrained(directory, use_fast=True, local_files_only=True)
    audit = audit_tokenizer(tokenizer, current_cases())
    audit["file_sha256"] = {name: _sha256_file(directory / name) for name in names if (directory / name).is_file()}
    return tokenizer, audit


def _identity() -> dict:
    return {
        "protocol": PROTOCOL, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "adapter": None, "dtype": "bfloat16", "logprob_dtype": "float32",
        "load_in_4bit": False, "attention": "sdpa", "enable_thinking": False,
        "system_prompt": SYSTEM_PROMPT, "seed": 42,
    }


def _signature(audit: dict, batch_size: int, environment: dict) -> dict:
    paths = (
        "scripts/llama_instruct_eval.py", "scripts/released_environment_eval.py",
        "scripts/harmony_eval/scoring.py", "scripts/harmony_eval/cases.py",
        "scripts/harmony_eval/validation.py",
        "scripts/ecological_prompt_sft/readout_evaluation.py",
        "scripts/ecological_prompt_sft/numeric_evaluation.py",
        "scripts/ecological_prompt_sft/abstention_evaluation.py",
    )
    return {
        **_identity(), "tokenizer_audit": audit, "batch_size": batch_size,
        "environment": environment,
        "implementation_sha256": {p: _sha256_file(REPO_ROOT / p) for p in paths},
        "case_set_sha256": {k: _case_set_sha256(v) for k, v in current_cases().items()},
    }


def _summary(rows: list[dict], suite: str) -> list[dict]:
    if suite == "abstention":
        return summarize_abstention_rows(rows)
    return summarize_numeric_threshold_rows(rows) if suite == "numeric" else choice_summary(rows)


def validate_instruct_bundle(artifacts: PosthocEvalArtifacts, *, signature: dict | None = None) -> dict:
    """Check hashes, checkpoint identity, the full single-model matrix and math."""
    validate_posthoc_eval(artifacts.output_dir)
    metadata = json.loads(artifacts.metadata_path.read_text())
    recorded = metadata.get("checkpoint_signature")
    if not isinstance(recorded, dict) or any(k not in recorded or recorded[k] != v for k, v in _identity().items()):
        raise RuntimeError("Incorrect official Instruct checkpoint or precision identity")
    if signature is not None and signature != recorded:
        raise RuntimeError("Instruct evaluation signature changed")
    suite = metadata.get("suite")
    if suite not in SUITES:
        raise RuntimeError("Unknown Instruct evaluation suite")
    cases = current_cases()[suite]
    _check_fields(metadata, {
        "status": "complete", "evaluation_kind": CONDITION, "pair_name": PAIR_NAME,
        "evaluation_slug": SUITES[suite], "case_count_per_model": len(cases),
        "case_set_sha256": _case_set_sha256(cases),
    })
    if (metadata.get("templates") != _template_manifest(cases)
            or metadata.get("cost_counts") != list(NUMERIC_COST_COUNTS if suite == "numeric" else DEFAULT_COST_COUNTS)):
        raise RuntimeError("Instruct template or cost-grid metadata mismatch")
    if metadata.get("model_roles") != {CONDITION: MODEL_ID}:
        raise RuntimeError("Instruct results must contain exactly one official model")
    if suite == "abstention" and (metadata.get("semantic_values") != SEMANTIC_VALUES
                                   or metadata.get("permutation_count") != 6):
        raise RuntimeError("Incorrect abstention semantic mapping or permutation count")
    if recorded.get("case_set_sha256", {}).get(suite) != _case_set_sha256(cases):
        raise RuntimeError("Instruct case signature mismatch")
    saved_cases = [json.loads(line) for line in artifacts.rendered_cases_path.read_text().splitlines()]
    if saved_cases != cases:
        raise RuntimeError("Instruct case manifest differs from the current questions")
    rows = _read_rows(artifacts)
    if metadata.get("score_row_count") != len(rows):
        raise RuntimeError("Incomplete Instruct score matrix")
    validate_case_rows(rows, cases, {
        "pair_name": PAIR_NAME, "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION, "model_role": CONDITION,
        "condition": CONDITION, "training_method": "official_instruct", "load_in_4bit": False,
    })
    with artifacts.thresholds_path.open(newline="") as file:
        summaries = list(csv.DictReader(file))
    recomputed = _summary(rows, suite)
    if len(summaries) != len(recomputed):
        raise RuntimeError("Incomplete Instruct summary")
    for row, expected_row in zip(summaries, recomputed):
        _check_fields(row, expected_row)
    return metadata


def _score_instruct(tokenizer, suites: tuple[str, ...], *, batch_size: int, token: str | None) -> dict:
    import torch
    from transformers import AutoModelForCausalLM

    model = None
    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"Loading official checkpoint: {MODEL_ID} @ {MODEL_REVISION}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, token=token, dtype=torch.bfloat16,
            attn_implementation="sdpa", device_map={"": 0}, low_cpu_mem_usage=True,
        )
        if getattr(model, "peft_config", None):
            raise RuntimeError("Official Instruct mode cannot load a PEFT adapter")
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False
        model.requires_grad_(False)
        model.eval()
        if any(p.dtype != torch.bfloat16 for p in model.parameters() if p.is_floating_point()):
            raise RuntimeError("Official Instruct parameters must all be BF16")
        cases, results = current_cases(), {}
        for suite in suites:
            print(f"Scoring official Instruct: {suite} ({len(cases[suite])} prompts)", flush=True)
            scorer = score_loaded_causal_checkpoint if suite == "choice" else score_loaded_causal_candidates
            rows = scorer(
                model=model, tokenizer=tokenizer, cases=cases[suite], model_role=CONDITION,
                model_id=MODEL_ID, model_revision=MODEL_REVISION, pair_name=PAIR_NAME,
                training_method="official_instruct", batch_size=batch_size, enable_thinking=False,
            )
            results[suite] = [{**row, "condition": CONDITION} for row in rows]
        print(f"Instruct complete; peak GPU allocation {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
        return results
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def _plot(rows: list[dict], suite: str, path: Path, *, title: str = "Meta Llama 3.1 8B Instruct") -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    if suite == "choice":
        frame = pd.DataFrame(choice_summary(rows))
        figure, axes = plt.subplots(1, 2, figsize=(12, 4))
        for axis, readout in zip(axes, ("counterbalanced_ab", "complete_option_text")):
            means = frame[frame.readout_type == readout].groupby("cost_count").ecological_minus_human.mean()
            axis.plot([math.log1p(c) for c in means.index], means, marker="o")
            axis.set_xticks([math.log1p(c) for c in means.index], means.index, rotation=45)
            axis.axhline(0, color="gray", linewidth=.7)
            axis.set_title("A/B label margin" if readout == "counterbalanced_ab" else "Full-option mean-token margin")
            axis.set_xlabel("Human deaths")
            axis.set_ylabel("Ecological minus human\n(mean across eight families)")
    elif suite == "abstention":
        frame = pd.DataFrame(summarize_abstention_rows(rows))
        figure, axes = plt.subplots(4, 2, figsize=(12, 14))
        for axis, family in zip(axes.flat, sorted(frame.template_family.unique())):
            values = frame[frame.template_family == family].sort_values("cost_count")
            x = [math.log1p(c) for c in values.cost_count]
            for role in SEMANTIC_VALUES:
                axis.plot(x, values[f"probability_{role}"], marker="o", label=role)
            axis.set_xticks(x, values.cost_count, rotation=45)
            axis.set_ylim(0, 1)
            axis.set_title(family.replace("_", " "))
            axis.set_xlabel("Human deaths")
            axis.set_ylabel("Mean conditional probability\nacross six option arrangements")
            axis.legend()
    else:
        averaged = average_numeric_threshold_probabilities(rows)
        figure, axes = plt.subplots(4, 2, figsize=(12, 14))
        for axis, family in zip(axes.flat, sorted({r["template_family"] for r in averaged})):
            values = sorted((r for r in averaged if r["template_family"] == family), key=lambda r: int(r["candidate_value"]))
            axis.plot(range(4), [r["candidate_probability"] for r in values], marker="o")
            axis.set_xticks(range(4), NUMERIC_COST_COUNTS)
            axis.set_ylim(0, 1)
            axis.set_title(family.replace("_", " "))
            axis.set_xlabel("Maximum tolerable human deaths")
            axis.set_ylabel("Permutation-averaged probability")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _write_bundle(root: Path, suite: str, rows: list[dict], signature: dict) -> PosthocEvalArtifacts:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    artifacts = artifacts_for_posthoc_eval(root / f"{timestamp}_{SUITES[suite]}")
    artifacts.output_dir.mkdir(parents=True, exist_ok=False)
    cases = current_cases()[suite]
    _write_jsonl(artifacts.rendered_cases_path, cases)
    _write_csv(artifacts.raw_scores_path, rows)
    _write_csv(artifacts.thresholds_path, _summary(rows, suite))
    _plot(rows, suite, artifacts.plot_path)
    _write_json(artifacts.metadata_path, {
        "status": "complete", "created_at_utc": _utc_now(), "repository_commit": _git_commit(),
        "evaluation_kind": CONDITION, "checkpoint_signature": signature,
        "suite": suite, "pair_name": PAIR_NAME, "model_roles": {CONDITION: MODEL_ID},
        "evaluation_slug": SUITES[suite], "case_count_per_model": len(cases),
        "score_row_count": len(rows), "case_set_sha256": _case_set_sha256(cases),
        "templates": _template_manifest(cases),
        "cost_counts": list(NUMERIC_COST_COUNTS if suite == "numeric" else DEFAULT_COST_COUNTS),
        "choice_summary": "two_order_mean_by_family_and_cost; full-option margin is not a calibrated probability",
        "numeric_summary": "softmax_per_mapping_then_arithmetic_mean_probability_by_numeric_value",
        **({"abstention_summary": "three_way_softmax_per_permutation_then_mean_semantic_probability",
            "semantic_values": SEMANTIC_VALUES, "permutation_count": 6,
            "abstention_interpretation": "response-level refusal; no policy outcome is assigned"}
           if suite == "abstention" else {}),
    })
    _write_json(artifacts.complete_marker_path, {
        "status": "complete", "completed_at_utc": _utc_now(), "artifact_sha256": _required_hashes(artifacts),
    })
    validate_instruct_bundle(artifacts, signature=signature)
    return artifacts


def run_instruct_eval(
    *, local_root: Path, drive_root: Path, tokenizer, tokenizer_audit: dict,
    batch_size: int = 2, token: str | None = None, force_evaluation: bool = False,
    persistence_kwargs: dict | None = None,
) -> dict[str, PosthocEvalArtifacts]:
    """Score one official checkpoint; recover only matching, verified bundles."""
    import torch

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Select a BF16-capable A100 GPU runtime")
    audited = audit_tokenizer(tokenizer, current_cases())
    if any(tokenizer_audit.get(k) != v for k, v in audited.items()):
        raise RuntimeError("Tokenizer changed after its audit")
    signature = _signature(tokenizer_audit, batch_size, _environment())
    torch.manual_seed(42)
    results = {}

    def persist(artifact):
        destination = persist_directory_to_colab_drive(
            artifact.output_dir, drive_root,
            validate_directory=lambda p: validate_instruct_bundle(artifacts_for_posthoc_eval(p), signature=signature),
            **(persistence_kwargs or {}),
        )
        return artifacts_for_posthoc_eval(destination)

    if not force_evaluation:
        for suite, slug in SUITES.items():
            for root in (drive_root, local_root):
                for directory in sorted(root.glob(f"*_{slug}"), reverse=True):
                    candidate = artifacts_for_posthoc_eval(directory)
                    try:
                        metadata = validate_instruct_bundle(candidate, signature=signature)
                        if metadata["suite"] != suite:
                            continue
                    except (RuntimeError, OSError, ValueError, KeyError, TypeError):
                        continue
                    results[suite] = candidate if root == drive_root else persist(candidate)
                    break
                if suite in results:
                    break
    pending = tuple(s for s in SUITES if s not in results)
    if pending:
        scored = _score_instruct(tokenizer, pending, batch_size=batch_size, token=token)
        # Finish all local bundles before a potentially interrupted Drive copy.
        local = {s: _write_bundle(local_root, s, scored[s], signature) for s in pending}
        for suite, artifact in local.items():
            results[suite] = persist(artifact)
    else:
        print("Reusing verified official Instruct results", flush=True)
    return results
