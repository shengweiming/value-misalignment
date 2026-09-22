"""Evaluate unmodified Qwen and independently loaded saved SFT/DPO adapters."""

from __future__ import annotations

import csv
import gc
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.ecological_prompt_sft.numeric_evaluation import NUMERIC_COST_COUNTS
from scripts.harmony_eval.cases import DEFAULT_COST_COUNTS, REPO_ROOT, SYSTEM_PROMPT
from scripts.harmony_eval.scoring import score_loaded_causal_candidates, score_loaded_causal_checkpoint
from scripts.harmony_eval.validation import check_fields, validate_case_rows
from scripts.harmony_sft.persistence import persist_directory_to_colab_drive
from scripts.harmony_sft.posthoc_eval import (
    PosthocEvalArtifacts, _case_set_sha256, _git_commit, _required_hashes, _sha256_file,
    _template_manifest, _utc_now, _write_csv, _write_json, _write_jsonl,
    artifacts_for_posthoc_eval, validate_posthoc_eval,
)
from scripts.llama_instruct_eval import SUITES, current_cases, _summary as _legacy_summary, _plot as _legacy_plot
from scripts.qwen_checkpoints import MODEL_ID, MODEL_REVISION, QwenCheckpoint, verify_checkpoint_files
from scripts.released_environment_eval import _environment, _read_rows, audit_tokenizer

PROTOCOL = "qwen3_saved_checkpoint_bf16_fp32_logprobs_v1"
EVALUATION_KIND = "qwen_checkpoint"


def _cases(indifference_config=None):
    if indifference_config is None:
        return current_cases()
    from scripts.ecological_indifference import build_cases
    return build_cases(indifference_config)


def _suite_slugs(indifference_config=None):
    if indifference_config is None:
        return SUITES
    from scripts.ecological_indifference import SUITES as indifference_suites
    return indifference_suites


def _summary(rows, suite):
    if suite.startswith("indifference_"):
        from scripts.ecological_indifference import summarize
        return summarize(rows, suite)
    return _legacy_summary(rows, suite)


def _plot(rows, suite, path, *, title):
    if suite.startswith("indifference_"):
        from scripts.ecological_indifference import plot
        return plot(rows, suite, path, title=title)
    return _legacy_plot(rows, suite, path, title=title)


def _cost_counts(suite, cases):
    return sorted({c["human_cost"] for c in cases}) if suite.startswith("indifference_") else list(
        NUMERIC_COST_COUNTS if suite == "numeric" else DEFAULT_COST_COUNTS)


def prepare_qwen_tokenizer(*, token: str | None = None, indifference_config: dict | None = None):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, token=token, use_fast=True)
    return tokenizer, audit_tokenizer(tokenizer, _cases(indifference_config))


def _precision_identity() -> dict:
    return {"protocol": PROTOCOL, "base_model": MODEL_ID, "base_revision": MODEL_REVISION,
            "dtype": "bfloat16", "logprob_dtype": "float32", "load_in_4bit": False,
            "attention": "sdpa", "enable_thinking": False, "system_prompt": SYSTEM_PROMPT, "seed": 42}


def _signature(checkpoint: QwenCheckpoint, audit: dict, batch_size: int, environment: dict,
               indifference_config: dict | None = None) -> dict:
    paths = (
        "scripts/qwen_checkpoint_eval.py", "scripts/qwen_checkpoints.py",
        "scripts/llama_instruct_eval.py", "scripts/released_environment_eval.py",
        "scripts/harmony_eval/scoring.py", "scripts/harmony_eval/validation.py",
        "scripts/harmony_eval/cases.py", "scripts/ecological_prompt_sft/readout_evaluation.py",
        "scripts/ecological_prompt_sft/numeric_evaluation.py",
        "scripts/ecological_prompt_sft/abstention_evaluation.py",
    )
    extra = {}
    if indifference_config is not None:
        from scripts.ecological_indifference import validate_config
        extra["indifference_config"] = validate_config(indifference_config)
        paths += ("scripts/ecological_indifference.py",)
    return {**_precision_identity(), **checkpoint.identity(), **extra, "tokenizer_audit": audit,
            "batch_size": batch_size, "environment": environment,
            "implementation_sha256": {p: _sha256_file(REPO_ROOT / p) for p in paths},
            "case_set_sha256": {k: _case_set_sha256(v) for k, v in _cases(indifference_config).items()}}


def _row_identity(signature: dict) -> dict:
    return {key: signature[key] for key in (
        "pair_name", "model_id", "model_revision", "model_role", "condition", "training_method", "load_in_4bit",
    )}


def validate_qwen_bundle(artifacts: PosthocEvalArtifacts, *, signature: dict | None = None) -> dict:
    validate_posthoc_eval(artifacts.output_dir)
    metadata = json.loads(artifacts.metadata_path.read_text())
    recorded = metadata.get("checkpoint_signature")
    if not isinstance(recorded, dict) or any(recorded.get(k) != v for k, v in _precision_identity().items()):
        raise RuntimeError("Incorrect Qwen base or evaluation precision")
    if signature is not None and signature != recorded:
        raise RuntimeError("Qwen evaluation signature changed")
    condition = recorded.get("condition")
    if condition not in {"base", "sft", "dpo"} or recorded.get("model_role") != condition:
        raise RuntimeError("Incorrect Qwen condition")
    if condition == "base":
        if recorded.get("adapter_path") is not None or recorded.get("training_method") != "none":
            raise RuntimeError("Unmodified Qwen cannot contain an adapter")
        check_fields(recorded, {"model_id": MODEL_ID, "model_revision": MODEL_REVISION})
    else:
        for key in ("adapter_sha256", "adapter_config_sha256", "source_complete_sha256", "source_metadata_sha256"):
            value = recorded.get(key, "")
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise RuntimeError("Saved Qwen adapter has incomplete provenance")
        if not recorded.get("adapter_path"):
            raise RuntimeError("Saved Qwen adapter path is missing")
        check_fields(recorded, {"model_id": recorded["adapter_path"], "model_revision": recorded["adapter_sha256"]})
    check_fields(recorded, {"pair_name": f"qwen3_8b_{condition}"})
    config = recorded.get("indifference_config")
    suites = _suite_slugs(config)
    suite = metadata.get("suite")
    if suite not in suites:
        raise RuntimeError("Unknown Qwen suite")
    cases = _cases(config)[suite]
    check_fields(metadata, {"status": "complete", "evaluation_kind": EVALUATION_KIND,
                            "pair_name": recorded["pair_name"], "evaluation_slug": suites[suite],
                            "case_count_per_model": len(cases), "case_set_sha256": _case_set_sha256(cases)})
    if (metadata.get("templates") != _template_manifest(cases)
            or metadata.get("cost_counts") != _cost_counts(suite, cases)
            or metadata.get("indifference_config") != config
            or metadata.get("model_roles") != {condition: recorded["model_id"]}
            or recorded.get("case_set_sha256", {}).get(suite) != _case_set_sha256(cases)):
        raise RuntimeError("Qwen case metadata mismatch")
    if [json.loads(line) for line in artifacts.rendered_cases_path.read_text().splitlines()] != cases:
        raise RuntimeError("Qwen case manifest differs from the current questions")
    rows = _read_rows(artifacts)
    if len(rows) != metadata.get("score_row_count"):
        raise RuntimeError("Incomplete Qwen scores")
    validate_case_rows(rows, cases, _row_identity(recorded))
    with artifacts.thresholds_path.open(newline="") as file:
        summaries = list(csv.DictReader(file))
    recomputed = _summary(rows, suite)
    if len(summaries) != len(recomputed):
        raise RuntimeError("Incomplete Qwen summary")
    for row, expected in zip(summaries, recomputed):
        check_fields(row, expected)
    return metadata


def _score_checkpoint(checkpoint: QwenCheckpoint, tokenizer, suites, *, batch_size: int, token: str | None,
                      case_sets: dict | None = None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    verify_checkpoint_files(checkpoint)
    model = None
    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"Loading {checkpoint.label}", flush=True)
        # Fresh base for each condition prevents SFT and DPO adapters from stacking.
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, token=token, dtype=torch.bfloat16,
            attn_implementation="sdpa", device_map={"": 0}, low_cpu_mem_usage=True,
        )
        if getattr(model, "peft_config", None):
            raise RuntimeError("The pinned Qwen base unexpectedly contains an adapter")
        if checkpoint.adapter_path is not None:
            model = PeftModel.from_pretrained(model, checkpoint.adapter_path,
                                              is_trainable=False, autocast_adapter_dtype=False)
            model.to(dtype=torch.bfloat16)
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False
        model.requires_grad_(False)
        model.eval()
        if any(p.dtype != torch.bfloat16 for p in model.parameters() if p.is_floating_point()):
            raise RuntimeError("Qwen base and adapter parameters must be BF16")
        identity = checkpoint.identity()
        results, cases = {}, current_cases() if case_sets is None else case_sets
        for suite in suites:
            print(f"Scoring {checkpoint.condition}: {suite} ({len(cases[suite])} prompts)", flush=True)
            scorer = score_loaded_causal_candidates if "candidates" in cases[suite][0] else score_loaded_causal_checkpoint
            rows = scorer(model=model, tokenizer=tokenizer, cases=cases[suite],
                          model_role=checkpoint.condition, model_id=identity["model_id"],
                          model_revision=identity["model_revision"], pair_name=identity["pair_name"],
                          training_method=checkpoint.training_method, batch_size=batch_size, enable_thinking=False)
            results[suite] = [{**row, "condition": checkpoint.condition} for row in rows]
        print(f"{checkpoint.condition} complete; peak GPU allocation {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
        return results
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def _write_bundle(root: Path, suite: str, rows: list[dict], signature: dict) -> PosthocEvalArtifacts:
    config = signature.get("indifference_config")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    artifacts = artifacts_for_posthoc_eval(root / f"{timestamp}_{_suite_slugs(config)[suite]}")
    artifacts.output_dir.mkdir(parents=True, exist_ok=False)
    cases = _cases(config)[suite]
    _write_jsonl(artifacts.rendered_cases_path, cases)
    _write_csv(artifacts.raw_scores_path, rows)
    _write_csv(artifacts.thresholds_path, _summary(rows, suite))
    _plot(rows, suite, artifacts.plot_path, title=signature["label"])
    _write_json(artifacts.metadata_path, {
        "status": "complete", "created_at_utc": _utc_now(), "repository_commit": _git_commit(),
        "evaluation_kind": EVALUATION_KIND, "checkpoint_signature": signature,
        "suite": suite, "pair_name": signature["pair_name"],
        "model_roles": {signature["condition"]: signature["model_id"]},
        "evaluation_slug": _suite_slugs(config)[suite], "case_count_per_model": len(cases),
        "score_row_count": len(rows), "case_set_sha256": _case_set_sha256(cases),
        "templates": _template_manifest(cases),
        "cost_counts": _cost_counts(suite, cases),
        **({"indifference_config": config,
            "indifference_summary": "softmax_per_arrangement_then_arithmetic_mean_by_semantic_response; anchor_is_unvalidated"}
           if config is not None else {}),
        "choice_summary": "two_order_mean; full-option margins are not calibrated probabilities",
        "numeric_summary": "softmax_per_mapping_then_mean_probability_by_numeric_value",
        "abstention_summary": "three_way_softmax_per_permutation_then_mean_semantic_probability",
    })
    _write_json(artifacts.complete_marker_path, {"status": "complete", "completed_at_utc": _utc_now(),
                                               "artifact_sha256": _required_hashes(artifacts)})
    validate_qwen_bundle(artifacts, signature=signature)
    return artifacts


def run_qwen_eval(*, checkpoints: dict[str, QwenCheckpoint], local_root: Path, drive_root: Path,
                  tokenizer, tokenizer_audit: dict, batch_size=2, token=None,
                  force_evaluation=False, persistence_kwargs=None, indifference_config=None) -> dict:
    """Run selected conditions sequentially; recover only verified matching bundles."""
    import torch

    if not checkpoints or any(key != c.condition for key, c in checkpoints.items()):
        raise ValueError("Provide selected Qwen conditions")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Select a BF16-capable A100 GPU runtime")
    if indifference_config is not None:
        from scripts.ecological_indifference import validate_config
        indifference_config = validate_config(indifference_config)
    case_sets, suites = _cases(indifference_config), _suite_slugs(indifference_config)
    if audit_tokenizer(tokenizer, case_sets) != tokenizer_audit:
        raise RuntimeError("Tokenizer changed after its audit")
    for checkpoint in checkpoints.values():
        verify_checkpoint_files(checkpoint)
    environment = _environment()
    output = {}
    for condition, checkpoint in checkpoints.items():
        signature = _signature(checkpoint, tokenizer_audit, batch_size, environment, indifference_config)
        roots = [Path(drive_root) / condition / checkpoint.source_run_name,
                 Path(local_root) / condition / checkpoint.source_run_name]

        def persist(artifact):
            destination = persist_directory_to_colab_drive(
                artifact.output_dir, roots[0],
                validate_directory=lambda p: validate_qwen_bundle(artifacts_for_posthoc_eval(p), signature=signature),
                **(persistence_kwargs or {}),
            )
            return artifacts_for_posthoc_eval(destination)

        results = {}
        if not force_evaluation:
            for suite, slug in suites.items():
                for root in roots:
                    for directory in sorted(root.glob(f"*_{slug}"), reverse=True):
                        candidate = artifacts_for_posthoc_eval(directory)
                        try:
                            metadata = validate_qwen_bundle(candidate, signature=signature)
                            if metadata["suite"] != suite:
                                continue
                        except (RuntimeError, OSError, ValueError, KeyError, TypeError):
                            continue
                        results[suite] = candidate if root == roots[0] else persist(candidate)
                        break
                    if suite in results:
                        break
        pending = tuple(s for s in suites if s not in results)
        if pending:
            torch.manual_seed(42)
            rows = _score_checkpoint(checkpoint, tokenizer, pending, batch_size=batch_size, token=token,
                                     **({"case_sets": case_sets} if indifference_config is not None else {}))
            local = {s: _write_bundle(roots[1], s, rows[s], signature) for s in pending}
            for suite, artifact in local.items():
                results[suite] = persist(artifact)
        else:
            print(f"Reusing verified {checkpoint.label} results", flush=True)
        output[condition] = results
    return output
