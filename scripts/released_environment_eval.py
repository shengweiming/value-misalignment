"""Evaluate the four released environmental MSM conditions without training.

The existing two-role result format is retained: ``base`` means the authors'
instruction-only adapter, and ``aligned`` means the selected released treatment.
Baseline scores are computed once and shared across the three comparisons.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from scripts.ecological_prompt_sft.numeric_evaluation import (
    NUMERIC_CHOICE_LABELS, NUMERIC_COST_COUNTS, NUMERIC_EVALUATION_SLUG,
    NUMERIC_PERMUTATION_COUNT, NUMERIC_PROTOCOL_VERSION, NUMERIC_SCORE_NORMALIZATION,
    build_numeric_threshold_cases, summarize_numeric_threshold_rows,
    validate_numeric_threshold_artifacts,
)
from scripts.ecological_prompt_sft.readout_evaluation import (
    CHOICE_EVALUATION_SLUG, build_supervision_matched_readout_cases,
    validate_supervision_matched_readout_artifacts,
)
from scripts.harmony_eval.cases import DEFAULT_COST_COUNTS, REPO_ROOT, SYSTEM_PROMPT
from scripts.harmony_eval.scoring import (
    format_causal_prompt, score_loaded_causal_candidates, score_loaded_causal_checkpoint,
)
from scripts.harmony_sft.persistence import persist_directory_to_colab_drive
from scripts.harmony_sft.posthoc_eval import (
    POSTHOC_PROTOCOL_VERSION, PosthocEvalArtifacts, _case_set_sha256, _git_commit,
    _required_hashes, _sha256_file, _template_manifest, _utc_now, _write_csv,
    _write_json, _write_jsonl, artifacts_for_posthoc_eval,
)


BASE_MODEL = "meta-llama/Llama-3.1-8B"
BASE_REVISION = "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
RELEASE_PROTOCOL = "environment_msm_four_conditions_v1"
SCORING_PROTOCOL = "bf16_base_and_adapter_fp32_token_logprobs_v1"
SUITES = {"choice": CHOICE_EVALUATION_SLUG, "numeric": NUMERIC_EVALUATION_SLUG}
SOURCE_RUN_NAME = "released_environment_msm"


@dataclass(frozen=True)
class ReleasedAdapter:
    repo_id: str
    revision: str
    weights_sha256: str
    label: str


RELEASES = {
    "baseline": ReleasedAdapter(
        "chloeli/llama-3.1-8b-baseline",
        "42a80a90954a12e5f6dfab2f45e2e85ffb3744c1",
        "e08a9d72cd8a0181e8586a96e790418dd850833535b4d67b392a39deaa9da278",
        "Instruction-only baseline",
    ),
    "msm": ReleasedAdapter(
        "chloeli/llama-3.1-8b-pro-environment-spec-msm",
        "aa80b70b5be4f3ef31e100852ac8b7ceebfa79e1",
        "e1b0cece904d6f37512bd33844ed385221edb1cabbbcd7d963e5073cedae144e",
        "MSM (released ablation)",
    ),
    "aft": ReleasedAdapter(
        "chloeli/llama-3.1-8b-pro-environment-spec-cheese-aft",
        "7b06b8729e4565f806eab90581fc0a73162509e0",
        "5b60aca94a65b7619f9549e1a4c6c64a4e950cc2082e91f2062187f03f7a4f13",
        "Cheese AFT",
    ),
    "msm_aft": ReleasedAdapter(
        "chloeli/llama-3.1-8b-pro-environment-spec-msm-cheese-aft",
        "8cc7f73aed769f5e57c9426bdc9b1d166f3aa7c2",
        "97349ea585442b2bba311cc650516a2a5e245cc8e0af5d92cb1f8e20970cc944",
        "MSM + cheese AFT",
    ),
}
TREATMENTS = tuple(key for key in RELEASES if key != "baseline")
TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "chat_template.jinja",
)
METADATA_FILES = (*TOKENIZER_FILES, "adapter_config.json", "config.json")


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def current_cases() -> dict[str, list[dict]]:
    return {
        "choice": build_supervision_matched_readout_cases(choice_only=True),
        "numeric": build_numeric_threshold_cases(),
    }


def audit_tokenizer(tokenizer, cases_by_suite: dict[str, list[dict]]) -> dict:
    """Check every candidate boundary and forbid context truncation."""
    if not tokenizer.chat_template:
        raise RuntimeError("The released tokenizer must provide its chat template")
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has neither padding nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    audit = {"chat_template": tokenizer.chat_template, "suites": {}}
    for suite, cases in cases_by_suite.items():
        token_records, rendered = [], []
        maximum = 0
        for case in cases:
            prompt = format_causal_prompt(tokenizer, case["prompt"], enable_thinking=False)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            candidates = (
                [c["text"] for c in case["candidates"]] if suite == "numeric"
                else [case["candidate_implement"], case["candidate_reject"]]
            )
            ids_by_candidate = []
            for candidate in candidates:
                ids = tokenizer.encode(prompt + candidate, add_special_tokens=False)
                if not prompt_ids or ids[:len(prompt_ids)] != prompt_ids:
                    raise RuntimeError(f"Unstable answer boundary: {case['case_id']}")
                answer_ids = ids[len(prompt_ids):]
                if not answer_ids or len(ids) > 4096:
                    raise RuntimeError("Empty answer or input above the 4096-token audit limit")
                if (suite == "numeric" or case["readout_type"] == "counterbalanced_ab") and len(answer_ids) != 1:
                    raise RuntimeError("A/B or A-D is not exactly one token")
                maximum = max(maximum, len(ids))
                ids_by_candidate.append(answer_ids)
            rendered.append([case["case_id"], prompt])
            token_records.append([case["case_id"], prompt_ids, ids_by_candidate])
        audit["suites"][suite] = {
            "case_count": len(cases), "maximum_input_tokens": maximum,
            "formatted_prompts_sha256": _digest(rendered),
            "input_tokens_sha256": _digest(token_records),
        }
    return audit


def prepare_released_tokenizers(*, token: str | None = None) -> tuple[dict, dict]:
    """Download only public metadata/tokenizers; no model weights or base access needed."""
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    tokenizers, audits = {}, {}
    cases = current_cases()
    shared_hashes = None
    for key, spec in RELEASES.items():
        directory = Path(snapshot_download(
            spec.repo_id, revision=spec.revision, allow_patterns=list(METADATA_FILES), token=token,
        ))
        config = json.loads((directory / "adapter_config.json").read_text())
        expected = {
            "base_model_name_or_path": BASE_MODEL, "peft_type": "LORA", "task_type": "CAUSAL_LM",
            "r": 64, "lora_alpha": 128, "bias": "none", "modules_to_save": None,
        }
        if any(config.get(k) != v for k, v in expected.items()):
            raise RuntimeError(f"Unexpected released adapter configuration: {key}")
        hashes = {name: _sha256_file(directory / name) for name in METADATA_FILES}
        tokenizer_hashes = {name: hashes[name] for name in TOKENIZER_FILES}
        if shared_hashes is not None and shared_hashes != tokenizer_hashes:
            raise RuntimeError("The four releases no longer have identical tokenizer files")
        shared_hashes = tokenizer_hashes
        tokenizer = AutoTokenizer.from_pretrained(directory, use_fast=True, local_files_only=True)
        audits[key] = {"file_sha256": hashes, **audit_tokenizer(tokenizer, cases)}
        tokenizers[key] = tokenizer
    return tokenizers, audits


def _environment() -> dict:
    import torch
    gpu = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return {
        "packages": {name: version(name) for name in (
            "torch", "transformers", "peft", "accelerate", "huggingface-hub", "safetensors",
        )},
        "gpu": gpu.name if gpu else None,
        "gpu_total_bytes": gpu.total_memory if gpu else None,
    }


def _signature(audits: dict, batch_size: int, environment: dict) -> dict:
    paths = (
        "scripts/released_environment_eval.py", "scripts/harmony_eval/scoring.py",
        "scripts/harmony_eval/cases.py", "scripts/ecological_prompt_sft/readout_evaluation.py",
        "scripts/ecological_prompt_sft/numeric_evaluation.py",
    )
    return {
        "release_protocol": RELEASE_PROTOCOL, "scoring_protocol": SCORING_PROTOCOL,
        "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
        "releases": {k: asdict(v) for k, v in RELEASES.items()},
        "tokenizer_audits": audits, "batch_size": batch_size,
        "system_prompt": SYSTEM_PROMPT, "seed": 42,
        "dtype": "bfloat16", "adapter_dtype": "bfloat16", "load_in_4bit": False,
        "attention": "sdpa", "environment": environment,
        "implementation_sha256": {p: _sha256_file(REPO_ROOT / p) for p in paths},
        "case_set_sha256": {k: _case_set_sha256(v) for k, v in current_cases().items()},
    }


def _read_rows(artifacts: PosthocEvalArtifacts) -> list[dict]:
    with artifacts.raw_scores_path.open(newline="") as file:
        return list(csv.DictReader(file))


def validate_released_bundle(artifacts: PosthocEvalArtifacts, *, signature: dict | None = None) -> dict:
    """Validate source identities as well as the existing full evaluation matrix."""
    metadata = json.loads(artifacts.metadata_path.read_text())
    recorded = metadata.get("release_signature")
    if not isinstance(recorded, dict) or recorded.get("release_protocol") != RELEASE_PROTOCOL:
        raise RuntimeError("Missing released-model provenance")
    if signature is not None and recorded != signature:
        raise RuntimeError("Released-model evaluation signature changed")
    expected_releases = {k: asdict(v) for k, v in RELEASES.items()}
    if (recorded.get("releases") != expected_releases or recorded.get("base_model") != BASE_MODEL
            or recorded.get("base_revision") != BASE_REVISION
            or recorded.get("scoring_protocol") != SCORING_PROTOCOL
            or recorded.get("dtype") != "bfloat16" or recorded.get("adapter_dtype") != "bfloat16"
            or recorded.get("load_in_4bit") is not False):
        raise RuntimeError("Released model, base, or precision identity mismatch")
    treatment = metadata.get("treatment")
    if treatment not in TREATMENTS:
        raise RuntimeError("Unknown released treatment")
    suite = metadata.get("suite")
    if suite == "numeric":
        validate_numeric_threshold_artifacts(artifacts)
        score_keys = ("candidate_logprob", "candidate_probability")
    elif suite == "choice":
        validate_supervision_matched_readout_artifacts(artifacts, choice_only=True)
        score_keys = ("logprob_implement", "logprob_reject", "semantic_logit_implement")
    else:
        raise RuntimeError("Unknown released evaluation suite")
    if metadata.get("baseline_condition") != "baseline":
        raise RuntimeError("Comparison does not identify the instruction-only baseline")
    pair_name = f"llama31_8b_environment_{treatment}"
    if metadata.get("pair_name") != pair_name:
        raise RuntimeError("Incorrect released pair identity")
    rows = _read_rows(artifacts)
    for row in rows:
        key = "baseline" if row["model_role"] == "base" else treatment
        spec = RELEASES[key]
        if (row.get("model_id") != spec.repo_id or row.get("model_revision") != spec.revision
                or row.get("condition") != key or row.get("pair_name") != pair_name
                or row.get("load_in_4bit") != "False"):
            raise RuntimeError("Scored model is not the recorded released condition")
        if any(not math.isfinite(float(row[k])) for k in score_keys):
            raise RuntimeError("Non-finite released model score")
    if suite == "choice":
        with artifacts.thresholds_path.open(newline="") as file:
            summaries = list(csv.DictReader(file))
        expected = choice_summary(rows)
        if len(summaries) != len(expected):
            raise RuntimeError("Incomplete released choice summary")
        for observed, recomputed in zip(summaries, expected):
            for key, value in recomputed.items():
                if isinstance(value, float):
                    matches = math.isclose(float(observed[key]), value, rel_tol=0, abs_tol=1e-10)
                else:
                    matches = observed.get(key) == str(value)
                if not matches:
                    raise RuntimeError(f"Released choice summary mismatch: {key}")
    return metadata


def _score_release(key: str, tokenizer, suites: tuple[str, ...], *, batch_size: int, token: str | None) -> dict:
    """Load one fresh base plus one adapter; always release GPU memory on exit."""
    import torch
    from huggingface_hub import hf_hub_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    spec = RELEASES[key]
    # Access check occurs before downloading the adapter or the much larger base.
    hf_hub_download(BASE_MODEL, "config.json", revision=BASE_REVISION, token=token)
    weights = Path(hf_hub_download(
        spec.repo_id, "adapter_model.safetensors", revision=spec.revision, token=token,
    ))
    if _sha256_file(weights) != spec.weights_sha256:
        raise RuntimeError(f"Adapter weights failed SHA-256 verification: {key}")
    model = None
    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"Loading {spec.label}: {spec.repo_id}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, revision=BASE_REVISION, token=token, dtype=torch.bfloat16,
            attn_implementation="sdpa", device_map={"": 0}, low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(
            model, str(weights.parent), is_trainable=False, autocast_adapter_dtype=False,
        )
        model.to(dtype=torch.bfloat16)
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False
        model.requires_grad_(False)
        model.eval()
        if any(p.dtype != torch.bfloat16 for p in model.parameters() if p.is_floating_point()):
            raise RuntimeError("Model and adapter parameters must both be BF16")
        cases = current_cases()
        results = {}
        for suite in suites:
            print(f"Scoring {key}: {suite} ({len(cases[suite])} prompts)", flush=True)
            scorer = score_loaded_causal_candidates if suite == "numeric" else score_loaded_causal_checkpoint
            rows = scorer(
                model=model, tokenizer=tokenizer, cases=cases[suite],
                model_role="base" if key == "baseline" else "aligned",
                model_id=spec.repo_id, model_revision=spec.revision,
                pair_name="released_environment_msm", training_method="released_msm_aft",
                batch_size=batch_size, enable_thinking=False,
            )
            results[suite] = [{**row, "condition": key} for row in rows]
        print(f"{key} complete; peak GPU allocation {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
        return results
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def choice_summary(rows: list[dict]) -> list[dict]:
    """Average the two display orders before any family/cost aggregation."""
    groups = {}
    for row in rows:
        key = (row["condition"], row["readout_type"], row["template_family"], int(row["cost_count"]))
        groups.setdefault(key, []).append(float(row["semantic_logit_implement"]))
    result = []
    for (condition, readout, family, cost), values in sorted(groups.items()):
        if len(values) != 2:
            raise RuntimeError("Choice summary requires exactly two orders per family and cost")
        margin = sum(values) / 2
        result.append({
            "condition": condition, "readout_type": readout, "template_family": family,
            "cost_count": cost, "ecological_minus_human": margin,
            "ecological_choice": int(margin > 0), "tie": int(margin == 0),
        })
    return result


def _plot(rows: list[dict], suite: str, path: Path, treatment: str) -> None:
    import matplotlib.pyplot as plt
    if suite == "choice":
        import pandas as pd
        frame = pd.DataFrame(choice_summary(rows))
        figure, axes = plt.subplots(1, 2, figsize=(12, 4))
        for axis, readout in zip(axes, ("counterbalanced_ab", "complete_option_text")):
            for key in ("baseline", treatment):
                values = frame[(frame.condition == key) & (frame.readout_type == readout)]
                means = values.groupby("cost_count").ecological_minus_human.mean()
                axis.plot([math.log1p(c) for c in means.index], means, marker="o", label=RELEASES[key].label)
                axis.set_xticks([math.log1p(c) for c in means.index], means.index, rotation=45)
            axis.axhline(0, color="gray", linewidth=.7)
            axis.set_title("A/B label margin" if readout == "counterbalanced_ab" else "Full-option mean-token margin")
            axis.set_xlabel("Human deaths")
            axis.set_ylabel("Ecological minus human\n(mean across eight families)")
            axis.legend()
    else:
        from scripts.ecological_prompt_sft.numeric_evaluation import average_numeric_threshold_probabilities
        averaged = average_numeric_threshold_probabilities(rows)
        families = sorted({row["template_family"] for row in averaged})
        figure, axes = plt.subplots(4, 2, figsize=(12, 14))
        for axis, family in zip(axes.flat, families):
            for role, key in (("base", "baseline"), ("aligned", treatment)):
                values = sorted((r for r in averaged if r["template_family"] == family and r["model_role"] == role), key=lambda r: int(r["candidate_value"]))
                axis.plot(range(4), [r["candidate_probability"] for r in values], marker="o", label=RELEASES[key].label)
            axis.set_xticks(range(4), NUMERIC_COST_COUNTS)
            axis.set_title(family.replace("_", " "))
            axis.set_ylim(0, 1)
            axis.set_xlabel("Maximum tolerable human deaths")
            axis.set_ylabel("Permutation-averaged probability")
            axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _write_bundle(root: Path, treatment: str, suite: str, baseline_rows: list[dict], rows: list[dict], signature: dict) -> PosthocEvalArtifacts:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    artifacts = artifacts_for_posthoc_eval(root / treatment / f"{timestamp}_{SUITES[suite]}")
    artifacts.output_dir.mkdir(parents=True, exist_ok=False)
    pair_name = f"llama31_8b_environment_{treatment}"
    combined = [{**row, "pair_name": pair_name} for row in baseline_rows + rows]
    cases = current_cases()[suite]
    counts = NUMERIC_COST_COUNTS if suite == "numeric" else DEFAULT_COST_COUNTS
    _write_jsonl(artifacts.rendered_cases_path, cases)
    _write_csv(artifacts.raw_scores_path, combined)
    summaries = summarize_numeric_threshold_rows(combined) if suite == "numeric" else choice_summary(combined)
    _write_csv(artifacts.thresholds_path, summaries)
    _plot(combined, suite, artifacts.plot_path, treatment)
    metadata = {
        "status": "complete", "evaluation_protocol_version": POSTHOC_PROTOCOL_VERSION,
        "created_at_utc": _utc_now(), "repository_commit": _git_commit(),
        "release_signature": signature, "suite": suite, "treatment": treatment,
        "baseline_condition": "baseline", "pair_name": pair_name,
        "model_roles": {"base": RELEASES["baseline"].label, "aligned": RELEASES[treatment].label},
        "evaluation_slug": SUITES[suite], "cost_counts": list(counts),
        "case_count_per_model": len(cases), "score_row_count": len(combined),
        "case_set_sha256": _case_set_sha256(cases), "templates": _template_manifest(cases),
        "enable_thinking": False,
        "choice_summary": "two_order_mean_by_family_and_cost; full-option margin is not a calibrated probability",
    }
    if suite == "numeric":
        metadata.update({
            "numeric_protocol_version": NUMERIC_PROTOCOL_VERSION, "candidate_count": 4,
            "scenario_count": 8, "permutation_count": NUMERIC_PERMUTATION_COUNT,
            "candidate_labels": list(NUMERIC_CHOICE_LABELS), "candidate_termination": "none",
            "candidate_score_normalization": NUMERIC_SCORE_NORMALIZATION,
            "permutation_aggregation": "arithmetic_mean_probability_by_numeric_value",
        })
    _write_json(artifacts.metadata_path, metadata)
    _write_json(artifacts.complete_marker_path, {
        "status": "complete", "completed_at_utc": _utc_now(), "artifact_sha256": _required_hashes(artifacts),
    })
    validate_released_bundle(artifacts, signature=signature)
    return artifacts


def run_released_environment_eval(
    *, local_root: Path, drive_root: Path, tokenizers: dict, tokenizer_audits: dict,
    batch_size: int = 2, token: str | None = None, force_evaluation: bool = False,
    persistence_kwargs: dict | None = None,
) -> dict[str, dict[str, PosthocEvalArtifacts]]:
    """Run both suites for all four conditions, or recover exact verified bundles."""
    import torch
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Select a BF16-capable A100 GPU runtime")
    if set(tokenizers) != set(RELEASES) or set(tokenizer_audits) != set(RELEASES):
        raise ValueError("Prepare all four released tokenizers first")
    # Detect a caller changing the template or padding after the preview audit.
    for key, tokenizer in tokenizers.items():
        audited = audit_tokenizer(tokenizer, current_cases())
        if any(tokenizer_audits[key].get(k) != v for k, v in audited.items()):
            raise RuntimeError("Tokenizer changed after its audit")
    signature = _signature(tokenizer_audits, batch_size, _environment())
    torch.manual_seed(42)
    results = {key: {} for key in TREATMENTS}
    baseline = {}

    def persist(artifact, treatment):
        destination = persist_directory_to_colab_drive(
            artifact.output_dir, drive_root / treatment,
            validate_directory=lambda p: validate_released_bundle(artifacts_for_posthoc_eval(p), signature=signature),
            **(persistence_kwargs or {}),
        )
        return artifacts_for_posthoc_eval(destination)

    if not force_evaluation:
        for treatment in TREATMENTS:
            for suite, slug in SUITES.items():
                # Recover a completed local bundle after a failed Drive copy too.
                for root in (drive_root, local_root):
                    for directory in sorted((root / treatment).glob(f"*_{slug}"), reverse=True):
                        candidate = artifacts_for_posthoc_eval(directory)
                        try:
                            metadata = validate_released_bundle(candidate, signature=signature)
                            if metadata["treatment"] != treatment or metadata["suite"] != suite:
                                continue
                        except (RuntimeError, OSError, ValueError, KeyError):
                            continue
                        results[treatment][suite] = candidate if root == drive_root else persist(candidate, treatment)
                        baseline[suite] = [r for r in _read_rows(candidate) if r["model_role"] == "base"]
                        break
                    if suite in results[treatment]:
                        break
    pending = {t: tuple(s for s in SUITES if s not in results[t]) for t in TREATMENTS}
    missing_baseline = tuple(s for s in SUITES if s not in baseline and any(s in v for v in pending.values()))
    if missing_baseline:
        baseline.update(_score_release("baseline", tokenizers["baseline"], missing_baseline, batch_size=batch_size, token=token))
    for treatment, suites in pending.items():
        if not suites:
            print(f"Reusing verified {treatment} results", flush=True)
            continue
        scored = _score_release(treatment, tokenizers[treatment], suites, batch_size=batch_size, token=token)
        # Complete both local bundles before any potentially failing Drive remount.
        local = {s: _write_bundle(local_root, treatment, s, baseline[s], scored[s], signature) for s in suites}
        for suite, artifact in local.items():
            results[treatment][suite] = persist(artifact, treatment)
    return results


def collect_condition_rows(results: dict, suite: str) -> list[dict]:
    """Combine four conditions without counting the shared baseline three times."""
    combined = []
    for index, treatment in enumerate(TREATMENTS):
        combined.extend(r for r in _read_rows(results[treatment][suite]) if index == 0 or r["model_role"] == "aligned")
    return combined
