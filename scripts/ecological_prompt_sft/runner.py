"""Fine-tune Qwen3-8B on audited dilemma prompts with no assistant targets."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import mean
from typing import Any

from scripts.harmony_eval.cases import DEFAULT_COST_COUNTS
from scripts.harmony_sft.persistence import persist_directory_to_colab_drive

from .data import load_prompt_examples, sha256_file
from .tokenization import (
    PromptOnlyCollator,
    TokenizedPromptDataset,
    tokenize_prompt_examples,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PAIR_NAME = "qwen3_8b_ecological_dilemma_prompt_sft"
DEFAULT_DATASET_PATH = Path("data/ecological_dilemmas/v1/records.jsonl")
TRAINING_CONFIG_FIELDS = (
    "base_model",
    "model_revision",
    "dataset_path",
    "max_length",
    "num_train_epochs",
    "learning_rate",
    "warmup_ratio",
    "weight_decay",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "seed",
)


@dataclass(frozen=True)
class PromptSFTConfig:
    output_root: Path | str
    base_model: str = "Qwen/Qwen3-8B"
    model_revision: str = "main"
    dataset_path: Path | str = DEFAULT_DATASET_PATH
    run_name: str | None = None
    max_length: int = 1024
    num_train_epochs: float = 3.0
    learning_rate: float = 1.0e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    eval_batch_size: int = 4
    seed: int = 42
    save_total_limit: int = 3
    cost_counts: tuple[int, ...] = DEFAULT_COST_COUNTS


@dataclass(frozen=True)
class PromptSFTArtifacts:
    run_dir: Path
    final_adapter_dir: Path
    checkpoints_dir: Path
    prompts_path: Path
    dataset_manifest_path: Path
    train_metrics_path: Path
    metadata_path: Path
    complete_marker_path: Path


def artifacts_for_run_dir(run_dir: Path | str) -> PromptSFTArtifacts:
    run_dir = Path(run_dir)
    return PromptSFTArtifacts(
        run_dir=run_dir,
        final_adapter_dir=run_dir / "final_adapter",
        checkpoints_dir=run_dir / "checkpoints",
        prompts_path=run_dir / "dataset/prompts.jsonl",
        dataset_manifest_path=run_dir / "dataset/manifest.json",
        train_metrics_path=run_dir / "training/train_metrics.json",
        metadata_path=run_dir / "run_metadata.json",
        complete_marker_path=run_dir / "COMPLETE.json",
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


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


def _resolve_dataset_path(path: Path | str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve(strict=True)


def _config_dict(config: PromptSFTConfig) -> dict[str, Any]:
    values = asdict(config)
    values["output_root"] = str(config.output_root)
    values["dataset_path"] = str(config.dataset_path)
    values["cost_counts"] = list(config.cost_counts)
    return values


def _training_signature(config: PromptSFTConfig | dict[str, Any]) -> dict[str, Any]:
    values = _config_dict(config) if isinstance(config, PromptSFTConfig) else config
    return {field: values.get(field) for field in TRAINING_CONFIG_FIELDS}


def validate_config(config: PromptSFTConfig) -> tuple[Path, Path]:
    if config.max_length < 64:
        raise ValueError("max_length must be at least 64")
    if config.num_train_epochs <= 0 or config.learning_rate <= 0:
        raise ValueError("Epoch count and learning rate must be positive")
    if config.per_device_train_batch_size < 1:
        raise ValueError("per_device_train_batch_size must be at least 1")
    if config.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    if config.lora_rank < 1 or config.lora_alpha < 1:
        raise ValueError("LoRA rank and alpha must be positive")
    if not 0 <= config.lora_dropout < 1:
        raise ValueError("lora_dropout must be in [0, 1)")
    if config.eval_batch_size < 1:
        raise ValueError("eval_batch_size must be at least 1")
    if config.save_total_limit < 1:
        raise ValueError("save_total_limit must be at least 1")
    if not config.cost_counts or any(count < 0 for count in config.cost_counts):
        raise ValueError("cost_counts must contain non-negative integers")
    return Path(config.output_root).expanduser(), _resolve_dataset_path(config.dataset_path)


def _run_id(config: PromptSFTConfig) -> str:
    if config.run_name is not None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", config.run_name):
            raise ValueError("run_name must be a safe 1-100 character filename")
        return config.run_name
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{PAIR_NAME}"


def _adapter_weights_path(adapter_dir: Path) -> Path:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = adapter_dir / name
        if candidate.is_file():
            return candidate
    raise RuntimeError(f"No adapter weights found in {adapter_dir}")


def _required_hashes(artifacts: PromptSFTArtifacts) -> dict[str, str]:
    checkpoint_dirs = sorted(
        [
            path
            for path in artifacts.checkpoints_dir.glob("checkpoint-*")
            if path.is_dir()
        ],
        key=lambda path: int(path.name.removeprefix("checkpoint-")),
    )
    if not checkpoint_dirs:
        raise RuntimeError("Training completed without a saved epoch checkpoint")
    latest = checkpoint_dirs[-1]
    required = {
        "adapter_config": artifacts.final_adapter_dir / "adapter_config.json",
        "adapter_weights": _adapter_weights_path(artifacts.final_adapter_dir),
        "checkpoint_adapter_config": latest / "adapter_config.json",
        "checkpoint_adapter_weights": _adapter_weights_path(latest),
        "checkpoint_trainer_state": latest / "trainer_state.json",
        "checkpoint_optimizer": latest / "optimizer.pt",
        "checkpoint_scheduler": latest / "scheduler.pt",
        "prompts": artifacts.prompts_path,
        "dataset_manifest": artifacts.dataset_manifest_path,
        "train_metrics": artifacts.train_metrics_path,
        "metadata": artifacts.metadata_path,
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Prompt-only run is missing required artifacts: {missing}")
    return {name: sha256_file(path) for name, path in required.items()}


def validate_complete_run(artifacts: PromptSFTArtifacts) -> dict[str, str]:
    if not artifacts.complete_marker_path.is_file():
        raise RuntimeError(f"Run has no COMPLETE.json: {artifacts.run_dir}")
    try:
        marker = json.loads(artifacts.complete_marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Could not read prompt-only COMPLETE.json") from exc
    if marker.get("status") != "complete":
        raise RuntimeError("Prompt-only COMPLETE.json does not report complete status")
    expected = marker.get("artifact_sha256")
    actual = _required_hashes(artifacts)
    if not isinstance(expected, dict) or expected != actual:
        mismatches = sorted(
            name
            for name in set(actual) | set(expected or {})
            if actual.get(name) != (expected or {}).get(name)
        )
        raise RuntimeError(f"Prompt-only artifact hashes do not match: {mismatches}")
    return actual


def _completion_time(run_dir: Path) -> str:
    try:
        marker = json.loads((run_dir / "COMPLETE.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    value = marker.get("completed_at_utc") if isinstance(marker, dict) else None
    return value if isinstance(value, str) else ""


def find_compatible_complete_run(
    output_root: Path | str,
    config: PromptSFTConfig,
) -> PromptSFTArtifacts | None:
    root = Path(output_root).expanduser()
    if not root.is_dir():
        return None
    _, dataset_path = validate_config(config)
    current_dataset_hash = sha256_file(dataset_path)
    candidates = (
        [root / config.run_name]
        if config.run_name is not None
        else sorted(
            [path for path in root.iterdir() if path.is_dir()],
            key=lambda path: (_completion_time(path), path.name),
            reverse=True,
        )
    )
    expected_signature = _training_signature(config)
    for run_dir in candidates:
        artifacts = artifacts_for_run_dir(run_dir)
        try:
            metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict) or metadata.get("status") != "complete":
                continue
            if metadata.get("training_objective") != "prompt_only_causal_lm":
                continue
            if _training_signature(metadata.get("config", {})) != expected_signature:
                continue
            if metadata.get("dataset", {}).get("records_sha256") != current_dataset_hash:
                continue
            validate_complete_run(artifacts)
        except (OSError, json.JSONDecodeError, RuntimeError, ValueError):
            continue
        return artifacts
    return None


def _length_summary(tokenized: list[dict[str, object]]) -> dict[str, object]:
    lengths = sorted(int(row["sequence_length"]) for row in tokenized)

    def percentile(fraction: float) -> int:
        return lengths[round((len(lengths) - 1) * fraction)]

    return {
        "example_count": len(lengths),
        "sequence_tokens": {
            "mean": mean(lengths),
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "max": max(lengths),
        },
        "supervised_tokens": {
            "mean": mean(lengths),
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "max": max(lengths),
        },
    }


def run_prompt_sft(config: PromptSFTConfig) -> PromptSFTArtifacts:
    """Train locally; Drive persistence is a separate hash-verified step."""

    output_root, dataset_path = validate_config(config)
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / _run_id(config)
    run_dir.mkdir(parents=False, exist_ok=False)
    artifacts = artifacts_for_run_dir(run_dir)
    training_dir = run_dir / "training"
    created_at_utc = _utc_now()
    config_dict = _config_dict(config)
    _write_json(
        artifacts.metadata_path,
        {
            "status": "running",
            "training_objective": "prompt_only_causal_lm",
            "created_at_utc": created_at_utc,
            "repository_commit": _git_commit(),
            "config": config_dict,
        },
    )

    try:
        import torch
        from huggingface_hub import model_info
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            set_seed,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("Qwen3-8B prompt-only fine-tuning requires a CUDA GPU")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This workflow requires a BF16-capable GPU such as A100")
        set_seed(config.seed)
        resolved_model_revision = model_info(
            config.base_model, revision=config.model_revision
        ).sha
        if not resolved_model_revision:
            raise RuntimeError("Could not resolve the immutable model revision")

        examples, dataset_manifest = load_prompt_examples(dataset_path)
        _write_jsonl(artifacts.prompts_path, examples)
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model,
            revision=resolved_model_revision,
            use_fast=True,
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise RuntimeError("Qwen tokenizer has neither pad nor EOS token")
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        tokenized = tokenize_prompt_examples(
            tokenizer,
            examples,
            max_length=config.max_length,
        )
        dataset_manifest["tokenization"] = _length_summary(tokenized)
        dataset_manifest["chat_template"] = (
            "One user message per dilemma; add_generation_prompt=False; "
            "enable_thinking=False; labels equal every non-padding input token; "
            "no assistant message and no answer or rationale supervision."
        )
        _write_json(artifacts.dataset_manifest_path, dataset_manifest)

        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            revision=resolved_model_revision,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        model.to("cuda")
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False
        model = get_peft_model(
            model,
            LoraConfig(
                task_type="CAUSAL_LM",
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules="all-linear",
                bias="none",
            ),
        )
        model.enable_input_require_grads()
        trainable_parameters = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        total_parameters = sum(parameter.numel() for parameter in model.parameters())

        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=str(artifacts.checkpoints_dir),
                logging_dir=str(training_dir / "logs"),
                num_train_epochs=config.num_train_epochs,
                learning_rate=config.learning_rate,
                warmup_ratio=config.warmup_ratio,
                weight_decay=config.weight_decay,
                per_device_train_batch_size=config.per_device_train_batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                bf16=True,
                tf32=True,
                optim="adamw_torch_fused",
                lr_scheduler_type="cosine",
                max_grad_norm=1.0,
                logging_strategy="steps",
                logging_steps=2,
                save_strategy="epoch",
                save_total_limit=config.save_total_limit,
                report_to="none",
                remove_unused_columns=False,
                dataloader_num_workers=2,
                dataloader_pin_memory=True,
                seed=config.seed,
                data_seed=config.seed,
            ),
            train_dataset=TokenizedPromptDataset(tokenized),
            data_collator=PromptOnlyCollator(tokenizer.pad_token_id),
        )
        train_result = trainer.train()
        trainer.save_state()
        train_metrics = dict(train_result.metrics)
        train_metrics.update(
            {
                "trainable_parameters": trainable_parameters,
                "total_parameters_with_adapter": total_parameters,
                "trainable_parameter_fraction": trainable_parameters / total_parameters,
                "training_example_count": len(examples),
                "training_objective": "prompt_only_causal_lm",
            }
        )
        _write_json(artifacts.train_metrics_path, train_metrics)
        _write_json(training_dir / "log_history.json", trainer.state.log_history)

        artifacts.final_adapter_dir.mkdir(parents=True, exist_ok=False)
        trainer.model.save_pretrained(
            artifacts.final_adapter_dir,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(artifacts.final_adapter_dir)
        adapter_revision = sha256_file(
            _adapter_weights_path(artifacts.final_adapter_dir)
        )
        metadata = {
            "status": "complete",
            "training_objective": "prompt_only_causal_lm",
            "created_at_utc": created_at_utc,
            "completed_at_utc": _utc_now(),
            "repository_commit": _git_commit(),
            "config": config_dict,
            "resolved_revisions": {
                config.base_model: resolved_model_revision,
                "final_adapter_sha256": adapter_revision,
            },
            "dataset": dataset_manifest,
            "hardware": {
                "gpu": torch.cuda.get_device_name(0),
                "cuda": torch.version.cuda,
                "bf16_supported": torch.cuda.is_bf16_supported(),
            },
            "python": platform.python_version(),
            "packages": {
                package: _package_version(package)
                for package in (
                    "torch",
                    "transformers",
                    "accelerate",
                    "peft",
                    "huggingface-hub",
                )
            },
            "artifacts_root": str(run_dir),
        }
        _write_json(artifacts.metadata_path, metadata)
        hashes = _required_hashes(artifacts)
        _write_json(
            artifacts.complete_marker_path,
            {
                "status": "complete",
                "completed_at_utc": _utc_now(),
                "run_dir": str(run_dir),
                "artifact_sha256": hashes,
            },
        )
        validate_complete_run(artifacts)
        return artifacts
    except Exception as exc:
        _write_json(
            run_dir / "FAILED.json",
            {
                "status": "failed",
                "failed_at_utc": _utc_now(),
                "exception_type": type(exc).__name__,
                "message": str(exc),
            },
        )
        raise


def persist_run_to_colab_drive(
    artifacts: PromptSFTArtifacts,
    drive_output_root: Path | str,
    **persistence_kwargs: object,
) -> PromptSFTArtifacts:
    """Copy, flush, remount, and freshly hash-verify a completed run."""

    def validate_directory(run_dir: Path) -> dict[str, str]:
        return validate_complete_run(artifacts_for_run_dir(run_dir))

    destination = persist_directory_to_colab_drive(
        artifacts.run_dir,
        drive_output_root,
        validate_directory=validate_directory,
        **persistence_kwargs,
    )
    return artifacts_for_run_dir(destination)
