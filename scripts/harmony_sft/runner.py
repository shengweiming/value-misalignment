"""Train Qwen3-8B on H4rmony R1 answers and persist the run to Google Drive."""

from __future__ import annotations

import csv
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

from scripts.harmony_eval.analysis import compare_thresholds, save_curve_plot
from scripts.harmony_eval.cases import DEFAULT_COST_COUNTS, REPO_ROOT, build_cases
from scripts.harmony_eval.scoring import score_loaded_causal_checkpoint

from .data import load_harmony_r1_examples
from .tokenization import ResponseOnlyCollator, TokenizedR1Dataset, tokenize_r1_examples


PAIR_NAME = "qwen3_8b_harmony_r1_sft"
GOOGLE_DRIVE_ROOT = Path("/content/drive/MyDrive")


@dataclass(frozen=True)
class SFTConfig:
    output_root: Path | str
    base_model: str = "Qwen/Qwen3-8B"
    model_revision: str = "main"
    dataset_id: str = "neovalle/H4rmony"
    dataset_revision: str = "main"
    dataset_split: str = "train"
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
    require_google_drive: bool = True


@dataclass(frozen=True)
class SFTArtifacts:
    run_dir: Path
    final_adapter_dir: Path
    checkpoints_dir: Path
    train_metrics_path: Path
    raw_scores_path: Path
    thresholds_path: Path
    plot_path: Path
    metadata_path: Path
    complete_marker_path: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def validate_config(config: SFTConfig) -> Path:
    if config.max_length < 64:
        raise ValueError("max_length must be at least 64")
    if config.num_train_epochs <= 0:
        raise ValueError("num_train_epochs must be positive")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
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
    if not config.cost_counts or any(count < 0 for count in config.cost_counts):
        raise ValueError("cost_counts must contain non-negative integers")

    output_root = Path(config.output_root).expanduser()
    if config.require_google_drive:
        if not GOOGLE_DRIVE_ROOT.is_dir():
            raise RuntimeError(
                "Google Drive is not mounted at /content/drive/MyDrive. Run "
                "google.colab.drive.mount('/content/drive') before training."
            )
        drive_root = GOOGLE_DRIVE_ROOT.resolve()
        resolved_output = output_root.resolve(strict=False)
        if resolved_output != drive_root and drive_root not in resolved_output.parents:
            raise ValueError(
                "output_root must be inside /content/drive/MyDrive so checkpoints "
                "and results survive the Colab runtime"
            )
    return output_root


def _run_id(config: SFTConfig) -> str:
    if config.run_name is not None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", config.run_name):
            raise ValueError(
                "run_name must be 1-100 filename-safe letters, numbers, dots, "
                "underscores, or hyphens"
            )
        return config.run_name
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{PAIR_NAME}"


def _length_summary(tokenized: list[dict[str, object]]) -> dict[str, object]:
    sequence_lengths = sorted(int(row["sequence_length"]) for row in tokenized)
    supervised_lengths = sorted(
        int(row["supervised_token_count"]) for row in tokenized
    )

    def percentile(values: list[int], fraction: float) -> int:
        index = round((len(values) - 1) * fraction)
        return values[index]

    return {
        "example_count": len(tokenized),
        "sequence_tokens": {
            "mean": mean(sequence_lengths),
            "p50": percentile(sequence_lengths, 0.50),
            "p95": percentile(sequence_lengths, 0.95),
            "max": max(sequence_lengths),
        },
        "supervised_tokens": {
            "mean": mean(supervised_lengths),
            "p50": percentile(supervised_lengths, 0.50),
            "p95": percentile(supervised_lengths, 0.95),
            "max": max(supervised_lengths),
        },
    }


def _adapter_weights_path(adapter_dir: Path) -> Path:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = adapter_dir / name
        if candidate.is_file():
            return candidate
    raise RuntimeError(f"No adapter weight file was saved in {adapter_dir}")


def _validate_complete_artifacts(artifacts: SFTArtifacts) -> dict[str, str]:
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
    latest_checkpoint = checkpoint_dirs[-1]
    resumable_checkpoint_files = {
        "checkpoint_adapter_config": latest_checkpoint / "adapter_config.json",
        "checkpoint_trainer_state": latest_checkpoint / "trainer_state.json",
        "checkpoint_optimizer": latest_checkpoint / "optimizer.pt",
        "checkpoint_scheduler": latest_checkpoint / "scheduler.pt",
    }
    missing_checkpoint_files = [
        name for name, path in resumable_checkpoint_files.items() if not path.is_file()
    ]
    if missing_checkpoint_files:
        raise RuntimeError(
            "The latest Drive checkpoint is not resumable; missing files: "
            f"{missing_checkpoint_files}"
        )

    required = {
        "adapter_config": artifacts.final_adapter_dir / "adapter_config.json",
        "adapter_weights": _adapter_weights_path(artifacts.final_adapter_dir),
        "checkpoint_adapter_weights": _adapter_weights_path(latest_checkpoint),
        "checkpoint_trainer_state": resumable_checkpoint_files[
            "checkpoint_trainer_state"
        ],
        "train_metrics": artifacts.train_metrics_path,
        "raw_scores": artifacts.raw_scores_path,
        "thresholds": artifacts.thresholds_path,
        "curves": artifacts.plot_path,
        "metadata": artifacts.metadata_path,
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Run completed without required Drive artifacts: {missing}")
    return {name: _sha256_file(path) for name, path in required.items()}


def run_harmony_r1_sft(config: SFTConfig) -> SFTArtifacts:
    """Run base evaluation, R1-only LoRA SFT, and aligned evaluation on one GPU."""

    output_root = validate_config(config)
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / _run_id(config)
    run_dir.mkdir(parents=False, exist_ok=False)
    checkpoints_dir = run_dir / "checkpoints"
    final_adapter_dir = run_dir / "final_adapter"
    dataset_dir = run_dir / "dataset"
    training_dir = run_dir / "training"
    evaluation_dir = run_dir / "evaluation"
    metadata_path = run_dir / "run_metadata.json"
    artifacts = SFTArtifacts(
        run_dir=run_dir,
        final_adapter_dir=final_adapter_dir,
        checkpoints_dir=checkpoints_dir,
        train_metrics_path=training_dir / "train_metrics.json",
        raw_scores_path=evaluation_dir / "raw_scores.csv",
        thresholds_path=evaluation_dir / "thresholds.csv",
        plot_path=evaluation_dir / "curves.png",
        metadata_path=metadata_path,
        complete_marker_path=run_dir / "COMPLETE.json",
    )
    config_dict = asdict(config)
    config_dict["output_root"] = str(config.output_root)
    created_at_utc = _utc_now()
    _write_json(
        metadata_path,
        {
            "status": "running",
            "created_at_utc": created_at_utc,
            "repository_commit": _git_commit(),
            "config": config_dict,
        },
    )

    try:
        import torch
        from huggingface_hub import HfApi, model_info
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            set_seed,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("Qwen3-8B SFT requires a CUDA GPU")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This workflow requires a BF16-capable GPU such as A100")
        set_seed(config.seed)

        resolved_model_revision = model_info(
            config.base_model, revision=config.model_revision
        ).sha
        resolved_dataset_revision = HfApi().dataset_info(
            repo_id=config.dataset_id,
            revision=config.dataset_revision,
        ).sha
        if not resolved_model_revision or not resolved_dataset_revision:
            raise RuntimeError("Could not resolve immutable model and dataset revisions")

        examples, dataset_manifest = load_harmony_r1_examples(
            config.dataset_id,
            revision=resolved_dataset_revision,
            split=config.dataset_split,
        )
        if dataset_manifest["r1_conflict_count"]:
            print(
                "H4rmony data audit: resolved "
                f"{dataset_manifest['r1_conflict_count']} conflicting R1 "
                "record(s); details will be saved in dataset/manifest.json."
            )
        _write_jsonl(dataset_dir / "r1_examples.jsonl", examples)

        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model,
            revision=resolved_model_revision,
            use_fast=True,
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise RuntimeError("Qwen tokenizer has neither a pad token nor EOS token")
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        tokenized = tokenize_r1_examples(
            tokenizer,
            examples,
            max_length=config.max_length,
        )
        dataset_manifest["tokenization"] = _length_summary(tokenized)
        dataset_manifest["chat_template"] = (
            "Qwen apply_chat_template with enable_thinking=False; prompt labels are "
            "masked to -100 and loss is applied only to the R1 assistant response "
            "and its EOS token."
        )
        _write_json(dataset_dir / "manifest.json", dataset_manifest)

        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            revision=resolved_model_revision,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        model.to("cuda")
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False

        cases = build_cases(config.cost_counts)
        base_rows = score_loaded_causal_checkpoint(
            model=model,
            tokenizer=tokenizer,
            cases=cases,
            model_role="base",
            model_id=config.base_model,
            model_revision=resolved_model_revision,
            pair_name=PAIR_NAME,
            training_method="sft",
            batch_size=config.eval_batch_size,
            enable_thinking=False,
        )
        _write_csv(evaluation_dir / "raw_scores_base.csv", base_rows)

        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules="all-linear",
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()
        trainable_parameters = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        total_parameters = sum(parameter.numel() for parameter in model.parameters())

        training_arguments = TrainingArguments(
            output_dir=str(checkpoints_dir),
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
            logging_steps=5,
            save_strategy="epoch",
            save_total_limit=config.save_total_limit,
            report_to="none",
            remove_unused_columns=False,
            dataloader_num_workers=2,
            dataloader_pin_memory=True,
            seed=config.seed,
            data_seed=config.seed,
        )
        trainer = Trainer(
            model=model,
            args=training_arguments,
            train_dataset=TokenizedR1Dataset(tokenized),
            data_collator=ResponseOnlyCollator(tokenizer.pad_token_id),
        )
        train_result = trainer.train()
        trainer.save_state()
        training_dir.mkdir(parents=True, exist_ok=True)
        train_metrics = dict(train_result.metrics)
        train_metrics.update(
            {
                "trainable_parameters": trainable_parameters,
                "total_parameters_with_adapter": total_parameters,
                "trainable_parameter_fraction": trainable_parameters
                / total_parameters,
            }
        )
        _write_json(artifacts.train_metrics_path, train_metrics)
        _write_json(training_dir / "log_history.json", trainer.state.log_history)

        final_adapter_dir.mkdir(parents=True, exist_ok=False)
        trainer.model.save_pretrained(final_adapter_dir, safe_serialization=True)
        tokenizer.save_pretrained(final_adapter_dir)
        adapter_weights = _adapter_weights_path(final_adapter_dir)
        adapter_revision = _sha256_file(adapter_weights)

        trainer.model.eval()
        aligned_rows = score_loaded_causal_checkpoint(
            model=trainer.model,
            tokenizer=tokenizer,
            cases=cases,
            model_role="aligned",
            model_id=str(final_adapter_dir),
            model_revision=adapter_revision,
            pair_name=PAIR_NAME,
            training_method="sft",
            batch_size=config.eval_batch_size,
            enable_thinking=False,
        )
        all_rows = base_rows + aligned_rows
        _write_csv(artifacts.raw_scores_path, all_rows)
        comparisons = compare_thresholds(all_rows)
        _write_csv(artifacts.thresholds_path, comparisons)
        save_curve_plot(all_rows, artifacts.plot_path)

        template_manifest = {
            str(case["template"]): {
                "path": case["template_path"],
                "sha256": case["template_sha256"],
            }
            for case in cases
        }
        metadata = {
            "status": "complete",
            "created_at_utc": created_at_utc,
            "completed_at_utc": _utc_now(),
            "repository_commit": _git_commit(),
            "config": config_dict,
            "resolved_revisions": {
                config.base_model: resolved_model_revision,
                config.dataset_id: resolved_dataset_revision,
                "final_adapter_sha256": adapter_revision,
            },
            "dataset": dataset_manifest,
            "evaluation": {
                "pair_name": PAIR_NAME,
                "case_count_per_model": len(cases),
                "templates": template_manifest,
                "cost_counts": list(config.cost_counts),
                "enable_thinking": False,
                "threshold_definition": (
                    "P(implement)=0.5 after a nonincreasing PAVA fit; interpolation "
                    "is linear in log(1 + cost)."
                ),
            },
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
                    "datasets",
                    "peft",
                    "huggingface-hub",
                )
            },
            "artifacts_root": str(run_dir),
        }
        _write_json(metadata_path, metadata)
        hashes = _validate_complete_artifacts(artifacts)
        _write_json(
            artifacts.complete_marker_path,
            {
                "status": "complete",
                "completed_at_utc": _utc_now(),
                "run_dir": str(run_dir),
                "artifact_sha256": hashes,
            },
        )
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
