"""Single-GPU BF16 LoRA DPO against the unchanged pinned Qwen reference."""

from __future__ import annotations

import gc
import json
import math
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from scripts.harmony_eval.cases import DEFAULT_COST_COUNTS
from scripts.ecological_prompt_sft.runner import (
    PromptSFTArtifacts, _adapter_weights_path, _completion_time, _git_commit,
    _package_version, _required_hashes, _utc_now, _write_json, _write_jsonl,
    artifacts_for_run_dir, validate_complete_run,
)
from scripts.ecological_prompt_sft.data import sha256_file
from .data import (
    PREFERRED_SIDES, audit_trainer_dataset, load_preference_examples,
    render_preference_examples,
)

TRAINING_OBJECTIVE = "paired_option_sigmoid_dpo_v1"
SCORE_PRECISION = "fp32_logits_logsoftmax_sum_v1"
INITIAL_LOGP_ATOL = 1e-4
RUNTIME_VERSIONS = {
    "transformers": "4.56.2", "trl": "0.24.0", "peft": "0.17.1",
    "accelerate": "1.10.1", "datasets": "4.1.1",
}


@dataclass(frozen=True)
class DilemmaDPOConfig:
    output_root: Path | str
    preferred_side: str = "ecological"
    base_model: str = "Qwen/Qwen3-8B"
    model_revision: str = "b968826d9c46dd6066d109eabc6255188de91218"
    max_length: int = 1024
    num_train_epochs: float = 3.0
    learning_rate: float = 5e-6
    beta: float = 0.1
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    lora_rank: int = 16
    lora_alpha: int = 32
    # DPO disables dropout in both policy and reference for stable comparisons.
    lora_dropout: float = 0.0
    seed: int = 42
    eval_batch_size: int = 2
    save_total_limit: int = 3
    cost_counts: tuple[int, ...] = DEFAULT_COST_COUNTS

    @property
    def pair_name(self):
        return f"qwen3_8b_ecological_dilemma_{self.preferred_side}_dpo"


def validate_config(config):
    if config.preferred_side not in PREFERRED_SIDES:
        raise ValueError(f"preferred_side must be one of {PREFERRED_SIDES}")
    if config.base_model != "Qwen/Qwen3-8B" or config.model_revision != (
        "b968826d9c46dd6066d109eabc6255188de91218"
    ):
        raise ValueError("This experiment requires the pinned historical Qwen3-8B base")
    for name in ("num_train_epochs", "learning_rate", "beta"):
        value = getattr(config, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in (
        "max_length", "per_device_train_batch_size", "gradient_accumulation_steps",
        "lora_rank", "lora_alpha", "eval_batch_size", "save_total_limit",
    ):
        value = getattr(config, name)
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config.max_length > 1024:
        raise ValueError("The A100 memory configuration supports max_length <= 1024")
    if config.per_device_train_batch_size != 1:
        raise ValueError("Use one preference pair per micro-batch on the A100")
    if config.lora_dropout != 0:
        raise ValueError("DPO disables dropout; set lora_dropout=0")
    if not 0 <= config.warmup_ratio <= 1 or not math.isfinite(config.weight_decay) or config.weight_decay < 0:
        raise ValueError("Invalid warmup_ratio or weight_decay")


def config_dict(config):
    result = asdict(config)
    result["output_root"] = str(config.output_root)
    result["cost_counts"] = list(config.cost_counts)
    return result


def training_signature(config):
    values = config_dict(config) if isinstance(config, DilemmaDPOConfig) else dict(config)
    for key in ("output_root", "eval_batch_size", "cost_counts", "save_total_limit"):
        values.pop(key, None)
    return values


def find_compatible_dpo_run(output_root, config) -> PromptSFTArtifacts | None:
    validate_config(config)
    _, manifest = load_preference_examples(config.preferred_side)
    root = Path(output_root)
    if not root.is_dir():
        return None
    for run_dir in sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: (_completion_time(path), path.name), reverse=True,
    ):
        artifacts = artifacts_for_run_dir(run_dir)
        try:
            metadata = json.loads(artifacts.metadata_path.read_text())
            if (
                metadata.get("status") != "complete"
                or metadata.get("training_objective") != TRAINING_OBJECTIVE
                or metadata.get("score_precision") != SCORE_PRECISION
                or metadata.get("initial_reference_audit", {}).get("status") != "passed"
                or metadata.get("initial_reference_audit", {}).get("score_precision") != SCORE_PRECISION
                or metadata.get("initial_reference_audit", {}).get("example_count") != manifest["example_count"]
                or metadata.get("pair_name") != config.pair_name
                or metadata.get("preferred_side") != config.preferred_side
                or training_signature(metadata.get("config", {})) != training_signature(config)
                or metadata.get("dataset", {}).get("records_sha256") != manifest["records_sha256"]
                or metadata.get("dataset", {}).get("source_releases") != manifest["source_releases"]
                or any(metadata.get("packages", {}).get(key) != value for key, value in RUNTIME_VERSIONS.items())
            ):
                continue
            validate_complete_run(artifacts)
        except (OSError, ValueError, RuntimeError, TypeError):
            continue
        return artifacts
    return None


def make_training_arguments(config, checkpoints_dir, *, smoke_test=False):
    """Shared real TRL arguments; a tiny CPU test changes only execution settings."""
    from trl import DPOConfig

    return DPOConfig(
        output_dir=str(checkpoints_dir),
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate, beta=config.beta,
        loss_type="sigmoid", label_smoothing=0.0, reference_free=False,
        precompute_ref_log_probs=True, precompute_ref_batch_size=1,
        sync_ref_model=False, disable_dropout=True,
        max_length=config.max_length, max_prompt_length=None, max_completion_length=None,
        # Qwen supports logits_to_keep; avoid projecting the whole prompt to its large vocabulary.
        use_logits_to_keep=True,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=not smoke_test, tf32=not smoke_test, use_cpu=smoke_test,
        optim="adamw_torch" if smoke_test else "adamw_torch_fused",
        warmup_ratio=config.warmup_ratio, weight_decay=config.weight_decay,
        lr_scheduler_type="cosine", max_grad_norm=1.0,
        logging_strategy="steps", logging_steps=1,
        save_strategy="epoch", save_total_limit=config.save_total_limit,
        report_to="none", remove_unused_columns=False,
        dataloader_num_workers=0, dataloader_pin_memory=not smoke_test,
        seed=config.seed, data_seed=config.seed,
    )


def build_trainer(model, tokenizer, rendered, expected_tokens, config, checkpoints_dir, *, smoke_test=False):
    from datasets import Dataset
    from peft import LoraConfig
    from trl import DPOTrainer
    from transformers import TrainerCallback

    trainer = DPOTrainer(
        model=model, ref_model=None,
        args=make_training_arguments(config, checkpoints_dir, smoke_test=smoke_test),
        train_dataset=Dataset.from_list([
            {key: row[key] for key in ("id", "prompt", "chosen", "rejected")}
            for row in rendered
        ]),
        processing_class=tokenizer,
        peft_config=LoraConfig(
            task_type="CAUSAL_LM", r=config.lora_rank, lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout, target_modules="all-linear", bias="none",
        ),
    )
    # Install on the PEFT model, after wrapping. This runs in both reference and
    # policy forwards, before TRL's log-softmax and sum, even before Accelerate
    # prepares the model. Casting an already-summed BF16 score would be too late.
    trainer._dpo_fp32_logits_hook = trainer.model.register_forward_hook(_fp32_output_logits)
    trainer.initial_reference_audit = None

    class InitialReferenceAudit(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            # Trainer has now applied Accelerate's actual mixed-precision wrapper,
            # but has not taken the first optimizer step.
            trainer.initial_reference_audit = audit_initial_reference_scores(trainer)
            audit = trainer.initial_reference_audit
            print(
                f"Initial reference/policy check passed on {audit['example_count']} pairs; "
                f"max log-probability difference={audit['max_abs_logp_difference']:.8g}, "
                f"mean DPO loss={audit['mean_dpo_loss']:.8f} (expected log(2))."
            )

    trainer.add_callback(InitialReferenceAudit())
    audit_trainer_dataset(trainer, expected_tokens)
    if not trainer.is_peft_model or trainer.ref_model is not None:
        raise RuntimeError("Expected one PEFT model with the adapter-disabled frozen base reference")
    trainable = [name for name, value in trainer.model.named_parameters() if value.requires_grad]
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError("Only LoRA parameters may be trainable")
    return trainer


def _fp32_output_logits(module, inputs, output):
    """Keep BF16 model computation, but use FP32 for both DPO score reductions."""
    output.logits = output.logits.float()
    return output


def audit_initial_reference_scores(trainer):
    """Fail before optimization if the fresh policy differs from its reference.

    Run after Accelerate prepares the model. Match reference precomputation's
    one-pair batches to avoid introducing padding/batching differences.
    """
    import torch

    was_training = trainer.model.training
    rows = []
    trainer.model.eval()
    try:
        with torch.no_grad(), trainer.compute_loss_context_manager():
            for tokens in trainer.train_dataset:
                batch = trainer._prepare_inputs(trainer.data_collator([tokens]))
                scores = trainer.concatenated_forward(trainer.model, batch)
                differences = []
                for side in ("chosen", "rejected"):
                    policy = scores[f"{side}_logps"]
                    reference = batch[f"ref_{side}_logps"]
                    if policy.dtype != torch.float32 or reference.dtype != torch.float32:
                        raise RuntimeError("Initial DPO scores must be FP32 for both policy and reference")
                    difference = float((policy - reference).abs().max())
                    if not math.isfinite(difference) or difference > INITIAL_LOGP_ATOL:
                        raise RuntimeError(
                            f"Initial reference/policy mismatch for {tokens['id']} ({side}): "
                            f"absolute log-probability difference {difference:.8g} exceeds "
                            f"{INITIAL_LOGP_ATOL}. Training stopped before the first update."
                        )
                    differences.append(difference)
                losses, chosen_rewards, rejected_rewards = trainer.dpo_loss(
                    scores["chosen_logps"], scores["rejected_logps"],
                    batch["ref_chosen_logps"], batch["ref_rejected_logps"],
                )
                rows.append({
                    "id": tokens["id"],
                    "chosen_logps": float(scores["chosen_logps"].item()),
                    "rejected_logps": float(scores["rejected_logps"].item()),
                    "max_abs_logp_difference": max(differences),
                    "reward_margin": float((chosen_rewards - rejected_rewards).item()),
                    "dpo_loss": float(losses.item()),
                })
    finally:
        trainer.model.train(was_training)
    if not rows:
        raise RuntimeError("Cannot audit an empty DPO dataset")
    return {
        "status": "passed", "score_precision": SCORE_PRECISION,
        "stage": "after Accelerate preparation, before first optimizer update",
        "example_count": len(rows), "absolute_logp_tolerance": INITIAL_LOGP_ATOL,
        "max_abs_logp_difference": max(row["max_abs_logp_difference"] for row in rows),
        "max_abs_reward_margin": max(abs(row["reward_margin"]) for row in rows),
        "mean_dpo_loss": sum(row["dpo_loss"] for row in rows) / len(rows),
        "per_example": rows,
    }


def precompute_reference_audit(trainer, rendered):
    """Freeze and retain base-reference scores before the first optimizer update."""
    trainer.model.eval()
    # Match the forward-computation autocast contexts used during training too.
    # The output hook keeps the subsequent log-softmax and sum in FP32.
    with trainer.accelerator.autocast(), trainer.compute_loss_context_manager():
        trainer.get_train_dataloader()
    rows = []
    for source, tokens in zip(rendered, trainer.train_dataset):
        row = {"id": source["id"]}
        for key in ("ref_chosen_logps", "ref_rejected_logps"):
            row[key] = float(tokens[key])
            if not math.isfinite(row[key]):
                raise RuntimeError("Nonfinite reference log probability")
        rows.append(row)
    return rows


def run_dilemma_dpo(config: DilemmaDPOConfig) -> PromptSFTArtifacts:
    """Complete training locally; the notebook then durably persists the run."""
    validate_config(config)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Select a Colab A100 GPU runtime with BF16 support")
    gpu = torch.cuda.get_device_properties(0)
    if gpu.total_memory / 2**30 < 38:
        raise RuntimeError("This BF16 Qwen3-8B DPO workflow needs an A100 40 GB or larger")
    if any(_package_version(key) != value for key, value in RUNTIME_VERSIONS.items()):
        raise RuntimeError("Install requirements-colab-dpo.txt and restart the runtime if packages were already imported")
    set_seed(config.seed)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = Path(config.output_root) / f"{timestamp}_{config.pair_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    artifacts = artifacts_for_run_dir(run_dir)
    metadata = {
        "status": "running", "created_at_utc": _utc_now(),
        "training_objective": TRAINING_OBJECTIVE, "preferred_side": config.preferred_side,
        "score_precision": SCORE_PRECISION,
        "pair_name": config.pair_name, "config": config_dict(config),
        "repository_commit": _git_commit(),
    }
    _write_json(artifacts.metadata_path, metadata)
    model = trainer = None
    try:
        examples, manifest = load_preference_examples(config.preferred_side)
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model, revision=config.model_revision, use_fast=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        rendered, tokenized, token_audit = render_preference_examples(
            tokenizer, examples, max_length=config.max_length,
        )
        _write_jsonl(artifacts.prompts_path, rendered)
        manifest["tokenization"] = token_audit
        _write_json(artifacts.dataset_manifest_path, manifest)
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model, revision=config.model_revision,
            dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True,
        ).to("cuda")
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False
        torch.cuda.reset_peak_memory_stats()
        trainer = build_trainer(
            model, tokenizer, rendered, tokenized, config, artifacts.checkpoints_dir,
        )
        manifest["reference_log_probs"] = precompute_reference_audit(trainer, rendered)
        _write_json(artifacts.dataset_manifest_path, manifest)
        train_result = trainer.train()
        if not trainer.initial_reference_audit or trainer.initial_reference_audit["status"] != "passed":
            raise RuntimeError("Missing initial reference/policy precision audit")
        trainer.save_state()
        metrics = dict(train_result.metrics)
        if not math.isfinite(float(metrics["train_loss"])):
            raise RuntimeError("DPO training produced a nonfinite loss")
        metrics.update({
            "training_example_count": len(examples), "preferred_side": config.preferred_side,
            "training_objective": TRAINING_OBJECTIVE,
            "score_precision": SCORE_PRECISION,
            "initial_reference_audit": trainer.initial_reference_audit,
            "trainable_parameters": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
            "peak_allocated_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gpu_gib": torch.cuda.max_memory_reserved() / 2**30,
            # Store the log in the hashed metrics artifact as well as its readable copy.
            "log_history": trainer.state.log_history,
        })
        _write_json(artifacts.train_metrics_path, metrics)
        _write_json(run_dir / "training/log_history.json", trainer.state.log_history)
        trainer.model.save_pretrained(artifacts.final_adapter_dir, safe_serialization=True)
        tokenizer.save_pretrained(artifacts.final_adapter_dir)
        metadata.update({
            "status": "complete", "completed_at_utc": _utc_now(), "dataset": manifest,
            "initial_reference_audit": trainer.initial_reference_audit,
            "resolved_revisions": {
                config.base_model: config.model_revision,
                "final_adapter_sha256": sha256_file(_adapter_weights_path(artifacts.final_adapter_dir)),
            },
            "reference": {"model": config.base_model, "revision": config.model_revision,
                          "mode": "frozen base, adapter disabled, scores precomputed before training"},
            "hardware": {"gpu": gpu.name, "gpu_memory_gib": gpu.total_memory / 2**30,
                         "cuda": torch.version.cuda},
            "python": platform.python_version(),
            "packages": {key: _package_version(key) for key in (*RUNTIME_VERSIONS, "torch", "huggingface-hub")},
            "dpo_training_arguments": trainer.args.to_dict(),
        })
        _write_json(artifacts.metadata_path, metadata)
        _write_json(artifacts.complete_marker_path, {
            "status": "complete", "completed_at_utc": _utc_now(),
            "artifact_sha256": _required_hashes(artifacts),
        })
        validate_complete_run(artifacts)
        return artifacts
    except Exception as exc:
        _write_json(run_dir / "FAILED.json", {
            "status": "failed", "failed_at_utc": _utc_now(),
            "exception_type": type(exc).__name__, "message": str(exc),
        })
        raise
    finally:
        # Evaluations reload one model at a time; no training model stays on the GPU.
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()
