"""Resolve saved, validated Qwen checkpoints for evaluation without training."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from scripts.ecological_dpo.runner import DilemmaDPOConfig, find_compatible_dpo_run
from scripts.ecological_prompt_sft.runner import (
    DilemmaSFTConfig, find_compatible_complete_run, validate_complete_run,
)
from scripts.harmony_sft import (
    SFTConfig as HarmonySFTConfig, find_compatible_complete_run as find_harmony_run,
    validate_complete_run as validate_harmony_run,
)
from scripts.harmony_sft.posthoc_eval import _adapter_weights, _sha256_file

MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
SFT_CHECKPOINT_SPECS = {
    "harmony_r1": {"source_kind": "harmony", "output_slug": "harmony_r1_qwen3_8b"},
    "ecological_prompt_only": {
        "training_arm": "prompt_only", "dataset_path": "data/ecological_dilemmas/v1/records.jsonl",
        "output_slug": "ecological_dilemma_prompt_qwen3_8b",
    },
    "ecological_option": {
        "training_arm": "ecological_option",
        "dataset_path": "data/ecological_dilemmas/sft/ecological_option/records.jsonl",
        "output_slug": "ecological_dilemma_ecological_option_qwen3_8b",
    },
    "ecological_option_10_epochs": {
        "training_arm": "ecological_option", "num_train_epochs": 10,
        "dataset_path": "data/ecological_dilemmas/sft/ecological_option/records.jsonl",
        "output_slug": "ecological_dilemma_ecological_option_qwen3_8b",
    },
    "human_option": {
        "training_arm": "human_option",
        "dataset_path": "data/ecological_dilemmas/sft/human_option/records.jsonl",
        "output_slug": "ecological_dilemma_human_option_qwen3_8b",
    },
    "clash_prompt_only": {
        "training_arm": "prompt_only", "dataset_path": "data/control_dilemmas/clash/v1/records.jsonl",
        "pair_name": "qwen3_8b_clash_prompt_control_sft", "output_slug": "clash_prompt_control_qwen3_8b",
    },
    "clash_action": {
        "training_arm": "action", "dataset_path": "data/control_dilemmas/clash/sft/action/records.jsonl",
        "pair_name": "qwen3_8b_clash_action_sft", "output_slug": "clash_action_qwen3_8b",
    },
}


def find_sft_checkpoint(drive_root: Path, checkpoint: str):
    """Use existing signature/hash checks, including corrected response masks."""
    if checkpoint not in SFT_CHECKPOINT_SPECS:
        raise ValueError(f"Unknown SFT checkpoint: {checkpoint}")
    spec = SFT_CHECKPOINT_SPECS[checkpoint]
    common = dict(output_root=Path("/content/evaluation-only-no-training"), base_model=MODEL_ID,
                  max_length=1024, num_train_epochs=spec.get("num_train_epochs", 3), learning_rate=1e-4,
                  per_device_train_batch_size=1, gradient_accumulation_steps=16,
                  lora_rank=16, lora_alpha=32, lora_dropout=.05, seed=42, eval_batch_size=4)
    root = Path(drive_root) / spec["output_slug"]
    if spec.get("source_kind") == "harmony":
        config = HarmonySFTConfig(**common, require_google_drive=False, dataset_id="neovalle/H4rmony")
        artifacts = find_harmony_run(root, config)
        validate = validate_harmony_run
    else:
        config = DilemmaSFTConfig(**common, model_revision=MODEL_REVISION,
                                  training_arm=spec["training_arm"], dataset_path=Path(spec["dataset_path"]),
                                  pair_name=spec.get("pair_name"))
        artifacts = find_compatible_complete_run(root, config)
        validate = validate_complete_run
    if artifacts is None:
        raise RuntimeError(f"No compatible, hash-verified {checkpoint} checkpoint under {root}. Evaluation will not train it.")
    validate(artifacts)
    return artifacts


@dataclass(frozen=True)
class QwenCheckpoint:
    condition: str
    label: str
    source_run_name: str
    training_method: str
    adapter_path: str | None = None
    adapter_sha256: str | None = None
    adapter_config_sha256: str | None = None
    source_complete_sha256: str | None = None
    source_metadata_sha256: str | None = None
    source_config: dict | None = None

    def identity(self) -> dict:
        return {**asdict(self), "base_model": MODEL_ID, "base_revision": MODEL_REVISION,
                "model_id": self.adapter_path or MODEL_ID,
                "model_revision": self.adapter_sha256 or MODEL_REVISION,
                "pair_name": f"qwen3_8b_{self.condition}", "model_role": self.condition}


def _from_artifacts(condition: str, label: str, artifacts) -> QwenCheckpoint:
    metadata = json.loads(artifacts.metadata_path.read_text())
    if (metadata["config"]["base_model"] != MODEL_ID
            or metadata["resolved_revisions"][MODEL_ID] != MODEL_REVISION):
        raise RuntimeError("Saved checkpoint does not use the pinned Qwen base")
    weights_hash = _sha256_file(_adapter_weights(artifacts.final_adapter_dir))
    if metadata["resolved_revisions"].get("final_adapter_sha256", weights_hash) != weights_hash:
        raise RuntimeError("Saved adapter hash differs from its recorded revision")
    return QwenCheckpoint(
        condition, label, artifacts.run_dir.name,
        metadata.get("training_objective") or "harmony_r1_response_only_sft",
        str(artifacts.final_adapter_dir), weights_hash,
        _sha256_file(artifacts.final_adapter_dir / "adapter_config.json"),
        _sha256_file(artifacts.complete_marker_path), _sha256_file(artifacts.metadata_path),
        metadata["config"],
    )


def resolve_qwen_checkpoints(
    drive_root: Path, *, conditions=("base", "sft", "dpo"),
    sft_checkpoint="ecological_option", dpo_preferred_side="ecological",
    dpo_epochs=3, dpo_beta=.1, dpo_learning_rate=5e-6,
) -> dict[str, QwenCheckpoint]:
    """Base-only evaluation never requires a saved adapter or training data."""
    if not conditions or len(set(conditions)) != len(conditions) or set(conditions) - {"base", "sft", "dpo"}:
        raise ValueError("Select a nonempty, unique subset of base, sft, dpo")
    result = {}
    for condition in conditions:
        if condition == "base":
            result[condition] = QwenCheckpoint("base", "Qwen3-8B (no project fine-tuning)", "qwen3_8b_unmodified", "none")
        elif condition == "sft":
            artifacts = find_sft_checkpoint(drive_root, sft_checkpoint)
            result[condition] = _from_artifacts(condition, f"Qwen3-8B SFT: {sft_checkpoint}", artifacts)
        else:
            config = DilemmaDPOConfig(output_root=Path("/content/evaluation-only-no-training"),
                                      preferred_side=dpo_preferred_side, num_train_epochs=dpo_epochs,
                                      beta=dpo_beta, learning_rate=dpo_learning_rate)
            root = Path(drive_root) / f"ecological_dilemma_{dpo_preferred_side}_dpo_qwen3_8b"
            artifacts = find_compatible_dpo_run(root, config)
            if artifacts is None:
                raise RuntimeError(f"No compatible, hash-verified corrected DPO checkpoint under {root}. Evaluation will not train it.")
            validate_complete_run(artifacts)
            result[condition] = _from_artifacts(condition, f"Qwen3-8B DPO: {dpo_preferred_side}", artifacts)
    return result


def verify_checkpoint_files(checkpoint: QwenCheckpoint) -> None:
    """Reject changed adapter/provenance files after checkpoint selection."""
    if checkpoint.condition not in {"base", "sft", "dpo"}:
        raise RuntimeError("Unknown Qwen condition")
    if checkpoint.condition == "base":
        if checkpoint.adapter_path is not None or checkpoint.training_method != "none":
            raise RuntimeError("Unmodified Qwen cannot contain an adapter")
        return
    if checkpoint.adapter_path is None:
        raise RuntimeError("Trained condition requires its saved adapter")
    adapter = Path(checkpoint.adapter_path)
    files = {
        _adapter_weights(adapter): checkpoint.adapter_sha256,
        adapter / "adapter_config.json": checkpoint.adapter_config_sha256,
        adapter.parent / "COMPLETE.json": checkpoint.source_complete_sha256,
        adapter.parent / "run_metadata.json": checkpoint.source_metadata_sha256,
    }
    if any(_sha256_file(path) != expected for path, expected in files.items()):
        raise RuntimeError("Selected Qwen checkpoint files changed")
