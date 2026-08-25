"""Matched base and H4rmony-aligned model definitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


Architecture = Literal["seq2seq", "causal"]
TrainingMethod = Literal["sft", "dpo"]


@dataclass(frozen=True)
class CheckpointPair:
    name: str
    base_model: str
    aligned_model: str
    tokenizer_model: str
    architecture: Architecture
    training_method: TrainingMethod
    default_load_in_4bit: bool
    note: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


CHECKPOINT_PAIRS: dict[str, CheckpointPair] = {
    "caramel_sft": CheckpointPair(
        name="caramel_sft",
        base_model="google/flan-t5-large",
        aligned_model="neovalle/H4rmoniousCaramel",
        tokenizer_model="google/flan-t5-large",
        architecture="seq2seq",
        training_method="sft",
        default_load_in_4bit=False,
        note="The smallest and cleanest first Colab run; this is the released SFT model.",
    ),
    "anthea_dpo": CheckpointPair(
        name="anthea_dpo",
        base_model="teknium/OpenHermes-2.5-Mistral-7B",
        aligned_model="neovalle/H4rmoniousAnthea",
        tokenizer_model="teknium/OpenHermes-2.5-Mistral-7B",
        architecture="causal",
        training_method="dpo",
        default_load_in_4bit=True,
        note="Released DPO comparison; a Colab GPU needs 4-bit loading and usually High RAM.",
    ),
    "breeze_dpo": CheckpointPair(
        name="breeze_dpo",
        base_model="HuggingFaceH4/zephyr-7b-beta",
        aligned_model="neovalle/H4rmoniousBreezeDPO",
        tokenizer_model="HuggingFaceH4/zephyr-7b-beta",
        architecture="causal",
        training_method="dpo",
        default_load_in_4bit=True,
        note="Released Zephyr-based DPO comparison; a Colab GPU needs 4-bit loading.",
    ),
}


def get_checkpoint_pair(name: str) -> CheckpointPair:
    try:
        return CHECKPOINT_PAIRS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(CHECKPOINT_PAIRS))
        raise ValueError(f"Unknown checkpoint pair {name!r}; choose one of: {choices}") from exc
