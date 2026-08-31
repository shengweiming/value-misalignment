"""Qwen chat formatting for prompt-only and option-answer dilemma SFT."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .data import PROMPT_ONLY_ARM, TRAINING_ARMS


IGNORE_INDEX = -100


def tokenize_prompt_examples(
    tokenizer: Any,
    examples: Sequence[dict[str, object]],
    *,
    max_length: int,
) -> list[dict[str, object]]:
    """Render user-only chats and use every non-padding token as its own label."""

    if max_length < 64:
        raise ValueError("max_length must be at least 64")
    tokenized: list[dict[str, object]] = []
    for example in examples:
        messages = [{"role": "user", "content": str(example["dilemma"])}]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        input_ids = [
            int(token)
            for token in tokenizer.encode(rendered, add_special_tokens=False)
        ]
        if not input_ids:
            raise ValueError(f"Prompt {example.get('id')!r} tokenized to an empty sequence")
        if len(input_ids) > max_length:
            raise ValueError(
                f"Prompt {example.get('id')!r} uses {len(input_ids)} tokens, above "
                f"max_length={max_length}; refusing to truncate the dilemma"
            )
        tokenized.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": list(input_ids),
                "id": example["id"],
                "sequence_length": len(input_ids),
                "supervised_token_count": len(input_ids),
            }
        )
    return tokenized


def tokenize_answer_examples(
    tokenizer: Any,
    examples: Sequence[dict[str, object]],
    *,
    max_length: int,
) -> list[dict[str, object]]:
    """Render user/assistant chats and supervise only the exact option answer."""

    if max_length < 64:
        raise ValueError("max_length must be at least 64")
    if tokenizer.eos_token is None:
        raise ValueError("The tokenizer must define an EOS token")
    tokenized: list[dict[str, object]] = []
    for example in examples:
        user_messages = [{"role": "user", "content": str(example["dilemma"])}]
        prefix_text = tokenizer.apply_chat_template(
            user_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        full_text = prefix_text + str(example["assistant_answer"]) + tokenizer.eos_token
        prefix_ids = [
            int(token)
            for token in tokenizer.encode(prefix_text, add_special_tokens=False)
        ]
        full_ids = [
            int(token)
            for token in tokenizer.encode(full_text, add_special_tokens=False)
        ]
        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError(
                "The assistant chat is not prefixed by its generation prompt; "
                "refusing to build response-only labels"
            )
        response_ids = full_ids[len(prefix_ids) :]
        if not response_ids:
            raise ValueError(
                f"Answer record {example.get('id')!r} tokenized to an empty response"
            )
        if len(full_ids) > max_length:
            raise ValueError(
                f"Answer record {example.get('id')!r} uses {len(full_ids)} tokens, "
                f"above max_length={max_length}; refusing to truncate the dilemma"
            )
        labels = [IGNORE_INDEX] * len(prefix_ids) + response_ids
        tokenized.append(
            {
                "input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels": labels,
                "id": example["id"],
                "sequence_length": len(full_ids),
                "supervised_token_count": len(response_ids),
            }
        )
    return tokenized


def tokenize_training_examples(
    tokenizer: Any,
    examples: Sequence[dict[str, object]],
    *,
    training_arm: str,
    max_length: int,
) -> list[dict[str, object]]:
    """Tokenize one of the three supported dilemma-training arms."""

    if training_arm not in TRAINING_ARMS:
        raise ValueError(f"training_arm must be one of {list(TRAINING_ARMS)}")
    if training_arm == PROMPT_ONLY_ARM:
        return tokenize_prompt_examples(tokenizer, examples, max_length=max_length)
    return tokenize_answer_examples(tokenizer, examples, max_length=max_length)


class TokenizedPromptDataset:
    def __init__(self, examples: Sequence[dict[str, object]]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.examples[index]


class PromptOnlyCollator:
    """Right-pad causal-LM inputs while preserving each arm's label mask."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: Sequence[dict[str, object]]) -> dict[str, Any]:
        import torch

        width = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_masks = []
        labels = []
        for feature in features:
            ids = list(feature["input_ids"])
            padding = width - len(ids)
            input_ids.append(ids + [self.pad_token_id] * padding)
            attention_masks.append([1] * len(ids) + [0] * padding)
            labels.append(ids + [IGNORE_INDEX] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
