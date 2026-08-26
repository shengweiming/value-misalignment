"""Qwen chat formatting with loss restricted to the H4rmony R1 answer."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


IGNORE_INDEX = -100


def response_only_features(
    prefix_ids: Sequence[int],
    full_ids: Sequence[int],
    *,
    max_length: int,
) -> dict[str, list[int]]:
    """Mask prompt tokens and retain the complete assistant response as labels."""

    prefix = [int(token) for token in prefix_ids]
    full = [int(token) for token in full_ids]
    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    if full[: len(prefix)] != prefix:
        raise ValueError(
            "The full chat is not prefixed by the generation prompt; refusing to "
            "construct potentially misaligned response-only labels"
        )
    response = full[len(prefix) :]
    if not response:
        raise ValueError("The assistant response tokenized to an empty sequence")
    if len(response) >= max_length:
        raise ValueError(
            f"The assistant response uses {len(response)} tokens, leaving no prompt "
            f"space at max_length={max_length}"
        )

    retained_prefix = prefix[-(max_length - len(response)) :]
    input_ids = retained_prefix + response
    labels = [IGNORE_INDEX] * len(retained_prefix) + response
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def tokenize_r1_examples(
    tokenizer: Any,
    examples: Sequence[dict[str, object]],
    *,
    max_length: int,
) -> list[dict[str, object]]:
    """Render Qwen non-thinking chats and supervise assistant R1 tokens only."""

    tokenized: list[dict[str, object]] = []
    if tokenizer.eos_token is None:
        raise ValueError("The tokenizer must define an EOS token")

    for example in examples:
        user_messages = [{"role": "user", "content": str(example["prompt"])}]
        prefix_text = tokenizer.apply_chat_template(
            user_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        full_text = prefix_text + str(example["r1_answer"]) + tokenizer.eos_token
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        features = response_only_features(
            prefix_ids,
            full_ids,
            max_length=max_length,
        )
        tokenized.append(
            {
                **features,
                "prompt_id": example["prompt_id"],
                "sequence_length": len(features["input_ids"]),
                "supervised_token_count": sum(
                    label != IGNORE_INDEX for label in features["labels"]
                ),
            }
        )
    return tokenized


class TokenizedR1Dataset:
    """Minimal dataset wrapper accepted by ``transformers.Trainer``."""

    def __init__(self, examples: Sequence[dict[str, object]]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.examples[index]


class ResponseOnlyCollator:
    """Right-pad tokenized examples while preserving -100 prompt labels."""

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
            mask = list(feature["attention_mask"])
            item_labels = list(feature["labels"])
            padding = width - len(ids)
            input_ids.append(ids + [self.pad_token_id] * padding)
            attention_masks.append(mask + [0] * padding)
            labels.append(item_labels + [IGNORE_INDEX] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
