"""Join the two audited answer releases; never invent or rewrite a response."""

from __future__ import annotations

import hashlib
import json
from statistics import mean

from scripts.ecological_prompt_sft.data import load_training_examples
from scripts.ecological_prompt_sft.runner import DEFAULT_DATASET_PATHS

PREFERRED_SIDES = ("ecological", "human")


def load_preference_examples(preferred_side: str):
    if preferred_side not in PREFERRED_SIDES:
        raise ValueError(f"preferred_side must be one of {PREFERRED_SIDES}")
    sources, manifests = {}, {}
    for side in PREFERRED_SIDES:
        arm = f"{side}_option"
        examples, manifest = load_training_examples(
            DEFAULT_DATASET_PATHS[arm], training_arm=arm,
        )
        sources[side] = {row["id"]: row for row in examples}
        manifests[side] = manifest
    if set(sources["ecological"]) != set(sources["human"]):
        raise ValueError("Ecological and human releases have different example IDs")
    rows = []
    other_side = "human" if preferred_side == "ecological" else "ecological"
    for example_id, ecological in sources["ecological"].items():
        human = sources["human"][example_id]
        if ecological["dilemma"] != human["dilemma"]:
            raise ValueError(f"Paired dilemmas differ for {example_id}")
        chosen = sources[preferred_side][example_id]["assistant_answer"]
        rejected = sources[other_side][example_id]["assistant_answer"]
        if not chosen.strip() or not rejected.strip() or chosen == rejected:
            raise ValueError(f"Invalid preference pair for {example_id}")
        rows.append({
            "id": example_id, "dilemma": ecological["dilemma"],
            "chosen": chosen, "rejected": rejected,
        })
    digest = hashlib.sha256(json.dumps(
        rows, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return rows, {
        "example_count": len(rows), "preferred_side": preferred_side,
        "records_sha256": digest, "source_releases": manifests,
        "contains_rationales": False, "contains_heldout_split": False,
        "chosen_source_field": f"{preferred_side}_protective_option"
        if preferred_side == "human" else "ecologically_protective_option",
        "rejected_source_field": f"{other_side}_protective_option"
        if other_side == "human" else "ecologically_protective_option",
    }


def render_preference_examples(tokenizer, examples, *, max_length: int):
    """Pre-render non-thinking prompts; TRL adds exactly one response EOS.

    Check separate tokenization against the complete chat before TRL can train.
    No prompt, response, or end token may be truncated.
    """
    if tokenizer.eos_token_id is None or tokenizer.eos_token is None:
        raise ValueError("DPO requires a terminating EOS token")
    rendered, tokenized, lengths = [], [], []
    for example in examples:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": example["dilemma"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        tokens = {"prompt_input_ids": prompt_ids}
        for name in ("chosen", "rejected"):
            response = example[name]
            response_ids = tokenizer.encode(response, add_special_tokens=False)
            if not response_ids or any(
                token in set(tokenizer.all_special_ids) for token in response_ids
            ):
                raise ValueError(f"Empty response or embedded special token: {example['id']}")
            completion_ids = response_ids + [tokenizer.eos_token_id]
            full_ids = tokenizer.encode(
                prompt + response + tokenizer.eos_token, add_special_tokens=False,
            )
            if full_ids != prompt_ids + completion_ids:
                raise ValueError(f"Prompt/response token boundary changed: {example['id']}")
            if len(full_ids) > max_length:
                raise ValueError(
                    f"Refusing to truncate {example['id']} {name}: "
                    f"{len(full_ids)} tokens exceeds max_length={max_length}"
                )
            tokens[f"{name}_input_ids"] = completion_ids
        rendered.append({**example, "prompt": prompt})
        tokenized.append(tokens)
        lengths.append({
            "id": example["id"], "prompt_tokens": len(prompt_ids),
            "chosen_tokens": len(tokens["chosen_input_ids"]),
            "rejected_tokens": len(tokens["rejected_input_ids"]),
            "max_sequence_tokens": len(prompt_ids) + max(
                len(tokens["chosen_input_ids"]), len(tokens["rejected_input_ids"]),
            ),
        })
    return rendered, tokenized, {
        "example_count": len(rendered), "per_example": lengths,
        "max_sequence_tokens": max(row["max_sequence_tokens"] for row in lengths),
        "mean_prompt_tokens": mean(row["prompt_tokens"] for row in lengths),
        "mean_chosen_tokens": mean(row["chosen_tokens"] for row in lengths),
        "mean_rejected_tokens": mean(row["rejected_tokens"] for row in lengths),
        "enable_thinking": False, "truncation": False,
        "completion_termination": "one tokenizer EOS; appended by TRL",
        "loss_scope": "completion sequence log probabilities only; prompt conditions both responses",
    }


def audit_trainer_dataset(trainer, expected_tokens):
    """Check the actual post-TRL dataset, not just an independent tokenizer."""
    if len(trainer.train_dataset) != len(expected_tokens):
        raise RuntimeError("TRL changed the preference-pair count")
    for index, expected in enumerate(expected_tokens):
        actual = trainer.train_dataset[index]
        for key, token_ids in expected.items():
            if actual[key] != token_ids:
                raise RuntimeError(f"TRL tokenization changed row {index} field {key}")
