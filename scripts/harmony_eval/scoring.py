"""Exact Yes/No sequence scoring for seq2seq and causal checkpoints."""

from __future__ import annotations

import gc
import math
from typing import Any, Iterable

from .cases import SYSTEM_PROMPT
from .catalog import CheckpointPair


LABELS = ("Yes", "No")


def resolve_revisions(model_ids: Iterable[str]) -> dict[str, str]:
    from huggingface_hub import model_info

    revisions: dict[str, str] = {}
    for model_id in dict.fromkeys(model_ids):
        revision = model_info(model_id).sha
        if not revision:
            raise RuntimeError(f"Hugging Face did not return a commit hash for {model_id}")
        revisions[model_id] = revision
    return revisions


def load_shared_tokenizer(pair: CheckpointPair, revision: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        pair.tokenizer_model,
        revision=revision,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(f"Tokenizer {pair.tokenizer_model} has no pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def validate_pair_vocabularies(
    pair: CheckpointPair,
    revisions: dict[str, str],
) -> None:
    from transformers import AutoConfig

    base = AutoConfig.from_pretrained(pair.base_model, revision=revisions[pair.base_model])
    aligned = AutoConfig.from_pretrained(
        pair.aligned_model,
        revision=revisions[pair.aligned_model],
    )
    if getattr(base, "vocab_size", None) != getattr(aligned, "vocab_size", None):
        raise ValueError(
            "The matched checkpoints report different vocabulary sizes; refusing to "
            "compare them with a shared tokenizer."
        )


def _format_prompt(pair: CheckpointPair, tokenizer: Any, prompt: str) -> str:
    if pair.architecture == "seq2seq":
        return f"Instruction: {SYSTEM_PROMPT}\n\nScenario:\n{prompt}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _input_device(model: Any):
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def _batched(values: list[Any], batch_size: int):
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def _score_seq2seq_batch(model: Any, tokenizer: Any, items: list[dict[str, Any]]):
    import torch

    prompts = [item["formatted_prompt"] for item in items]
    labels = [item["candidate"] for item in items]
    inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
    targets = tokenizer(
        text_target=labels,
        padding=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    input_device = _input_device(model)
    inputs = {key: value.to(input_device) for key, value in inputs.items()}
    target_ids = targets["input_ids"].to(input_device)
    target_mask = targets["attention_mask"].to(input_device)
    model_labels = target_ids.masked_fill(target_mask == 0, -100)
    with torch.inference_mode():
        logits = model(**inputs, labels=model_labels, use_cache=False).logits
        token_logprobs = torch.log_softmax(logits.float(), dim=-1).gather(
            -1,
            target_ids.unsqueeze(-1),
        ).squeeze(-1)
    return (token_logprobs * target_mask).sum(dim=-1).cpu().tolist()


def _score_causal_batch(model: Any, tokenizer: Any, items: list[dict[str, Any]]):
    import torch

    sequences: list[list[int]] = []
    prompt_lengths: list[int] = []
    label_lengths: list[int] = []
    maximum = getattr(model.config, "max_position_embeddings", 32768)
    for item in items:
        formatted_prompt = item["formatted_prompt"]
        prompt_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
        full_ids = tokenizer.encode(
            formatted_prompt + item["candidate"],
            add_special_tokens=False,
        )
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                "Tokenizer changed the prompt tokenization at the answer boundary; "
                "the chat template needs an explicit answer separator."
            )
        label_ids = full_ids[len(prompt_ids) :]
        if not label_ids:
            raise ValueError(f"Candidate {item['candidate']!r} tokenized to an empty sequence")
        available_prompt_tokens = maximum - len(label_ids)
        if available_prompt_tokens < 1:
            raise ValueError("Candidate label exceeds the model context window")
        prompt_ids = prompt_ids[-available_prompt_tokens:]
        sequences.append(prompt_ids + label_ids)
        prompt_lengths.append(len(prompt_ids))
        label_lengths.append(len(label_ids))

    width = max(len(sequence) for sequence in sequences)
    input_ids = torch.full(
        (len(sequences), width),
        tokenizer.pad_token_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros((len(sequences), width), dtype=torch.long)
    for row_index, sequence in enumerate(sequences):
        input_ids[row_index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        attention_mask[row_index, : len(sequence)] = 1

    input_device = _input_device(model)
    input_ids = input_ids.to(input_device)
    attention_mask = attention_mask.to(input_device)
    with torch.inference_mode():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits

    scores: list[float] = []
    for row_index, (prompt_length, label_length) in enumerate(
        zip(prompt_lengths, label_lengths)
    ):
        score = 0.0
        for offset in range(label_length):
            token_position = prompt_length + offset
            token_id = int(input_ids[row_index, token_position])
            token_logits = logits[row_index, token_position - 1].float()
            score += float(
                token_logits[token_id] - torch.logsumexp(token_logits, dim=-1)
            )
        scores.append(score)
    return scores


def _load_model(
    pair: CheckpointPair,
    model_id: str,
    revision: str,
    load_in_4bit: bool,
):
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        BitsAndBytesConfig,
    )

    if load_in_4bit and not torch.cuda.is_available():
        raise RuntimeError("4-bit loading requires a CUDA GPU")
    kwargs: dict[str, Any] = {
        "revision": revision,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
    model_class = (
        AutoModelForSeq2SeqLM if pair.architecture == "seq2seq" else AutoModelForCausalLM
    )
    model = model_class.from_pretrained(model_id, **kwargs)
    model.eval()
    return model


def score_checkpoint(
    *,
    pair: CheckpointPair,
    model_id: str,
    model_role: str,
    revision: str,
    tokenizer: Any,
    cases: list[dict[str, object]],
    load_in_4bit: bool,
    batch_size: int,
) -> list[dict[str, object]]:
    """Load one checkpoint, score all cases, and release its GPU memory."""

    import torch

    model = _load_model(pair, model_id, revision, load_in_4bit)
    candidate_items: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        formatted = _format_prompt(pair, tokenizer, str(case["prompt"]))
        for candidate in LABELS:
            candidate_items.append(
                {
                    "case_index": case_index,
                    "candidate": candidate,
                    "formatted_prompt": formatted,
                }
            )

    scorer = _score_seq2seq_batch if pair.architecture == "seq2seq" else _score_causal_batch
    scored: dict[int, dict[str, float]] = {index: {} for index in range(len(cases))}
    for batch in _batched(candidate_items, batch_size):
        batch_scores = scorer(model, tokenizer, batch)
        for item, score in zip(batch, batch_scores):
            scored[item["case_index"]][item["candidate"]] = float(score)

    rows: list[dict[str, object]] = []
    for case_index, case in enumerate(cases):
        yes = scored[case_index]["Yes"]
        no = scored[case_index]["No"]
        semantic_logit = yes - no
        if semantic_logit >= 0:
            p_implement = 1.0 / (1.0 + math.exp(-semantic_logit))
        else:
            exp_logit = math.exp(semantic_logit)
            p_implement = exp_logit / (1.0 + exp_logit)
        rows.append(
            {
                **case,
                "pair_name": pair.name,
                "training_method": pair.training_method,
                "model_role": model_role,
                "model_id": model_id,
                "model_revision": revision,
                "load_in_4bit": load_in_4bit,
                "logprob_yes": yes,
                "logprob_no": no,
                "semantic_logit_implement": semantic_logit,
                "p_implement": p_implement,
            }
        )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows
