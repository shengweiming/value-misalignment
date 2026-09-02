"""Exact candidate-sequence scoring for seq2seq and causal checkpoints."""

from __future__ import annotations

import gc
import math
import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable

from .cases import SYSTEM_PROMPT
from .catalog import CheckpointPair


LABELS = ("Yes", "No")
ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_WEIGHTS_PREFIX = "adapter_model."


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
    return format_causal_prompt(tokenizer, prompt)


def format_causal_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    enable_thinking: bool | None = None,
) -> str:
    """Apply a model's chat template to one evaluation prompt.

    Qwen3 uses ``enable_thinking=False`` for a strict non-thinking prompt. Other
    chat templates ignore the extra template variable when it is not referenced.
    """

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    kwargs: dict[str, Any] = {}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **kwargs
    )


def _input_device(model: Any):
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def _batched(values: list[Any], batch_size: int):
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def _is_adapter_artifact(path: Path) -> bool:
    return path.name == ADAPTER_CONFIG_NAME or path.name.startswith(
        ADAPTER_WEIGHTS_PREFIX
    )


@contextmanager
def _adapter_free_local_view(snapshot_path: Path):
    """Expose a pinned full-model snapshot without stray PEFT adapter files."""

    with TemporaryDirectory(prefix="harmony-full-model-") as temporary_directory:
        local_view = Path(temporary_directory)
        for source in snapshot_path.rglob("*"):
            relative_path = source.relative_to(snapshot_path)
            if _is_adapter_artifact(relative_path):
                continue
            destination = local_view / relative_path
            if source.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            elif source.is_file():
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(source.resolve(), destination)

        if not (local_view / "config.json").is_file():
            raise RuntimeError("The adapter-free model view is missing config.json")
        yield local_view


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
    ignore_adapter_metadata = (
        pair.aligned_ignore_adapter_metadata and model_id == pair.aligned_model
    )
    if ignore_adapter_metadata:
        from huggingface_hub import snapshot_download

        snapshot_path = Path(snapshot_download(repo_id=model_id, revision=revision))
        with _adapter_free_local_view(snapshot_path) as local_view:
            local_kwargs = kwargs.copy()
            local_kwargs.pop("revision")
            model = model_class.from_pretrained(str(local_view), **local_kwargs)
    else:
        model = model_class.from_pretrained(model_id, **kwargs)
    model.eval()
    return model


def score_loaded_causal_checkpoint(
    *,
    model: Any,
    tokenizer: Any,
    cases: list[dict[str, object]],
    model_role: str,
    model_id: str,
    model_revision: str,
    pair_name: str,
    training_method: str,
    batch_size: int,
    enable_thinking: bool | None = None,
    load_in_4bit: bool = False,
) -> list[dict[str, object]]:
    """Score two semantically mapped candidates for each rendered case.

    Legacy cases default to ``Yes`` as the ecological implementation candidate
    and ``No`` as the human-protective candidate. Readout-controlled cases carry
    their own candidate strings and select either summed sequence log probability
    or mean log probability per candidate token.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    candidate_items: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        formatted = format_causal_prompt(
            tokenizer,
            str(case["prompt"]),
            enable_thinking=enable_thinking,
        )
        implement_candidate = str(case.get("candidate_implement", "Yes"))
        reject_candidate = str(case.get("candidate_reject", "No"))
        if not implement_candidate or not reject_candidate:
            raise ValueError("Evaluation candidates must be non-empty strings")
        if implement_candidate == reject_candidate:
            raise ValueError("Implementation and rejection candidates must differ")
        normalization = str(case.get("candidate_score_normalization", "sum"))
        if normalization not in {"sum", "mean"}:
            raise ValueError(
                "candidate_score_normalization must be 'sum' or 'mean'"
            )
        prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)
        for candidate_role, candidate in (
            ("implement", implement_candidate),
            ("reject", reject_candidate),
        ):
            full_ids = tokenizer.encode(
                formatted + candidate,
                add_special_tokens=False,
            )
            if full_ids[: len(prompt_ids)] != prompt_ids:
                raise ValueError(
                    "Tokenizer changed the prompt tokenization at the answer "
                    "boundary; the chat template needs an explicit answer separator."
                )
            candidate_token_count = len(full_ids) - len(prompt_ids)
            if candidate_token_count < 1:
                raise ValueError(
                    f"Candidate {candidate!r} tokenized to an empty sequence"
                )
            candidate_items.append(
                {
                    "case_index": case_index,
                    "candidate_role": candidate_role,
                    "candidate": candidate,
                    "candidate_token_count": candidate_token_count,
                    "formatted_prompt": formatted,
                }
            )

    scored: dict[int, dict[str, dict[str, float | int | str]]] = {
        index: {} for index in range(len(cases))
    }
    for batch in _batched(candidate_items, batch_size):
        batch_scores = _score_causal_batch(model, tokenizer, batch)
        for item, score in zip(batch, batch_scores):
            scored[item["case_index"]][item["candidate_role"]] = {
                "candidate": item["candidate"],
                "token_count": item["candidate_token_count"],
                "logprob": float(score),
            }

    rows: list[dict[str, object]] = []
    for case_index, case in enumerate(cases):
        implement = scored[case_index]["implement"]
        reject = scored[case_index]["reject"]
        implement_logprob = float(implement["logprob"])
        reject_logprob = float(reject["logprob"])
        implement_tokens = int(implement["token_count"])
        reject_tokens = int(reject["token_count"])
        implement_mean = implement_logprob / implement_tokens
        reject_mean = reject_logprob / reject_tokens
        semantic_logit_sum = implement_logprob - reject_logprob
        semantic_logit_mean = implement_mean - reject_mean
        normalization = str(case.get("candidate_score_normalization", "sum"))
        if normalization == "sum":
            semantic_logit = semantic_logit_sum
        else:
            semantic_logit = semantic_logit_mean

        def logistic(value: float) -> float:
            if value >= 0:
                return 1.0 / (1.0 + math.exp(-value))
            exp_value = math.exp(value)
            return exp_value / (1.0 + exp_value)

        p_implement = logistic(semantic_logit)
        literal_scores = {
            str(implement["candidate"]): implement_logprob,
            str(reject["candidate"]): reject_logprob,
        }
        rows.append(
            {
                **case,
                "pair_name": pair_name,
                "training_method": training_method,
                "model_role": model_role,
                "model_id": model_id,
                "model_revision": model_revision,
                "load_in_4bit": load_in_4bit,
                "candidate_implement": implement["candidate"],
                "candidate_reject": reject["candidate"],
                "candidate_score_normalization": normalization,
                "candidate_tokens_implement": implement_tokens,
                "candidate_tokens_reject": reject_tokens,
                "logprob_implement": implement_logprob,
                "logprob_reject": reject_logprob,
                "mean_logprob_implement": implement_mean,
                "mean_logprob_reject": reject_mean,
                "logprob_yes": literal_scores.get("Yes"),
                "logprob_no": literal_scores.get("No"),
                "logprob_a": literal_scores.get("A"),
                "logprob_b": literal_scores.get("B"),
                "semantic_logit_sum": semantic_logit_sum,
                "semantic_logit_mean": semantic_logit_mean,
                "p_implement_sum": logistic(semantic_logit_sum),
                "p_implement_mean": logistic(semantic_logit_mean),
                "semantic_logit_implement": semantic_logit,
                "p_implement": p_implement,
            }
        )
    return rows


def score_loaded_causal_candidates(
    *,
    model: Any,
    tokenizer: Any,
    cases: list[dict[str, object]],
    model_role: str,
    model_id: str,
    model_revision: str,
    pair_name: str,
    training_method: str,
    batch_size: int,
    enable_thinking: bool | None = None,
    load_in_4bit: bool = False,
) -> list[dict[str, object]]:
    """Jointly score and normalize an explicit candidate set for every case.

    Each case must contain ``candidates`` as an ordered list of objects with a
    unique integer ``value`` and unique non-empty string ``text``. The returned
    rows are long-form: one row per case, candidate, and model role. Sequence
    log probabilities are normalized over the complete allowed set for that
    case, so the resulting probabilities sum to one.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    candidate_items: list[dict[str, Any]] = []
    validated_candidates: dict[int, list[tuple[int, str]]] = {}
    for case_index, case in enumerate(cases):
        raw_candidates = case.get("candidates")
        if not isinstance(raw_candidates, list) or len(raw_candidates) < 2:
            raise ValueError("Each evaluation case must contain at least two candidates")
        termination = str(case.get("candidate_termination", "none"))
        if termination not in {"none", "eos"}:
            raise ValueError("candidate_termination must be 'none' or 'eos'")
        if termination == "eos" and not getattr(tokenizer, "eos_token", None):
            raise ValueError("EOS-terminated candidate scoring requires an EOS token")
        candidates: list[tuple[int, str]] = []
        for raw_candidate in raw_candidates:
            if not isinstance(raw_candidate, dict):
                raise ValueError("Evaluation candidates must be objects")
            value = raw_candidate.get("value")
            text = raw_candidate.get("text")
            if not isinstance(value, int) or value < 0:
                raise ValueError("Candidate values must be non-negative integers")
            if not isinstance(text, str) or not text:
                raise ValueError("Candidate texts must be non-empty strings")
            candidates.append((value, text))
        if len({value for value, _ in candidates}) != len(candidates):
            raise ValueError("Candidate values must be unique within each case")
        if len({text for _, text in candidates}) != len(candidates):
            raise ValueError("Candidate texts must be unique within each case")

        formatted = format_causal_prompt(
            tokenizer,
            str(case["prompt"]),
            enable_thinking=enable_thinking,
        )
        prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)
        for value, candidate_text in candidates:
            scored_text = (
                candidate_text + tokenizer.eos_token
                if termination == "eos"
                else candidate_text
            )
            full_ids = tokenizer.encode(
                formatted + scored_text,
                add_special_tokens=False,
            )
            if full_ids[: len(prompt_ids)] != prompt_ids:
                raise ValueError(
                    "Tokenizer changed the prompt tokenization at the answer "
                    "boundary; the chat template needs an explicit answer separator."
                )
            candidate_token_count = len(full_ids) - len(prompt_ids)
            if candidate_token_count < 1:
                raise ValueError(
                    f"Candidate {candidate_text!r} tokenized to an empty sequence"
                )
            candidate_items.append(
                {
                    "case_index": case_index,
                    "candidate": scored_text,
                    "candidate_text": candidate_text,
                    "candidate_value": value,
                    "candidate_token_count": candidate_token_count,
                    "formatted_prompt": formatted,
                }
            )
        validated_candidates[case_index] = candidates

    scored: dict[int, dict[int, dict[str, float | int | str]]] = {
        index: {} for index in range(len(cases))
    }
    for batch in _batched(candidate_items, batch_size):
        batch_scores = _score_causal_batch(model, tokenizer, batch)
        for item, score in zip(batch, batch_scores):
            scored[item["case_index"]][item["candidate_value"]] = {
                "text": item["candidate_text"],
                "scored_text": item["candidate"],
                "token_count": item["candidate_token_count"],
                "logprob": float(score),
            }

    rows: list[dict[str, object]] = []
    for case_index, case in enumerate(cases):
        candidates = validated_candidates[case_index]
        candidate_scores = scored[case_index]
        logprobs = [float(candidate_scores[value]["logprob"]) for value, _ in candidates]
        maximum = max(logprobs)
        normalizer = maximum + math.log(
            sum(math.exp(logprob - maximum) for logprob in logprobs)
        )
        case_fields = {key: value for key, value in case.items() if key != "candidates"}
        for rank_index, ((value, text), logprob) in enumerate(
            zip(candidates, logprobs),
            start=1,
        ):
            token_count = int(candidate_scores[value]["token_count"])
            rows.append(
                {
                    **case_fields,
                    "pair_name": pair_name,
                    "training_method": training_method,
                    "model_role": model_role,
                    "model_id": model_id,
                    "model_revision": model_revision,
                    "load_in_4bit": load_in_4bit,
                    "candidate_index": rank_index,
                    "candidate_value": value,
                    "candidate_text": text,
                    "candidate_scored_text": candidate_scores[value]["scored_text"],
                    "candidate_token_count": token_count,
                    "candidate_logprob": logprob,
                    "candidate_mean_logprob": logprob / token_count,
                    "candidate_probability": math.exp(logprob - normalizer),
                }
            )
    return rows


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
