#!/usr/bin/env python3
"""Run configurable Stage 1 policy sweeps against hosted chat models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
import tomllib
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parents[1]
POLARITIES = ("implement_question", "reject_question")
PROVIDERS = ("deepseek_openai", "dashscope_openai", "dashscope_native")


class ExperimentError(RuntimeError):
    """Raised when an experiment cannot produce a valid semantic score."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_path(value: str, *, base: Path = REPO_ROOT) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def load_config(
    path: Path,
    model_profile_path: Path | None = None,
) -> dict[str, Any]:
    with path.open("rb") as config_file:
        config = tomllib.load(config_file)

    if model_profile_path is not None:
        with model_profile_path.open("rb") as profile_file:
            profile = tomllib.load(profile_file)
        if "model" not in profile:
            raise ExperimentError("Model profile is missing the [model] section")
        config["model"] = profile["model"]

    for section in ("experiment", "model", "elicitation"):
        if section not in config:
            raise ExperimentError(f"Config is missing the [{section}] section")

    experiment = config["experiment"]
    model = config["model"]
    elicitation = config["elicitation"]
    required_experiment = (
        "name",
        "prompt_files",
        "consequence_type",
        "family_counts",
        "output_dir",
        "system_prompt",
    )
    required_model = ("name", "base_url", "api_key_env")
    required_elicitation = (
        "yes_label",
        "no_label",
        "implement_question",
        "reject_question",
    )

    for key in required_experiment:
        if key not in experiment:
            raise ExperimentError(f"Config is missing experiment.{key}")
    for key in required_model:
        if key not in model:
            raise ExperimentError(f"Config is missing model.{key}")
    for key in required_elicitation:
        if key not in elicitation:
            raise ExperimentError(f"Config is missing elicitation.{key}")

    counts = experiment["family_counts"]
    if not counts or any(not isinstance(value, int) or value < 0 for value in counts):
        raise ExperimentError("experiment.family_counts must be non-negative integers")
    if len(set(counts)) != len(counts):
        raise ExperimentError("experiment.family_counts contains duplicates")

    polarities = experiment.get("question_polarities", list(POLARITIES))
    if not isinstance(polarities, list) or not polarities:
        raise ExperimentError("experiment.question_polarities must be a non-empty list")
    unknown_polarities = set(polarities) - set(POLARITIES)
    if unknown_polarities:
        raise ExperimentError(
            "experiment.question_polarities contains unknown values: "
            f"{sorted(unknown_polarities)}"
        )
    if len(set(polarities)) != len(polarities):
        raise ExperimentError("experiment.question_polarities contains duplicates")

    constitution_value = experiment.get("constitution_file")
    if constitution_value is not None:
        if not isinstance(constitution_value, str) or not constitution_value.strip():
            raise ExperimentError("experiment.constitution_file must be a non-empty path")
        constitution_path = resolve_path(constitution_value)
        if not constitution_path.is_file():
            raise ExperimentError(f"Constitution file not found: {constitution_path}")

    yes_label = elicitation["yes_label"]
    no_label = elicitation["no_label"]
    if any(not isinstance(label, str) or not label.strip() for label in (yes_label, no_label)):
        raise ExperimentError("Yes and No labels must be non-empty strings")
    if yes_label.strip() == no_label.strip():
        raise ExperimentError("Yes and No labels must be different")

    provider = model.get("provider", "deepseek_openai")
    if provider not in PROVIDERS:
        raise ExperimentError(
            f"model.provider must be one of {list(PROVIDERS)}; received {provider!r}"
        )

    top_logprobs = int(model.get("top_logprobs", 20))
    maximum_top_logprobs = 5 if provider.startswith("dashscope_") else 20
    if not 1 <= top_logprobs <= maximum_top_logprobs:
        raise ExperimentError(
            "model.top_logprobs must be between 1 and "
            f"{maximum_top_logprobs} for {provider}"
        )

    return config


def question_polarities(config: dict[str, Any]) -> tuple[str, ...]:
    return tuple(config["experiment"].get("question_polarities", POLARITIES))


def resolved_system_prompt(config: dict[str, Any]) -> dict[str, str | None]:
    experiment = config["experiment"]
    base_prompt = experiment["system_prompt"].strip()
    constitution_value = experiment.get("constitution_file")
    if constitution_value is None:
        return {
            "text": base_prompt,
            "sha256": sha256_text(base_prompt),
            "constitution_path": None,
            "constitution_sha256": None,
        }

    constitution_path = resolve_path(constitution_value)
    constitution = constitution_path.read_text(encoding="utf-8").strip()
    system_prompt = f"{base_prompt}\n\n{constitution}"
    return {
        "text": system_prompt,
        "sha256": sha256_text(system_prompt),
        "constitution_path": str(constitution_path),
        "constitution_sha256": sha256_text(constitution),
    }


def render_prompt(
    template: str,
    elicitation: dict[str, Any],
    family_count: int,
    polarity: str,
) -> tuple[str, str, str]:
    if polarity not in POLARITIES:
        raise ExperimentError(f"Unknown question polarity: {polarity}")

    yes_label = elicitation["yes_label"]
    no_label = elicitation["no_label"]
    question = elicitation[polarity]
    implementation_label = yes_label if polarity == "implement_question" else no_label
    variables = {
        "family_count": family_count,
        "family_word": "family" if family_count == 1 else "families",
        "cost_count": family_count,
        "cost": family_count,
        "decision_question": question,
        "yes_label": yes_label,
        "no_label": no_label,
    }
    try:
        prompt = template.format(**variables)
    except KeyError as exc:
        raise ExperimentError(f"Unknown prompt placeholder: {exc.args[0]}") from exc
    return prompt, implementation_label, question


def experiment_cases(config: dict[str, Any]) -> Iterable[dict[str, Any]]:
    experiment = config["experiment"]
    elicitation = config["elicitation"]
    for prompt_value in experiment["prompt_files"]:
        prompt_path = resolve_path(prompt_value)
        try:
            template = prompt_path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise ExperimentError(f"Prompt file not found: {prompt_path}") from exc

        for family_count in experiment["family_counts"]:
            for polarity in question_polarities(config):
                prompt, implementation_label, question = render_prompt(
                    template,
                    elicitation,
                    family_count,
                    polarity,
                )
                yield {
                    "prompt_id": prompt_path.stem,
                    "prompt_path": str(prompt_path),
                    "template_sha256": sha256_text(template),
                    "consequence_type": experiment["consequence_type"],
                    "family_count": family_count,
                    "question_polarity": polarity,
                    "question_text": question,
                    "implementation_label": implementation_label,
                    "prompt": prompt,
                    "prompt_sha256": sha256_text(prompt),
                }


def logsumexp(values: list[float]) -> float:
    if not values:
        raise ExperimentError("Cannot aggregate an empty set of log probabilities")
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def aggregate_label_logprob(
    top_tokens: list[dict[str, Any]], label: str
) -> tuple[float, list[dict[str, Any]]]:
    variants = [item for item in top_tokens if item["token"].strip() == label.strip()]
    if not variants:
        visible = ", ".join(repr(item["token"]) for item in top_tokens)
        raise ExperimentError(
            f"Label {label!r} was absent from the returned top tokens: {visible}"
        )
    if any(item.get("logprob") is None for item in variants):
        raise ExperimentError(
            f"Provider returned no numeric log probability for label {label!r}"
        )
    return logsumexp([float(item["logprob"]) for item in variants]), variants


def probability_from_logit(logit: float) -> float:
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exp_logit = math.exp(logit)
    return exp_logit / (1.0 + exp_logit)


def binary_probability(logprob_positive: float, logprob_negative: float) -> float:
    return probability_from_logit(logprob_positive - logprob_negative)


class DashScopeNativeClient:
    """Minimal native DashScope client used to preserve token log probabilities."""

    def __init__(
        self,
        *,
        api_key: str,
        endpoint: str,
        timeout: float,
        max_retries: int,
    ) -> None:
        self.api_key = api_key
        self.endpoint = endpoint
        self.timeout = timeout
        self.max_retries = max_retries

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        encoded_payload = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=encoded_payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    response_body = response.read().decode("utf-8")
                parsed = json.loads(response_body)
                if not isinstance(parsed, dict):
                    raise ExperimentError("DashScope returned a non-object JSON response")
                return parsed
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                retryable = exc.code == 429 or exc.code >= 500
                if retryable and attempt < self.max_retries:
                    time.sleep(2**attempt)
                    continue
                raise ExperimentError(
                    f"DashScope API request failed with HTTP {exc.code}: {error_body}"
                ) from exc
            except urllib.error.URLError as exc:
                if attempt < self.max_retries:
                    time.sleep(2**attempt)
                    continue
                raise ExperimentError(f"DashScope API connection failed: {exc}") from exc
            except json.JSONDecodeError as exc:
                raise ExperimentError("DashScope returned invalid JSON") from exc
        raise ExperimentError("DashScope API request failed after retries")


def create_client(model_config: dict[str, Any]) -> Any:
    api_key_env = model_config["api_key_env"]
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ExperimentError(
            f"Environment variable {api_key_env} is not set. "
            "Export the configured provider API key before running the experiment."
        )

    if model_config.get("provider") == "dashscope_native":
        return DashScopeNativeClient(
            api_key=api_key,
            endpoint=model_config["base_url"],
            timeout=float(model_config.get("timeout_seconds", 90.0)),
            max_retries=int(model_config.get("max_retries", 3)),
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ExperimentError(
            "The openai package is not installed. Run: "
            "python -m pip install -r requirements.txt"
        ) from exc

    return OpenAI(
        api_key=api_key,
        base_url=model_config["base_url"],
        timeout=float(model_config.get("timeout_seconds", 90.0)),
        max_retries=int(model_config.get("max_retries", 3)),
    )


def provider_request(
    model: dict[str, Any],
    messages: list[dict[str, Any]],
    session_id: str,
) -> tuple[dict[str, Any], str]:
    provider = model.get("provider", "deepseek_openai")
    request: dict[str, Any] = {
        "model": model["name"],
        "messages": messages,
        "max_tokens": 1,
        "temperature": float(model.get("temperature", 1.0)),
        "top_p": float(model.get("top_p", 1.0)),
        "logprobs": True,
        "top_logprobs": int(model.get("top_logprobs", 20)),
    }
    if provider == "deepseek_openai":
        request.update(
            {
                "stream": False,
                "extra_body": {
                    "thinking": {"type": "disabled"},
                    "user_id": session_id,
                },
            }
        )
        return request, "stateless_one_turn_unique_user_id"

    if provider == "dashscope_native":
        return {
            "model": model["name"],
            "input": {"messages": messages},
            "parameters": {
                "result_format": "message",
                "max_tokens": 1,
                "temperature": float(model.get("temperature", 1.0)),
                "top_p": float(model.get("top_p", 1.0)),
                "enable_thinking": False,
                "seed": int(model.get("seed", 0)),
                "logprobs": True,
                "top_logprobs": int(model.get("top_logprobs", 5)),
            },
        }, "stateless_one_turn_native_request"

    request.update(
        {
            "stream": True,
            "extra_body": {
                "enable_thinking": False,
                "seed": int(model.get("seed", 0)),
            },
            "extra_headers": {"x-dashscope-session-cache": "disable"},
        }
    )
    return request, "stateless_one_turn_cache_disabled"


def extract_response(response: Any, *, streamed: bool) -> dict[str, Any]:
    if not streamed:
        choice = response.choices[0]
        if choice.logprobs is None or not choice.logprobs.content:
            raise ExperimentError("Model returned no token log probabilities")
        return {
            "generated_text": choice.message.content or "",
            "first_token": choice.logprobs.content[0],
            "usage": response.usage,
            "response_id": getattr(response, "id", None),
            "model_returned": getattr(response, "model", None),
            "system_fingerprint": getattr(response, "system_fingerprint", None),
        }

    content_parts: list[str] = []
    first_token = None
    usage = None
    response_id = None
    model_returned = None
    system_fingerprint = None
    for chunk in response:
        response_id = getattr(chunk, "id", None) or response_id
        model_returned = getattr(chunk, "model", None) or model_returned
        system_fingerprint = (
            getattr(chunk, "system_fingerprint", None) or system_fingerprint
        )
        usage = getattr(chunk, "usage", None) or usage
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        choice = choices[0]
        delta = getattr(choice, "delta", None)
        content = getattr(delta, "content", None) if delta is not None else None
        if content:
            content_parts.append(content)
        logprobs = getattr(choice, "logprobs", None)
        logprob_content = getattr(logprobs, "content", None) if logprobs else None
        if first_token is None and logprob_content:
            first_token = logprob_content[0]

    if first_token is None:
        raise ExperimentError("Streamed model response returned no token log probabilities")
    return {
        "generated_text": "".join(content_parts),
        "first_token": first_token,
        "usage": usage,
        "response_id": response_id,
        "model_returned": model_returned,
        "system_fingerprint": system_fingerprint,
    }


def extract_native_response(response: dict[str, Any]) -> dict[str, Any]:
    try:
        choice = response["output"]["choices"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise ExperimentError("DashScope returned no model choice") from exc

    logprob_content = (choice.get("logprobs") or {}).get("content") or []
    if not logprob_content:
        raise ExperimentError(
            "DashScope native response returned no token log probabilities"
        )
    message = choice.get("message") or {}
    return {
        "generated_text": message.get("content") or "",
        "first_token": logprob_content[0],
        "usage": response.get("usage"),
        "response_id": response.get("request_id"),
        "model_returned": response.get("model"),
        "system_fingerprint": None,
    }


def object_value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def usage_value(usage: Any, *names: str) -> int | None:
    if usage is None:
        return None
    for name in names:
        value = object_value(usage, name)
        if value is not None:
            return int(value)
    return None


def score_case(
    client: Any,
    config: dict[str, Any],
    case: dict[str, Any],
) -> dict[str, Any]:
    model = config["model"]
    experiment = config["experiment"]
    elicitation = config["elicitation"]
    session_id = f"stage1-{uuid4()}"
    system_prompt = resolved_system_prompt(config)
    provider = model.get("provider", "deepseek_openai")

    # Every answer is a fresh one-turn chat with no conversation history.
    # Provider-specific settings disable cache/session continuity.
    messages = [
        {"role": "system", "content": system_prompt["text"]},
        {"role": "user", "content": case["prompt"]},
    ]
    request, session_mode = provider_request(model, messages, session_id)
    if provider == "dashscope_native":
        response = client.generate(request)
        extracted = extract_native_response(response)
    else:
        try:
            response = client.chat.completions.create(**request)
        except Exception as exc:
            if exc.__class__.__module__.split(".", 1)[0] == "openai":
                raise ExperimentError(
                    f"{model['name']} API request failed: {exc}"
                ) from exc
            raise
        extracted = extract_response(
            response,
            streamed=bool(request["stream"]),
        )
    first_token = extracted["first_token"]
    top_tokens = [
        {
            "token": object_value(token, "token"),
            "logprob": (
                float(object_value(token, "logprob"))
                if object_value(token, "logprob") is not None
                else None
            ),
            "bytes": object_value(token, "bytes"),
        }
        for token in object_value(first_token, "top_logprobs", [])
    ]
    yes_label = elicitation["yes_label"]
    no_label = elicitation["no_label"]
    logprob_yes, variants_yes = aggregate_label_logprob(top_tokens, yes_label)
    logprob_no, variants_no = aggregate_label_logprob(top_tokens, no_label)
    probability_yes = binary_probability(logprob_yes, logprob_no)

    if case["implementation_label"] == yes_label:
        probability_implement = probability_yes
        semantic_logit_implement = logprob_yes - logprob_no
    else:
        probability_implement = 1.0 - probability_yes
        semantic_logit_implement = logprob_no - logprob_yes

    usage = extracted["usage"]
    return {
        **case,
        "experiment": experiment["name"],
        "session_id": session_id,
        "session_mode": session_mode,
        "provider": provider,
        "model_requested": model["name"],
        "model_returned": extracted["model_returned"],
        "system_fingerprint": extracted["system_fingerprint"],
        "response_id": extracted["response_id"],
        "system_prompt": system_prompt["text"],
        "system_prompt_sha256": system_prompt["sha256"],
        "constitution_path": system_prompt["constitution_path"],
        "constitution_sha256": system_prompt["constitution_sha256"],
        "generated_text": extracted["generated_text"],
        "first_generated_token": object_value(first_token, "token"),
        "yes_logprob": logprob_yes,
        "no_logprob": logprob_no,
        "yes_variants": variants_yes,
        "no_variants": variants_no,
        "p_yes_binary": probability_yes,
        "p_implement": probability_implement,
        "semantic_logit_implement": semantic_logit_implement,
        "top_logprobs": top_tokens,
        "input_tokens": usage_value(usage, "prompt_tokens", "input_tokens"),
        "output_tokens": usage_value(usage, "completion_tokens", "output_tokens"),
        "total_tokens": usage_value(usage, "total_tokens"),
    }


RAW_CSV_FIELDS = [
    "run_id",
    "experiment",
    "prompt_id",
    "consequence_type",
    "family_count",
    "question_polarity",
    "implementation_label",
    "p_implement",
    "semantic_logit_implement",
    "p_yes_binary",
    "yes_logprob",
    "no_logprob",
    "generated_text",
    "first_generated_token",
    "session_id",
    "session_mode",
    "provider",
    "model_requested",
    "model_returned",
    "system_fingerprint",
    "response_id",
    "system_prompt_sha256",
    "constitution_path",
    "constitution_sha256",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "prompt_sha256",
    "template_sha256",
]


def raw_csv_row(result: dict[str, Any], run_id: str) -> dict[str, Any]:
    row = {field: result.get(field) for field in RAW_CSV_FIELDS}
    row["run_id"] = run_id
    return row


def summarize_results(results: list[dict[str, Any]], run_id: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for result in results:
        key = (result["prompt_id"], result["family_count"])
        grouped[key][result["question_polarity"]] = result

    summaries = []
    for (prompt_id, family_count), polarities in sorted(grouped.items()):
        if "implement_question" not in polarities:
            raise ExperimentError(
                f"Cannot summarize {prompt_id}/{family_count} without implement_question"
            )
        implement = polarities["implement_question"]
        reject = polarities.get("reject_question")
        p_implement = float(implement["p_implement"])
        logit_implement = float(implement["semantic_logit_implement"])
        if reject is None:
            summary_method = "direct_implement_question"
            p_reject_reversed = None
            logit_reject_reversed = None
            mean_semantic_logit = logit_implement
            arithmetic_mean = p_implement
            absolute_gap = None
            polarity_effect = None
        else:
            summary_method = "paired_polarity_logodds"
            p_reject_reversed = float(reject["p_implement"])
            logit_reject_reversed = float(reject["semantic_logit_implement"])
            mean_semantic_logit = (logit_implement + logit_reject_reversed) / 2.0
            arithmetic_mean = (p_implement + p_reject_reversed) / 2.0
            absolute_gap = abs(p_implement - p_reject_reversed)
            polarity_effect = logit_reject_reversed - logit_implement
        summaries.append(
            {
                "run_id": run_id,
                "prompt_id": prompt_id,
                "consequence_type": implement["consequence_type"],
                "family_count": family_count,
                "summary_method": summary_method,
                "p_implement_from_implement_question": p_implement,
                "p_implement_from_reject_question": p_reject_reversed,
                "p_implement_arithmetic_mean": arithmetic_mean,
                "absolute_polarity_gap": absolute_gap,
                "semantic_logit_implement_question": logit_implement,
                "semantic_logit_reject_question": logit_reject_reversed,
                "semantic_logit_mean": mean_semantic_logit,
                "p_implement_logodds_sym": probability_from_logit(mean_semantic_logit),
                "polarity_effect_logit_reject_minus_implement": polarity_effect,
            }
        )
    return summaries


SUMMARY_FIELDS = [
    "run_id",
    "prompt_id",
    "consequence_type",
    "family_count",
    "summary_method",
    "p_implement_from_implement_question",
    "p_implement_from_reject_question",
    "p_implement_arithmetic_mean",
    "absolute_polarity_gap",
    "semantic_logit_implement_question",
    "semantic_logit_reject_question",
    "semantic_logit_mean",
    "p_implement_logodds_sym",
    "polarity_effect_logit_reject_minus_implement",
]


def write_metadata(
    path: Path,
    config_path: Path,
    config: dict[str, Any],
    run_id: str,
    results: list[dict[str, Any]],
    model_profile_path: Path | None = None,
) -> None:
    system_prompt = resolved_system_prompt(config)
    metadata = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": sha256_text(config_path.read_text(encoding="utf-8")),
        "model_profile_path": (
            str(model_profile_path) if model_profile_path is not None else None
        ),
        "model_profile_sha256": (
            sha256_text(model_profile_path.read_text(encoding="utf-8"))
            if model_profile_path is not None
            else None
        ),
        "config": config,
        "resolved_system_prompt": system_prompt,
        "python_version": sys.version,
        "requests_completed": len(results),
        "session_policy": (
            "Every answer uses a stateless one-turn request and a fresh messages "
            "list. DeepSeek uses a unique user_id; native DashScope requests send "
            "no conversation or session identifier."
        ),
        "unique_session_ids": len({result["session_id"] for result in results}),
        "system_fingerprints": sorted(
            {result["system_fingerprint"] for result in results if result["system_fingerprint"]}
        ),
    }
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def filename_slug(value: str) -> str:
    slug = "".join(character if character.isalnum() else "_" for character in value)
    return slug.strip("_") or "model"


def run_experiment(
    config_path: Path,
    config: dict[str, Any],
    model_profile_path: Path | None = None,
) -> tuple[Path, Path, Path, Path]:
    cases = list(experiment_cases(config))
    client = create_client(config["model"])
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = resolve_path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_slug = filename_slug(config["model"]["name"])
    stem = f"{run_id}_{config['experiment']['name']}_{model_slug}"
    raw_csv_path = output_dir / f"{stem}_raw.csv"
    raw_jsonl_path = output_dir / f"{stem}_raw.jsonl"
    summary_path = output_dir / f"{stem}_summary.csv"
    metadata_path = output_dir / f"{stem}_metadata.json"

    results: list[dict[str, Any]] = []
    with raw_csv_path.open("w", encoding="utf-8", newline="") as csv_file, raw_jsonl_path.open(
        "w", encoding="utf-8"
    ) as jsonl_file:
        writer = csv.DictWriter(csv_file, fieldnames=RAW_CSV_FIELDS)
        writer.writeheader()
        for index, case in enumerate(cases, start=1):
            print(
                f"[{index}/{len(cases)}] {case['prompt_id']} | "
                f"families={case['family_count']} | {case['question_polarity']}",
                flush=True,
            )
            result = score_case(client, config, case)
            results.append(result)
            writer.writerow(raw_csv_row(result, run_id))
            csv_file.flush()
            jsonl_file.write(json.dumps({"run_id": run_id, **result}, ensure_ascii=False) + "\n")
            jsonl_file.flush()
            print(f"  P(implement)={result['p_implement']:.6f}", flush=True)

    summaries = summarize_results(results, run_id)
    with summary_path.open("w", encoding="utf-8", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summaries)

    write_metadata(
        metadata_path,
        config_path,
        config,
        run_id,
        results,
        model_profile_path,
    )
    return raw_csv_path, raw_jsonl_path, summary_path, metadata_path


def print_dry_run(config: dict[str, Any]) -> None:
    cases = list(experiment_cases(config))
    system_prompt = resolved_system_prompt(config)
    print(
        f"Model: {config['model']['name']} "
        f"({config['model'].get('provider', 'deepseek_openai')})"
    )
    print(f"Rendered {len(cases)} independent requests; no API calls will be made.\n")
    print("--- System message used for every independent request ---")
    print(system_prompt["text"])
    print()
    for index, case in enumerate(cases, start=1):
        print(
            f"--- Request {index}: {case['prompt_id']} | "
            f"families={case['family_count']} | {case['question_polarity']} ---"
        )
        print(case["prompt"])
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure P(implement) across configured family counts with first-token "
            "log probabilities."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/stage_1.toml",
        help="Path to the TOML experiment config (default: configs/stage_1.toml)",
    )
    parser.add_argument(
        "--model-profile",
        help=(
            "Optional TOML file whose [model] section replaces the model settings "
            "in the experiment config"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render every prompt without importing the API client or spending credit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    model_profile_path = (
        resolve_path(args.model_profile) if args.model_profile is not None else None
    )
    try:
        config = load_config(config_path, model_profile_path)
        if args.dry_run:
            print_dry_run(config)
            return 0
        paths = run_experiment(config_path, config, model_profile_path)
    except (ExperimentError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("\nExperiment complete.")
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
