#!/usr/bin/env python3
"""Run paired-polarity Stage 1 probability sweeps against DeepSeek."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tomllib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parents[1]
POLARITIES = ("implement_question", "reject_question")


class ExperimentError(RuntimeError):
    """Raised when an experiment cannot produce a valid semantic score."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_path(value: str, *, base: Path = REPO_ROOT) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as config_file:
        config = tomllib.load(config_file)

    for section in ("experiment", "model", "elicitation"):
        if section not in config:
            raise ExperimentError(f"Config is missing the [{section}] section")

    experiment = config["experiment"]
    model = config["model"]
    elicitation = config["elicitation"]
    required_experiment = (
        "name",
        "prompt_files",
        "displacement_counts",
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

    counts = experiment["displacement_counts"]
    if not counts or any(not isinstance(value, int) or value < 0 for value in counts):
        raise ExperimentError("experiment.displacement_counts must be non-negative integers")
    if len(set(counts)) != len(counts):
        raise ExperimentError("experiment.displacement_counts contains duplicates")

    yes_label = elicitation["yes_label"]
    no_label = elicitation["no_label"]
    if any(not isinstance(label, str) or not label.strip() for label in (yes_label, no_label)):
        raise ExperimentError("Yes and No labels must be non-empty strings")
    if yes_label.strip() == no_label.strip():
        raise ExperimentError("Yes and No labels must be different")

    top_logprobs = int(model.get("top_logprobs", 20))
    if not 1 <= top_logprobs <= 20:
        raise ExperimentError("model.top_logprobs must be between 1 and 20")

    return config


def render_prompt(
    template: str,
    elicitation: dict[str, Any],
    displaced_families: int,
    polarity: str,
) -> tuple[str, str, str]:
    if polarity not in POLARITIES:
        raise ExperimentError(f"Unknown question polarity: {polarity}")

    yes_label = elicitation["yes_label"]
    no_label = elicitation["no_label"]
    question = elicitation[polarity]
    implementation_label = yes_label if polarity == "implement_question" else no_label
    variables = {
        "displaced_families": displaced_families,
        "family_word": "family" if displaced_families == 1 else "families",
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

        for displaced_families in experiment["displacement_counts"]:
            for polarity in POLARITIES:
                prompt, implementation_label, question = render_prompt(
                    template,
                    elicitation,
                    displaced_families,
                    polarity,
                )
                yield {
                    "prompt_id": prompt_path.stem,
                    "prompt_path": str(prompt_path),
                    "template_sha256": sha256_text(template),
                    "displaced_families": displaced_families,
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
    return logsumexp([float(item["logprob"]) for item in variants]), variants


def probability_from_logit(logit: float) -> float:
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exp_logit = math.exp(logit)
    return exp_logit / (1.0 + exp_logit)


def binary_probability(logprob_positive: float, logprob_negative: float) -> float:
    return probability_from_logit(logprob_positive - logprob_negative)


def create_client(model_config: dict[str, Any]) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ExperimentError(
            "The openai package is not installed. Run: "
            "python -m pip install -r requirements.txt"
        ) from exc

    api_key_env = model_config["api_key_env"]
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ExperimentError(
            f"Environment variable {api_key_env} is not set. "
            "Export your DeepSeek API key before running the experiment."
        )

    return OpenAI(
        api_key=api_key,
        base_url=model_config["base_url"],
        timeout=float(model_config.get("timeout_seconds", 90.0)),
        max_retries=int(model_config.get("max_retries", 3)),
    )


def score_case(
    client: Any,
    config: dict[str, Any],
    case: dict[str, Any],
) -> dict[str, Any]:
    model = config["model"]
    experiment = config["experiment"]
    elicitation = config["elicitation"]
    session_id = f"stage1-{uuid4()}"

    # Each answer is a fresh one-turn chat. The unique user_id also isolates
    # DeepSeek's request-side KV cache; no conversation history is ever reused.
    messages = [
        {"role": "system", "content": experiment["system_prompt"]},
        {"role": "user", "content": case["prompt"]},
    ]
    response = client.chat.completions.create(
        model=model["name"],
        messages=messages,
        max_tokens=1,
        temperature=float(model.get("temperature", 1.0)),
        top_p=float(model.get("top_p", 1.0)),
        logprobs=True,
        top_logprobs=int(model.get("top_logprobs", 20)),
        stream=False,
        extra_body={
            "thinking": {"type": "disabled"},
            "user_id": session_id,
        },
    )

    choice = response.choices[0]
    if choice.logprobs is None or not choice.logprobs.content:
        raise ExperimentError("DeepSeek returned no token log probabilities")

    first_token = choice.logprobs.content[0]
    top_tokens = [
        {
            "token": token.token,
            "logprob": float(token.logprob),
            "bytes": token.bytes,
        }
        for token in first_token.top_logprobs
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

    usage = response.usage
    return {
        **case,
        "experiment": experiment["name"],
        "session_id": session_id,
        "session_mode": "stateless_one_turn_unique_user_id",
        "model_requested": model["name"],
        "model_returned": getattr(response, "model", None),
        "system_fingerprint": getattr(response, "system_fingerprint", None),
        "response_id": getattr(response, "id", None),
        "generated_text": choice.message.content,
        "first_generated_token": first_token.token,
        "yes_logprob": logprob_yes,
        "no_logprob": logprob_no,
        "yes_variants": variants_yes,
        "no_variants": variants_no,
        "p_yes_binary": probability_yes,
        "p_implement": probability_implement,
        "semantic_logit_implement": semantic_logit_implement,
        "top_logprobs": top_tokens,
        "input_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "output_tokens": getattr(usage, "completion_tokens", None) if usage else None,
        "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
    }


RAW_CSV_FIELDS = [
    "run_id",
    "experiment",
    "prompt_id",
    "displaced_families",
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
    "model_requested",
    "model_returned",
    "system_fingerprint",
    "response_id",
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
        key = (result["prompt_id"], result["displaced_families"])
        grouped[key][result["question_polarity"]] = result

    summaries = []
    for (prompt_id, displaced_families), polarities in sorted(grouped.items()):
        missing = set(POLARITIES) - set(polarities)
        if missing:
            raise ExperimentError(
                f"Cannot symmetrize {prompt_id}/{displaced_families}; "
                f"missing {sorted(missing)}"
            )
        implement = polarities["implement_question"]
        reject = polarities["reject_question"]
        p_implement = float(implement["p_implement"])
        p_reject_reversed = float(reject["p_implement"])
        logit_implement = float(implement["semantic_logit_implement"])
        logit_reject_reversed = float(reject["semantic_logit_implement"])
        mean_semantic_logit = (logit_implement + logit_reject_reversed) / 2.0
        summaries.append(
            {
                "run_id": run_id,
                "prompt_id": prompt_id,
                "displaced_families": displaced_families,
                "p_implement_from_implement_question": p_implement,
                "p_implement_from_reject_question": p_reject_reversed,
                "p_implement_arithmetic_mean": (p_implement + p_reject_reversed) / 2.0,
                "absolute_polarity_gap": abs(p_implement - p_reject_reversed),
                "semantic_logit_implement_question": logit_implement,
                "semantic_logit_reject_question": logit_reject_reversed,
                "semantic_logit_mean": mean_semantic_logit,
                "p_implement_logodds_sym": probability_from_logit(mean_semantic_logit),
                "polarity_effect_logit_reject_minus_implement": (
                    logit_reject_reversed - logit_implement
                ),
            }
        )
    return summaries


SUMMARY_FIELDS = [
    "run_id",
    "prompt_id",
    "displaced_families",
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
) -> None:
    metadata = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": sha256_text(config_path.read_text(encoding="utf-8")),
        "config": config,
        "python_version": sys.version,
        "requests_completed": len(results),
        "session_policy": (
            "Every answer uses a stateless one-turn request, a fresh messages list, "
            "and a unique DeepSeek user_id."
        ),
        "unique_session_ids": len({result["session_id"] for result in results}),
        "system_fingerprints": sorted(
            {result["system_fingerprint"] for result in results if result["system_fingerprint"]}
        ),
    }
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_experiment(config_path: Path, config: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    cases = list(experiment_cases(config))
    client = create_client(config["model"])
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = resolve_path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{run_id}_{config['experiment']['name']}"
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
                f"families={case['displaced_families']} | {case['question_polarity']}",
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

    write_metadata(metadata_path, config_path, config, run_id, results)
    return raw_csv_path, raw_jsonl_path, summary_path, metadata_path


def print_dry_run(config: dict[str, Any]) -> None:
    cases = list(experiment_cases(config))
    print(f"Rendered {len(cases)} independent requests; no API calls will be made.\n")
    for index, case in enumerate(cases, start=1):
        print(
            f"--- Request {index}: {case['prompt_id']} | "
            f"families={case['displaced_families']} | {case['question_polarity']} ---"
        )
        print(case["prompt"])
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure paired-polarity P(implement) across displacement levels "
            "with DeepSeek logprobs."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/stage_1.toml",
        help="Path to the TOML experiment config (default: configs/stage_1.toml)",
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
    try:
        config = load_config(config_path)
        if args.dry_run:
            print_dry_run(config)
            return 0
        paths = run_experiment(config_path, config)
    except (ExperimentError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("\nExperiment complete.")
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
