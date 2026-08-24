#!/usr/bin/env python3
"""Run the Stage 1 A/B displacement sweep against the DeepSeek API."""

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


REPO_ROOT = Path(__file__).resolve().parents[1]


class ExperimentError(RuntimeError):
    """Raised when an experiment cannot produce a valid binary score."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_path(value: str, *, base: Path = REPO_ROOT) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as config_file:
        config = tomllib.load(config_file)

    for section in ("experiment", "model", "choices"):
        if section not in config:
            raise ExperimentError(f"Config is missing the [{section}] section")

    experiment = config["experiment"]
    model = config["model"]
    choices = config["choices"]

    required_experiment = (
        "name",
        "prompt_files",
        "displacement_counts",
        "output_dir",
        "system_prompt",
    )
    required_model = ("name", "base_url", "api_key_env")
    required_choices = ("label_a", "label_b", "displace", "do_not_displace")

    for key in required_experiment:
        if key not in experiment:
            raise ExperimentError(f"Config is missing experiment.{key}")
    for key in required_model:
        if key not in model:
            raise ExperimentError(f"Config is missing model.{key}")
    for key in required_choices:
        if key not in choices:
            raise ExperimentError(f"Config is missing choices.{key}")

    counts = experiment["displacement_counts"]
    if not counts or any(not isinstance(value, int) or value < 0 for value in counts):
        raise ExperimentError("experiment.displacement_counts must be non-negative integers")
    if len(set(counts)) != len(counts):
        raise ExperimentError("experiment.displacement_counts contains duplicates")

    labels = (choices["label_a"], choices["label_b"])
    if any(not isinstance(label, str) or len(label) != 1 for label in labels):
        raise ExperimentError("Choice labels must each be exactly one character")
    if labels[0] == labels[1]:
        raise ExperimentError("Choice labels must be different")

    top_logprobs = int(model.get("top_logprobs", 20))
    if not 1 <= top_logprobs <= 20:
        raise ExperimentError("model.top_logprobs must be between 1 and 20")

    return config


def render_choice(template: str, displaced_families: int) -> str:
    family_word = "family" if displaced_families == 1 else "families"
    return template.format(
        displaced_families=displaced_families,
        family_word=family_word,
    )


def render_prompt(
    template: str,
    choices: dict[str, Any],
    displaced_families: int,
    ordering: str,
) -> tuple[str, str]:
    displace = render_choice(choices["displace"], displaced_families)
    do_not_displace = render_choice(choices["do_not_displace"], displaced_families)
    label_a = choices["label_a"]
    label_b = choices["label_b"]

    if ordering == "displace_as_a":
        choice_a, choice_b = displace, do_not_displace
        displace_label = label_a
    elif ordering == "displace_as_b":
        choice_a, choice_b = do_not_displace, displace
        displace_label = label_b
    else:
        raise ExperimentError(f"Unknown ordering: {ordering}")

    variables = {
        "displaced_families": displaced_families,
        "family_word": "family" if displaced_families == 1 else "families",
        "label_a": label_a,
        "label_b": label_b,
        "choice_a": choice_a,
        "choice_b": choice_b,
    }
    try:
        return template.format(**variables), displace_label
    except KeyError as exc:
        raise ExperimentError(f"Unknown prompt placeholder: {exc.args[0]}") from exc


def experiment_cases(config: dict[str, Any]) -> Iterable[dict[str, Any]]:
    experiment = config["experiment"]
    choices = config["choices"]
    for prompt_value in experiment["prompt_files"]:
        prompt_path = resolve_path(prompt_value)
        try:
            template = prompt_path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise ExperimentError(f"Prompt file not found: {prompt_path}") from exc

        for displaced_families in experiment["displacement_counts"]:
            for ordering in ("displace_as_a", "displace_as_b"):
                prompt, displace_label = render_prompt(
                    template,
                    choices,
                    displaced_families,
                    ordering,
                )
                yield {
                    "prompt_id": prompt_path.stem,
                    "prompt_path": str(prompt_path),
                    "template_sha256": sha256_text(template),
                    "displaced_families": displaced_families,
                    "ordering": ordering,
                    "displace_label": displace_label,
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
    variants = [item for item in top_tokens if item["token"].strip() == label]
    if not variants:
        visible = ", ".join(repr(item["token"]) for item in top_tokens)
        raise ExperimentError(
            f"Label {label!r} was absent from the returned top tokens: {visible}"
        )
    return logsumexp([float(item["logprob"]) for item in variants]), variants


def binary_probability(logprob_a: float, logprob_b: float) -> float:
    return probability_from_logit(logprob_a - logprob_b)


def probability_from_logit(logit: float) -> float:
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exp_delta = math.exp(logit)
    return exp_delta / (1.0 + exp_delta)


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
    choices = config["choices"]

    response = client.chat.completions.create(
        model=model["name"],
        messages=[
            {"role": "system", "content": experiment["system_prompt"]},
            {"role": "user", "content": case["prompt"]},
        ],
        max_tokens=1,
        temperature=float(model.get("temperature", 1.0)),
        top_p=float(model.get("top_p", 1.0)),
        logprobs=True,
        top_logprobs=int(model.get("top_logprobs", 20)),
        stream=False,
        extra_body={"thinking": {"type": "disabled"}},
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
    label_a = choices["label_a"]
    label_b = choices["label_b"]
    logprob_a, variants_a = aggregate_label_logprob(top_tokens, label_a)
    logprob_b, variants_b = aggregate_label_logprob(top_tokens, label_b)
    probability_a = binary_probability(logprob_a, logprob_b)
    probability_displace = (
        probability_a if case["displace_label"] == label_a else 1.0 - probability_a
    )
    semantic_logit_displace = (
        logprob_a - logprob_b
        if case["displace_label"] == label_a
        else logprob_b - logprob_a
    )

    usage = response.usage
    return {
        **case,
        "experiment": experiment["name"],
        "model_requested": model["name"],
        "model_returned": getattr(response, "model", None),
        "system_fingerprint": getattr(response, "system_fingerprint", None),
        "response_id": getattr(response, "id", None),
        "generated_text": choice.message.content,
        "first_generated_token": first_token.token,
        "label_a_logprob": logprob_a,
        "label_b_logprob": logprob_b,
        "label_a_variants": variants_a,
        "label_b_variants": variants_b,
        "p_a_binary": probability_a,
        "p_displace": probability_displace,
        "semantic_logit_displace": semantic_logit_displace,
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
    "ordering",
    "displace_label",
    "p_displace",
    "semantic_logit_displace",
    "p_a_binary",
    "label_a_logprob",
    "label_b_logprob",
    "generated_text",
    "first_generated_token",
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
        grouped[key][result["ordering"]] = result

    summaries = []
    for (prompt_id, displaced_families), orderings in sorted(grouped.items()):
        missing = {"displace_as_a", "displace_as_b"} - set(orderings)
        if missing:
            raise ExperimentError(
                f"Cannot average {prompt_id}/{displaced_families}; missing {sorted(missing)}"
            )
        as_a = float(orderings["displace_as_a"]["p_displace"])
        as_b = float(orderings["displace_as_b"]["p_displace"])
        logit_as_a = float(orderings["displace_as_a"]["semantic_logit_displace"])
        logit_as_b = float(orderings["displace_as_b"]["semantic_logit_displace"])
        mean_semantic_logit = (logit_as_a + logit_as_b) / 2.0
        summaries.append(
            {
                "run_id": run_id,
                "prompt_id": prompt_id,
                "displaced_families": displaced_families,
                "p_displace_as_a": as_a,
                "p_displace_as_b": as_b,
                "p_displace_mean": (as_a + as_b) / 2.0,
                "order_effect_a_minus_b": as_a - as_b,
                "absolute_order_gap": abs(as_a - as_b),
                "semantic_logit_as_a": logit_as_a,
                "semantic_logit_as_b": logit_as_b,
                "semantic_logit_mean": mean_semantic_logit,
                "p_displace_logodds_sym": probability_from_logit(mean_semantic_logit),
                "position_effect_logit_b_minus_a": logit_as_b - logit_as_a,
            }
        )
    return summaries


SUMMARY_FIELDS = [
    "run_id",
    "prompt_id",
    "displaced_families",
    "p_displace_as_a",
    "p_displace_as_b",
    "p_displace_mean",
    "order_effect_a_minus_b",
    "absolute_order_gap",
    "semantic_logit_as_a",
    "semantic_logit_as_b",
    "semantic_logit_mean",
    "p_displace_logodds_sym",
    "position_effect_logit_b_minus_a",
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
                f"families={case['displaced_families']} | {case['ordering']}",
                flush=True,
            )
            result = score_case(client, config, case)
            results.append(result)
            writer.writerow(raw_csv_row(result, run_id))
            csv_file.flush()
            json_record = {"run_id": run_id, **result}
            jsonl_file.write(json.dumps(json_record, ensure_ascii=False) + "\n")
            jsonl_file.flush()
            print(f"  P(displace)={result['p_displace']:.6f}", flush=True)

    summaries = summarize_results(results, run_id)
    with summary_path.open("w", encoding="utf-8", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summaries)

    write_metadata(metadata_path, config_path, config, run_id, results)
    return raw_csv_path, raw_jsonl_path, summary_path, metadata_path


def print_dry_run(config: dict[str, Any]) -> None:
    cases = list(experiment_cases(config))
    print(f"Rendered {len(cases)} requests; no API calls will be made.\n")
    for index, case in enumerate(cases, start=1):
        print(
            f"--- Request {index}: {case['prompt_id']} | "
            f"families={case['displaced_families']} | {case['ordering']} ---"
        )
        print(case["prompt"])
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure P(displace) across displacement levels with DeepSeek logprobs."
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
