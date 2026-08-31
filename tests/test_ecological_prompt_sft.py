import csv
import hashlib
import json
import shutil
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from scripts.build_ecological_answer_sft_datasets import build_datasets
from scripts.ecological_prompt_sft.data import (
    ECOLOGICAL_OPTION_ARM,
    HUMAN_OPTION_ARM,
    load_answer_examples,
    load_prompt_examples,
)
from scripts.ecological_prompt_sft.evaluation import (
    build_extreme_v2_cases,
    build_extreme_v2_control_cases,
    publish_results_to_github,
    run_extreme_v2_control_workflow,
    run_extreme_v2_workflow,
)
from scripts.ecological_prompt_sft.runner import (
    PromptSFTConfig,
    _required_hashes,
    artifacts_for_run_dir,
    find_compatible_complete_run,
    pair_name_for_arm,
    training_objective_for_arm,
    validate_complete_run,
)
from scripts.ecological_prompt_sft.tokenization import (
    IGNORE_INDEX,
    tokenize_answer_examples,
    tokenize_prompt_examples,
)
from scripts.harmony_eval.cases import DEFAULT_COST_COUNTS
from scripts.harmony_sft.posthoc_eval import (
    POSTHOC_PROTOCOL_VERSION,
    _required_hashes as posthoc_hashes,
    _template_manifest,
    artifacts_for_posthoc_eval,
)


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls = []
        self.eos_token = "</s>"

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        rendered = f"<user>{messages[0]['content']}</user>"
        if kwargs["add_generation_prompt"]:
            rendered += "<assistant>"
        return rendered

    def encode(self, value, *, add_special_tokens):
        self.assert_false(add_special_tokens)
        return list(value.encode("utf-8"))

    @staticmethod
    def assert_false(value):
        if value:
            raise AssertionError("add_special_tokens must be false")


def write_prompt_release(root: Path, count: int = 3) -> Path:
    root.mkdir(parents=True)
    records_path = root / "records.jsonl"
    rows = [
        {
            "id": f"case-{index}",
            "dilemma": f"Dilemma setup {index}?",
            "source": {"run": "test", "index": index},
            "assignment": {"ecological_object": "Wetland continuity"},
            "title": f"Case {index}",
        }
        for index in range(1, count + 1)
    ]
    records_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    records_hash = hashlib.sha256(records_path.read_bytes()).hexdigest()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_type": "prompt_only",
                "contains_normative_labels": False,
                "contains_assistant_responses": False,
                "released_count": count,
                "artifacts": {"records.jsonl": records_hash},
            }
        )
    )
    return records_path


def make_complete_prompt_run(run_dir: Path, config: PromptSFTConfig):
    artifacts = artifacts_for_run_dir(run_dir)
    checkpoint = artifacts.checkpoints_dir / "checkpoint-10"
    for directory in (
        artifacts.final_adapter_dir,
        checkpoint,
        artifacts.prompts_path.parent,
        artifacts.train_metrics_path.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    for path in (
        artifacts.final_adapter_dir / "adapter_config.json",
        artifacts.final_adapter_dir / "adapter_model.safetensors",
        checkpoint / "adapter_config.json",
        checkpoint / "adapter_model.safetensors",
        checkpoint / "trainer_state.json",
        checkpoint / "optimizer.pt",
        checkpoint / "scheduler.pt",
        artifacts.prompts_path,
        artifacts.dataset_manifest_path,
        artifacts.train_metrics_path,
    ):
        path.write_bytes(f"test artifact: {path.name}".encode())
    dataset_hash = hashlib.sha256(Path(config.dataset_path).read_bytes()).hexdigest()
    config_dict = asdict(config)
    config_dict["output_root"] = str(config.output_root)
    config_dict["dataset_path"] = str(config.dataset_path)
    config_dict["cost_counts"] = list(config.cost_counts)
    artifacts.metadata_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "training_objective": training_objective_for_arm(
                    config.training_arm
                ),
                "pair_name": pair_name_for_arm(config.training_arm),
                "config": config_dict,
                "dataset": {"records_sha256": dataset_hash},
            }
        )
    )
    artifacts.complete_marker_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "completed_at_utc": run_dir.name,
                "artifact_sha256": _required_hashes(artifacts),
            }
        )
    )
    return artifacts


def make_complete_eval(output_dir: Path, source_complete: Path, *, control: bool):
    artifacts = artifacts_for_posthoc_eval(output_dir)
    output_dir.mkdir(parents=True)
    cases = (
        build_extreme_v2_control_cases(DEFAULT_COST_COUNTS)
        if control
        else build_extreme_v2_cases(DEFAULT_COST_COUNTS)
    )
    artifacts.rendered_cases_path.write_text(
        "".join(json.dumps(case) + "\n" for case in cases)
    )
    with artifacts.raw_scores_path.open("w", newline="") as output:
        fields = (
            "case_id",
            "template",
            "cost_count",
            "model_role",
            "p_implement",
            "semantic_logit_implement",
        )
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for role in ("base", "aligned"):
            for case in cases:
                writer.writerow(
                    {
                        "case_id": case["case_id"],
                        "template": case["template"],
                        "cost_count": case["cost_count"],
                        "model_role": role,
                        "p_implement": 0.5,
                        "semantic_logit_implement": 0,
                    }
                )
    artifacts.thresholds_path.write_text("template,base_status,aligned_status\n")
    artifacts.plot_path.write_bytes(b"png")
    artifacts.metadata_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "evaluation_protocol_version": POSTHOC_PROTOCOL_VERSION,
                "source_complete_sha256": hashlib.sha256(
                    source_complete.read_bytes()
                ).hexdigest(),
                "cost_counts": list(DEFAULT_COST_COUNTS),
                "case_count_per_model": len(cases),
                "enable_thinking": False,
                "templates": _template_manifest(cases),
            }
        )
    )
    artifacts.complete_marker_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "completed_at_utc": output_dir.name,
                "artifact_sha256": posthoc_hashes(artifacts),
            }
        )
    )
    return artifacts


class EcologicalPromptSFTTests(unittest.TestCase):
    def test_repository_release_has_98_prompts_and_no_supervision(self):
        examples, manifest = load_prompt_examples(
            Path("data/ecological_dilemmas/v1/records.jsonl")
        )

        self.assertEqual(len(examples), 98)
        self.assertEqual(manifest["dataset_type"], "prompt_only")
        self.assertFalse(manifest["contains_normative_labels"])
        self.assertFalse(manifest["contains_assistant_responses"])

    def test_repository_answer_arms_exactly_copy_the_two_option_fields(self):
        source_rows = {
            row["id"]: row
            for row in (
                json.loads(line)
                for line in Path(
                    "data/ecological_dilemmas/v1/records.jsonl"
                ).read_text().splitlines()
                if line.strip()
            )
        }
        for arm, target_field in (
            (ECOLOGICAL_OPTION_ARM, "ecologically_protective_option"),
            (HUMAN_OPTION_ARM, "human_protective_option"),
        ):
            examples, manifest = load_answer_examples(
                Path(f"data/ecological_dilemmas/sft/{arm}/records.jsonl"),
                training_arm=arm,
            )
            self.assertEqual(len(examples), 98)
            self.assertEqual(manifest["training_arm"], arm)
            self.assertTrue(manifest["contains_normative_labels"])
            self.assertTrue(manifest["contains_assistant_responses"])
            self.assertFalse(manifest["contains_rationales"])
            for example in examples:
                source = source_rows[example["id"]]
                self.assertEqual(example["dilemma"], source["dilemma"])
                self.assertEqual(
                    example["assistant_answer"], source[target_field]
                )

    def test_answer_arm_builder_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temp:
            output_root = Path(temp) / "sft"
            manifests = build_datasets(output_root=output_root)
            for arm in (ECOLOGICAL_OPTION_ARM, HUMAN_OPTION_ARM):
                generated = output_root / arm / "records.jsonl"
                tracked = Path(
                    f"data/ecological_dilemmas/sft/{arm}/records.jsonl"
                )
                self.assertEqual(generated.read_bytes(), tracked.read_bytes())
                self.assertEqual(
                    manifests[arm]["artifacts"]["records.jsonl"],
                    hashlib.sha256(tracked.read_bytes()).hexdigest(),
                )

    def test_answer_loader_rejects_a_rehashed_answer_not_found_in_source(self):
        with tempfile.TemporaryDirectory() as temp:
            release = Path(temp) / ECOLOGICAL_OPTION_ARM
            shutil.copytree(
                Path("data/ecological_dilemmas/sft") / ECOLOGICAL_OPTION_ARM,
                release,
            )
            records_path = release / "records.jsonl"
            rows = [json.loads(line) for line in records_path.read_text().splitlines()]
            rows[0]["messages"][1]["content"] = "Choose something else."
            records_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
            )
            manifest_path = release / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["artifacts"]["records.jsonl"] = hashlib.sha256(
                records_path.read_bytes()
            ).hexdigest()
            manifest_path.write_text(json.dumps(manifest))

            with self.assertRaisesRegex(ValueError, "does not exactly copy"):
                load_answer_examples(
                    records_path,
                    training_arm=ECOLOGICAL_OPTION_ARM,
                )

    def test_loader_rejects_an_invented_supervision_field(self):
        with tempfile.TemporaryDirectory() as temp:
            release = Path(temp) / "release"
            records = write_prompt_release(release, count=1)
            row = json.loads(records.read_text())
            row["target_label"] = "Human"
            records.write_text(json.dumps(row) + "\n")
            manifest = json.loads((release / "manifest.json").read_text())
            manifest["artifacts"]["records.jsonl"] = hashlib.sha256(
                records.read_bytes()
            ).hexdigest()
            (release / "manifest.json").write_text(json.dumps(manifest))

            with self.assertRaisesRegex(ValueError, "supervision fields"):
                load_prompt_examples(records, expected_count=1)

    def test_tokenization_has_user_only_chat_and_full_prompt_loss(self):
        tokenizer = FakeTokenizer()
        tokenized = tokenize_prompt_examples(
            tokenizer,
            [{"id": "one", "dilemma": "Which policy?"}],
            max_length=128,
        )

        self.assertEqual(tokenized[0]["labels"], tokenized[0]["input_ids"])
        messages, kwargs = tokenizer.calls[0]
        self.assertEqual(messages, [{"role": "user", "content": "Which policy?"}])
        self.assertFalse(kwargs["add_generation_prompt"])
        self.assertFalse(kwargs["enable_thinking"])

    def test_answer_tokenization_masks_user_and_supervises_exact_option(self):
        tokenizer = FakeTokenizer()
        answer = "Choose habitat restoration."
        tokenized = tokenize_answer_examples(
            tokenizer,
            [
                {
                    "id": "one",
                    "dilemma": "Which policy?",
                    "assistant_answer": answer,
                }
            ],
            max_length=128,
        )

        row = tokenized[0]
        supervised = [label for label in row["labels"] if label != IGNORE_INDEX]
        self.assertEqual(bytes(supervised).decode(), answer + tokenizer.eos_token)
        self.assertEqual(
            row["labels"][: -len(supervised)],
            [IGNORE_INDEX] * (len(row["labels"]) - len(supervised)),
        )
        messages, kwargs = tokenizer.calls[0]
        self.assertEqual(messages, [{"role": "user", "content": "Which policy?"}])
        self.assertTrue(kwargs["add_generation_prompt"])
        self.assertFalse(kwargs["enable_thinking"])

    def test_tokenization_refuses_to_truncate(self):
        with self.assertRaisesRegex(ValueError, "refusing to truncate"):
            tokenize_prompt_examples(
                FakeTokenizer(),
                [{"id": "long", "dilemma": "x" * 200}],
                max_length=64,
            )

    def test_complete_run_reuse_and_hash_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = write_prompt_release(root / "release")
            config = PromptSFTConfig(
                output_root=root / "local",
                dataset_path=records,
            )
            artifacts = make_complete_prompt_run(root / "drive" / "run", config)

            self.assertEqual(
                find_compatible_complete_run(root / "drive", config), artifacts
            )
            validate_complete_run(artifacts)
            (artifacts.final_adapter_dir / "adapter_model.safetensors").write_bytes(
                b"tampered"
            )
            with self.assertRaisesRegex(RuntimeError, "hashes do not match"):
                validate_complete_run(artifacts)

    def test_primary_and_control_workflows_reuse_verified_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = write_prompt_release(root / "release")
            config = PromptSFTConfig(output_root=root / "local", dataset_path=records)
            sft = make_complete_prompt_run(root / "drive" / "run", config)
            primary = make_complete_eval(
                sft.run_dir / "posthoc_evaluations/primary",
                sft.complete_marker_path,
                control=False,
            )
            control = make_complete_eval(
                sft.run_dir / "posthoc_evaluations/control",
                sft.complete_marker_path,
                control=True,
            )

            primary_result = run_extreme_v2_workflow(
                sft,
                cost_counts=DEFAULT_COST_COUNTS,
                batch_size=4,
            )
            control_result = run_extreme_v2_control_workflow(
                sft,
                cost_counts=DEFAULT_COST_COUNTS,
                batch_size=4,
            )

            self.assertTrue(primary_result.evaluation_reused)
            self.assertEqual(primary_result.evaluation_artifacts, primary)
            self.assertEqual(primary_result.validation.score_row_count, 128)
            self.assertTrue(control_result.evaluation_reused)
            self.assertEqual(control_result.evaluation_artifacts, control)
            self.assertEqual(control_result.validation.score_row_count, 68)

    def test_publication_uses_the_evaluated_arm_result_root(self):
        with tempfile.TemporaryDirectory() as temp:
            artifacts = artifacts_for_posthoc_eval(Path(temp) / "eval")
            artifacts.output_dir.mkdir()
            pair_name = pair_name_for_arm(ECOLOGICAL_OPTION_ARM)
            artifacts.metadata_path.write_text(json.dumps({"pair_name": pair_name}))
            with patch(
                "scripts.ecological_prompt_sft.evaluation."
                "publish_extreme_v2_results_to_github"
            ) as publish:
                publish.return_value = object()
                result = publish_results_to_github(
                    artifacts,
                    source_run_name="run",
                    github_repository="owner/repo",
                    branch="main",
                    github_token="token",
                    repo_root=Path(temp),
                )

            self.assertIs(result, publish.return_value)
            self.assertEqual(
                publish.call_args.kwargs["results_root"],
                Path("results/harmony_eval") / pair_name,
            )


if __name__ == "__main__":
    unittest.main()
