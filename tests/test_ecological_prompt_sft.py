import csv
import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from scripts.ecological_prompt_sft.data import load_prompt_examples
from scripts.ecological_prompt_sft.evaluation import (
    build_extreme_v2_cases,
    build_extreme_v2_control_cases,
    run_extreme_v2_control_workflow,
    run_extreme_v2_workflow,
)
from scripts.ecological_prompt_sft.runner import (
    PromptSFTConfig,
    _required_hashes,
    artifacts_for_run_dir,
    find_compatible_complete_run,
    validate_complete_run,
)
from scripts.ecological_prompt_sft.tokenization import tokenize_prompt_examples
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

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return f"<user>{messages[0]['content']}</user>"

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
                "training_objective": "prompt_only_causal_lm",
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


if __name__ == "__main__":
    unittest.main()
