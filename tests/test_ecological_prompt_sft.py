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
from scripts.ecological_prompt_sft.readout_evaluation import (
    READOUT_EVALUATION_SLUG,
    build_supervision_matched_readout_cases,
    run_supervision_matched_readout_workflow,
    validate_supervision_matched_readout_artifacts,
)
from scripts.ecological_prompt_sft.runner import (
    PromptSFTConfig,
    _required_hashes,
    artifacts_for_run_dir,
    find_complete_runs_for_arms,
    find_compatible_complete_run,
    pair_name_for_arm,
    pair_name_for_config,
    training_objective_for_arm,
    validate_complete_run,
)
from scripts.ecological_prompt_sft.tokenization import (
    IGNORE_INDEX,
    PromptOnlyCollator,
    tokenize_answer_examples,
    tokenize_prompt_examples,
)
from scripts.harmony_eval.cases import DEFAULT_COST_COUNTS
from scripts.harmony_eval.analysis import _readout_matrix_layout
from scripts.harmony_eval.scoring import score_loaded_causal_checkpoint
from scripts.harmony_sft.github_publish import _publication_sources
from scripts.harmony_sft.posthoc_eval import (
    POSTHOC_PROTOCOL_VERSION,
    _case_set_sha256,
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
                "training_arm": config.training_arm,
                "pair_name": pair_name_for_config(config),
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


def make_complete_readout_eval(output_dir: Path, source_complete: Path):
    artifacts = artifacts_for_posthoc_eval(output_dir)
    output_dir.mkdir(parents=True)
    cases = build_supervision_matched_readout_cases(DEFAULT_COST_COUNTS)
    artifacts.rendered_cases_path.write_text(
        "".join(json.dumps(case) + "\n" for case in cases)
    )
    with artifacts.raw_scores_path.open("w", newline="") as output:
        fields = (
            "case_id",
            "template",
            "cost_count",
            "model_role",
            "readout_type",
            "readout_variant",
            "candidate_implement",
            "candidate_reject",
            "candidate_score_normalization",
            "candidate_tokens_implement",
            "candidate_tokens_reject",
            "p_implement",
            "semantic_logit_implement",
        )
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for role in ("base", "aligned"):
            for case in cases:
                writer.writerow(
                    {
                        key: case[key]
                        for key in (
                            "case_id",
                            "template",
                            "cost_count",
                            "readout_type",
                            "readout_variant",
                            "candidate_implement",
                            "candidate_reject",
                            "candidate_score_normalization",
                        )
                    }
                    | {
                        "model_role": role,
                        "candidate_tokens_implement": 1,
                        "candidate_tokens_reject": 1,
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
                "evaluation_slug": READOUT_EVALUATION_SLUG,
                "source_complete_sha256": hashlib.sha256(
                    source_complete.read_bytes()
                ).hexdigest(),
                "cost_counts": list(DEFAULT_COST_COUNTS),
                "case_count_per_model": len(cases),
                "case_set_sha256": _case_set_sha256(cases),
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
    def test_colab_discovers_three_arms_and_demotes_legacy_controls(self):
        notebook = json.loads(
            Path("notebooks/ecological_dilemma_prompt_sft_colab.ipynb").read_text()
        )
        code_cells = [
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        combined = "\n".join(code_cells)

        self.assertIn("find_complete_runs_for_arms", combined)
        self.assertIn("artifacts_by_arm = find_complete_runs_for_arms", combined)
        self.assertIn("for arm in EVALUATION_ARMS", combined)
        self.assertIn("run_supervision_matched_readout_workflow", combined)
        legacy_control_cells = [
            source
            for source in code_cells
            if source.startswith("# Legacy six-control")
        ]
        self.assertEqual(len(legacy_control_cells), 3)
        for source in legacy_control_cells:
            executable_lines = [
                line
                for line in source.splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
            self.assertEqual(executable_lines, [])

    def test_supervision_matched_readouts_cover_all_orders_and_costs(self):
        cases = build_supervision_matched_readout_cases()

        self.assertEqual(len(cases), 320)
        self.assertEqual(len({case["case_id"] for case in cases}), 320)
        self.assertEqual(len({case["template"] for case in cases}), 40)
        by_readout = {
            readout: [case for case in cases if case["readout_type"] == readout]
            for readout in {case["readout_type"] for case in cases}
        }
        self.assertEqual(len(by_readout["reversed_yes_no"]), 64)
        self.assertEqual(len(by_readout["counterbalanced_ab"]), 128)
        self.assertEqual(len(by_readout["complete_option_text"]), 128)
        layout = _readout_matrix_layout(
            cases,
            sorted({str(case["template"]) for case in cases}),
        )
        self.assertIsNotNone(layout)
        self.assertEqual(len(layout[0]), 8)
        self.assertEqual(len(layout[1]), 5)

        for case in by_readout["reversed_yes_no"]:
            self.assertEqual(case["candidate_implement"], "No")
            self.assertEqual(case["candidate_reject"], "Yes")
            self.assertEqual(case["candidate_score_normalization"], "sum")
        ab_mappings = {
            (case["readout_variant"], case["candidate_implement"], case["candidate_reject"])
            for case in by_readout["counterbalanced_ab"]
        }
        self.assertEqual(
            ab_mappings,
            {("ecological_a", "A", "B"), ("ecological_b", "B", "A")},
        )
        for case in by_readout["complete_option_text"]:
            self.assertEqual(case["candidate_score_normalization"], "mean")
            self.assertGreater(len(case["candidate_implement"].split()), 2)
            self.assertGreater(len(case["candidate_reject"].split()), 2)

    def test_readouts_preserve_the_original_scenario_body(self):
        cases = build_supervision_matched_readout_cases((10,))
        for case in cases:
            source = Path(case["source_template_path"]).read_text()
            scenario, _ = source.strip().rsplit("\n\n", maxsplit=1)
            with self.subTest(case=case["case_id"]):
                self.assertTrue(case["prompt"].startswith(scenario.format(cost=10) + "\n\n"))

    def test_semantic_scoring_maps_polarity_and_normalizes_full_text(self):
        tokenizer = FakeTokenizer()
        cases = [
            {
                "case_id": "reversed",
                "template": "reversed",
                "cost_count": 1,
                "prompt": "Question?",
                "candidate_implement": "No",
                "candidate_reject": "Yes",
                "candidate_score_normalization": "sum",
            },
            {
                "case_id": "full",
                "template": "full",
                "cost_count": 1,
                "prompt": "Question?",
                "candidate_implement": "long",
                "candidate_reject": "x",
                "candidate_score_normalization": "mean",
            },
        ]

        def fake_scores(_model, _tokenizer, items):
            totals = {"No": -1.0, "Yes": -3.0, "long": -8.0, "x": -3.0}
            return [totals[item["candidate"]] for item in items]

        with patch(
            "scripts.harmony_eval.scoring._score_causal_batch",
            side_effect=fake_scores,
        ):
            rows = score_loaded_causal_checkpoint(
                model=object(),
                tokenizer=tokenizer,
                cases=cases,
                model_role="base",
                model_id="model",
                model_revision="revision",
                pair_name="pair",
                training_method="test",
                batch_size=2,
                enable_thinking=False,
            )

        self.assertEqual(rows[0]["semantic_logit_implement"], 2.0)
        self.assertEqual(rows[0]["logprob_no"], -1.0)
        self.assertEqual(rows[0]["logprob_yes"], -3.0)
        # FakeTokenizer uses bytes: -8/4 minus -3/1 = +1.
        self.assertEqual(rows[1]["semantic_logit_implement"], 1.0)
        self.assertEqual(rows[1]["logprob_implement"], -8.0)
        self.assertEqual(rows[1]["candidate_tokens_implement"], 4)

    def test_readout_validation_requires_the_full_640_row_matrix(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_complete = root / "COMPLETE.json"
            source_complete.write_text('{"status":"complete"}')
            artifacts = make_complete_readout_eval(root / "readout", source_complete)

            validation = validate_supervision_matched_readout_artifacts(artifacts)

            self.assertEqual(validation.case_count_per_model, 320)
            self.assertEqual(validation.score_row_count, 640)
            self.assertEqual(validation.template_count, 40)
            self.assertEqual(set(_publication_sources(artifacts)), {
                "rendered_cases.jsonl",
                "raw_scores.csv",
                "thresholds.csv",
                "curves.png",
                "metadata.json",
                "COMPLETE.json",
            })

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

    def test_answer_mask_survives_tokenization_and_collation(self):
        tokenizer = FakeTokenizer()
        tokenized = tokenize_answer_examples(
            tokenizer,
            [
                {
                    "id": "one",
                    "dilemma": "Which policy?",
                    "assistant_answer": "Choose habitat restoration.",
                }
            ],
            max_length=128,
        )

        class FakeTorch:
            long = "long"

            @staticmethod
            def tensor(values, *, dtype):
                if dtype != FakeTorch.long:
                    raise AssertionError("The collator must construct long tensors")
                return values

        with patch.dict("sys.modules", {"torch": FakeTorch}):
            batch = PromptOnlyCollator(pad_token_id=0)(tokenized)

        self.assertEqual(batch["input_ids"][0], tokenized[0]["input_ids"])
        self.assertEqual(batch["labels"][0], tokenized[0]["labels"])
        self.assertEqual(
            batch["labels"][0][: -tokenized[0]["supervised_token_count"]],
            [IGNORE_INDEX]
            * (
                len(tokenized[0]["labels"])
                - tokenized[0]["supervised_token_count"]
            ),
        )

    def test_collator_rejects_misaligned_labels(self):
        class FakeTorch:
            long = "long"

            @staticmethod
            def tensor(values, *, dtype):
                return values

        with patch.dict("sys.modules", {"torch": FakeTorch}):
            with self.assertRaisesRegex(ValueError, "one label per input token"):
                PromptOnlyCollator(pad_token_id=0)(
                    [{"input_ids": [1, 2], "labels": [1]}]
                )

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

    def test_legacy_prompt_run_without_pair_name_is_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = write_prompt_release(root / "release")
            config = PromptSFTConfig(
                output_root=root / "local",
                dataset_path=records,
            )
            artifacts = make_complete_prompt_run(root / "drive" / "run", config)
            metadata = json.loads(artifacts.metadata_path.read_text())
            metadata.pop("pair_name")
            metadata.pop("training_arm")
            metadata["config"].pop("training_arm")
            artifacts.metadata_path.write_text(json.dumps(metadata))
            complete = json.loads(artifacts.complete_marker_path.read_text())
            complete["artifact_sha256"] = _required_hashes(artifacts)
            artifacts.complete_marker_path.write_text(json.dumps(complete))

            self.assertEqual(
                find_compatible_complete_run(root / "drive", config),
                artifacts,
            )

    def test_custom_prompt_pair_name_is_isolated_from_the_legacy_pair(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = write_prompt_release(root / "release")
            legacy_config = PromptSFTConfig(
                output_root=root / "local",
                dataset_path=records,
            )
            control_config = PromptSFTConfig(
                output_root=root / "local",
                dataset_path=records,
                pair_name="qwen3_8b_clash_prompt_control_sft",
            )
            legacy = make_complete_prompt_run(
                root / "drive" / "legacy", legacy_config
            )
            control = make_complete_prompt_run(
                root / "drive" / "control", control_config
            )

            self.assertEqual(
                pair_name_for_config(legacy_config),
                pair_name_for_arm("prompt_only"),
            )
            self.assertEqual(
                pair_name_for_config(control_config),
                "qwen3_8b_clash_prompt_control_sft",
            )
            self.assertEqual(
                find_compatible_complete_run(root / "drive", control_config),
                control,
            )
            self.assertNotEqual(control, legacy)

            metadata = json.loads(control.metadata_path.read_text())
            metadata.pop("pair_name")
            control.metadata_path.write_text(json.dumps(metadata))
            complete = json.loads(control.complete_marker_path.read_text())
            complete["artifact_sha256"] = _required_hashes(control)
            control.complete_marker_path.write_text(json.dumps(complete))
            self.assertIsNone(
                find_compatible_complete_run(root / "drive", control_config)
            )

    def test_custom_pair_name_rejects_unsafe_identifiers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = write_prompt_release(root / "release")
            config = PromptSFTConfig(
                output_root=root / "local",
                dataset_path=records,
                pair_name="../shared-results",
            )
            with self.assertRaisesRegex(ValueError, "pair_name"):
                find_compatible_complete_run(root / "drive", config)

    def test_three_arm_discovery_rejects_cross_arm_or_incomplete_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            configs = {
                arm: PromptSFTConfig(
                    output_root=root / "local" / arm,
                    training_arm=arm,
                    dataset_path=Path(
                        "data/ecological_dilemmas/v1/records.jsonl"
                        if arm == "prompt_only"
                        else f"data/ecological_dilemmas/sft/{arm}/records.jsonl"
                    ),
                )
                for arm in ("prompt_only", "ecological_option", "human_option")
            }
            drive_roots = {arm: root / "drive" / arm for arm in configs}
            expected = {
                arm: make_complete_prompt_run(
                    drive_roots[arm] / f"run-{arm}", config
                )
                for arm, config in configs.items()
            }

            self.assertEqual(
                find_complete_runs_for_arms(drive_roots, configs),
                expected,
            )

            ecological = expected["ecological_option"]
            metadata = json.loads(ecological.metadata_path.read_text())
            metadata["pair_name"] = pair_name_for_arm(HUMAN_OPTION_ARM)
            ecological.metadata_path.write_text(json.dumps(metadata))
            complete = json.loads(ecological.complete_marker_path.read_text())
            complete["artifact_sha256"] = _required_hashes(ecological)
            ecological.complete_marker_path.write_text(json.dumps(complete))

            with self.assertRaisesRegex(RuntimeError, "ecological_option"):
                find_complete_runs_for_arms(drive_roots, configs)

    def test_buggy_answer_checkpoint_is_not_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = PromptSFTConfig(
                output_root=root / "local",
                training_arm=ECOLOGICAL_OPTION_ARM,
                dataset_path=Path(
                    "data/ecological_dilemmas/sft/ecological_option/records.jsonl"
                ),
            )
            artifacts = make_complete_prompt_run(root / "drive" / "old-run", config)
            metadata = json.loads(artifacts.metadata_path.read_text())
            metadata["training_objective"] = "ecological_option_response_only_sft"
            artifacts.metadata_path.write_text(json.dumps(metadata))
            complete = json.loads(artifacts.complete_marker_path.read_text())
            complete["artifact_sha256"] = _required_hashes(artifacts)
            artifacts.complete_marker_path.write_text(json.dumps(complete))

            self.assertIsNone(find_compatible_complete_run(root / "drive", config))

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

    def test_supervision_matched_workflow_reuses_exact_case_set(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = write_prompt_release(root / "release")
            config = PromptSFTConfig(output_root=root / "local", dataset_path=records)
            sft = make_complete_prompt_run(root / "drive" / "run", config)
            readout = make_complete_readout_eval(
                sft.run_dir / "posthoc_evaluations/readout",
                sft.complete_marker_path,
            )

            result = run_supervision_matched_readout_workflow(
                sft,
                cost_counts=DEFAULT_COST_COUNTS,
                batch_size=4,
            )

            self.assertTrue(result.evaluation_reused)
            self.assertEqual(result.evaluation_artifacts, readout)
            self.assertEqual(result.validation.score_row_count, 640)

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
