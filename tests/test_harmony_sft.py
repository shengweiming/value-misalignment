import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.harmony_eval.cases import DEFAULT_COST_COUNTS, build_cases
from scripts.harmony_eval.scoring import format_causal_prompt
from scripts.harmony_sft.data import extract_r1_examples
from scripts.harmony_sft.persistence import persist_run_to_colab_drive
from scripts.harmony_sft.posthoc_eval import (
    POSTHOC_PROTOCOL_VERSION,
    _required_hashes as posthoc_required_hashes,
    _template_manifest,
    artifacts_for_posthoc_eval,
    find_compatible_posthoc_eval,
    persist_posthoc_eval_to_colab_drive,
    validate_posthoc_eval,
)
from scripts.harmony_sft.runner import (
    SFTArtifacts,
    SFTConfig,
    _validate_complete_artifacts,
    artifacts_for_run_dir,
    find_compatible_complete_run,
    validate_config,
    validate_complete_run,
)
from scripts.harmony_sft.tokenization import (
    IGNORE_INDEX,
    response_only_features,
    tokenize_r1_examples,
)


def harmony_row(
    prompt_id: int,
    comparison: str,
    better: str,
    *,
    worse: str = "Lower-ranked answer",
    prompt: str = "What should we do?",
) -> dict[str, object]:
    return {
        "PromptID": prompt_id,
        "Prompt": prompt,
        "BetterCompletion": better,
        "WorseCompletion": worse,
        "ComparedRanks": comparison,
    }


def make_complete_run(run_dir: Path) -> SFTArtifacts:
    final_adapter = run_dir / "final_adapter"
    checkpoint = run_dir / "checkpoints" / "checkpoint-10"
    training = run_dir / "training"
    evaluation = run_dir / "evaluation"
    for directory in (final_adapter, checkpoint, training, evaluation):
        directory.mkdir(parents=True, exist_ok=True)
    for path in (
        final_adapter / "adapter_config.json",
        final_adapter / "adapter_model.safetensors",
        checkpoint / "adapter_config.json",
        checkpoint / "adapter_model.safetensors",
        checkpoint / "trainer_state.json",
        checkpoint / "optimizer.pt",
        checkpoint / "scheduler.pt",
        training / "train_metrics.json",
        evaluation / "raw_scores.csv",
        evaluation / "thresholds.csv",
        evaluation / "curves.png",
        run_dir / "run_metadata.json",
    ):
        path.write_bytes(f"test artifact: {path.name}".encode())

    artifacts = artifacts_for_run_dir(run_dir)
    hashes = _validate_complete_artifacts(artifacts)
    artifacts.complete_marker_path.write_text(
        json.dumps({"status": "complete", "artifact_sha256": hashes}),
        encoding="utf-8",
    )
    return artifacts


def write_complete_metadata(artifacts: SFTArtifacts, config: SFTConfig) -> None:
    artifacts.metadata_path.write_text(
        json.dumps({"status": "complete", "config": config.__dict__}, default=str),
        encoding="utf-8",
    )
    hashes = _validate_complete_artifacts(artifacts)
    artifacts.complete_marker_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "completed_at_utc": artifacts.run_dir.name,
                "artifact_sha256": hashes,
            }
        ),
        encoding="utf-8",
    )


def make_complete_posthoc_eval(output_dir: Path):
    artifacts = artifacts_for_posthoc_eval(output_dir)
    output_dir.mkdir(parents=True)
    for path in (
        artifacts.raw_scores_path,
        artifacts.thresholds_path,
        artifacts.plot_path,
        artifacts.metadata_path,
    ):
        path.write_bytes(f"test artifact: {path.name}".encode())
    artifacts.complete_marker_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "artifact_sha256": posthoc_required_hashes(artifacts),
            }
        ),
        encoding="utf-8",
    )
    return artifacts


def write_posthoc_metadata(
    artifacts,
    source_run: SFTArtifacts,
    *,
    template_names=None,
) -> None:
    source_complete_hash = hashlib.sha256(
        source_run.complete_marker_path.read_bytes()
    ).hexdigest()
    artifacts.metadata_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "evaluation_protocol_version": POSTHOC_PROTOCOL_VERSION,
                "source_complete_sha256": source_complete_hash,
                "cost_counts": list(DEFAULT_COST_COUNTS),
                "enable_thinking": False,
                "templates": _template_manifest(
                    build_cases(DEFAULT_COST_COUNTS, template_names)
                ),
            }
        ),
        encoding="utf-8",
    )
    artifacts.complete_marker_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "completed_at_utc": artifacts.output_dir.name,
                "artifact_sha256": posthoc_required_hashes(artifacts),
            }
        ),
        encoding="utf-8",
    )


class FakeQwenTokenizer:
    eos_token = "<eos>"

    def __init__(self) -> None:
        self.template_calls: list[dict[str, object]] = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append(kwargs)
        return f"USER:{messages[-1]['content']}\nASSISTANT:"

    def encode(self, value: str, *, add_special_tokens: bool):
        if add_special_tokens:
            raise AssertionError("Tests expect special tokens to be disabled")
        return list(value.encode("utf-8"))


class HarmonySFTDataTests(unittest.TestCase):
    def test_extracts_one_r1_answer_per_prompt_id(self):
        rows = [
            harmony_row(1, "R1-R2", "R1 answer"),
            harmony_row(1, "R1-R3", "R1 answer"),
            harmony_row(1, "R2-R3", "R2 answer"),
            harmony_row(2, "R1-R3", "Second R1", prompt="Another prompt"),
        ]

        examples = extract_r1_examples(rows)

        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0]["r1_answer"], "R1 answer")
        self.assertEqual(examples[0]["source_comparisons"], ["R1-R2", "R1-R3"])
        self.assertEqual(examples[0]["r1_selection_method"], "source_agreement")
        self.assertEqual(examples[1]["r1_answer"], "Second R1")

    def test_applies_known_prompt_131_r1_correction(self):
        environmental = (
            "Overfishing stresses the necessity for harmony in our relationship "
            "with the marine world, ensuring abundance for all future generations."
        )
        ambivalent = (
            "Sustainable fishing practices have implications for global food "
            "security and socio-cultural dynamics."
        )
        rows = [
            harmony_row(131, "R1-R2", environmental, worse=ambivalent),
            harmony_row(131, "R1-R3", ambivalent, worse="Harmful answer"),
            harmony_row(131, "R2-R3", ambivalent, worse="Harmful answer"),
        ]

        example = extract_r1_examples(rows)[0]

        self.assertEqual(example["r1_answer"], environmental)
        self.assertTrue(example["r1_conflict"])
        self.assertEqual(
            example["r1_selection_method"],
            "known_dataset_correction",
        )

    def test_resolves_other_conflict_from_cross_pair_rank_evidence(self):
        rows = [
            harmony_row(7, "R1-R2", "Environmental", worse="Ambivalent"),
            harmony_row(7, "R1-R3", "Ambivalent", worse="Harmful"),
            harmony_row(7, "R2-R3", "Ambivalent", worse="Harmful"),
        ]

        example = extract_r1_examples(rows)[0]

        self.assertEqual(example["r1_answer"], "Environmental")
        self.assertEqual(
            example["r1_selection_method"],
            "cross_pair_rank_consistency",
        )

    def test_uses_r1_r2_as_an_audited_fallback_for_unresolved_conflict(self):
        rows = [
            harmony_row(1, "R1-R2", "Canonical answer", worse="Second answer"),
            harmony_row(1, "R1-R3", "Other R1 answer", worse="Third answer"),
            harmony_row(1, "R2-R3", "Unrelated second", worse="Unrelated third"),
        ]

        example = extract_r1_examples(rows)[0]

        self.assertEqual(example["r1_answer"], "Canonical answer")
        self.assertEqual(
            example["r1_selection_method"],
            "canonical_r1_r2_conflict_fallback",
        )
        self.assertEqual(
            example["r1_source_answers"],
            {"R1-R2": "Canonical answer", "R1-R3": "Other R1 answer"},
        )

    def test_rejects_multiple_answers_within_the_canonical_pair(self):
        rows = [
            harmony_row(1, "R1-R2", "First answer"),
            harmony_row(1, "R1-R2", "Different answer"),
        ]

        with self.assertRaisesRegex(ValueError, "within R1-R2"):
            extract_r1_examples(rows)

    def test_can_require_both_r1_comparisons(self):
        rows = [harmony_row(1, "R1-R2", "R1 answer")]

        with self.assertRaisesRegex(ValueError, "missing R1 comparison"):
            extract_r1_examples(rows, require_both_comparisons=True)


class HarmonySFTTokenizationTests(unittest.TestCase):
    def test_response_only_loss_masks_prompt(self):
        features = response_only_features([1, 2, 3], [1, 2, 3, 4, 5], max_length=8)

        self.assertEqual(features["input_ids"], [1, 2, 3, 4, 5])
        self.assertEqual(features["labels"], [IGNORE_INDEX] * 3 + [4, 5])

    def test_left_truncates_only_the_prompt(self):
        features = response_only_features(
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 6, 7],
            max_length=4,
        )

        self.assertEqual(features["input_ids"], [4, 5, 6, 7])
        self.assertEqual(features["labels"], [IGNORE_INDEX, IGNORE_INDEX, 6, 7])

    def test_rejects_a_nonmatching_generation_prefix(self):
        with self.assertRaisesRegex(ValueError, "not prefixed"):
            response_only_features([1, 9], [1, 2, 3], max_length=8)

    def test_qwen_format_disables_thinking_and_supervises_answer(self):
        tokenizer = FakeQwenTokenizer()

        rows = tokenize_r1_examples(
            tokenizer,
            [{"prompt_id": "1", "prompt": "Prompt", "r1_answer": "Answer"}],
            max_length=128,
        )

        self.assertEqual(tokenizer.template_calls[0]["enable_thinking"], False)
        supervised = [
            token for token, label in zip(rows[0]["input_ids"], rows[0]["labels"])
            if label != IGNORE_INDEX
        ]
        self.assertEqual(bytes(supervised).decode("utf-8"), "Answer<eos>")


class HarmonySFTConfigurationTests(unittest.TestCase):
    def test_local_output_is_allowed_when_drive_requirement_is_disabled(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "runs"
            config = SFTConfig(output_root=output, require_google_drive=False)

            self.assertEqual(validate_config(config), output)

    def test_rejects_invalid_lora_rank(self):
        config = SFTConfig(
            output_root="unused",
            require_google_drive=False,
            lora_rank=0,
        )

        with self.assertRaisesRegex(ValueError, "rank and alpha"):
            validate_config(config)

    def test_completion_validation_requires_and_hashes_drive_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory)
            final_adapter = run_dir / "final_adapter"
            checkpoint = run_dir / "checkpoints" / "checkpoint-10"
            training = run_dir / "training"
            evaluation = run_dir / "evaluation"
            for directory in (final_adapter, checkpoint, training, evaluation):
                directory.mkdir(parents=True, exist_ok=True)
            for path in (
                final_adapter / "adapter_config.json",
                final_adapter / "adapter_model.safetensors",
                checkpoint / "adapter_config.json",
                checkpoint / "adapter_model.safetensors",
                checkpoint / "trainer_state.json",
                checkpoint / "optimizer.pt",
                checkpoint / "scheduler.pt",
                training / "train_metrics.json",
                evaluation / "raw_scores.csv",
                evaluation / "thresholds.csv",
                evaluation / "curves.png",
                run_dir / "run_metadata.json",
            ):
                path.write_bytes(b"test artifact")
            artifacts = SFTArtifacts(
                run_dir=run_dir,
                final_adapter_dir=final_adapter,
                checkpoints_dir=run_dir / "checkpoints",
                train_metrics_path=training / "train_metrics.json",
                raw_scores_path=evaluation / "raw_scores.csv",
                thresholds_path=evaluation / "thresholds.csv",
                plot_path=evaluation / "curves.png",
                metadata_path=run_dir / "run_metadata.json",
                complete_marker_path=run_dir / "COMPLETE.json",
            )

            hashes = _validate_complete_artifacts(artifacts)

            self.assertIn("checkpoint_adapter_weights", hashes)
            self.assertIn("checkpoint_optimizer", hashes)
            self.assertIn("checkpoint_trainer_state", hashes)

    def test_complete_run_rechecks_hash_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifacts = make_complete_run(Path(temporary_directory) / "run")

            hashes = validate_complete_run(artifacts)
            self.assertIn("adapter_weights", hashes)

            (artifacts.final_adapter_dir / "adapter_model.safetensors").write_bytes(
                b"tampered"
            )
            with self.assertRaisesRegex(RuntimeError, "do not match"):
                validate_complete_run(artifacts)

    def test_finds_newest_valid_run_with_matching_training_config(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            config = SFTConfig(
                output_root="/content/local-runs",
                require_google_drive=False,
            )
            older = make_complete_run(output_root / "2026-08-25")
            write_complete_metadata(older, config)
            newer = make_complete_run(output_root / "2026-08-26")
            write_complete_metadata(newer, config)

            found = find_compatible_complete_run(output_root, config)

            self.assertIsNotNone(found)
            self.assertEqual(found.run_dir, newer.run_dir)

    def test_reuse_ignores_output_and_evaluation_only_config_changes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            saved_config = SFTConfig(
                output_root="/content/old-local-runs",
                require_google_drive=False,
                eval_batch_size=2,
                cost_counts=(0, 10),
            )
            completed = make_complete_run(output_root / "completed")
            write_complete_metadata(completed, saved_config)
            current_config = SFTConfig(
                output_root="/content/new-local-runs",
                require_google_drive=False,
                eval_batch_size=8,
                cost_counts=(0, 100, 1000),
            )

            found = find_compatible_complete_run(output_root, current_config)

            self.assertIsNotNone(found)
            self.assertEqual(found.run_dir, completed.run_dir)

    def test_reuse_rejects_changed_training_config_and_corrupt_runs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            config = SFTConfig(
                output_root="/content/local-runs",
                require_google_drive=False,
            )
            wrong_rank = make_complete_run(output_root / "wrong-rank")
            write_complete_metadata(
                wrong_rank,
                SFTConfig(
                    output_root="elsewhere",
                    require_google_drive=False,
                    lora_rank=8,
                ),
            )
            corrupt = make_complete_run(output_root / "corrupt")
            write_complete_metadata(corrupt, config)
            (corrupt.final_adapter_dir / "adapter_model.safetensors").write_bytes(
                b"corrupt"
            )

            self.assertIsNone(find_compatible_complete_run(output_root, config))

    def test_persistence_flushes_remounts_and_verifies_drive_copy(self):
        class FakeDrive:
            def __init__(self):
                self.flush_calls = []
                self.mount_calls = []

            def flush_and_unmount(self, *, timeout_ms):
                self.flush_calls.append(timeout_ms)

            def mount(self, mountpoint, *, timeout_ms):
                self.mount_calls.append((mountpoint, timeout_ms))

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = make_complete_run(root / "local" / "test-run")
            mountpoint = root / "drive"
            my_drive = mountpoint / "MyDrive"
            drive_output = my_drive / "value-misalignment" / "runs"
            my_drive.mkdir(parents=True)
            fake_drive = FakeDrive()

            persisted = persist_run_to_colab_drive(
                source,
                drive_output,
                drive_mountpoint=mountpoint,
                drive_module=fake_drive,
            )

            self.assertEqual(persisted.run_dir, (drive_output / "test-run").resolve())
            self.assertTrue(persisted.complete_marker_path.is_file())
            self.assertTrue(source.complete_marker_path.is_file())
            self.assertEqual(len(fake_drive.flush_calls), 1)
            self.assertEqual(len(fake_drive.mount_calls), 1)
            validate_complete_run(persisted)

    def test_persistence_retries_from_unchanged_local_run(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = make_complete_run(root / "local" / "test-run")
            mountpoint = root / "drive"
            my_drive = mountpoint / "MyDrive"
            drive_output = my_drive / "value-misalignment" / "runs"
            my_drive.mkdir(parents=True)
            remote_weights = (
                drive_output
                / "test-run"
                / "final_adapter"
                / "adapter_model.safetensors"
            )

            class OneBadFreshMount:
                def __init__(self):
                    self.flush_count = 0
                    self.mount_count = 0

                def flush_and_unmount(self, *, timeout_ms):
                    self.flush_count += 1

                def mount(self, mountpoint, *, timeout_ms):
                    self.mount_count += 1
                    if self.mount_count == 1:
                        remote_weights.write_bytes(b"corrupted after first upload")

            fake_drive = OneBadFreshMount()
            persisted = persist_run_to_colab_drive(
                source,
                drive_output,
                drive_mountpoint=mountpoint,
                drive_module=fake_drive,
            )

            self.assertEqual(fake_drive.flush_count, 2)
            self.assertEqual(fake_drive.mount_count, 2)
            validate_complete_run(source)
            validate_complete_run(persisted)

    def test_posthoc_eval_persists_beneath_source_run(self):
        class FakeDrive:
            def __init__(self):
                self.flush_count = 0
                self.mount_count = 0

            def flush_and_unmount(self, *, timeout_ms):
                self.flush_count += 1

            def mount(self, mountpoint, *, timeout_ms):
                self.mount_count += 1

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = make_complete_posthoc_eval(root / "local" / "eval-run")
            mountpoint = root / "drive"
            sft_run = mountpoint / "MyDrive" / "runs" / "sft-run"
            sft_run.mkdir(parents=True)
            fake_drive = FakeDrive()

            persisted = persist_posthoc_eval_to_colab_drive(
                source,
                sft_run,
                drive_mountpoint=mountpoint,
                drive_module=fake_drive,
            )

            self.assertEqual(
                persisted.output_dir,
                (sft_run / "posthoc_evaluations" / "eval-run").resolve(),
            )
            self.assertEqual(fake_drive.flush_count, 1)
            self.assertEqual(fake_drive.mount_count, 1)
            validate_posthoc_eval(source.output_dir)
            validate_posthoc_eval(persisted.output_dir)

    def test_posthoc_eval_detects_tampered_results(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifacts = make_complete_posthoc_eval(Path(temporary_directory) / "eval")

            artifacts.thresholds_path.write_bytes(b"tampered")

            with self.assertRaisesRegex(RuntimeError, "do not match"):
                validate_posthoc_eval(artifacts.output_dir)

    def test_posthoc_eval_hashes_rendered_case_manifest_when_present(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifacts = make_complete_posthoc_eval(Path(temporary_directory) / "eval")
            artifacts.rendered_cases_path.write_text(
                '{"case_id":"example","prompt":"Rendered prompt"}\n',
                encoding="utf-8",
            )
            artifacts.complete_marker_path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "artifact_sha256": posthoc_required_hashes(artifacts),
                    }
                ),
                encoding="utf-8",
            )

            hashes = validate_posthoc_eval(artifacts.output_dir)
            self.assertIn("rendered_cases", hashes)

            artifacts.rendered_cases_path.write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "do not match"):
                validate_posthoc_eval(artifacts.output_dir)

    def test_finds_current_verified_posthoc_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_run = make_complete_run(Path(temporary_directory) / "sft-run")
            older = make_complete_posthoc_eval(
                source_run.run_dir / "posthoc_evaluations" / "2026-08-25"
            )
            write_posthoc_metadata(older, source_run)
            newer = make_complete_posthoc_eval(
                source_run.run_dir / "posthoc_evaluations" / "2026-08-26"
            )
            write_posthoc_metadata(newer, source_run)

            found = find_compatible_posthoc_eval(source_run.run_dir)

            self.assertIsNotNone(found)
            self.assertEqual(found.output_dir, newer.output_dir)

    def test_posthoc_reuse_rejects_changed_templates_and_tampering(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_run = make_complete_run(Path(temporary_directory) / "sft-run")
            wrong_templates = make_complete_posthoc_eval(
                source_run.run_dir / "posthoc_evaluations" / "wrong-templates"
            )
            write_posthoc_metadata(wrong_templates, source_run)
            metadata = json.loads(wrong_templates.metadata_path.read_text())
            metadata["templates"] = {}
            wrong_templates.metadata_path.write_text(json.dumps(metadata))
            wrong_templates.complete_marker_path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "artifact_sha256": posthoc_required_hashes(wrong_templates),
                    }
                )
            )
            tampered = make_complete_posthoc_eval(
                source_run.run_dir / "posthoc_evaluations" / "tampered"
            )
            write_posthoc_metadata(tampered, source_run)
            tampered.raw_scores_path.write_bytes(b"tampered")

            self.assertIsNone(find_compatible_posthoc_eval(source_run.run_dir))

    def test_posthoc_reuse_distinguishes_explicit_template_suite(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_run = make_complete_run(Path(temporary_directory) / "sft-run")
            default_eval = make_complete_posthoc_eval(
                source_run.run_dir / "posthoc_evaluations" / "default"
            )
            write_posthoc_metadata(default_eval, source_run)
            v2_templates = (
                "extreme_v2/wetland_relocation",
                "extreme_v2/marine_reserve",
                "extreme_v2/pesticide_ban",
                "extreme_v2/oil_extraction_ban",
                "extreme_v2/dam_removal",
                "extreme_v2/wildfire_restoration",
                "extreme_v2/river_water_allocation",
                "extreme_v2/island_biosecurity",
            )
            v2_eval = make_complete_posthoc_eval(
                source_run.run_dir / "posthoc_evaluations" / "extreme-v2"
            )
            write_posthoc_metadata(
                v2_eval,
                source_run,
                template_names=v2_templates,
            )

            found_default = find_compatible_posthoc_eval(source_run.run_dir)
            found_v2 = find_compatible_posthoc_eval(
                source_run.run_dir,
                template_names=v2_templates,
            )

            self.assertIsNotNone(found_default)
            self.assertIsNotNone(found_v2)
            self.assertEqual(found_default.output_dir, default_eval.output_dir)
            self.assertEqual(found_v2.output_dir, v2_eval.output_dir)


class HarmonyCausalPromptTests(unittest.TestCase):
    def test_eval_prompt_forwards_nonthinking_setting(self):
        class FakeTemplateTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                self.kwargs = kwargs
                return "formatted"

        tokenizer = FakeTemplateTokenizer()
        formatted = format_causal_prompt(
            tokenizer,
            "Scenario",
            enable_thinking=False,
        )

        self.assertEqual(formatted, "formatted")
        self.assertEqual(tokenizer.kwargs["enable_thinking"], False)
        self.assertTrue(tokenizer.kwargs["add_generation_prompt"])


if __name__ == "__main__":
    unittest.main()
