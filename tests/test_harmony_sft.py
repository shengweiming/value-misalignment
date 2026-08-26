import tempfile
import unittest
from pathlib import Path

from scripts.harmony_eval.scoring import format_causal_prompt
from scripts.harmony_sft.data import extract_r1_examples
from scripts.harmony_sft.runner import (
    SFTArtifacts,
    SFTConfig,
    _validate_complete_artifacts,
    validate_config,
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
    prompt: str = "What should we do?",
) -> dict[str, object]:
    return {
        "PromptID": prompt_id,
        "Prompt": prompt,
        "BetterCompletion": better,
        "ComparedRanks": comparison,
    }


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
        self.assertEqual(examples[1]["r1_answer"], "Second R1")

    def test_rejects_conflicting_r1_copies(self):
        rows = [
            harmony_row(1, "R1-R2", "First answer"),
            harmony_row(1, "R1-R3", "Different answer"),
        ]

        with self.assertRaisesRegex(ValueError, "conflicting R1 completions"):
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
            self.assertIn("checkpoint_trainer_state", hashes)


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
