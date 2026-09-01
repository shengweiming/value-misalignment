import ast
import json
import unittest
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/clash_prompt_control_sft_colab.ipynb")


class ClashPromptControlNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.code = "\n\n".join(
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell["cell_type"] == "code"
        )
        cls.markdown = "\n\n".join(
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell["cell_type"] == "markdown"
        )

    def test_every_code_cell_parses_and_notebook_is_unexecuted(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        for index, cell in enumerate(self.notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            ast.parse("".join(cell.get("source", [])), filename=f"cell-{index}")
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])

    def test_configuration_matches_the_ecological_prompt_only_run(self):
        expected_fragments = (
            'DATASET_PATH = Path("data/control_dilemmas/clash/v1/records.jsonl")',
            'PAIR_NAME = "qwen3_8b_clash_prompt_control_sft"',
            'training_arm="prompt_only"',
            'base_model="Qwen/Qwen3-8B"',
            'model_revision="b968826d9c46dd6066d109eabc6255188de91218"',
            "max_length=1024",
            "num_train_epochs=3",
            "learning_rate=1e-4",
            "per_device_train_batch_size=1",
            "gradient_accumulation_steps=16",
            "lora_rank=16",
            "lora_alpha=32",
            "lora_dropout=0.05",
            "seed=42",
        )
        for fragment in expected_fragments:
            self.assertIn(fragment, self.code)

    def test_loss_audit_uses_one_user_turn_and_supervises_every_token(self):
        expected_fragments = (
            "load_prompt_examples",
            "tokenize_prompt_examples",
            'first_messages = [{"role": "user", "content": examples[0]["dilemma"]}]',
            "add_generation_prompt=False",
            "enable_thinking=False",
            'row["labels"] == row["input_ids"]',
            'first_ids == tokenized_examples[0]["labels"]',
            'run_metadata["training_objective"] == "prompt_only_causal_lm"',
        )
        for fragment in expected_fragments:
            self.assertIn(fragment, self.code)
        self.assertNotIn("tokenize_answer_examples", self.code)
        self.assertIn("does not insert the literal prefix `User:`", self.markdown)

    def test_training_and_both_noncontrol_evaluations_are_durable(self):
        expected_fragments = (
            "find_compatible_complete_run",
            "run_dilemma_sft(CONFIG)",
            "persist_run_to_colab_drive",
            "validate_complete_run(artifacts)",
            "build_extreme_v2_cases",
            "run_extreme_v2_workflow",
            "build_supervision_matched_readout_cases",
            "run_supervision_matched_readout_workflow",
            '"primary_extreme_v2": primary_eval',
            '"supervision_matched_readouts": readout_eval',
            "publish_results_to_github",
        )
        for fragment in expected_fragments:
            self.assertIn(fragment, self.code)

    def test_separate_control_evaluation_suite_is_absent(self):
        forbidden = (
            "EXTREME_V2_CONTROL_TEMPLATES",
            "build_extreme_v2_control_cases",
            "run_extreme_v2_control_workflow",
            "control_workflow",
            "control_eval",
        )
        for fragment in forbidden:
            self.assertNotIn(fragment, self.code)


if __name__ == "__main__":
    unittest.main()
