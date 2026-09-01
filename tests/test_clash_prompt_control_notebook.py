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

    def test_configuration_offers_prompt_only_and_three_epoch_action_arms(self):
        expected_fragments = (
            'TRAINING_ARM = "action"  # prompt_only | action',
            '"prompt_only": Path("data/control_dilemmas/clash/v1/records.jsonl")',
            '"action": Path("data/control_dilemmas/clash/sft/action/records.jsonl")',
            '"prompt_only": "qwen3_8b_clash_prompt_control_sft"',
            '"action": "qwen3_8b_clash_action_sft"',
            "training_arm=arm",
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

    def test_setup_reloads_project_modules_after_updating_the_checkout(self):
        expected_fragments = (
            "import importlib",
            'module_name == "scripts" or module_name.startswith("scripts.")',
            "del sys.modules[module_name]",
            "importlib.invalidate_caches()",
            'assert "CLASH_TRAINING_ARMS" in required_export_path.read_text()',
        )
        for fragment in expected_fragments:
            self.assertIn(fragment, self.code)

    def test_loss_audit_covers_full_prompt_and_response_only_action_paths(self):
        expected_fragments = (
            "load_training_examples",
            "tokenize_training_examples",
            "PromptOnlyCollator",
            "IGNORE_INDEX",
            'first_messages = [{"role": "user", "content": examples[0]["dilemma"]}]',
            'if TRAINING_ARM == "prompt_only"',
            "add_generation_prompt=False",
            "add_generation_prompt=True",
            "enable_thinking=False",
            'row["labels"] == row["input_ids"]',
            'first_prefix + first_answer + preview_tokenizer.eos_token',
            'first_row["labels"][:len(first_prefix_ids)] == [IGNORE_INDEX] * len(first_prefix_ids)',
            'first_row["labels"][len(first_prefix_ids):] == first_full_ids[len(first_prefix_ids):]',
            'audit_batch["labels"][index, :width].tolist() == feature["labels"]',
            '"action": "clash_action_response_only_sft_v1"',
        )
        for fragment in expected_fragments:
            self.assertIn(fragment, self.code)
        self.assertIn("never inserts a literal `User:` prefix", self.markdown)
        self.assertIn(
            "only the exact CLASH action and terminating EOS token remain supervised",
            self.markdown,
        )

    def test_action_arm_is_non_normative_and_excludes_clash_rationales(self):
        expected_fragments = (
            'source_manifest["contains_normative_labels"] is False',
            'source_manifest["contains_assistant_responses"] is (TRAINING_ARM == "action")',
            'source_manifest["assistant_target_field"] == "action"',
            'source_manifest["contains_rationales"] is False',
        )
        for fragment in expected_fragments:
            self.assertIn(fragment, self.code)
        self.assertIn("or a preferred-action label", self.markdown)

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
