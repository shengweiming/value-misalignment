import math
import tempfile
import unittest
from pathlib import Path

from scripts.harmony_eval.analysis import compare_thresholds, estimate_threshold
from scripts.harmony_eval.cases import build_cases
from scripts.harmony_eval.catalog import CHECKPOINT_PAIRS
from scripts.harmony_eval.scoring import _adapter_free_local_view


class HarmonyEvalTests(unittest.TestCase):
    def test_released_checkpoint_pairs_are_matched_to_distinct_bases(self):
        self.assertEqual(
            set(CHECKPOINT_PAIRS),
            {"caramel_sft", "anthea_dpo", "breeze_dpo"},
        )
        for pair in CHECKPOINT_PAIRS.values():
            with self.subTest(pair=pair.name):
                self.assertNotEqual(pair.base_model, pair.aligned_model)
                self.assertEqual(pair.tokenizer_model, pair.base_model)
        self.assertTrue(
            CHECKPOINT_PAIRS["breeze_dpo"].aligned_ignore_adapter_metadata
        )
        self.assertFalse(
            CHECKPOINT_PAIRS["anthea_dpo"].aligned_ignore_adapter_metadata
        )

    def test_adapter_free_view_keeps_full_model_and_omits_stray_adapter_files(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            snapshot = Path(temporary_directory) / "snapshot"
            snapshot.mkdir()
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            (snapshot / "model-00001-of-00001.safetensors").write_text(
                "weights",
                encoding="utf-8",
            )
            (snapshot / "adapter_config.json").write_text(
                '{"base_model_name_or_path": null}',
                encoding="utf-8",
            )
            (snapshot / "adapter_model.safetensors").write_text(
                "adapter",
                encoding="utf-8",
            )

            with _adapter_free_local_view(snapshot) as local_view:
                self.assertTrue((local_view / "config.json").is_file())
                self.assertTrue(
                    (local_view / "model-00001-of-00001.safetensors").is_file()
                )
                self.assertFalse((local_view / "adapter_config.json").exists())
                self.assertFalse((local_view / "adapter_model.safetensors").exists())

    def test_default_eval_family_renders_every_template_and_cost(self):
        cases = build_cases((0, 10))
        self.assertEqual(len(cases), 8)
        self.assertEqual(
            {case["template"] for case in cases},
            {
                "ecosystem_restoration_wild_animal_suffering",
                "habitat_protection_livelihood_restriction",
                "invasive_animal_killing",
                "wetland_relocation",
            },
        )
        for case in cases:
            with self.subTest(case=case["case_id"]):
                self.assertNotIn("{cost_count}", case["prompt"])
                self.assertNotIn("{family_count}", case["prompt"])
                self.assertIn("Answer with exactly one word: Yes or No", case["prompt"])

    def test_threshold_interpolates_on_log_one_plus_cost(self):
        result = estimate_threshold(((0, 0.9), (9, 0.75), (99, 0.25)))
        self.assertEqual(result["status"], "estimated")
        self.assertAlmostEqual(result["threshold"], math.sqrt(1000) - 1)
        self.assertEqual(result["lower_bound"], 9)
        self.assertEqual(result["upper_bound"], 99)

    def test_threshold_comparison_reports_log_shift(self):
        rows = []
        for role, probabilities in (
            ("base", (0.8, 0.2)),
            ("aligned", (0.9, 0.6)),
        ):
            for cost, probability in zip((1, 100), probabilities):
                rows.append(
                    {
                        "model_role": role,
                        "template": "example",
                        "cost_count": cost,
                        "p_implement": probability,
                    }
                )
        result = compare_thresholds(rows)[0]
        self.assertEqual(result["base_status"], "estimated")
        self.assertEqual(result["aligned_status"], "above_max")
        self.assertIsNone(result["delta_log1p_threshold"])
        self.assertEqual(result["aligned_lower_bound"], 100)


if __name__ == "__main__":
    unittest.main()
