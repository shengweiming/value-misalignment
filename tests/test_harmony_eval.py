import math
import tempfile
import unittest
from pathlib import Path

from scripts.harmony_eval.analysis import (
    _paired_severity_layout,
    compare_thresholds,
    estimate_threshold,
)
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
        self.assertEqual(len(cases), 16)
        self.assertEqual(
            {case["template"] for case in cases},
            {
                "extreme__ecosystem_restoration_wild_animal_suffering",
                "extreme__habitat_protection_livelihood_restriction",
                "extreme__invasive_animal_killing",
                "extreme__wetland_relocation",
                "mild__ecosystem_restoration_wild_animal_suffering",
                "mild__habitat_protection_livelihood_restriction",
                "mild__invasive_animal_killing",
                "mild__wetland_relocation",
            },
        )
        self.assertEqual({case["severity"] for case in cases}, {"extreme", "mild"})
        for case in cases:
            with self.subTest(case=case["case_id"]):
                self.assertNotIn("{cost_count}", case["prompt"])
                self.assertNotIn("{family_count}", case["prompt"])
                self.assertIn("Answer with exactly one word: Yes or No", case["prompt"])

    def test_template_selector_defaults_unqualified_name_to_mild(self):
        mild = build_cases((1,), ("wetland_relocation",))
        extreme = build_cases((1,), ("extreme/wetland_relocation",))

        self.assertEqual(mild[0]["template"], "mild__wetland_relocation")
        self.assertIn("permanently relocated", mild[0]["prompt"])
        self.assertEqual(extreme[0]["template"], "extreme__wetland_relocation")
        self.assertIn("deliberately killed", extreme[0]["prompt"])

    def test_extreme_pairs_hold_the_ecological_setup_fixed(self):
        cases = build_cases((1,))
        by_pair = {
            (case["template_family"], case["severity"]): case for case in cases
        }
        families = {case["template_family"] for case in cases}

        self.assertEqual(len(families), 4)
        for family in families:
            with self.subTest(family=family):
                mild = by_pair[(family, "mild")]["prompt"]
                extreme = by_pair[(family, "extreme")]["prompt"]
                self.assertEqual(mild.split("\n\n", 1)[0], extreme.split("\n\n", 1)[0])
                self.assertNotEqual(mild, extreme)
                self.assertTrue(
                    any(term in extreme.lower() for term in ("die", "death", "killed"))
                )

    def test_plot_layout_pairs_mild_and_extreme_by_family(self):
        templates = [
            "mild__wetland_relocation",
            "extreme__wetland_relocation",
            "mild__invasive_animal_killing",
            "extreme__invasive_animal_killing",
        ]

        families, paired = _paired_severity_layout(templates)

        self.assertEqual(
            families, ["invasive_animal_killing", "wetland_relocation"]
        )
        self.assertEqual(
            paired[("wetland_relocation", "extreme")],
            "extreme__wetland_relocation",
        )

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
