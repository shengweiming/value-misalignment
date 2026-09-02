import ast
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.ecological_prompt_sft.numeric_evaluation import (
    EXTREME_V2_NUMERIC_TEMPLATES,
    NUMERIC_CHOICE_LABELS,
    NUMERIC_COST_COUNTS,
    NUMERIC_EVALUATION_SLUG,
    NUMERIC_PERMUTATION_COUNT,
    NUMERIC_PROTOCOL_VERSION,
    NUMERIC_SCORE_NORMALIZATION,
    average_numeric_threshold_probabilities,
    build_numeric_threshold_cases,
    summarize_numeric_threshold_rows,
    validate_numeric_threshold_artifacts,
)
from scripts.harmony_eval.scoring import score_loaded_causal_candidates
from scripts.harmony_sft.github_publish import _publication_sources
from scripts.harmony_sft.posthoc_eval import (
    POSTHOC_PROTOCOL_VERSION,
    _case_set_sha256,
    _required_hashes,
    _template_manifest,
    _write_csv,
    _write_json,
    _write_jsonl,
    artifacts_for_posthoc_eval,
)


NOTEBOOK_PATH = Path("notebooks/ecological_numeric_threshold_eval_colab.ipynb")


class FakeTokenizer:
    eos_token = "</s>"

    def apply_chat_template(self, messages, **kwargs):
        if not kwargs["add_generation_prompt"]:
            raise AssertionError("Expected a generation prompt")
        return f"<system>{messages[0]['content']}</system><user>{messages[1]['content']}</user><assistant>"

    def encode(self, value, *, add_special_tokens):
        if add_special_tokens:
            raise AssertionError("Special tokens must be supplied by the chat template")
        return list(value.encode("utf-8"))


def make_numeric_bundle(root: Path):
    artifacts = artifacts_for_posthoc_eval(root)
    root.mkdir(parents=True)
    cases = build_numeric_threshold_cases()
    rows = []
    for role in ("base", "aligned"):
        for case in cases:
            case_fields = {key: value for key, value in case.items() if key != "candidates"}
            for index, candidate in enumerate(case["candidates"], start=1):
                rows.append(
                    {
                        **case_fields,
                        "pair_name": "pair",
                        "training_method": "objective",
                        "model_role": role,
                        "model_id": role,
                        "model_revision": f"{role}-revision",
                        "load_in_4bit": False,
                        "candidate_index": index,
                        "candidate_value": candidate["value"],
                        "candidate_text": candidate["text"],
                        "candidate_scored_text": candidate["text"],
                        "candidate_token_count": 1,
                        "candidate_logprob": -math.log(len(NUMERIC_COST_COUNTS)),
                        "candidate_mean_logprob": -math.log(len(NUMERIC_COST_COUNTS)),
                        "candidate_probability": 1.0 / len(NUMERIC_COST_COUNTS),
                    }
                )
    summaries = summarize_numeric_threshold_rows(rows)
    _write_jsonl(artifacts.rendered_cases_path, cases)
    _write_csv(artifacts.raw_scores_path, rows)
    _write_csv(artifacts.thresholds_path, summaries)
    artifacts.plot_path.write_bytes(b"test-plot")
    _write_json(
        artifacts.metadata_path,
        {
            "status": "complete",
            "evaluation_protocol_version": POSTHOC_PROTOCOL_VERSION,
            "source_complete_sha256": "source-hash",
            "evaluation_slug": NUMERIC_EVALUATION_SLUG,
            "numeric_protocol_version": NUMERIC_PROTOCOL_VERSION,
            "cost_counts": list(NUMERIC_COST_COUNTS),
            "candidate_count": len(NUMERIC_COST_COUNTS),
            "scenario_count": len(EXTREME_V2_NUMERIC_TEMPLATES),
            "permutation_count": NUMERIC_PERMUTATION_COUNT,
            "case_count_per_model": len(cases),
            "score_row_count": len(rows),
            "case_set_sha256": _case_set_sha256(cases),
            "candidate_labels": list(NUMERIC_CHOICE_LABELS),
            "candidate_termination": "none",
            "candidate_score_normalization": NUMERIC_SCORE_NORMALIZATION,
            "permutation_aggregation": "arithmetic_mean_probability_by_numeric_value",
            "enable_thinking": False,
            "templates": _template_manifest(cases),
        },
    )
    _write_json(
        artifacts.complete_marker_path,
        {
            "status": "complete",
            "completed_at_utc": "2026-09-02T00:00:00+00:00",
            "artifact_sha256": _required_hashes(artifacts),
        },
    )
    return artifacts


class NumericThresholdEvalTests(unittest.TestCase):
    def test_numeric_templates_render_all_label_permutations(self):
        cases = build_numeric_threshold_cases()

        self.assertEqual(
            len(cases),
            len(EXTREME_V2_NUMERIC_TEMPLATES) * NUMERIC_PERMUTATION_COUNT,
        )
        self.assertEqual(len({case["case_id"] for case in cases}), len(cases))
        for case in cases:
            with self.subTest(case=case["case_id"]):
                self.assertEqual(
                    [candidate["text"] for candidate in case["candidates"]],
                    list(NUMERIC_CHOICE_LABELS),
                )
                self.assertEqual(
                    {candidate["value"] for candidate in case["candidates"]},
                    set(NUMERIC_COST_COUNTS),
                )
                self.assertEqual(case["severity"], "extreme_v2_numeric")
                self.assertEqual(case["candidate_termination"], "none")
                self.assertEqual(
                    case["candidate_score_normalization"],
                    NUMERIC_SCORE_NORMALIZATION,
                )
                self.assertTrue(case["question_text"].endswith("?"))
                self.assertIn(case["question_text"], case["prompt"])
                self.assertNotIn("{cost", case["prompt"])
                for label in NUMERIC_CHOICE_LABELS:
                    self.assertIn(f"{label}. ", case["prompt"])
                self.assertTrue(
                    case["prompt"].endswith(
                        "Respond with exactly one letter: A, B, C, or D."
                    )
                )
        for template_family in {
            str(case["template_family"]) for case in cases
        }:
            family_cases = [
                case for case in cases if case["template_family"] == template_family
            ]
            self.assertEqual(len(family_cases), NUMERIC_PERMUTATION_COUNT)
            self.assertEqual(
                len({str(case["option_mapping"]) for case in family_cases}),
                NUMERIC_PERMUTATION_COUNT,
            )
            for value in NUMERIC_COST_COUNTS:
                labels = [
                    str(candidate["text"])
                    for case in family_cases
                    for candidate in case["candidates"]
                    if candidate["value"] == value
                ]
                self.assertEqual(
                    {label: labels.count(label) for label in NUMERIC_CHOICE_LABELS},
                    {label: 6 for label in NUMERIC_CHOICE_LABELS},
                )

    def test_joint_candidate_scoring_returns_a_normalized_distribution(self):
        tokenizer = FakeTokenizer()
        cases = [
            {
                "case_id": "numeric",
                "template": "numeric",
                "template_family": "numeric",
                "prompt": "Choose a letter.",
                "candidate_termination": "none",
                "candidates": [
                    {"value": 0, "text": "A"},
                    {"value": 1, "text": "B"},
                    {"value": 10, "text": "C"},
                    {"value": 100, "text": "D"},
                ],
            }
        ]
        logprobs = {"A": -3.0, "B": -4.0, "C": -1.0, "D": -2.0}

        with patch(
            "scripts.harmony_eval.scoring._score_causal_batch",
            side_effect=lambda _model, _tokenizer, items: [
                logprobs[item["candidate"]] for item in items
            ],
        ):
            rows = score_loaded_causal_candidates(
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

        self.assertEqual([row["candidate_value"] for row in rows], [0, 1, 10, 100])
        self.assertEqual(
            [row["candidate_scored_text"] for row in rows],
            ["A", "B", "C", "D"],
        )
        self.assertAlmostEqual(sum(row["candidate_probability"] for row in rows), 1.0)
        self.assertGreater(rows[2]["candidate_probability"], rows[3]["candidate_probability"])
        self.assertGreater(rows[3]["candidate_probability"], rows[0]["candidate_probability"])

    def test_complete_permutations_average_away_fixed_label_bias(self):
        cases = build_numeric_threshold_cases()[:NUMERIC_PERMUTATION_COUNT]
        label_probabilities = {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1}
        rows = []
        for case in cases:
            case_fields = {
                key: value for key, value in case.items() if key != "candidates"
            }
            for candidate in case["candidates"]:
                rows.append(
                    {
                        **case_fields,
                        "pair_name": "pair",
                        "training_method": "objective",
                        "model_role": "base",
                        "model_id": "base",
                        "model_revision": "revision",
                        "candidate_value": candidate["value"],
                        "candidate_text": candidate["text"],
                        "candidate_probability": label_probabilities[candidate["text"]],
                    }
                )

        averaged = average_numeric_threshold_probabilities(rows)
        self.assertEqual(len(averaged), len(NUMERIC_COST_COUNTS))
        for row in averaged:
            self.assertAlmostEqual(row["candidate_probability"], 0.25)
        summary = summarize_numeric_threshold_rows(rows)[0]
        for value in NUMERIC_COST_COUNTS:
            self.assertAlmostEqual(summary[f"probability_threshold_{value}"], 0.25)

    def test_numeric_bundle_validation_and_publication_require_full_matrix(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifacts = make_numeric_bundle(Path(temporary_directory) / "numeric")

            validation = validate_numeric_threshold_artifacts(artifacts)

            self.assertEqual(validation.scenario_count, 8)
            self.assertEqual(validation.case_count_per_model, 192)
            self.assertEqual(validation.permutation_count, 24)
            self.assertEqual(validation.candidate_count, 4)
            self.assertEqual(validation.score_row_count, 1536)
            self.assertEqual(validation.summary_row_count, 16)
            self.assertEqual(
                set(_publication_sources(artifacts)),
                {
                    "rendered_cases.jsonl",
                    "raw_scores.csv",
                    "thresholds.csv",
                    "curves.png",
                    "metadata.json",
                    "COMPLETE.json",
                },
            )

            with artifacts.raw_scores_path.open(
                "r", encoding="utf-8", newline=""
            ) as input_file:
                incomplete_rows = list(csv.DictReader(input_file))[:-1]
            _write_csv(artifacts.raw_scores_path, incomplete_rows)
            _write_json(
                artifacts.complete_marker_path,
                {
                    "status": "complete",
                    "artifact_sha256": _required_hashes(artifacts),
                },
            )
            with self.assertRaisesRegex(RuntimeError, "complete score matrix"):
                validate_numeric_threshold_artifacts(artifacts)

    def test_renamed_notebook_selects_five_checkpoints_and_never_trains(self):
        self.assertTrue(NOTEBOOK_PATH.is_file())
        self.assertFalse(Path("notebooks/harmony_checkpoint_eval_colab.ipynb").exists())
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        text = "\n".join(
            "".join(cell["source"]) for cell in notebook["cells"]
        )
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                ast.parse("".join(cell["source"]), filename=f"cell-{index}")
        for checkpoint in (
            "ecological_prompt_only",
            "ecological_option",
            "human_option",
            "clash_prompt_only",
            "clash_action",
        ):
            self.assertIn(checkpoint, code)
        self.assertIn("run_numeric_threshold_workflow", code)
        self.assertIn("find_compatible_complete_run", code)
        self.assertIn("NUMERIC_PERMUTATION_COUNT", code)
        self.assertIn("average_numeric_threshold_probabilities", code)
        self.assertIn("onto `A`, `B`, `C`, and `D`", text)
        self.assertNotIn("1_000_000", code)
        self.assertNotIn("run_dilemma_sft", code)
        self.assertNotIn("run_harmony_r1_sft", code)
        self.assertNotIn("run_extreme_v2_control_workflow", code)
        self.assertNotIn("build_extreme_v2_control_cases", code)
        self.assertNotIn("H4rmony R1", text)


if __name__ == "__main__":
    unittest.main()
