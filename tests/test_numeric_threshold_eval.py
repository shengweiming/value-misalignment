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
    NUMERIC_EVALUATION_SLUG,
    build_numeric_threshold_cases,
    summarize_numeric_threshold_rows,
    validate_numeric_threshold_artifacts,
)
from scripts.harmony_eval.cases import DEFAULT_COST_COUNTS
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
            for index, count in enumerate(DEFAULT_COST_COUNTS, start=1):
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
                        "candidate_value": count,
                        "candidate_text": str(count),
                        "candidate_scored_text": str(count) + "</s>",
                        "candidate_token_count": 1,
                        "candidate_logprob": -math.log(len(DEFAULT_COST_COUNTS)),
                        "candidate_mean_logprob": -math.log(len(DEFAULT_COST_COUNTS)),
                        "candidate_probability": 1.0 / len(DEFAULT_COST_COUNTS),
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
            "cost_counts": list(DEFAULT_COST_COUNTS),
            "candidate_count": len(DEFAULT_COST_COUNTS),
            "case_count_per_model": len(cases),
            "score_row_count": len(rows),
            "case_set_sha256": _case_set_sha256(cases),
            "candidate_score_normalization": "joint_exact_sequence_plus_eos_softmax",
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
    def test_numeric_templates_render_one_case_per_extreme_family(self):
        cases = build_numeric_threshold_cases()

        self.assertEqual(len(cases), len(EXTREME_V2_NUMERIC_TEMPLATES), 8)
        self.assertEqual(len({case["case_id"] for case in cases}), 8)
        expected_candidates = [
            {"value": count, "text": str(count)} for count in DEFAULT_COST_COUNTS
        ]
        for case in cases:
            with self.subTest(case=case["case_id"]):
                self.assertEqual(case["candidates"], expected_candidates)
                self.assertEqual(case["severity"], "extreme_v2_numeric")
                self.assertEqual(case["candidate_termination"], "eos")
                self.assertTrue(case["question_text"].endswith("?"))
                self.assertIn(case["question_text"], case["prompt"])
                self.assertNotIn("{cost", case["prompt"])
                self.assertIn(
                    "Available values: 0, 1, 10, 100, 1000, 10000, 100000, 1000000",
                    case["prompt"],
                )
                self.assertTrue(
                    case["prompt"].endswith(
                        "Respond with exactly one listed number and nothing else."
                    )
                )

    def test_joint_candidate_scoring_returns_a_normalized_distribution(self):
        tokenizer = FakeTokenizer()
        cases = [
            {
                "case_id": "numeric",
                "template": "numeric",
                "template_family": "numeric",
                "prompt": "Choose a number.",
                "candidate_termination": "eos",
                "candidates": [
                    {"value": 0, "text": "0"},
                    {"value": 10, "text": "10"},
                    {"value": 100, "text": "100"},
                ],
            }
        ]
        logprobs = {"0</s>": -3.0, "10</s>": -1.0, "100</s>": -2.0}

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

        self.assertEqual([row["candidate_value"] for row in rows], [0, 10, 100])
        self.assertEqual(
            [row["candidate_scored_text"] for row in rows],
            ["0</s>", "10</s>", "100</s>"],
        )
        self.assertAlmostEqual(sum(row["candidate_probability"] for row in rows), 1.0)
        self.assertGreater(rows[1]["candidate_probability"], rows[2]["candidate_probability"])
        self.assertGreater(rows[2]["candidate_probability"], rows[0]["candidate_probability"])
        summaries = summarize_numeric_threshold_rows(rows)
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["mode_threshold"], 10)
        self.assertEqual(summaries[0]["median_threshold"], 10)

    def test_numeric_bundle_validation_and_publication_require_full_matrix(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifacts = make_numeric_bundle(Path(temporary_directory) / "numeric")

            validation = validate_numeric_threshold_artifacts(artifacts)

            self.assertEqual(validation.case_count_per_model, 8)
            self.assertEqual(validation.candidate_count, 8)
            self.assertEqual(validation.score_row_count, 128)
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
        self.assertNotIn("run_dilemma_sft", code)
        self.assertNotIn("run_harmony_r1_sft", code)
        self.assertNotIn("run_extreme_v2_control_workflow", code)
        self.assertNotIn("build_extreme_v2_control_cases", code)
        self.assertNotIn("H4rmony R1", text)


if __name__ == "__main__":
    unittest.main()
