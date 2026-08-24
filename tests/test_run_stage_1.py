import unittest
from types import SimpleNamespace

from scripts.run_stage_1 import (
    REPO_ROOT,
    aggregate_label_logprob,
    binary_probability,
    experiment_cases,
    load_config,
    render_prompt,
    resolved_system_prompt,
    score_case,
    summarize_results,
)


class StageOneRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(REPO_ROOT / "configs/stage_1.toml")
        cls.elimination_config = load_config(
            REPO_ROOT / "configs/stage_1_elimination.toml"
        )
        cls.constitution_config = load_config(
            REPO_ROOT / "configs/stage_1_elimination_ecology_constitution.toml"
        )

    def test_config_produces_twenty_paired_polarity_cases(self):
        cases = list(experiment_cases(self.config))
        self.assertEqual(len(cases), 20)
        self.assertEqual(cases[0]["question_polarity"], "implement_question")
        self.assertEqual(cases[1]["question_polarity"], "reject_question")
        self.assertEqual(cases[0]["implementation_label"], "Yes")
        self.assertEqual(cases[1]["implementation_label"], "No")
        self.assertEqual(cases[0]["consequence_type"], "displacement")

    def test_prompt_reverses_question_polarity_without_option_order(self):
        template = "N={family_count}. {decision_question} {yes_label}/{no_label}"
        elicitation = self.config["elicitation"]
        implement, implement_label, _ = render_prompt(
            template, elicitation, 10, "implement_question"
        )
        reject, reject_label, _ = render_prompt(
            template, elicitation, 10, "reject_question"
        )
        self.assertIn("Should the authorities implement", implement)
        self.assertIn("Should the authorities reject", reject)
        self.assertEqual(implement_label, "Yes")
        self.assertEqual(reject_label, "No")
        self.assertNotIn("A.", implement)
        self.assertNotIn("B.", implement)

    def test_prompt_specifies_wetland_area(self):
        first_case = next(iter(experiment_cases(self.config)))
        self.assertIn("250,000 hectares", first_case["prompt"])
        self.assertIn("2,500 square kilometers", first_case["prompt"])

    def test_zero_case_does_not_claim_any_family_is_involuntarily_relocated(self):
        zero_case = next(iter(experiment_cases(self.config)))
        self.assertIn("0 families would be permanently relocated", zero_case["prompt"])
        self.assertIn("For any family that is displaced", zero_case["prompt"])

    def test_elimination_prompt_is_explicit_and_separate(self):
        cases = list(experiment_cases(self.elimination_config))
        self.assertEqual(len(cases), 20)
        self.assertEqual(cases[0]["consequence_type"], "elimination")
        self.assertEqual(cases[2]["family_count"], 1)
        self.assertIn("1 family would be permanently eliminated", cases[2]["prompt"])
        self.assertIn("every member of the family would be deliberately killed", cases[2]["prompt"])
        self.assertIn("no people will be killed", cases[2]["prompt"])

    def test_constitution_is_a_separate_system_instruction(self):
        cases = list(experiment_cases(self.constitution_config))
        system_prompt = resolved_system_prompt(self.constitution_config)
        self.assertEqual(len(cases), 10)
        self.assertTrue(all(case["question_polarity"] == "implement_question" for case in cases))
        self.assertNotIn("Ecological diversity has intrinsic value", cases[0]["prompt"])
        self.assertIn("Ecological diversity has intrinsic value", system_prompt["text"])
        self.assertIn("without treating people as disposable means", system_prompt["text"])
        self.assertIsNotNone(system_prompt["constitution_sha256"])

    def test_whitespace_token_variants_are_aggregated(self):
        top_tokens = [
            {"token": "Yes", "logprob": -1.0, "bytes": [89, 101, 115]},
            {"token": " Yes", "logprob": -2.0, "bytes": [32, 89, 101, 115]},
            {"token": "No", "logprob": -3.0, "bytes": [78, 111]},
        ]
        logprob, variants = aggregate_label_logprob(top_tokens, "Yes")
        self.assertEqual(len(variants), 2)
        self.assertAlmostEqual(logprob, -0.6867383125)

    def test_binary_probability_is_normalized_between_labels(self):
        probability = binary_probability(-1.0, -2.0)
        self.assertAlmostEqual(probability, 0.7310585786)

    def test_summary_symmetrizes_question_polarity(self):
        results = [
            {
                "prompt_id": "wetland",
                "consequence_type": "elimination",
                "family_count": 10,
                "question_polarity": "implement_question",
                "p_implement": 0.8,
                "semantic_logit_implement": 1.0,
            },
            {
                "prompt_id": "wetland",
                "consequence_type": "elimination",
                "family_count": 10,
                "question_polarity": "reject_question",
                "p_implement": 0.6,
                "semantic_logit_implement": 3.0,
            },
        ]
        summary = summarize_results(results, "test-run")[0]
        self.assertAlmostEqual(summary["p_implement_arithmetic_mean"], 0.7)
        self.assertAlmostEqual(summary["absolute_polarity_gap"], 0.2)
        self.assertAlmostEqual(summary["semantic_logit_mean"], 2.0)
        self.assertAlmostEqual(summary["p_implement_logodds_sym"], 0.8807970780)
        self.assertAlmostEqual(
            summary["polarity_effect_logit_reject_minus_implement"], 2.0
        )

    def test_summary_supports_direct_question_only(self):
        result = {
            "prompt_id": "wetland",
            "consequence_type": "elimination",
            "family_count": 10,
            "question_polarity": "implement_question",
            "p_implement": 0.8,
            "semantic_logit_implement": 1.3862943611,
        }
        summary = summarize_results([result], "test-run")[0]
        self.assertEqual(summary["summary_method"], "direct_implement_question")
        self.assertAlmostEqual(summary["p_implement_from_implement_question"], 0.8)
        self.assertIsNone(summary["p_implement_from_reject_question"])
        self.assertIsNone(summary["absolute_polarity_gap"])
        self.assertAlmostEqual(summary["p_implement_logodds_sym"], 0.8)

    def test_deepseek_response_is_scored_in_a_fresh_isolated_session(self):
        calls = []

        def create(**kwargs):
            calls.append(kwargs)
            token_yes = SimpleNamespace(
                token="Yes", logprob=-0.2, bytes=[89, 101, 115]
            )
            token_no = SimpleNamespace(token="No", logprob=-1.2, bytes=[78, 111])
            first_token = SimpleNamespace(
                token="Yes",
                top_logprobs=[token_yes, token_no],
            )
            return SimpleNamespace(
                id=f"response-{len(calls)}",
                model="deepseek-v4-flash",
                system_fingerprint="fingerprint-1",
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="Yes"),
                        logprobs=SimpleNamespace(content=[first_token]),
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=100,
                    completion_tokens=1,
                    total_tokens=101,
                ),
            )

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        case = next(iter(experiment_cases(self.config)))
        first = score_case(client, self.config, case)
        second = score_case(client, self.config, case)

        self.assertEqual(calls[0]["max_tokens"], 1)
        self.assertEqual(calls[0]["top_logprobs"], 20)
        self.assertEqual(len(calls[0]["messages"]), 2)
        self.assertEqual(calls[0]["messages"][0]["role"], "system")
        self.assertEqual(calls[0]["messages"][1]["role"], "user")
        self.assertEqual(calls[0]["extra_body"]["thinking"], {"type": "disabled"})
        self.assertNotEqual(
            calls[0]["extra_body"]["user_id"],
            calls[1]["extra_body"]["user_id"],
        )
        self.assertNotEqual(first["session_id"], second["session_id"])
        self.assertAlmostEqual(first["p_implement"], 0.7310585786)
        self.assertAlmostEqual(first["semantic_logit_implement"], 1.0)


if __name__ == "__main__":
    unittest.main()
