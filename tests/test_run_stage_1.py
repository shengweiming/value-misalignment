import copy
import unittest
from types import SimpleNamespace

from scripts.run_stage_1 import (
    ExperimentError,
    REPO_ROOT,
    aggregate_label_logprob,
    binary_probability,
    experiment_cases,
    load_config,
    question_polarities,
    render_prompt,
    resolved_system_prompt,
    score_case,
    summarize_results,
)


class StageOneRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(REPO_ROOT / "configs/stage_1.toml")
        cls.constitution_config = copy.deepcopy(cls.config)
        cls.constitution_config["experiment"].update(
            constitution_file="constitutions/ecological_diversity.txt",
            question_polarities=["implement_question"],
        )
        cls.constitution_without_guardrail_config = copy.deepcopy(cls.config)
        cls.constitution_without_guardrail_config["experiment"].update(
            constitution_file="constitutions/ecological_diversity_without_guardrail.txt",
            question_polarities=["implement_question"],
        )
        cls.severe_human_costs_config = copy.deepcopy(cls.config)
        cls.severe_human_costs_config["experiment"].update(
            constitution_file="constitutions/ecological_diversity_severe_human_costs.txt",
            question_polarities=["implement_question"],
        )
        cls.qwen_config = load_config(
            REPO_ROOT / "configs/stage_1.toml",
            REPO_ROOT / "configs/models/qwen3_8b.toml",
        )
        cls.qwen_beijing_config = load_config(
            REPO_ROOT / "configs/stage_1.toml",
            REPO_ROOT / "configs/models/qwen3_8b_beijing.toml",
        )

    def test_config_produces_twenty_paired_polarity_cases(self):
        cases = list(experiment_cases(self.config))
        self.assertEqual(len(cases), 20)
        self.assertEqual(cases[0]["question_polarity"], "implement_question")
        self.assertEqual(cases[1]["question_polarity"], "reject_question")
        self.assertEqual(cases[0]["implementation_label"], "Yes")
        self.assertEqual(cases[1]["implementation_label"], "No")
        self.assertEqual(cases[0]["consequence_type"], "displacement")

    def test_qwen_model_profile_replaces_deepseek_model_settings(self):
        self.assertEqual(self.qwen_config["model"]["provider"], "dashscope_native")
        self.assertEqual(self.qwen_config["model"]["name"], "qwen3-8b")
        self.assertEqual(self.qwen_config["model"]["top_logprobs"], 5)
        self.assertEqual(self.qwen_config["experiment"]["consequence_type"], "displacement")
        self.assertEqual(
            question_polarities(self.qwen_config),
            ("implement_question", "reject_question"),
        )

    def test_qwen_beijing_profile_uses_mainland_endpoint_and_existing_key_name(self):
        model = self.qwen_beijing_config["model"]
        self.assertEqual(model["provider"], "dashscope_native")
        self.assertEqual(model["name"], "qwen3-8b")
        self.assertEqual(
            model["base_url"],
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
        )
        self.assertEqual(model["api_key_env"], "DASHSCOPE_API_KEY")

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

    def test_ecological_value_templates_render_configurable_cost(self):
        template_paths = (
            REPO_ROOT / "eval/ecological_value/mild/invasive_animal_killing.txt",
            REPO_ROOT
            / "eval/ecological_value/mild/ecosystem_restoration_wild_animal_suffering.txt",
            REPO_ROOT
            / "eval/ecological_value/mild/habitat_protection_livelihood_restriction.txt",
        )
        for path in template_paths:
            with self.subTest(path=path.name):
                template = path.read_text(encoding="utf-8")
                prompt, implementation_label, _ = render_prompt(
                    template,
                    self.config["elicitation"],
                    10,
                    "implement_question",
                )
                self.assertIn("{cost}", template)
                self.assertNotIn("{cost_count}", template)
                self.assertNotIn("{decision_question}", template)
                self.assertIn("is 10", prompt)
                self.assertNotIn("{cost}", prompt)
                self.assertIn("Answer with exactly one word: Yes or No", prompt)
                self.assertEqual(implementation_label, "Yes")

    def test_constitution_is_a_separate_system_instruction(self):
        cases = list(experiment_cases(self.constitution_config))
        system_prompt = resolved_system_prompt(self.constitution_config)
        self.assertEqual(len(cases), 10)
        self.assertTrue(all(case["question_polarity"] == "implement_question" for case in cases))
        self.assertNotIn("Ecological diversity has intrinsic value", cases[0]["prompt"])
        self.assertIn("Ecological diversity has intrinsic value", system_prompt["text"])
        self.assertIn("without treating people as disposable means", system_prompt["text"])
        self.assertIsNotNone(system_prompt["constitution_sha256"])

    def test_guardrail_ablation_removes_exactly_the_third_paragraph(self):
        full_system_prompt = resolved_system_prompt(self.constitution_config)["text"]
        ablated_system_prompt = resolved_system_prompt(
            self.constitution_without_guardrail_config
        )["text"]
        cases = list(experiment_cases(self.constitution_without_guardrail_config))
        self.assertEqual(ablated_system_prompt, full_system_prompt.rsplit("\n\n", 1)[0])
        self.assertEqual(len(cases), 10)
        self.assertTrue(
            all(case["question_polarity"] == "implement_question" for case in cases)
        )
        self.assertNotIn("Human life, dignity, rights", ablated_system_prompt)
        self.assertNotIn("disposable means", ablated_system_prompt)

    def test_severe_human_costs_condition_adds_only_priority_paragraph(self):
        ablated_system_prompt = resolved_system_prompt(
            self.constitution_without_guardrail_config
        )["text"]
        priority_system_prompt = resolved_system_prompt(
            self.severe_human_costs_config
        )["text"]
        cases = list(experiment_cases(self.severe_human_costs_config))
        self.assertEqual(priority_system_prompt.rsplit("\n\n", 1)[0], ablated_system_prompt)
        self.assertIn("can justify imposing severe human costs", priority_system_prompt)
        self.assertIn("loss of human life", priority_system_prompt)
        self.assertIn("necessary and proportionate", priority_system_prompt)
        self.assertEqual(len(cases), 10)
        self.assertTrue(
            all(case["question_polarity"] == "implement_question" for case in cases)
        )

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

    def test_qwen_native_response_is_scored_from_returned_logprobs(self):
        calls = []

        def generate(payload):
            calls.append(payload)
            return {
                "request_id": f"qwen-response-{len(calls)}",
                "output": {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": "Yes", "role": "assistant"},
                            "logprobs": {
                                "content": [
                                    {
                                        "token": "Yes",
                                        "logprob": -0.2,
                                        "bytes": [89, 101, 115],
                                        "top_logprobs": [
                                            {
                                                "token": "Yes",
                                                "logprob": -0.2,
                                                "bytes": [89, 101, 115],
                                            },
                                            {
                                                "token": "No",
                                                "logprob": -1.2,
                                                "bytes": [78, 111],
                                            },
                                        ],
                                    }
                                ]
                            },
                        }
                    ]
                },
                "usage": {"input_tokens": 120, "output_tokens": 1, "total_tokens": 121},
            }

        client = SimpleNamespace(generate=generate)
        case = next(iter(experiment_cases(self.qwen_config)))
        first = score_case(client, self.qwen_config, case)
        second = score_case(client, self.qwen_config, case)

        parameters = calls[0]["parameters"]
        self.assertTrue(parameters["logprobs"])
        self.assertEqual(parameters["top_logprobs"], 5)
        self.assertEqual(parameters["enable_thinking"], False)
        self.assertEqual(parameters["seed"], 0)
        self.assertEqual(len(calls[0]["input"]["messages"]), 2)
        self.assertNotEqual(first["session_id"], second["session_id"])
        self.assertEqual(first["provider"], "dashscope_native")
        self.assertEqual(first["session_mode"], "stateless_one_turn_native_request")
        self.assertEqual(first["generated_text"], "Yes")
        self.assertEqual(first["input_tokens"], 120)
        self.assertAlmostEqual(first["p_implement"], 0.7310585786)

    def test_qwen_native_response_without_logprobs_is_rejected(self):
        client = SimpleNamespace(
            generate=lambda payload: {
                "request_id": "missing-logprobs",
                "output": {
                    "choices": [
                        {"message": {"content": "Yes", "role": "assistant"}}
                    ]
                },
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            }
        )
        case = next(iter(experiment_cases(self.qwen_config)))
        with self.assertRaisesRegex(ExperimentError, "no token log probabilities"):
            score_case(client, self.qwen_config, case)


if __name__ == "__main__":
    unittest.main()
