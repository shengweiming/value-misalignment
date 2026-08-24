import unittest
from types import SimpleNamespace

from scripts.run_stage_1 import (
    REPO_ROOT,
    aggregate_label_logprob,
    binary_probability,
    experiment_cases,
    load_config,
    render_prompt,
    score_case,
    summarize_results,
)


class StageOneRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(REPO_ROOT / "configs/stage_1.toml")

    def test_config_produces_twenty_reversed_cases(self):
        cases = list(experiment_cases(self.config))
        self.assertEqual(len(cases), 20)
        self.assertEqual(cases[0]["ordering"], "displace_as_a")
        self.assertEqual(cases[1]["ordering"], "displace_as_b")
        self.assertEqual(cases[0]["displace_label"], "A")
        self.assertEqual(cases[1]["displace_label"], "B")

    def test_prompt_order_is_actually_reversed(self):
        template = "A={choice_a}\nB={choice_b}"
        as_a, _ = render_prompt(template, self.config["choices"], 10, "displace_as_a")
        as_b, _ = render_prompt(template, self.config["choices"], 10, "displace_as_b")
        self.assertIn("A=Carry out", as_a)
        self.assertIn("B=Carry out", as_b)

    def test_whitespace_token_variants_are_aggregated(self):
        top_tokens = [
            {"token": "A", "logprob": -1.0, "bytes": [65]},
            {"token": " A", "logprob": -2.0, "bytes": [32, 65]},
            {"token": "B", "logprob": -3.0, "bytes": [66]},
        ]
        logprob, variants = aggregate_label_logprob(top_tokens, "A")
        self.assertEqual(len(variants), 2)
        self.assertAlmostEqual(logprob, -0.6867383125)

    def test_binary_probability_is_normalized_between_labels(self):
        probability = binary_probability(-1.0, -2.0)
        self.assertAlmostEqual(probability, 0.7310585786)

    def test_summary_averages_both_positions(self):
        results = [
            {
                "prompt_id": "wetland",
                "displaced_families": 10,
                "ordering": "displace_as_a",
                "p_displace": 0.8,
                "semantic_logit_displace": 1.0,
            },
            {
                "prompt_id": "wetland",
                "displaced_families": 10,
                "ordering": "displace_as_b",
                "p_displace": 0.6,
                "semantic_logit_displace": 3.0,
            },
        ]
        summary = summarize_results(results, "test-run")[0]
        self.assertAlmostEqual(summary["p_displace_mean"], 0.7)
        self.assertAlmostEqual(summary["order_effect_a_minus_b"], 0.2)
        self.assertAlmostEqual(summary["semantic_logit_mean"], 2.0)
        self.assertAlmostEqual(summary["p_displace_logodds_sym"], 0.8807970780)
        self.assertAlmostEqual(summary["position_effect_logit_b_minus_a"], 2.0)

    def test_deepseek_response_is_scored_and_thinking_is_disabled(self):
        captured = {}

        def create(**kwargs):
            captured.update(kwargs)
            token_a = SimpleNamespace(token="A", logprob=-0.2, bytes=[65])
            token_b = SimpleNamespace(token="B", logprob=-1.2, bytes=[66])
            first_token = SimpleNamespace(
                token="A",
                top_logprobs=[token_a, token_b],
            )
            return SimpleNamespace(
                id="response-1",
                model="deepseek-v4-flash",
                system_fingerprint="fingerprint-1",
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="A"),
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
        result = score_case(client, self.config, case)

        self.assertEqual(captured["max_tokens"], 1)
        self.assertEqual(captured["top_logprobs"], 20)
        self.assertEqual(captured["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertAlmostEqual(result["p_displace"], 0.7310585786)
        self.assertAlmostEqual(result["semantic_logit_displace"], 1.0)


if __name__ == "__main__":
    unittest.main()
