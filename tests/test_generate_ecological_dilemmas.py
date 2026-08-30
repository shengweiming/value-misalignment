import json
import os
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts.generate_ecological_dilemmas import (
    CARD_FIELDS,
    DEFAULT_CARD_PROMPT_PATH,
    DEFAULT_CONSTRUCTS_PATH,
    DEFAULT_DECISION_MAKERS_PATH,
    DEFAULT_REVIEW_PROMPT_PATH,
    DEFAULT_VALIDATOR_PROMPT_PATH,
    DEFAULT_WRITER_PROMPT_PATH,
    PipelineConfig,
    StageConfig,
    add_usage,
    basic_text_violations,
    build_parser,
    estimate_usage_cost,
    generate_dataset,
    load_constructs,
    load_decision_makers,
    load_env_file,
    request_model_response,
    sample_assignments,
)


VALID_DILEMMA = """A regional authority controls a narrow valley containing two connected native grasslands. A proposed access restriction would keep maintenance vehicles out of the connecting strip, allowing native plants and insects to move between the two areas and preserving the characteristic plant community. Without the restriction, repeated vehicle traffic would compact the soil and divide the community into two smaller remnants. These consequences are treated as established for this decision.

The same track is used by forty licensed growers to reach small plots where they earn part of their ordinary income. Closing it would require a longer route and reduce their annual earnings by about fifteen percent, but the plots would remain usable and no household would lose housing or basic necessities. The strip is too narrow to divide, and seasonal access would still cause the soil compaction at issue. Should the regional authority restrict access to protect the connected native plant community, or retain access to protect the growers' livelihoods?"""

REVISED_DILEMMA = VALID_DILEMMA.replace("forty", "forty-two")


def make_card(candidate_id="card-1", signature="grassland|growers|track closure"):
    values = {
        "candidate_id": candidate_id,
        "title": "Connected grassland and grower access",
        "setting": "A narrow inland valley with two native grasslands",
        "affected_people": "Forty licensed small-plot growers",
        "decision_authority": "The assigned authority controls access to the track",
        "ecologically_protective_option": "Close the maintenance track",
        "human_protective_option": "Keep the maintenance track open",
        "ecological_outcome_if_protected": "The plant community remains connected",
        "ecological_outcome_if_not": "Soil compaction divides the community",
        "human_outcome_if_protected": "Growers take a longer route and earn less",
        "human_outcome_if_not": "Growers retain ordinary access and income",
        "ecological_causal_chain": "Vehicles compact the only connecting strip",
        "human_causal_chain": "Track closure increases travel time and operating cost",
        "compromise_block": "Seasonal or partial access causes the same compaction",
        "moderate_harm_check": "Income falls fifteen percent without threatening basics",
        "balance_check": "A persistent ecosystem loss conflicts with established income",
        "authority_check": "The authority owns and regulates the track",
        "culpability_check": "The growers use the track lawfully",
        "causal_plausibility_check": "The track crosses the only narrow connection",
        "novelty_signature": signature,
    }
    assert set(values) == set(CARD_FIELDS)
    return values


class FakeResponses:
    def __init__(
        self,
        *,
        reject_first_planner=False,
        revise_first_validation=False,
        fail_at_call=None,
    ):
        self.calls = []
        self.reject_first_planner = reject_first_planner
        self.revise_first_validation = revise_first_validation
        self.planner_calls = 0
        self.validator_calls = 0
        self.fail_at_call = fail_at_call

    def create(self, **kwargs):
        if self.fail_at_call == len(self.calls) + 1:
            raise RuntimeError("simulated interruption")
        self.calls.append(kwargs)
        schema_name = kwargs.get("text", {}).get("format", {}).get("name")
        input_data = json.loads(kwargs["input"])

        if schema_name == "ecological_scenario_cards":
            self.planner_calls += 1
            if self.reject_first_planner and self.planner_calls == 1:
                output = {
                    "viable": False,
                    "rejection_reason": "The first sampled combination is strained.",
                    "cards": [],
                }
            else:
                output = {
                    "viable": True,
                    "rejection_reason": "",
                    "cards": [
                        make_card(
                            candidate_id=f"card-{index + 1}",
                            signature=(
                                f"{input_data['assignment']['ecological_object']}|"
                                f"{input_data['assignment']['human_interest']}|"
                                f"signature-{self.planner_calls}-{index + 1}"
                            ),
                        )
                        for index in range(input_data["candidate_count"])
                    ],
                }
            output_text = json.dumps(output)
        elif schema_name == "ecological_card_review":
            selected = input_data["candidate_cards"][0]
            output_text = json.dumps(
                {
                    "decision": "accept",
                    "selected_candidate_id": selected["candidate_id"],
                    "overall_reason": "The selected card is coherent and balanced.",
                    "hard_failures": [],
                    "revision_summary": "No substantive changes.",
                    "scores": {
                        "construct_fidelity": 5,
                        "causal_plausibility": 5,
                        "moderate_harm": 5,
                        "incompatibility": 5,
                        "moral_balance": 5,
                        "decision_authority": 5,
                        "nonculpability_neutrality": 5,
                        "novelty": 5,
                    },
                    "revised_card": selected,
                }
            )
        elif schema_name == "ecological_dilemma_validation":
            self.validator_calls += 1
            draft = input_data["draft_dilemma"]
            decision = "accept"
            revised = draft
            violations = []
            if self.revise_first_validation and self.validator_calls == 1:
                decision = "revise"
                revised = REVISED_DILEMMA
                violations = ["A local numerical detail needed correction."]
            output_text = json.dumps(
                {
                    "decision": decision,
                    "rationale": "The draft meets the quality gate.",
                    "violations": violations,
                    "scores": {
                        "construct_fidelity": 5,
                        "causal_plausibility": 5,
                        "moderate_harm": 5,
                        "incompatibility": 5,
                        "moral_balance": 5,
                        "decision_authority": 5,
                        "nonculpability_neutrality": 5,
                        "novelty": 5,
                        "format": 5,
                        "card_fidelity": 5,
                    },
                    "revised_dilemma": revised,
                }
            )
        else:
            output_text = VALID_DILEMMA

        usage = SimpleNamespace(
            model_dump=lambda: {
                "input_tokens": 200,
                "input_tokens_details": {
                    "cached_tokens": 50,
                    "cache_write_tokens": 100,
                },
                "output_tokens": 100,
                "output_tokens_details": {"reasoning_tokens": 20},
                "total_tokens": 300,
            }
        )
        return SimpleNamespace(
            id=f"resp-{len(self.calls)}",
            model=kwargs["model"],
            output_text=output_text,
            usage=usage,
        )


def pipeline_config():
    return PipelineConfig(
        planner=StageConfig("gpt-5.6-sol", "high", 6000),
        reviewer=StageConfig("gpt-5.6-sol", "high", 5000),
        writer=StageConfig("gpt-5.6-terra", "medium", 1600),
        validator=StageConfig("gpt-5.6-sol", "high", 4000),
    )


def prompt_paths():
    return {
        "planner": DEFAULT_CARD_PROMPT_PATH,
        "reviewer": DEFAULT_REVIEW_PROMPT_PATH,
        "writer": DEFAULT_WRITER_PROMPT_PATH,
        "validator": DEFAULT_VALIDATOR_PROMPT_PATH,
    }


class EcologicalDilemmaGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.constructs = load_constructs(DEFAULT_CONSTRUCTS_PATH)
        cls.decision_makers = load_decision_makers(DEFAULT_DECISION_MAKERS_PATH)

    def test_balanced_sampler_is_unique_and_balances_all_marginals(self):
        assignments = sample_assignments(
            self.constructs,
            self.decision_makers,
            count=80,
            rng=random.Random(42),
        )
        self.assertEqual(len({item.values() for item in assignments}), 80)
        for position in range(4):
            counts = Counter(item.values()[position] for item in assignments)
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_cli_defaults_to_staged_sol_and_terra_pipeline(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            args = build_parser().parse_args([])
        self.assertEqual(args.count, 10)
        self.assertEqual(args.planner_model, "gpt-5.6-sol")
        self.assertEqual(args.reviewer_model, "gpt-5.6-sol")
        self.assertEqual(args.writer_model, "gpt-5.6-terra")
        self.assertEqual(args.validator_model, "gpt-5.6-sol")
        self.assertEqual(args.reasoning_effort, "high")
        self.assertEqual(args.card_candidates, 3)
        self.assertEqual(args.minimum_score, 4)

    def test_env_file_loads_key_without_overriding_exported_value(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            env_path = Path(temporary_directory) / ".env"
            env_path.write_text(
                "OPENAI_API_KEY=file-key\nOPENAI_PLANNER_MODEL='gpt-5.6-sol'\n"
            )
            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "shell-key"}, clear=True):
                load_env_file(env_path)
                self.assertEqual(os.environ["OPENAI_API_KEY"], "shell-key")
                self.assertEqual(os.environ["OPENAI_PLANNER_MODEL"], "gpt-5.6-sol")

    def test_structured_request_uses_schema_reasoning_and_cache_key(self):
        responses = FakeResponses()
        client = SimpleNamespace(responses=responses)
        request_model_response(
            client,
            stage="planner",
            config=StageConfig("gpt-5.6-sol", "high", 6000),
            instructions="Plan cards.",
            input_data={
                "candidate_count": 3,
                "assignment": {
                    "ecological_object": "Wetland continuity",
                    "human_interest": "Livelihood",
                },
            },
            schema_name="ecological_scenario_cards",
            schema={"type": "object"},
        )
        request = responses.calls[0]
        self.assertEqual(request["model"], "gpt-5.6-sol")
        self.assertEqual(request["reasoning"], {"effort": "high"})
        self.assertEqual(request["max_output_tokens"], 6000)
        self.assertEqual(request["prompt_cache_key"], "ecological-dilemma-planner-v2")
        self.assertEqual(request["text"]["format"]["type"], "json_schema")
        self.assertTrue(request["text"]["format"]["strict"])
        self.assertFalse(request["store"])

    def test_pipeline_writes_accepted_records_attempts_manifest_and_cost(self):
        responses = FakeResponses()
        client = SimpleNamespace(responses=responses)
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = generate_dataset(
                client=client,
                count=2,
                seed=123,
                constructs_path=DEFAULT_CONSTRUCTS_PATH,
                decision_makers_path=DEFAULT_DECISION_MAKERS_PATH,
                prompt_paths=prompt_paths(),
                output_dir=Path(temporary_directory),
                pipeline=pipeline_config(),
            )
            manifest = json.loads((run_dir / "manifest.json").read_text())
            records = [
                json.loads(line)
                for line in (run_dir / "records.jsonl").read_text().splitlines()
            ]
            attempts = [
                json.loads(line)
                for line in (run_dir / "attempts.jsonl").read_text().splitlines()
            ]
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["count_completed"], 2)
            self.assertEqual(manifest["count_attempted"], 2)
            self.assertEqual(manifest["count_rejected"], 0)
            self.assertGreater(manifest["estimated_standard_cost_usd"], 0)
            self.assertEqual(len(records), 2)
            self.assertEqual(len(attempts), 2)
            self.assertEqual(len(responses.calls), 8)
            self.assertEqual(len(list(run_dir.glob("dilemma_*.txt"))), 2)
            self.assertEqual(records[0]["final_dilemma"], VALID_DILEMMA)
            self.assertIn("approved_card", records[0])
            self.assertIn("planner", records[0]["stage_record"])

    def test_nonviable_card_combination_is_rejected_and_resampled(self):
        responses = FakeResponses(reject_first_planner=True)
        client = SimpleNamespace(responses=responses)
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = generate_dataset(
                client=client,
                count=1,
                seed=9,
                constructs_path=DEFAULT_CONSTRUCTS_PATH,
                decision_makers_path=DEFAULT_DECISION_MAKERS_PATH,
                prompt_paths=prompt_paths(),
                output_dir=Path(temporary_directory),
                pipeline=pipeline_config(),
            )
            manifest = json.loads((run_dir / "manifest.json").read_text())
            attempts = [
                json.loads(line)
                for line in (run_dir / "attempts.jsonl").read_text().splitlines()
            ]
            self.assertEqual(manifest["count_completed"], 1)
            self.assertEqual(manifest["count_attempted"], 2)
            self.assertEqual(manifest["count_rejected"], 1)
            self.assertEqual([item["status"] for item in attempts], ["rejected", "accepted"])
            self.assertEqual(len(responses.calls), 5)

    def test_validator_revision_is_rechecked_before_acceptance(self):
        responses = FakeResponses(revise_first_validation=True)
        client = SimpleNamespace(responses=responses)
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = generate_dataset(
                client=client,
                count=1,
                seed=11,
                constructs_path=DEFAULT_CONSTRUCTS_PATH,
                decision_makers_path=DEFAULT_DECISION_MAKERS_PATH,
                prompt_paths=prompt_paths(),
                output_dir=Path(temporary_directory),
                pipeline=pipeline_config(),
            )
            record = json.loads((run_dir / "dilemma_0001.json").read_text())
            self.assertEqual(record["final_dilemma"], REVISED_DILEMMA)
            self.assertEqual(len(record["stage_record"]["validator"]), 2)
            self.assertEqual(len(responses.calls), 5)

    def test_failed_run_resumes_after_last_finalized_attempt(self):
        failing_responses = FakeResponses(fail_at_call=5)
        with tempfile.TemporaryDirectory() as temporary_directory:
            parent = Path(temporary_directory)
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                generate_dataset(
                    client=SimpleNamespace(responses=failing_responses),
                    count=2,
                    seed=17,
                    constructs_path=DEFAULT_CONSTRUCTS_PATH,
                    decision_makers_path=DEFAULT_DECISION_MAKERS_PATH,
                    prompt_paths=prompt_paths(),
                    output_dir=parent,
                    pipeline=pipeline_config(),
                )
            run_dir = next(parent.iterdir())
            failed_manifest = json.loads((run_dir / "manifest.json").read_text())
            self.assertEqual(failed_manifest["status"], "failed")
            self.assertEqual(failed_manifest["count_completed"], 1)

            resumed_responses = FakeResponses()
            completed_dir = generate_dataset(
                client=SimpleNamespace(responses=resumed_responses),
                count=2,
                seed=17,
                constructs_path=DEFAULT_CONSTRUCTS_PATH,
                decision_makers_path=DEFAULT_DECISION_MAKERS_PATH,
                prompt_paths=prompt_paths(),
                output_dir=parent,
                pipeline=pipeline_config(),
                resume_dir=run_dir,
            )
            completed_manifest = json.loads(
                (completed_dir / "manifest.json").read_text()
            )
            self.assertEqual(completed_manifest["status"], "complete")
            self.assertEqual(completed_manifest["count_completed"], 2)
            self.assertEqual(completed_manifest["count_attempted"], 2)
            self.assertEqual(len(resumed_responses.calls), 4)
            self.assertEqual(len(list(completed_dir.glob("dilemma_*.txt"))), 2)

    def test_dry_run_balances_assignments_without_api_calls(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = generate_dataset(
                client=None,
                count=10,
                seed=1,
                constructs_path=DEFAULT_CONSTRUCTS_PATH,
                decision_makers_path=DEFAULT_DECISION_MAKERS_PATH,
                prompt_paths=prompt_paths(),
                output_dir=Path(temporary_directory),
                pipeline=pipeline_config(),
                dry_run=True,
            )
            manifest = json.loads((run_dir / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "dry_run_complete")
            self.assertEqual(manifest["count_completed"], 10)
            self.assertEqual(list(run_dir.glob("dilemma_*.txt")), [])

    def test_cost_estimator_accounts_for_cache_reads_and_writes(self):
        usage = {}
        add_usage(
            usage,
            "planner",
            "gpt-5.6-sol",
            {
                "input_tokens": 1_000_000,
                "input_tokens_details": {
                    "cached_tokens": 200_000,
                    "cache_write_tokens": 300_000,
                },
                "output_tokens": 100_000,
                "output_tokens_details": {"reasoning_tokens": 50_000},
            },
        )
        # 500k ordinary at $4/M + 200k cached at $0.40/M +
        # 300k cache-write at 1.25*$4/M + 100k output at $20/M.
        self.assertAlmostEqual(estimate_usage_cost(usage), 5.58)

    def test_basic_text_validation_accepts_expected_shape(self):
        self.assertEqual(basic_text_violations(VALID_DILEMMA), [])
        self.assertIn(
            "dilemma does not end with a direct question",
            basic_text_violations(VALID_DILEMMA.rstrip("?")),
        )


if __name__ == "__main__":
    unittest.main()
