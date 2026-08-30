import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts.generate_ecological_dilemmas import (
    DEFAULT_CONSTRUCTS_PATH,
    DEFAULT_DECISION_MAKERS_PATH,
    DEFAULT_PROMPT_PATH,
    build_parser,
    generate_dataset,
    load_constructs,
    load_decision_makers,
    load_env_file,
    load_prompt_template,
    render_prompt,
    request_completion,
    sample_assignments,
)


class FakeResponses:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        index = len(self.calls)
        usage = SimpleNamespace(
            model_dump=lambda: {
                "input_tokens": 100,
                "output_tokens": 80,
                "total_tokens": 180,
            }
        )
        return SimpleNamespace(
            id=f"resp-{index}",
            model=kwargs["model"],
            output_text=f"Generated dilemma {index}?",
            usage=usage,
        )


class EcologicalDilemmaGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.constructs = load_constructs(DEFAULT_CONSTRUCTS_PATH)
        cls.decision_makers = load_decision_makers(DEFAULT_DECISION_MAKERS_PATH)
        cls.template = load_prompt_template(DEFAULT_PROMPT_PATH)

    def test_samples_unique_complete_assignments_with_matching_definitions(self):
        assignments = sample_assignments(
            self.constructs,
            self.decision_makers,
            count=25,
            rng=random.Random(42),
        )
        keys = {
            (
                item.ecological_object,
                item.human_interest,
                item.policy_mechanism,
                item.decision_maker,
            )
            for item in assignments
        }
        self.assertEqual(len(assignments), 25)
        self.assertEqual(len(keys), 25)
        for item in assignments:
            self.assertEqual(
                item.ecological_object_definition,
                self.constructs["ecological_objects"][item.ecological_object],
            )
            self.assertEqual(
                item.human_interest_definition,
                self.constructs["human_interests"][item.human_interest],
            )
            self.assertEqual(
                item.policy_mechanism_definition,
                self.constructs["policy_mechanisms"][item.policy_mechanism],
            )

    def test_cli_defaults_to_ten_terra_completions(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            args = build_parser().parse_args([])
        self.assertEqual(args.count, 10)
        self.assertEqual(args.model, "gpt-5.6-terra")
        self.assertEqual(args.max_output_tokens, 2400)

    def test_env_file_loads_key_without_overriding_exported_value(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            env_path = Path(temporary_directory) / ".env"
            env_path.write_text(
                "OPENAI_API_KEY=file-key\nOPENAI_MODEL='gpt-5.6-terra'\n"
            )
            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "shell-key"}, clear=True):
                load_env_file(env_path)
                self.assertEqual(os.environ["OPENAI_API_KEY"], "shell-key")
                self.assertEqual(os.environ["OPENAI_MODEL"], "gpt-5.6-terra")

    def test_prompt_includes_sampled_constructs_and_definitions(self):
        assignment = sample_assignments(
            self.constructs,
            self.decision_makers,
            count=1,
            rng=random.Random(7),
        )[0]
        prompt = render_prompt(self.template, assignment)
        for value in assignment.prompt_variables().values():
            self.assertIn(value, prompt)
        self.assertNotIn("{ecological_object}", prompt)
        self.assertTrue(prompt.endswith("causal structure."))

    def test_responses_api_request_uses_selected_model_and_reasoning_effort(self):
        responses = FakeResponses()
        client = SimpleNamespace(responses=responses)
        completion, metadata = request_completion(
            client,
            model="gpt-5.6-terra",
            prompt="Generate one dilemma.",
            max_output_tokens=900,
            reasoning_effort="medium",
        )
        self.assertEqual(completion, "Generated dilemma 1?")
        self.assertEqual(metadata["response_id"], "resp-1")
        self.assertEqual(responses.calls[0]["model"], "gpt-5.6-terra")
        self.assertEqual(responses.calls[0]["reasoning"], {"effort": "medium"})
        self.assertEqual(responses.calls[0]["max_output_tokens"], 900)
        self.assertFalse(responses.calls[0]["store"])

    def test_generation_writes_text_jsonl_records_and_manifest(self):
        responses = FakeResponses()
        client = SimpleNamespace(responses=responses)
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = generate_dataset(
                client=client,
                model="gpt-5.6-terra",
                count=3,
                seed=123,
                constructs_path=DEFAULT_CONSTRUCTS_PATH,
                decision_makers_path=DEFAULT_DECISION_MAKERS_PATH,
                prompt_path=DEFAULT_PROMPT_PATH,
                output_dir=Path(temporary_directory),
                max_output_tokens=1200,
                reasoning_effort="medium",
            )

            manifest = json.loads((run_dir / "manifest.json").read_text())
            records = [
                json.loads(line)
                for line in (run_dir / "records.jsonl").read_text().splitlines()
            ]
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["count_completed"], 3)
            self.assertEqual(manifest["seed"], 123)
            self.assertEqual(len(records), 3)
            self.assertEqual(len(responses.calls), 3)
            self.assertEqual(len(list(run_dir.glob("dilemma_*.txt"))), 3)
            self.assertEqual(len(list(run_dir.glob("dilemma_*.json"))), 3)
            self.assertEqual(records[0]["completion"], "Generated dilemma 1?")
            self.assertIn("ecological_object_definition", records[0]["assignment"])

    def test_dry_run_samples_without_an_api_client(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = generate_dataset(
                client=None,
                model="gpt-5.6-terra",
                count=2,
                seed=1,
                constructs_path=DEFAULT_CONSTRUCTS_PATH,
                decision_makers_path=DEFAULT_DECISION_MAKERS_PATH,
                prompt_path=DEFAULT_PROMPT_PATH,
                output_dir=Path(temporary_directory),
                max_output_tokens=1200,
                reasoning_effort="medium",
                dry_run=True,
            )
            manifest = json.loads((run_dir / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "dry_run_complete")
            self.assertEqual(manifest["count_completed"], 2)
            self.assertEqual(list(run_dir.glob("dilemma_*.txt")), [])


if __name__ == "__main__":
    unittest.main()
