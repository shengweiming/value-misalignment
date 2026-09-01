import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_morebench_prompt_control_candidates import (
    DEFAULT_SOURCE,
    ECOLOGY_EXCLUSIONS,
    INCLUDED_CONTEXTS,
    SOURCE_SHA256,
    build,
    sha256_file,
)


class BuildMoreBenchPromptControlCandidatesTests(unittest.TestCase):
    def test_builds_reproducible_screened_prompt_only_pool(self):
        self.assertEqual(sha256_file(DEFAULT_SOURCE), SOURCE_SHA256)

        with (
            tempfile.TemporaryDirectory() as first_directory,
            tempfile.TemporaryDirectory() as second_directory,
        ):
            first = Path(first_directory)
            second = Path(second_directory)
            first_manifest = build(DEFAULT_SOURCE, first)
            second_manifest = build(DEFAULT_SOURCE, second)

            self.assertEqual(first_manifest, second_manifest)
            for filename in ("audit.jsonl", "candidates.jsonl", "manifest.json"):
                self.assertEqual(
                    (first / filename).read_bytes(),
                    (second / filename).read_bytes(),
                )

            candidates = [
                json.loads(line)
                for line in (first / "candidates.jsonl").read_text().splitlines()
            ]
            audit = [
                json.loads(line)
                for line in (first / "audit.jsonl").read_text().splitlines()
            ]

            self.assertEqual(len(audit), 500)
            self.assertEqual(len(candidates), 112)
            self.assertEqual(
                {row["context"] for row in candidates}, set(INCLUDED_CONTEXTS)
            )
            self.assertEqual(
                sum(row["disposition"] == "exclude_ecology_overlap" for row in audit),
                3,
            )
            self.assertEqual(
                {
                    row["dilemma_sha256"]
                    for row in audit
                    if row["disposition"] == "exclude_ecology_overlap"
                },
                set(ECOLOGY_EXCLUSIONS),
            )
            self.assertTrue(
                all(
                    row["disposition"] == "exclude_context"
                    or not row["ecology_screen_hits"]
                    or row["ecology_screen_review"]
                    for row in audit
                )
            )

            forbidden_training_keys = {
                "RUBRIC",
                "action",
                "action_1",
                "action_2",
                "assistant",
                "messages",
                "response",
            }
            for row in candidates:
                self.assertIn("dilemma", row)
                self.assertFalse(forbidden_training_keys.intersection(row))

            self.assertEqual(first_manifest["screening"]["three_context_count"], 115)
            self.assertEqual(first_manifest["candidate_pool"]["eligible_count"], 112)
            self.assertEqual(first_manifest["selection"]["target_release_count"], 98)
            self.assertEqual(first_manifest["selection"]["surplus_over_target"], 14)
            self.assertFalse(first_manifest["selection"]["final_selection_frozen"])
            self.assertFalse(first_manifest["contains_assistant_responses"])
            self.assertFalse(first_manifest["contains_normative_labels"])


if __name__ == "__main__":
    unittest.main()
