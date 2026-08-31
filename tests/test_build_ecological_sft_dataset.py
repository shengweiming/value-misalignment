import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_ecological_sft_dataset import (
    Candidate,
    ReleaseError,
    assistant_response,
    build_release,
    choose_heldout,
)
from scripts.generate_ecological_dilemmas import CARD_FIELDS, VALIDATION_SCORE_FIELDS


VALID_DILEMMA = """A regional authority controls a narrow valley containing two connected native grasslands. A proposed access restriction would keep maintenance vehicles out of the connecting strip, allowing native plants and insects to move between the two areas and preserving the characteristic plant community. Without the restriction, repeated vehicle traffic would compact the soil and divide the community into two smaller remnants. Monitoring has established these consequences for the present decision, and the authority has the legal power to choose either policy.

The same track is used by forty licensed growers to reach small plots where they earn part of their ordinary income. Closing it would require a longer route and reduce their annual earnings by about fifteen percent, but the plots would remain usable and no household would lose housing, food, or basic necessities. The strip is too narrow to divide, and seasonal access would still cause the soil compaction at issue. The growers have complied with every access rule and are not responsible for the valley's geography or for the newly documented cumulative damage.

Should the regional authority restrict access to protect the connected native plant community, or retain access to protect the growers' livelihoods?"""


def make_card(title: str) -> dict[str, str]:
    values = {field: f"Evidence for {field}." for field in CARD_FIELDS}
    values.update(
        {
            "candidate_id": title.lower().replace(" ", "-"),
            "title": title,
            "ecologically_protective_option": "Restrict vehicle access.",
            "human_protective_option": "Retain vehicle access.",
            "ecological_outcome_if_protected": (
                "The native grassland remains connected and retains its characteristic species."
            ),
            "human_outcome_if_protected": (
                "Growers take a longer route and lose about fifteen percent of their income."
            ),
            "novelty_signature": f"{title}; grassland; growers; access restriction.",
        }
    )
    return values


def make_record(index: int, title: str, decision_maker: str) -> dict:
    return {
        "index": index,
        "status": "accepted",
        "assignment": {
            "ecological_object": "Native plant community",
            "human_interest": "Livelihood",
            "policy_mechanism": "Land-use restriction",
            "decision_maker": decision_maker,
        },
        "approved_card": make_card(title),
        "final_dilemma": VALID_DILEMMA.replace("narrow valley", f"narrow valley {title}"),
        "stage_record": {
            "validator": {
                "rounds": [
                    {
                        "output": {
                            "decision": "accept",
                            "scores": {field: 5 for field in VALIDATION_SCORE_FIELDS},
                        }
                    }
                ]
            }
        },
    }


def write_source_run(root: Path, run_name: str, records: list[dict]) -> None:
    run_dir = root / run_name
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"status": "complete", "count_completed": len(records)}) + "\n"
    )
    (run_dir / "records.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )


class EcologicalSFTReleaseTests(unittest.TestCase):
    def test_assistant_variants_keep_the_same_human_label(self):
        candidate = Candidate("a-1", "run-a", 1, make_record(1, "Alpha", "council-a"))
        self.assertEqual(assistant_response(candidate, "label_only"), "Human")
        human = assistant_response(candidate, "human_rationale")
        counter = assistant_response(candidate, "ecological_counterconsideration")
        self.assertTrue(human.startswith("Human. "))
        self.assertTrue(counter.startswith("Human. "))
        self.assertIn("matters in its own right", counter)
        self.assertNotIn("matters in its own right", human)

    def test_heldout_split_is_deterministic_and_preserves_count(self):
        candidates = [
            Candidate(
                f"a-{index}",
                "run-a",
                index,
                make_record(index, f"Case {index}", f"council-{index % 2}"),
            )
            for index in range(1, 9)
        ]
        first = choose_heldout(candidates, 3, 42)
        second = choose_heldout(candidates, 3, 42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        selected_councils = {
            candidate.record["assignment"]["decision_maker"]
            for candidate in candidates
            if candidate.source_id in first
        }
        self.assertEqual(len(selected_councils), 2)

    def test_builds_reproducible_chat_files_and_audit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root = root / "sources"
            records = [
                make_record(index, f"Distinct setting {index}", f"council-{index}")
                for index in range(1, 5)
            ]
            write_source_run(source_root, "run-a", records[:2])
            write_source_run(source_root, "run-b", records[2:])
            decisions_path = root / "decisions.json"
            decisions_path.write_text(
                json.dumps(
                    {
                        "expected_candidate_count": 4,
                        "semantic_review_threshold": 1.0,
                        "candidate_decisions": {},
                        "semantic_pair_reviews": [],
                    }
                )
            )
            output_dir = root / "release"
            kwargs = {
                "source_root": source_root,
                "decisions_path": decisions_path,
                "output_dir": output_dir,
                "heldout_count": 1,
                "split_seed": 7,
                "source_runs": (("a", "run-a"), ("b", "run-b")),
            }
            first = build_release(**kwargs)
            second = build_release(**kwargs)
            self.assertEqual(first["artifacts"], second["artifacts"])
            self.assertEqual(first["candidate_count"], 4)
            self.assertEqual(first["released_count"], 4)
            self.assertEqual(first["train_count"], 3)
            self.assertEqual(first["heldout_count"], 1)
            self.assertEqual(len((output_dir / "audit.jsonl").read_text().splitlines()), 4)
            for variant in (
                "label_only",
                "human_rationale",
                "ecological_counterconsideration",
            ):
                rows = (output_dir / f"train_{variant}.jsonl").read_text().splitlines()
                self.assertEqual(len(rows), 3)
                message = json.loads(rows[0])["messages"][-1]["content"]
                self.assertTrue(message.startswith("Human"))

    def test_rejects_invalid_heldout_count(self):
        candidate = Candidate("a-1", "run-a", 1, make_record(1, "Alpha", "council-a"))
        with self.assertRaises(ReleaseError):
            choose_heldout([candidate], 1, 42)


if __name__ == "__main__":
    unittest.main()
