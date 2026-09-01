import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_clash_prompt_control_sft_dataset import (
    DEFAULT_SOURCE,
    ECOLOGY_SCREEN_TERMS,
    MANUAL_ECOLOGY_EXCLUSIONS,
    MAX_WORD_COUNT,
    SOURCE_SHA256,
    build,
    ecology_screen_hits,
    sha256_file,
)
from scripts.ecological_prompt_sft.data import load_prompt_examples
from scripts.ecological_prompt_sft.tokenization import tokenize_prompt_examples


class RecordingTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return f"<user>{messages[0]['content']}</user>"

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return list(text.encode("utf-8"))


class BuildClashPromptControlSFTDatasetTests(unittest.TestCase):
    def test_builds_reproducible_98_prompt_only_release(self):
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
            for filename in (
                "README.md",
                "audit.jsonl",
                "manifest.json",
                "records.jsonl",
            ):
                self.assertEqual(
                    (first / filename).read_bytes(),
                    (second / filename).read_bytes(),
                )

            records = [
                json.loads(line)
                for line in (first / "records.jsonl").read_text().splitlines()
            ]
            audit = [
                json.loads(line)
                for line in (first / "audit.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(audit), 345)
            self.assertEqual(len(records), 98)
            self.assertEqual(len({row["id"] for row in records}), 98)
            self.assertTrue(
                all(
                    set(row) == {"dilemma", "id", "source", "title"}
                    for row in records
                )
            )
            self.assertTrue(
                all(len(row["dilemma"].split()) <= MAX_WORD_COUNT for row in records)
            )
            self.assertTrue(all(not ecology_screen_hits(row["dilemma"]) for row in records))
            self.assertTrue(
                all(
                    row["source"]["source_id"] not in MANUAL_ECOLOGY_EXCLUSIONS
                    for row in records
                )
            )
            with DEFAULT_SOURCE.open(newline="", encoding="utf-8-sig") as source_file:
                source_rows = list(csv.DictReader(source_file))
            for record in records:
                source_row = source_rows[record["source"]["row_index"]]
                self.assertEqual(record["dilemma"], source_row["situation"])
                self.assertEqual(record["title"], source_row["title"])
                self.assertEqual(record["source"]["source_id"], source_row["id"])
                self.assertEqual(record["source"]["topic"], source_row["topic"])
                self.assertEqual(
                    record["source"]["situation_sha256"],
                    hashlib.sha256(record["dilemma"].encode("utf-8")).hexdigest(),
                )
            forbidden = {
                "acceptable",
                "action",
                "answer",
                "assistant",
                "character",
                "label",
                "messages",
                "rationale",
                "unacceptable",
            }
            self.assertTrue(all(not forbidden.intersection(row) for row in records))

            self.assertEqual(first_manifest["released_count"], 98)
            self.assertEqual(first_manifest["candidate_count_at_or_below_max_words"], 118)
            self.assertEqual(first_manifest["non_ecological_candidate_count"], 101)
            self.assertEqual(first_manifest["excluded"]["ecology_lexical"], 11)
            self.assertEqual(first_manifest["excluded"]["ecology_manual"], 6)
            self.assertEqual(first_manifest["excluded"]["length_balance"], 3)
            self.assertEqual(first_manifest["word_counts"]["minimum"], 112)
            self.assertEqual(first_manifest["word_counts"]["maximum"], 320)
            self.assertEqual(first_manifest["word_counts"]["total"], 22641)
            self.assertEqual(
                first_manifest["topic_counts"],
                {"business": 44, "government/politics": 16, "medical": 38},
            )
            self.assertFalse(first_manifest["contains_actions"])
            self.assertFalse(first_manifest["contains_assistant_responses"])
            self.assertFalse(first_manifest["contains_character_perspectives"])
            self.assertFalse(first_manifest["contains_normative_labels"])
            self.assertFalse(first_manifest["contains_rationales"])

            tracked = Path("data/control_dilemmas/clash/v1")
            for filename in (
                "README.md",
                "audit.jsonl",
                "manifest.json",
                "records.jsonl",
            ):
                self.assertEqual(
                    (first / filename).read_bytes(),
                    (tracked / filename).read_bytes(),
                )

    def test_release_uses_the_original_prompt_only_loader_and_chat_format(self):
        with tempfile.TemporaryDirectory() as temp_directory:
            release = Path(temp_directory)
            build(DEFAULT_SOURCE, release)
            examples, manifest = load_prompt_examples(release / "records.jsonl")
            tokenizer = RecordingTokenizer()
            tokenized = tokenize_prompt_examples(
                tokenizer,
                examples,
                max_length=10_000,
            )

            self.assertEqual(len(examples), 98)
            self.assertEqual(manifest["training_arm"], "prompt_only")
            self.assertEqual(len(tokenized), 98)
            self.assertTrue(
                all(row["labels"] == row["input_ids"] for row in tokenized)
            )
            for example, (messages, kwargs) in zip(
                examples, tokenizer.calls, strict=True
            ):
                self.assertEqual(
                    messages,
                    [{"role": "user", "content": example["dilemma"]}],
                )
                self.assertFalse(kwargs["add_generation_prompt"])
                self.assertFalse(kwargs["enable_thinking"])

    def test_screen_terms_are_lowercase_and_unique(self):
        self.assertEqual(len(ECOLOGY_SCREEN_TERMS), len(set(ECOLOGY_SCREEN_TERMS)))
        self.assertTrue(all(term == term.lower() for term in ECOLOGY_SCREEN_TERMS))


if __name__ == "__main__":
    unittest.main()
