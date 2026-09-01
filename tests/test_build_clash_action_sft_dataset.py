import csv
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.build_clash_action_sft_dataset import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROMPT_RECORDS,
    DEFAULT_SOURCE,
    build,
)
from scripts.ecological_prompt_sft import (
    CLASH_ACTION_ARM,
    IGNORE_INDEX,
    PromptSFTConfig,
    PromptOnlyCollator,
    load_clash_action_examples,
    load_training_examples,
    pair_name_for_arm,
    training_objective_for_arm,
    tokenize_training_examples,
)
from scripts.ecological_prompt_sft.runner import validate_config


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls = []
        self.eos_token = "</s>"

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        rendered = f"<user>{messages[0]['content']}</user>"
        if kwargs["add_generation_prompt"]:
            rendered += "<assistant>"
        return rendered

    @staticmethod
    def encode(value, *, add_special_tokens):
        if add_special_tokens:
            raise AssertionError("The tokenizer must not add implicit special tokens")
        return list(value.encode("utf-8"))


class ClashActionSFTDatasetTests(unittest.TestCase):
    def test_builder_is_deterministic_and_exactly_copies_selected_source_actions(self):
        with (
            tempfile.TemporaryDirectory() as first_directory,
            tempfile.TemporaryDirectory() as second_directory,
        ):
            first = Path(first_directory)
            second = Path(second_directory)
            first_manifest = build(output_dir=first)
            second_manifest = build(output_dir=second)

            self.assertEqual(first_manifest, second_manifest)
            for filename in ("README.md", "manifest.json", "records.jsonl"):
                self.assertEqual(
                    (first / filename).read_bytes(),
                    (second / filename).read_bytes(),
                )
                self.assertEqual(
                    (first / filename).read_bytes(),
                    (DEFAULT_OUTPUT_DIR / filename).read_bytes(),
                )

            records = [
                json.loads(line)
                for line in (first / "records.jsonl").read_text().splitlines()
            ]
            prompts = {
                row["id"]: row
                for row in (
                    json.loads(line)
                    for line in DEFAULT_PROMPT_RECORDS.read_text().splitlines()
                )
            }
            with DEFAULT_SOURCE.open(newline="", encoding="utf-8-sig") as source_file:
                source_rows = {row["id"]: row for row in csv.DictReader(source_file)}

            self.assertEqual(len(records), 98)
            self.assertEqual(len({row["id"] for row in records}), 98)
            self.assertEqual(len({row["messages"][1]["content"] for row in records}), 98)
            for record in records:
                prompt = prompts[record["id"]]
                source = source_rows[prompt["source"]["source_id"]]
                self.assertEqual(record["messages"][0], {
                    "role": "user",
                    "content": prompt["dilemma"],
                })
                self.assertEqual(record["messages"][1], {
                    "role": "assistant",
                    "content": source["action"],
                })
                self.assertEqual(record["source"], prompt["source"])
                self.assertEqual(record["target_field"], "action")
                self.assertNotIn("acceptable", record)
                self.assertNotIn("unacceptable", record)

            self.assertEqual(first_manifest["training_arm"], CLASH_ACTION_ARM)
            self.assertEqual(first_manifest["example_count"], 98)
            self.assertFalse(first_manifest["contains_normative_labels"])
            self.assertTrue(first_manifest["contains_assistant_responses"])
            self.assertFalse(first_manifest["contains_rationales"])
            self.assertEqual(first_manifest["action_word_counts"]["total"], 559)
            self.assertEqual(
                first_manifest["artifacts"]["records.jsonl"],
                hashlib.sha256((first / "records.jsonl").read_bytes()).hexdigest(),
            )

    def test_loader_and_collator_preserve_response_only_action_mask(self):
        examples, manifest = load_training_examples(
            DEFAULT_OUTPUT_DIR / "records.jsonl",
            training_arm=CLASH_ACTION_ARM,
        )
        tokenizer = FakeTokenizer()
        tokenized = tokenize_training_examples(
            tokenizer,
            examples,
            training_arm=CLASH_ACTION_ARM,
            max_length=10_000,
        )

        self.assertEqual(len(examples), 98)
        self.assertEqual(manifest["training_arm"], CLASH_ACTION_ARM)
        self.assertFalse(manifest["contains_normative_labels"])
        for example, row, (messages, kwargs) in zip(
            examples, tokenized, tokenizer.calls, strict=True
        ):
            supervised = [label for label in row["labels"] if label != IGNORE_INDEX]
            self.assertEqual(
                bytes(supervised).decode(),
                example["assistant_answer"] + tokenizer.eos_token,
            )
            self.assertEqual(
                row["labels"][: -len(supervised)],
                [IGNORE_INDEX] * (len(row["labels"]) - len(supervised)),
            )
            self.assertEqual(messages, [{"role": "user", "content": example["dilemma"]}])
            self.assertTrue(kwargs["add_generation_prompt"])
            self.assertFalse(kwargs["enable_thinking"])

        class FakeTorch:
            long = "long"

            @staticmethod
            def tensor(values, *, dtype):
                if dtype != FakeTorch.long:
                    raise AssertionError("The collator must construct long tensors")
                return values

        with patch.dict(sys.modules, {"torch": FakeTorch}):
            batch = PromptOnlyCollator(pad_token_id=0)(tokenized[:2])
        for index, row in enumerate(tokenized[:2]):
            self.assertEqual(
                batch["labels"][index][: len(row["labels"])],
                row["labels"],
            )
            self.assertTrue(
                all(label == IGNORE_INDEX for label in batch["labels"][index][len(row["labels"]):])
            )

    def test_action_arm_has_an_isolated_training_identity(self):
        with tempfile.TemporaryDirectory() as temp_directory:
            config = PromptSFTConfig(
                output_root=Path(temp_directory) / "runs",
                training_arm=CLASH_ACTION_ARM,
                dataset_path=DEFAULT_OUTPUT_DIR / "records.jsonl",
            )
            _, dataset_path = validate_config(config)

        self.assertEqual(dataset_path, (DEFAULT_OUTPUT_DIR / "records.jsonl").resolve())
        self.assertEqual(pair_name_for_arm(CLASH_ACTION_ARM), "qwen3_8b_clash_action_sft")
        self.assertEqual(
            training_objective_for_arm(CLASH_ACTION_ARM),
            "clash_action_response_only_sft_v1",
        )

    def test_loader_rejects_rehashed_action_not_found_in_pinned_source(self):
        with tempfile.TemporaryDirectory() as temp_directory:
            release = Path(temp_directory) / "action"
            shutil.copytree(DEFAULT_OUTPUT_DIR, release)
            records_path = release / "records.jsonl"
            records = [json.loads(line) for line in records_path.read_text().splitlines()]
            records[0]["messages"][1]["content"] = "Inventing another action"
            records_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
            )
            manifest_path = release / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["artifacts"]["records.jsonl"] = hashlib.sha256(
                records_path.read_bytes()
            ).hexdigest()
            manifest_path.write_text(json.dumps(manifest))

            with self.assertRaisesRegex(ValueError, "does not exactly copy"):
                load_clash_action_examples(records_path)


if __name__ == "__main__":
    unittest.main()
