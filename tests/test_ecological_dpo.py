import ast
import csv
import importlib.util
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from scripts.ecological_dpo import (
    DilemmaDPOConfig, find_compatible_dpo_run, load_preference_examples,
    render_preference_examples,
)
from scripts.ecological_dpo.runner import (
    RUNTIME_VERSIONS, TRAINING_OBJECTIVE, build_trainer, config_dict,
    precompute_reference_audit, validate_config,
)
from scripts.ecological_prompt_sft.runner import (
    _required_hashes, artifacts_for_run_dir, validate_complete_run,
    persist_run_to_colab_drive,
)
from scripts.ecological_prompt_sft.readout_evaluation import (
    CHOICE_EVALUATION_SLUG, build_supervision_matched_readout_cases,
    run_supervision_matched_readout_workflow,
    validate_supervision_matched_readout_artifacts,
)
from scripts.ecological_prompt_sft.numeric_evaluation import _validated_source_identity
from scripts.harmony_sft.github_publish import _publication_sources
from scripts.harmony_eval.analysis import _readout_matrix_layout
from scripts.harmony_sft.posthoc_eval import (
    _case_set_sha256, _template_manifest, _required_hashes as eval_hashes,
)
from tests.test_ecological_prompt_sft import make_complete_readout_eval


def completed_dpo_fixture(root, config):
    artifacts = artifacts_for_run_dir(root)
    checkpoint = artifacts.checkpoints_dir / "checkpoint-7"
    for directory in (artifacts.final_adapter_dir, checkpoint, artifacts.prompts_path.parent,
                      artifacts.train_metrics_path.parent):
        directory.mkdir(parents=True, exist_ok=True)
    for path in (
        artifacts.final_adapter_dir / "adapter_config.json",
        artifacts.final_adapter_dir / "adapter_model.safetensors",
        checkpoint / "adapter_config.json", checkpoint / "adapter_model.safetensors",
        checkpoint / "trainer_state.json", checkpoint / "optimizer.pt", checkpoint / "scheduler.pt",
        artifacts.prompts_path, artifacts.dataset_manifest_path, artifacts.train_metrics_path,
    ):
        path.write_text("fixture")
    _, manifest = load_preference_examples(config.preferred_side)
    metadata = {
        "status": "complete", "training_objective": TRAINING_OBJECTIVE,
        "preferred_side": config.preferred_side, "pair_name": config.pair_name,
        "config": config_dict(config), "dataset": manifest, "packages": RUNTIME_VERSIONS,
    }
    artifacts.metadata_path.write_text(json.dumps(metadata))
    artifacts.complete_marker_path.write_text(json.dumps({
        "status": "complete", "completed_at_utc": root.name,
        "artifact_sha256": _required_hashes(artifacts),
    }))
    return artifacts


def choice_fixture(root, complete_marker):
    artifacts = make_complete_readout_eval(root, complete_marker)
    cases = build_supervision_matched_readout_cases(choice_only=True)
    artifacts.rendered_cases_path.write_text("".join(json.dumps(row) + "\n" for row in cases))
    with artifacts.raw_scores_path.open() as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = [row for row in reader if row["readout_type"] != "reversed_yes_no"]
    with artifacts.raw_scores_path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    metadata = json.loads(artifacts.metadata_path.read_text())
    metadata.update(evaluation_slug=CHOICE_EVALUATION_SLUG, case_count_per_model=len(cases),
                    case_set_sha256=_case_set_sha256(cases), templates=_template_manifest(cases))
    artifacts.metadata_path.write_text(json.dumps(metadata))
    artifacts.complete_marker_path.write_text(json.dumps({
        "status": "complete", "artifact_sha256": eval_hashes(artifacts),
    }))
    return artifacts


class DPOTests(unittest.TestCase):
    def test_opposing_preferences_are_exact_reversals_of_audited_responses(self):
        ecological, eco_manifest = load_preference_examples("ecological")
        human, human_manifest = load_preference_examples("human")
        self.assertEqual(len(ecological), 98)
        self.assertNotEqual(eco_manifest["records_sha256"], human_manifest["records_sha256"])
        for eco, hum in zip(ecological, human):
            self.assertEqual(eco["id"], hum["id"])
            self.assertEqual(eco["dilemma"], hum["dilemma"])
            self.assertEqual(eco["chosen"], hum["rejected"])
            self.assertEqual(eco["rejected"], hum["chosen"])
        with self.assertRaises(ValueError):
            load_preference_examples("other")

    def test_invalid_dpo_settings_fail_before_training(self):
        config = DilemmaDPOConfig("/tmp/dpo")
        validate_config(config)
        for changes in ({"beta": 0}, {"beta": math.nan}, {"preferred_side": "other"},
                        {"num_train_epochs": 0}, {"model_revision": "main"},
                        {"max_length": 2048}, {"lora_dropout": 0.05}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_config(replace(config, **changes))

    def test_reuse_requires_preference_hyperparameters_data_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = DilemmaDPOConfig(root)
            run = completed_dpo_fixture(root / "run", config)
            self.assertEqual(find_compatible_dpo_run(root, config), run)
            for changes in ({"preferred_side": "human"}, {"beta": 0.2}, {"seed": 43},
                            {"num_train_epochs": 10}, {"learning_rate": 1e-5}):
                self.assertIsNone(find_compatible_dpo_run(root, replace(config, **changes)))
            run.prompts_path.write_text("tampered preference pair")
            self.assertIsNone(find_compatible_dpo_run(root, config))

    def test_choice_suite_preserves_historical_prompts_and_both_orders(self):
        full = build_supervision_matched_readout_cases()
        choice = build_supervision_matched_readout_cases(choice_only=True)
        self.assertEqual(choice, [row for row in full if row["readout_type"] != "reversed_yes_no"])
        self.assertEqual(len(choice), 256)
        self.assertEqual(len({row["template_family"] for row in choice}), 8)
        layout = _readout_matrix_layout(choice, sorted({row["template"] for row in choice}))
        self.assertEqual(len(layout[0]), 8)
        self.assertEqual(len(layout[1]), 4)

    def test_dpo_persistence_numeric_identity_and_choice_validation_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = DilemmaDPOConfig(root)
            run = completed_dpo_fixture(root / "local/run", config)
            calls = []
            class FakeDrive:
                def flush_and_unmount(self, **kwargs):
                    calls.append("flush")

                def mount(self, mountpoint, **kwargs):
                    calls.append("mount")
                    (Path(mountpoint) / "MyDrive").mkdir(parents=True, exist_ok=True)

            persisted = persist_run_to_colab_drive(
                run, root / "drive/MyDrive/dpo", drive_module=FakeDrive(),
                drive_mountpoint=root / "drive",
            )
            validate_complete_run(persisted)
            self.assertIn("flush", calls)
            self.assertEqual(_validated_source_identity(persisted)[:2], (config.pair_name, TRAINING_OBJECTIVE))
            evaluation = choice_fixture(
                persisted.run_dir / "posthoc_evaluations/choice", persisted.complete_marker_path,
            )
            validation = validate_supervision_matched_readout_artifacts(evaluation, choice_only=True)
            self.assertEqual(validation.score_row_count, 512)
            self.assertIn("raw_scores.csv", _publication_sources(evaluation))
            with patch("scripts.ecological_prompt_sft.readout_evaluation.run_saved_adapter_eval") as run_eval:
                result = run_supervision_matched_readout_workflow(
                    persisted, cost_counts=config.cost_counts, batch_size=2, choice_only=True,
                )
                self.assertTrue(result.evaluation_reused)
                run_eval.assert_not_called()
            with self.assertRaises(RuntimeError):
                validate_supervision_matched_readout_artifacts(evaluation)

    def test_notebook_parses_and_starts_unexecuted(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "notebooks/ecological_dilemma_dpo_colab.ipynb").read_text())
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                ast.parse("".join(cell["source"]))


@unittest.skipUnless(importlib.util.find_spec("trl"), "optional pinned DPO dependencies not installed")
class DPORuntimeTests(unittest.TestCase):
    def test_real_trl_masks_reference_loss_direction_training_and_reload(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import WhitespaceSplit
        from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM, set_seed
        from peft import PeftModel

        torch.set_num_threads(1)
        set_seed(42)
        vocab = {token: i for i, token in enumerate([
            "<pad>", "<eos>", "<unk>", "<assistant>", "which", "policy", "protect", "trees", "people",
        ])}
        raw = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
        raw.pre_tokenizer = WhitespaceSplit()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
        # A trailing space makes the independent tokenization boundary explicit.
        tokenizer.chat_template = "{{ messages[0]['content'] }} <assistant> "
        model_config = Qwen3Config(
            vocab_size=len(vocab), hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
            max_position_embeddings=128, pad_token_id=0, eos_token_id=1, use_cache=False,
        )
        # Whitespace tokenizer needs a separator before EOS, unlike the real Qwen tokenizer.
        examples = [{"id": str(i), "dilemma": "which policy", "chosen": "protect trees ",
                     "rejected": "protect people policy "} for i in range(4)]
        rendered, tokens, _ = render_preference_examples(tokenizer, examples, max_length=64)
        with self.assertRaisesRegex(ValueError, "Refusing to truncate"):
            render_preference_examples(tokenizer, examples, max_length=3)
        with tempfile.TemporaryDirectory() as tmp:
            model = Qwen3ForCausalLM(model_config)
            frozen = {key: value.detach().clone() for key, value in model.state_dict().items()}
            config = DilemmaDPOConfig(tmp, num_train_epochs=2, learning_rate=0.01,
                                      gradient_accumulation_steps=1, lora_rank=2, lora_alpha=4, max_length=64)
            trainer = build_trainer(model, tokenizer, rendered, tokens, config, Path(tmp) / "checkpoints", smoke_test=True)
            refs = precompute_reference_audit(trainer, rendered)
            batch = trainer.data_collator([trainer.train_dataset[0]])
            trainer.model.eval()
            # Independent teacher-forced calculation scores completion+EOS, excluding the prompt.
            prompt_ids = tokens[0]["prompt_input_ids"]
            completion = tokens[0]["chosen_input_ids"]
            sequence = torch.tensor([prompt_ids + completion])
            with torch.no_grad(), trainer.model.disable_adapter():
                logits = trainer.model(sequence).logits.float()
                log_probs = logits[:, len(prompt_ids)-1:-1].log_softmax(-1)
                manual = log_probs.gather(-1, torch.tensor([[completion]]).reshape(1, -1, 1)).sum()
            self.assertAlmostEqual(refs[0]["ref_chosen_logps"], manual.item(), places=5)
            with torch.no_grad():
                initial = trainer.concatenated_forward(trainer.model, batch)
                loss = trainer.compute_loss(trainer.model, batch)
            self.assertAlmostEqual(loss.item(), math.log(2), places=5)
            trainer.train()
            trainer.model.eval()
            with torch.no_grad():
                final = trainer.concatenated_forward(trainer.model, batch)
                new_ref_chosen, new_ref_rejected = trainer.compute_ref_log_probs(batch)
            self.assertGreater(
                (final["chosen_logps"] - final["rejected_logps"]).item(),
                (initial["chosen_logps"] - initial["rejected_logps"]).item(),
            )
            self.assertAlmostEqual(new_ref_chosen.item(), refs[0]["ref_chosen_logps"], places=5)
            self.assertAlmostEqual(new_ref_rejected.item(), refs[0]["ref_rejected_logps"], places=5)
            adapter = Path(tmp) / "adapter"
            trainer.model.save_pretrained(adapter)
            base = Qwen3ForCausalLM(model_config)
            base.load_state_dict(frozen)
            reloaded = PeftModel.from_pretrained(base, adapter).eval()
            with torch.no_grad():
                self.assertTrue(torch.allclose(reloaded(sequence).logits, trainer.model(sequence).logits, atol=1e-6))
            checkpoint = next((Path(tmp) / "checkpoints").glob("checkpoint-*"))
            self.assertTrue((checkpoint / "optimizer.pt").is_file())
            self.assertTrue((checkpoint / "scheduler.pt").is_file())


if __name__ == "__main__":
    unittest.main()
