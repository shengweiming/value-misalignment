import ast
import csv
import importlib.util
import json
import math
import os
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
    RUNTIME_VERSIONS, SCORE_PRECISION, TRAINING_OBJECTIVE, build_trainer, config_dict,
    audit_initial_reference_scores, make_training_arguments, precompute_reference_audit, validate_config,
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
        "score_precision": SCORE_PRECISION,
        "initial_reference_audit": {"status": "passed", "score_precision": SCORE_PRECISION, "example_count": 98},
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

    def test_reuse_rejects_legacy_precision_and_missing_or_failed_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = DilemmaDPOConfig(root)
            run = completed_dpo_fixture(root / "run", config)
            original = json.loads(run.metadata_path.read_text())
            for updates in (
                {"score_precision": None}, {"score_precision": "bf16"},
                {"initial_reference_audit": {}}, {"initial_reference_audit": {"status": "failed"}},
            ):
                run.metadata_path.write_text(json.dumps({**original, **updates}))
                # Rehash so this is a compatibility rejection, not a corrupt-file rejection.
                marker = json.loads(run.complete_marker_path.read_text())
                marker["artifact_sha256"] = _required_hashes(run)
                run.complete_marker_path.write_text(json.dumps(marker))
                validate_complete_run(run)
                self.assertIsNone(find_compatible_dpo_run(root, config))

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
    def setUp(self):
        from accelerate.state import AcceleratorState

        # These tests deliberately alternate FP32 and BF16 Trainer instances.
        AcceleratorState._reset_state(reset_partial_state=True)
        self.addCleanup(AcceleratorState._reset_state, reset_partial_state=True)
        environment = patch.dict(os.environ, {"ACCELERATE_MIXED_PRECISION": "no"})
        environment.start()
        self.addCleanup(environment.stop)

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
            self.assertEqual(trainer.initial_reference_audit["status"], "passed")
            self.assertEqual(trainer.initial_reference_audit["example_count"], 4)
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

    def test_bf16_reference_matches_manual_fp32_and_accelerate_then_trains(self):
        import torch
        from accelerate.utils import convert_outputs_to_fp32
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import WhitespaceSplit
        from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM, set_seed

        torch.set_num_threads(1)
        set_seed(42)
        vocab = {t: i for i, t in enumerate([
            "<pad>", "<eos>", "<unk>", "<assistant>", "which", "policy", "protect", "trees", "people",
        ])}
        raw = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
        raw.pre_tokenizer = WhitespaceSplit()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
        tokenizer.chat_template = "{{ messages[0]['content'] }} <assistant> "
        model_config = Qwen3Config(
            vocab_size=len(vocab), hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
            max_position_embeddings=128, pad_token_id=0, eos_token_id=1, use_cache=False,
        )
        # Long, unequal responses make BF16 accumulation errors visible.
        examples = [{"id": str(i), "dilemma": "which policy", "chosen": "protect trees " * (12+i),
                     "rejected": "protect people policy " * (8+i)} for i in range(4)]
        rendered, tokens, _ = render_preference_examples(tokenizer, examples, max_length=64)
        with tempfile.TemporaryDirectory() as tmp:
            model = Qwen3ForCausalLM(model_config).to(torch.bfloat16)
            config = DilemmaDPOConfig(tmp, max_length=64, num_train_epochs=2,
                                      learning_rate=.001, warmup_ratio=0,
                                      gradient_accumulation_steps=1, lora_rank=2, lora_alpha=4)
            cpu_bf16_args = replace(make_training_arguments(config, Path(tmp) / "checkpoints", smoke_test=True), bf16=True)
            with patch("scripts.ecological_dpo.runner.make_training_arguments", return_value=cpu_bf16_args):
                trainer = build_trainer(model, tokenizer, rendered, tokens, config, Path(tmp) / "checkpoints", smoke_test=True)
            self.assertEqual(trainer.accelerator.mixed_precision, "bf16")
            self.assertTrue(trainer.accelerator.native_amp)
            refs = precompute_reference_audit(trainer, rendered)
            manual_scores = []
            legacy_scores = []
            for row in tokens:
                values = {}
                old_values = {}
                prompt = row["prompt_input_ids"]
                sequences = [prompt + row[f"{side}_input_ids"] for side in ("chosen", "rejected")]
                max_length = max(map(len, sequences))
                input_ids = torch.tensor([seq + [0] * (max_length-len(seq)) for seq in sequences])
                attention_mask = torch.tensor([[1] * len(seq) + [0] * (max_length-len(seq)) for seq in sequences])
                # Use the same shapes/AMP as the scorer, but independently select
                # the completion positions and calculate log-softmax and sums.
                with torch.no_grad(), trainer.accelerator.autocast(), trainer.compute_loss_context_manager(), trainer.model.disable_adapter():
                    logits = trainer.model.get_base_model()(
                        input_ids, attention_mask=attention_mask,
                        logits_to_keep=max_length-len(prompt)+1,
                    ).logits
                self.assertEqual(logits.dtype, torch.bfloat16)
                for index, side in enumerate(("chosen", "rejected")):
                    completion = row[f"{side}_input_ids"]
                    response_logits = logits[index, :len(completion)]
                    labels = torch.tensor(completion).unsqueeze(-1)
                    values[side] = response_logits.float().log_softmax(-1).gather(-1, labels).sum().item()
                    old_values[side] = response_logits.log_softmax(-1).gather(-1, labels).sum().item()
                manual_scores.append(values)
                legacy_scores.append(old_values)
            for ref, manual in zip(refs, manual_scores):
                for side in ("chosen", "rejected"):
                    self.assertAlmostEqual(ref[f"ref_{side}_logps"], manual[side], delta=2e-5)
            # A post-hoc cast of the old BF16 response sums cannot pass this check.
            self.assertGreater(max(abs(old[s]-new[s]) for old, new in zip(legacy_scores, manual_scores)
                                   for s in ("chosen", "rejected")), .01)
            # Exercise Accelerate's real output converter on the unchanged BF16 model.
            trainer.model.forward = convert_outputs_to_fp32(trainer.model.forward)
            with trainer.accelerator.autocast():
                initial_audit = audit_initial_reference_scores(trainer)
            self.assertLessEqual(initial_audit["max_abs_logp_difference"], 1e-5)
            self.assertAlmostEqual(initial_audit["mean_dpo_loss"], math.log(2), places=6)
            self.assertLessEqual(initial_audit["max_abs_reward_margin"], 1e-6)
            trainer.train()
            self.assertEqual(trainer.initial_reference_audit["status"], "passed")
            trainer.model.eval()
            improvements = []
            for row, ref in zip(trainer.train_dataset, refs):
                batch = trainer.data_collator([row])
                with torch.no_grad():
                    scores = trainer.concatenated_forward(trainer.model, batch)
                    ref_chosen, ref_rejected = trainer.compute_ref_log_probs(batch)
                self.assertEqual(scores["chosen_logps"].dtype, torch.float32)
                self.assertAlmostEqual(ref_chosen.item(), ref["ref_chosen_logps"], places=5)
                self.assertAlmostEqual(ref_rejected.item(), ref["ref_rejected_logps"], places=5)
                improvements.append((scores["chosen_logps"] - scores["rejected_logps"]).item()
                                    - (ref["ref_chosen_logps"] - ref["ref_rejected_logps"]))
            self.assertGreater(sum(improvements)/len(improvements), 0)

    def test_initial_audit_rejects_wrong_reference_before_any_update(self):
        # A corrupted cached score must fail even if the model forward succeeds.
        import torch
        from types import SimpleNamespace
        from contextlib import nullcontext
        model = torch.nn.Linear(1, 1)
        trainer = SimpleNamespace(
            model=model, train_dataset=[{"id": "corrupt"}],
            data_collator=lambda rows: {"ref_chosen_logps": torch.tensor([-10.25]),
                                       "ref_rejected_logps": torch.tensor([-10.0])},
            _prepare_inputs=lambda batch: batch, compute_loss_context_manager=nullcontext,
            concatenated_forward=lambda model, batch: {"chosen_logps": torch.tensor([-10.0]),
                                                       "rejected_logps": torch.tensor([-10.0])},
        )
        with self.assertRaisesRegex(RuntimeError, "before the first update"):
            audit_initial_reference_scores(trainer)
        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
