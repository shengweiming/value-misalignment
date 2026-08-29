import csv
import hashlib
import json
import subprocess
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from scripts.harmony_eval.cases import DEFAULT_COST_COUNTS
from scripts.harmony_sft.extreme_v2_eval import (
    EXTREME_V2_CONTROL_TEMPLATES,
    EXTREME_V2_TEMPLATES,
    build_extreme_v2_control_cases,
    build_extreme_v2_cases,
    run_extreme_v2_workflow,
    validate_extreme_v2_artifacts,
)
from scripts.harmony_sft.github_publish import publish_extreme_v2_results_to_github
from scripts.harmony_sft.posthoc_eval import (
    POSTHOC_PROTOCOL_VERSION,
    _required_hashes,
    _template_manifest,
    artifacts_for_posthoc_eval,
)
from scripts.harmony_sft.runner import (
    SFTConfig,
    _validate_complete_artifacts,
    artifacts_for_run_dir,
)


def _run_git(arguments, *, cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def make_complete_sft_run(run_dir: Path, config: SFTConfig):
    artifacts = artifacts_for_run_dir(run_dir)
    checkpoint = artifacts.checkpoints_dir / "checkpoint-10"
    for directory in (
        artifacts.final_adapter_dir,
        checkpoint,
        artifacts.train_metrics_path.parent,
        artifacts.raw_scores_path.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    for path in (
        artifacts.final_adapter_dir / "adapter_config.json",
        artifacts.final_adapter_dir / "adapter_model.safetensors",
        checkpoint / "adapter_config.json",
        checkpoint / "adapter_model.safetensors",
        checkpoint / "trainer_state.json",
        checkpoint / "optimizer.pt",
        checkpoint / "scheduler.pt",
        artifacts.train_metrics_path,
        artifacts.raw_scores_path,
        artifacts.thresholds_path,
        artifacts.plot_path,
    ):
        path.write_bytes(f"test artifact: {path.name}".encode("utf-8"))
    config_dict = asdict(config)
    config_dict["output_root"] = str(config.output_root)
    artifacts.metadata_path.write_text(
        json.dumps({"status": "complete", "config": config_dict}),
        encoding="utf-8",
    )
    hashes = _validate_complete_artifacts(artifacts)
    artifacts.complete_marker_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "completed_at_utc": run_dir.name,
                "artifact_sha256": hashes,
            }
        ),
        encoding="utf-8",
    )
    return artifacts


def make_complete_extreme_v2_eval(
    output_dir: Path,
    *,
    source_complete_path: Path | None = None,
):
    artifacts = artifacts_for_posthoc_eval(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = build_extreme_v2_cases(DEFAULT_COST_COUNTS)
    artifacts.rendered_cases_path.write_text(
        "".join(json.dumps(case) + "\n" for case in cases),
        encoding="utf-8",
    )
    with artifacts.raw_scores_path.open("w", encoding="utf-8", newline="") as output:
        fieldnames = (
            "case_id",
            "template",
            "cost_count",
            "model_role",
            "p_implement",
            "semantic_logit_implement",
        )
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for role in ("base", "aligned"):
            for case in cases:
                writer.writerow(
                    {
                        "case_id": case["case_id"],
                        "template": case["template"],
                        "cost_count": case["cost_count"],
                        "model_role": role,
                        "p_implement": 0.5,
                        "semantic_logit_implement": 0.0,
                    }
                )
    artifacts.thresholds_path.write_text("template,base_status,aligned_status\n")
    artifacts.plot_path.write_bytes(b"png")
    source_hash = (
        hashlib.sha256(source_complete_path.read_bytes()).hexdigest()
        if source_complete_path is not None
        else "test-source-complete-sha256"
    )
    artifacts.metadata_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "evaluation_protocol_version": POSTHOC_PROTOCOL_VERSION,
                "completed_at_utc": output_dir.name,
                "source_complete_sha256": source_hash,
                "cost_counts": list(DEFAULT_COST_COUNTS),
                "case_count_per_model": len(cases),
                "enable_thinking": False,
                "templates": _template_manifest(cases),
            }
        ),
        encoding="utf-8",
    )
    artifacts.complete_marker_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "completed_at_utc": output_dir.name,
                "artifact_sha256": _required_hashes(artifacts),
            }
        ),
        encoding="utf-8",
    )
    return artifacts


class ExtremeV2WorkflowTests(unittest.TestCase):
    def test_suite_has_eight_templates_and_64_cases(self):
        cases = build_extreme_v2_cases()

        self.assertEqual(len(EXTREME_V2_TEMPLATES), 8)
        self.assertEqual(len(cases), 64)
        self.assertEqual(len({case["case_id"] for case in cases}), 64)

    def test_control_suite_has_three_two_prompt_categories(self):
        cases = build_extreme_v2_control_cases()
        by_template = {
            template: [case for case in cases if case["template"] == template]
            for template in {case["template"] for case in cases}
        }

        self.assertEqual(len(EXTREME_V2_CONTROL_TEMPLATES), 6)
        self.assertEqual(len(cases), 34)
        self.assertEqual(len(by_template), 6)
        zero_cost_templates = {
            template for template in by_template if "__zero_cost_ecological__" in template
        }
        self.assertEqual(len(zero_cost_templates), 2)
        for template, template_cases in by_template.items():
            with self.subTest(template=template):
                if template in zero_cost_templates:
                    self.assertEqual(
                        [case["cost_count"] for case in template_cases],
                        [0],
                    )
                else:
                    self.assertEqual(len(template_cases), len(DEFAULT_COST_COUNTS))

    def test_validation_requires_base_and_aligned_row_for_every_case(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifacts = make_complete_extreme_v2_eval(
                Path(temporary_directory) / "eval"
            )
            validation = validate_extreme_v2_artifacts(artifacts)
            self.assertEqual(validation.score_row_count, 128)

            with artifacts.raw_scores_path.open(
                "r", encoding="utf-8", newline=""
            ) as input_file:
                rows = list(csv.DictReader(input_file))
            with artifacts.raw_scores_path.open(
                "w", encoding="utf-8", newline=""
            ) as output:
                writer = csv.DictWriter(output, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows[:-1])
            artifacts.complete_marker_path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "artifact_sha256": _required_hashes(artifacts),
                    }
                )
            )

            with self.assertRaisesRegex(RuntimeError, "exactly one base"):
                validate_extreme_v2_artifacts(artifacts)

    def test_workflow_reuses_verified_sft_and_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            drive_root = Path(temporary_directory) / "drive-runs"
            config = SFTConfig(
                output_root="unused",
                require_google_drive=False,
            )
            sft = make_complete_sft_run(drive_root / "sft-run", config)
            evaluation = make_complete_extreme_v2_eval(
                sft.run_dir / "posthoc_evaluations" / "extreme-v2",
                source_complete_path=sft.complete_marker_path,
            )

            workflow = run_extreme_v2_workflow(drive_root, config)

            self.assertTrue(workflow.evaluation_reused)
            self.assertEqual(workflow.sft_artifacts.run_dir, sft.run_dir)
            self.assertEqual(
                workflow.evaluation_artifacts.output_dir,
                evaluation.output_dir,
            )
            self.assertEqual(workflow.validation.score_row_count, 128)

    def test_workflow_refuses_to_train_when_checkpoint_is_missing(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = SFTConfig(
                output_root="unused",
                require_google_drive=False,
            )
            with self.assertRaisesRegex(RuntimeError, "will not retrain"):
                run_extreme_v2_workflow(temporary_directory, config)

    def test_github_publication_pushes_and_reuses_exact_bundle(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            remote = root / "remote.git"
            checkout = root / "checkout"
            _run_git(["init", "--bare", str(remote)], cwd=root)
            _run_git(["init", str(checkout)], cwd=root)
            _run_git(["config", "user.name", "Test User"], cwd=checkout)
            _run_git(["config", "user.email", "test@example.com"], cwd=checkout)
            (checkout / "README.md").write_text("test\n")
            _run_git(["add", "README.md"], cwd=checkout)
            _run_git(["commit", "-m", "Initial"], cwd=checkout)
            _run_git(["branch", "-M", "main"], cwd=checkout)
            _run_git(["remote", "add", "origin", str(remote)], cwd=checkout)
            _run_git(["push", "-u", "origin", "main"], cwd=checkout)
            artifacts = make_complete_extreme_v2_eval(root / "evaluation")

            first = publish_extreme_v2_results_to_github(
                artifacts,
                source_run_name="sft-run",
                github_repository="example/value-misalignment",
                repo_root=checkout,
            )
            second = publish_extreme_v2_results_to_github(
                artifacts,
                source_run_name="sft-run",
                github_repository="example/value-misalignment",
                repo_root=checkout,
            )

            self.assertTrue(first.created_commit)
            self.assertFalse(second.created_commit)
            self.assertEqual(first.commit_sha, second.commit_sha)
            remote_head = _run_git(
                ["--git-dir", str(remote), "rev-parse", "refs/heads/main"],
                cwd=root,
            )
            self.assertEqual(remote_head, first.commit_sha)
            published_raw_scores = _run_git(
                [
                    "--git-dir",
                    str(remote),
                    "show",
                    f"main:{first.repository_path}/raw_scores.csv",
                ],
                cwd=root,
            )
            self.assertIn("model_role", published_raw_scores)


if __name__ == "__main__":
    unittest.main()
