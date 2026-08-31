"""Publish a verified compact evaluation bundle from Colab to GitHub."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

from scripts.harmony_eval.cases import REPO_ROOT

from .extreme_v2_eval import (
    build_extreme_v2_cases,
    build_extreme_v2_control_cases,
    validate_extreme_v2_artifacts,
    validate_extreme_v2_control_artifacts,
)
from .posthoc_eval import PosthocEvalArtifacts, _template_manifest


DEFAULT_RESULTS_ROOT = Path("results/harmony_eval/qwen3_8b_harmony_r1_sft")
PUBLISHED_RESULT_FILES = (
    "rendered_cases.jsonl",
    "raw_scores.csv",
    "thresholds.csv",
    "curves.png",
    "metadata.json",
    "COMPLETE.json",
)
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}")
_SAFE_BRANCH = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,199}")
_SAFE_REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class GitHubPublication:
    repository: str
    branch: str
    commit_sha: str
    repository_path: str
    html_url: str
    created_commit: bool


def _git(
    arguments: Sequence[str],
    *,
    repo_root: Path,
    environment: Mapping[str, str] | None = None,
) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        env=dict(environment) if environment is not None else None,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout.strip()


@contextmanager
def _git_auth_environment(github_token: str | None) -> Iterator[dict[str, str]]:
    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    if github_token is None:
        yield environment
        return
    if not github_token or "\n" in github_token or "\r" in github_token:
        raise ValueError("github_token must be a non-empty single-line token")
    with tempfile.TemporaryDirectory(prefix="value-misalignment-askpass-") as temp:
        askpass = Path(temp) / "askpass.sh"
        askpass.write_text(
            "#!/bin/sh\n"
            "case \"$1\" in\n"
            "  *Username*) printf '%s\\n' 'x-access-token' ;;\n"
            "  *Password*) printf '%s\\n' \"$VALUE_MISALIGNMENT_GITHUB_TOKEN\" ;;\n"
            "  *) exit 1 ;;\n"
            "esac\n",
            encoding="utf-8",
        )
        askpass.chmod(0o700)
        environment["GIT_ASKPASS"] = str(askpass)
        environment["VALUE_MISALIGNMENT_GITHUB_TOKEN"] = github_token
        yield environment


def _validate_destination_parts(
    source_run_name: str,
    evaluation_name: str,
    results_root: Path,
) -> Path:
    if not _SAFE_COMPONENT.fullmatch(source_run_name):
        raise ValueError(f"Unsafe source run directory name: {source_run_name!r}")
    if not _SAFE_COMPONENT.fullmatch(evaluation_name):
        raise ValueError(f"Unsafe evaluation directory name: {evaluation_name!r}")
    if results_root.is_absolute() or ".." in results_root.parts:
        raise ValueError("results_root must be a safe repository-relative path")
    return results_root / source_run_name / evaluation_name


def _publication_sources(artifacts: PosthocEvalArtifacts) -> dict[str, Path]:
    try:
        metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
        cost_counts = tuple(int(value) for value in metadata["cost_counts"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Could not read the evaluation cost grid before GitHub publication"
        ) from exc
    observed_templates = metadata.get("templates")
    primary_templates = _template_manifest(build_extreme_v2_cases(cost_counts))
    control_templates = _template_manifest(
        build_extreme_v2_control_cases(cost_counts)
    )
    if observed_templates == primary_templates:
        validate_extreme_v2_artifacts(artifacts, cost_counts=cost_counts)
    elif observed_templates == control_templates:
        validate_extreme_v2_control_artifacts(artifacts, cost_counts=cost_counts)
    elif metadata.get("evaluation_slug") == (
        "extreme_v2_supervision_matched_readouts_eval"
    ):
        # Import lazily: the ecological workflow itself reuses this generic
        # publication module, so importing its validator at module load time
        # would create a cycle.
        from scripts.ecological_prompt_sft.readout_evaluation import (
            validate_supervision_matched_readout_artifacts,
        )

        validate_supervision_matched_readout_artifacts(
            artifacts,
            cost_counts=cost_counts,
        )
    else:
        raise RuntimeError(
            "GitHub publication accepts only the current primary, control, or "
            "supervision-matched extreme-v2 suite"
        )
    sources = {name: artifacts.output_dir / name for name in PUBLISHED_RESULT_FILES}
    missing = [name for name, path in sources.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Evaluation is missing GitHub publication files: {missing}")
    return sources


def _copy_or_verify_bundle(sources: Mapping[str, Path], destination: Path) -> None:
    if destination.exists():
        if not destination.is_dir():
            raise RuntimeError(f"GitHub result destination is not a directory: {destination}")
        existing_entries = list(destination.iterdir())
        existing_names = {path.name for path in existing_entries}
        if existing_names != set(sources):
            raise RuntimeError(
                "An existing GitHub result bundle has a different file set: "
                f"{destination}"
            )
        if any(not path.is_file() for path in existing_entries):
            raise RuntimeError(
                f"An existing GitHub result bundle contains non-file entries: {destination}"
            )
        mismatches = [
            name
            for name, source in sources.items()
            if (destination / name).read_bytes() != source.read_bytes()
        ]
        if mismatches:
            raise RuntimeError(
                "An existing GitHub result bundle differs from the verified Drive "
                f"bundle: {mismatches}"
            )
        return

    destination.mkdir(parents=True, exist_ok=False)
    for name, source in sources.items():
        shutil.copy2(source, destination / name)


def publish_extreme_v2_results_to_github(
    artifacts: PosthocEvalArtifacts,
    *,
    source_run_name: str,
    github_repository: str = "shengweiming/value-misalignment",
    branch: str = "main",
    github_token: str | None = None,
    repo_root: Path | str = REPO_ROOT,
    remote_name: str = "origin",
    results_root: Path | str = DEFAULT_RESULTS_ROOT,
) -> GitHubPublication:
    """Commit, push, and remotely verify one approved evaluation result bundle.

    HTTPS authentication uses a transient ``GIT_ASKPASS`` helper and an
    environment variable. The token is never written into the repository, remote
    URL, command arguments, commit, or result bundle.
    """

    if not _SAFE_REPOSITORY.fullmatch(github_repository):
        raise ValueError("github_repository must have the form owner/name")
    if not _SAFE_BRANCH.fullmatch(branch) or ".." in branch or branch.endswith("/"):
        raise ValueError(f"Unsafe Git branch name: {branch!r}")
    repository_root = Path(repo_root).expanduser().resolve(strict=True)
    if not (repository_root / ".git").exists():
        raise ValueError(f"repo_root is not a Git checkout: {repository_root}")

    results_root_path = Path(results_root)
    relative_destination = _validate_destination_parts(
        source_run_name,
        artifacts.output_dir.name,
        results_root_path,
    )
    sources = _publication_sources(artifacts)

    status = _git(["status", "--porcelain"], repo_root=repository_root)
    if status:
        raise RuntimeError(
            "The Colab repository has uncommitted changes; refusing to mix them "
            "with evaluation results"
        )
    remote_url = _git(
        ["remote", "get-url", remote_name], repo_root=repository_root
    )
    if remote_url.startswith(("http://", "https://")) and github_token is None:
        raise ValueError(
            "A GITHUB_TOKEN with Contents read/write access is required to publish "
            "from the Colab HTTPS clone"
        )

    with _git_auth_environment(github_token) as environment:
        _git(
            ["fetch", "--no-tags", remote_name, branch],
            repo_root=repository_root,
            environment=environment,
        )
        _git(
            ["merge", "--ff-only", "FETCH_HEAD"],
            repo_root=repository_root,
            environment=environment,
        )

        destination = repository_root / relative_destination
        _copy_or_verify_bundle(sources, destination)
        _git(
            ["add", "--", relative_destination.as_posix()],
            repo_root=repository_root,
            environment=environment,
        )
        staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet", "--", relative_destination.as_posix()],
            cwd=repository_root,
            env=environment,
            check=False,
        )
        if staged.returncode not in {0, 1}:
            raise RuntimeError("Could not determine whether GitHub result files changed")
        created_commit = staged.returncode == 1
        if created_commit:
            _git(
                [
                    "-c",
                    "user.name=value-misalignment-colab",
                    "-c",
                    "user.email=value-misalignment-colab@users.noreply.github.com",
                    "commit",
                    "-m",
                    f"Add evaluation results {artifacts.output_dir.name}",
                    "--",
                    relative_destination.as_posix(),
                ],
                repo_root=repository_root,
                environment=environment,
            )

        local_head = _git(
            ["rev-parse", "HEAD"],
            repo_root=repository_root,
            environment=environment,
        )
        _git(
            ["push", remote_name, f"HEAD:refs/heads/{branch}"],
            repo_root=repository_root,
            environment=environment,
        )
        remote_ref = _git(
            ["ls-remote", remote_name, f"refs/heads/{branch}"],
            repo_root=repository_root,
            environment=environment,
        )
    remote_head = remote_ref.split(maxsplit=1)[0] if remote_ref else ""
    if remote_head != local_head:
        raise RuntimeError(
            "GitHub push returned without placing the local result commit at the "
            f"tip of {branch}"
        )

    repository_path = relative_destination.as_posix()
    return GitHubPublication(
        repository=github_repository,
        branch=branch,
        commit_sha=local_head,
        repository_path=repository_path,
        html_url=(
            f"https://github.com/{github_repository}/tree/{local_head}/"
            f"{repository_path}"
        ),
        created_commit=created_commit,
    )
