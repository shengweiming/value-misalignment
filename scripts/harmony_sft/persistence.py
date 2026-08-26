"""Durably copy a completed local SFT run into Colab-mounted Google Drive."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from .runner import SFTArtifacts, artifacts_for_run_dir, validate_complete_run


DEFAULT_DRIVE_MOUNTPOINT = Path("/content/drive")
DEFAULT_FLUSH_TIMEOUT_MS = 2 * 60 * 60 * 1000
DEFAULT_MOUNT_TIMEOUT_MS = 5 * 60 * 1000


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _mounted_my_drive(mountpoint: Path) -> Path:
    preferred = mountpoint / "MyDrive"
    if preferred.is_dir():
        return preferred
    legacy = mountpoint / "My Drive"
    if legacy.is_dir():
        return legacy
    return preferred


def _mount_if_needed(drive_module: Any, mountpoint: Path, timeout_ms: int) -> Path:
    my_drive = _mounted_my_drive(mountpoint)
    if not my_drive.is_dir():
        drive_module.mount(str(mountpoint), timeout_ms=timeout_ms)
        my_drive = _mounted_my_drive(mountpoint)
    if not my_drive.is_dir():
        raise RuntimeError(f"Google Drive is not mounted beneath {mountpoint}")
    return my_drive


def _copy_run(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, dirs_exist_ok=True, copy_function=shutil.copy2)


def persist_run_to_colab_drive(
    artifacts: SFTArtifacts,
    drive_output_root: Path | str,
    *,
    drive_mountpoint: Path | str = DEFAULT_DRIVE_MOUNTPOINT,
    flush_timeout_ms: int = DEFAULT_FLUSH_TIMEOUT_MS,
    mount_timeout_ms: int = DEFAULT_MOUNT_TIMEOUT_MS,
    max_attempts: int = 2,
    drive_module: Any | None = None,
) -> SFTArtifacts:
    """Copy, flush, remount, and hash-verify a run before declaring Drive success.

    The source run must be on local runtime storage rather than inside the Drive
    mount. It is deliberately retained after success and after failure, providing
    a recovery copy until the Colab runtime is disconnected.
    """

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if flush_timeout_ms < 1 or mount_timeout_ms < 1:
        raise ValueError("Drive timeouts must be positive")

    source = artifacts.run_dir.expanduser().resolve(strict=True)
    source_artifacts = artifacts_for_run_dir(source)
    validate_complete_run(source_artifacts)

    mountpoint = Path(drive_mountpoint).expanduser().resolve(strict=False)
    if drive_module is None:
        try:
            from google.colab import drive as drive_module
        except ImportError as exc:
            raise RuntimeError(
                "Durable Drive persistence requires a Google Colab runtime"
            ) from exc

    my_drive = _mount_if_needed(drive_module, mountpoint, mount_timeout_ms)
    my_drive = my_drive.resolve(strict=True)
    if _is_within(source, my_drive):
        raise ValueError(
            "The completed source run must be on local /content storage, not "
            "inside the Drive mount"
        )

    output_root = Path(drive_output_root).expanduser().resolve(strict=False)
    if not _is_within(output_root, my_drive):
        raise ValueError(
            f"drive_output_root must be inside the mounted Drive root {my_drive}"
        )
    destination = output_root / source.name
    destination_artifacts = artifacts_for_run_dir(destination)

    print(f"Local recovery copy retained at: {source}")
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            print(
                f"Drive persistence attempt {attempt}/{max_attempts}: copying "
                f"{source.name}"
            )
            _copy_run(source, destination)
            os.sync()
            print("Flushing outstanding Google Drive writes...")
            drive_module.flush_and_unmount(timeout_ms=flush_timeout_ms)
            drive_module.mount(str(mountpoint), timeout_ms=mount_timeout_ms)

            # Reconstruct paths only after a fresh mount. Reading and hashing these
            # files now verifies the remote Drive view, not the pre-flush FUSE cache.
            destination_artifacts = artifacts_for_run_dir(destination)
            validate_complete_run(destination_artifacts)
            print(f"Fresh-mount hash verification passed: {destination}")
            return destination_artifacts
        except Exception as exc:
            last_error = exc
            print(
                "Drive persistence has not been verified. Do not disconnect the "
                f"runtime; the complete local run remains at {source}."
            )
            if attempt == max_attempts:
                break
            _mount_if_needed(drive_module, mountpoint, mount_timeout_ms)

    raise RuntimeError(
        "Could not verify the run after a fresh Google Drive mount. The complete "
        f"local recovery copy remains at {source}."
    ) from last_error
