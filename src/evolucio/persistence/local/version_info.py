"""Small operational environment metadata without importing JAX or Equinox."""

import platform
import subprocess
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from evolucio import __version__


@dataclass(frozen=True, slots=True)
class CodeVersionInfo:
    package_version: str
    git_commit: str | None
    git_dirty: bool | None


@dataclass(frozen=True, slots=True)
class EnvironmentInfo:
    python_version: str
    jax_version: str | None
    equinox_version: str | None
    platform: str


def code_version_info(repo_path: Path | None = None) -> CodeVersionInfo:
    """Keep unavailable Git information explicitly unknown, with bounded calls."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return CodeVersionInfo(__version__, None, None)
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return CodeVersionInfo(__version__, commit, None)
    return CodeVersionInfo(__version__, commit, bool(status))


def _version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def environment_info() -> EnvironmentInfo:
    return EnvironmentInfo(
        platform.python_version(), _version("jax"), _version("equinox"), platform.platform()
    )
