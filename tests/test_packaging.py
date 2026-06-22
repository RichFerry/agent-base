"""Packaging and installation smoke tests for release hygiene."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tomllib
import venv
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).parents[1]


def _load_pyproject() -> dict[str, Any]:
    return tomllib.loads((_repo_root() / "pyproject.toml").read_text(encoding="utf-8"))


def test_pyproject_packaging_metadata_contract() -> None:
    """Package metadata stays minimal and runtime dependencies stay empty."""
    pyproject = _load_pyproject()
    project = pyproject["project"]

    assert pyproject["build-system"]["requires"] == ["setuptools>=61", "wheel"]
    assert project["name"] == "agent-kernel"
    assert project["version"] == "0.3.0"
    assert "Agent Kernel" in project["description"]
    assert project["requires-python"] == ">=3.11"
    assert project["dependencies"] == []
    assert project["scripts"] == {"agent-kernel-local": "examples.local_agent:main"}
    assert pyproject["project"]["optional-dependencies"]["test"] == ["pytest>=8.0"]
    assert pyproject["tool"]["setuptools"]["packages"]["find"]["include"] == ["agent_kernel*", "examples"]


def test_local_runner_help_starts_without_model_credentials() -> None:
    """The example runner can show help without real model credentials."""
    result = subprocess.run(
        [sys.executable, "examples/local_agent.py", "--help"],
        cwd=_repo_root(),
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Run one prompt through the local Python Agent Kernel." in result.stdout
    assert "--permission-mode" in result.stdout
    assert "--enable-web-search" in result.stdout
    assert "--enable-web-fetch" in result.stdout
    assert "--mcp-fixture" in result.stdout


def test_editable_install_exposes_local_runner_console_script(tmp_path: Path) -> None:
    """An editable install exposes the console script and help stays offline."""
    source = tmp_path / "source"
    shutil.copytree(
        _repo_root(),
        source,
        ignore=shutil.ignore_patterns(
            ".git",
            ".pytest_cache",
            "__pycache__",
            "*.pyc",
            ".DS_Store",
            "*.egg-info",
            ".venv",
            "venv",
            "build",
            "dist",
        ),
    )

    venv_dir = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(venv_dir)
    if os.name == "nt":
        venv_python = venv_dir / "Scripts" / "python.exe"
        console_script = venv_dir / "Scripts" / "agent-kernel-local.exe"
    else:
        venv_python = venv_dir / "bin" / "python"
        console_script = venv_dir / "bin" / "agent-kernel-local"

    pip_probe = subprocess.run(
        [str(venv_python), "-m", "pip", "--version"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert pip_probe.returncode == 0, pip_probe.stderr

    env = os.environ.copy()
    env.update(
        {
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INDEX": "1",
        }
    )
    install = subprocess.run(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "--no-deps",
            "-e",
            str(source),
        ],
        text=True,
        capture_output=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert install.returncode == 0, install.stderr
    assert console_script.exists()

    help_result = subprocess.run(
        [str(console_script), "--help"],
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )

    assert "Run one prompt through the local Python Agent Kernel." in help_result.stdout
    assert "--skills-dir" in help_result.stdout
    assert "--mcp-fixture" in help_result.stdout
