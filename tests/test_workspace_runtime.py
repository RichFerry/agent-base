"""Workspace runtime boundary tests for v0.7."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from agent_kernel.config import KernelConfig
from agent_kernel.memory import MemoryLoader
from agent_kernel.model_provider import FakeModelProvider
from agent_kernel.path_utils import sanitize_path
from agent_kernel.prompt_composer import PromptComposer
from agent_kernel.session import SessionStore
from agent_kernel.tools import BashTool, FileReadTool
from examples.local_agent import build_local_engine, latest_local_session_id, list_local_sessions, main, run_local_agent_once


def _git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    return repo


def test_workspace_runtime_anchors_project_storage_to_git_root(tmp_path: Path) -> None:
    """Sessions, memory, and artifacts share one workspace project store."""
    repo = _git_repo(tmp_path)
    subdir = repo / "packages" / "agent"
    subdir.mkdir(parents=True)
    config_home = tmp_path / ".agent"
    settings = repo / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    skill_dir = repo / "skills"
    mcp_config = repo / "mcp.json"

    config = KernelConfig(
        cwd=subdir,
        config_home=config_home,
        settings_paths=(settings,),
        skill_paths=(skill_dir,),
        mcp_config_paths=(mcp_config,),
    )
    runtime = config.workspace_runtime

    expected_store = config_home / "projects" / sanitize_path(repo.resolve())
    assert runtime.cwd == subdir.resolve()
    assert runtime.workspace_root == repo.resolve()
    assert runtime.workspace_root_source == "git"
    assert runtime.project_store_dir == expected_store
    assert runtime.sessions_dir == expected_store
    assert runtime.memory_dir == expected_store / "memory"
    assert runtime.artifacts_dir == expected_store / "artifacts"
    assert runtime.bash_output_dir == expected_store / "artifacts" / "bash-output"
    assert runtime.agent_output_dir == expected_store / "artifacts" / "agent-output"
    assert runtime.allowed_working_directories == (subdir.resolve(),)
    assert runtime.settings_sources[0].scope == "project"
    assert runtime.skill_sources[0].scope == "project"
    assert runtime.mcp_sources[0].scope == "project"
    assert SessionStore(config, session_id="s1").project_dir == runtime.sessions_dir
    assert MemoryLoader(config).get_auto_mem_path() == runtime.memory_dir


def test_workspace_runtime_can_disable_project_memory_scope(tmp_path: Path) -> None:
    """Disabled memory remains explicit in runtime diagnostics."""
    repo = _git_repo(tmp_path)
    config = KernelConfig(cwd=repo, config_home=tmp_path / ".agent", auto_memory_enabled=False)

    runtime = config.workspace_runtime

    assert runtime.memory_scope == "disabled"
    assert runtime.memory_dir is None
    assert MemoryLoader(config).load_memory_prompt() is None


def test_prompt_environment_includes_workspace_runtime_boundaries(tmp_path: Path) -> None:
    """The model-facing environment section names root, transcript, memory, and artifact paths."""
    repo = _git_repo(tmp_path)
    config = KernelConfig(cwd=repo, config_home=tmp_path / ".agent", session_start_date="2026-06-24")
    composer = PromptComposer(config, MemoryLoader(config))

    system_prompt = composer.get_system_prompt(tools=[BashTool(), FileReadTool()], model="agent-kernel-frontier")
    env_section = next(section for section in system_prompt if section.startswith("# Environment"))

    assert f"Workspace root: {repo.resolve()}" in env_section
    assert f"Session transcripts directory: {config.workspace_runtime.transcript_dir}" in env_section
    assert f"Workspace artifacts directory: {config.workspace_runtime.artifacts_dir}" in env_section
    assert f"Memory directory: {config.workspace_runtime.memory_dir}" in env_section
    assert str(config.workspace_runtime.allowed_working_directories[0]) in env_section


def test_cli_workspace_doctor_reports_settings_and_capability_scopes(
    capsys,
    tmp_path: Path,
) -> None:
    """workspace doctor answers where settings, skills, memory, sessions, MCP, and artifacts live."""
    repo = _git_repo(tmp_path)
    subdir = repo / "pkg"
    subdir.mkdir()
    config_home = tmp_path / ".agent"
    config_home.mkdir()
    user_settings = config_home / "settings.json"
    project_settings = repo / "settings.json"
    explicit_settings = tmp_path / "explicit-settings.json"
    for path in (user_settings, project_settings, explicit_settings):
        path.write_text("{}", encoding="utf-8")
    skill_dir = repo / "skills"
    skill_dir.mkdir()
    mcp_config = repo / "mcp.json"
    mcp_config.write_text('{"mcpServers":{"echo":{"command":"python3","args":["server.py"]}}}', encoding="utf-8")
    fixture = tmp_path / "fixture.json"
    fixture.write_text("{}", encoding="utf-8")

    exit_code = main(
        [
            "workspace",
            "doctor",
            "--json",
            "--cwd",
            str(subdir),
            "--config-home",
            str(config_home),
            "--agent-config",
            str(explicit_settings),
            "--skills-dir",
            str(skill_dir),
            "--mcp-config",
            str(mcp_config),
            "--mcp-fixture",
            str(fixture),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    workspace = payload["workspace"]

    assert exit_code == 0
    assert payload["schemaVersion"] == "0.7"
    assert workspace["cwd"] == str(subdir.resolve())
    assert workspace["workspaceRoot"] == str(repo.resolve())
    assert workspace["sessions"]["scope"] == "project"
    assert workspace["memory"]["scope"] == "project"
    assert workspace["artifacts"]["dir"].endswith("/artifacts")
    assert workspace["actMode"]["allowedWorkingDirectories"] == [str(subdir.resolve())]
    assert [Path(item["path"]).name for item in workspace["settingsSources"]] == [
        "settings.json",
        "settings.json",
        "explicit-settings.json",
    ]
    assert [item["scope"] for item in workspace["settingsSources"]] == ["user", "project", "explicit"]
    assert workspace["skillSources"][0]["scope"] == "project"
    assert workspace["mcpSources"][0]["scope"] == "project"
    assert workspace["mcpSources"][1]["scope"] == "explicit"
    assert captured.err == ""


def test_sdk_init_reports_workspace_summary(tmp_path: Path) -> None:
    """SDK init carries additive workspace metadata for session tooling."""
    repo = _git_repo(tmp_path)
    engine = build_local_engine(
        cwd=repo,
        config_home=tmp_path / ".agent",
        model_provider=FakeModelProvider(["done"]),
        require_api_key=False,
    )

    result = asyncio.run(run_local_agent_once("hello", engine=engine))
    init = next(event for event in result.events if event.get("type") == "system" and event.get("subtype") == "init")

    assert init["workspace"]["root"] == str(repo.resolve())
    assert init["workspace"]["rootSource"] == "git"
    assert init["workspace"]["sessionsDir"] == str(engine.config.workspace_runtime.sessions_dir)
    assert init["workspace"]["memoryScope"] == "project"
    assert init["workspace"]["artifactsDir"] == str(engine.config.workspace_runtime.artifacts_dir)


def test_v07_session_resume_can_read_legacy_cwd_bucket(tmp_path: Path) -> None:
    """Explicit resume and continue discovery still see pre-v0.7 cwd-keyed transcripts."""
    repo = _git_repo(tmp_path)
    subdir = repo / "pkg"
    subdir.mkdir()
    config_home = tmp_path / ".agent"
    session_id = "legacy-session"
    legacy_dir = config_home / "projects" / sanitize_path(subdir)
    legacy_dir.mkdir(parents=True)
    legacy_path = legacy_dir / f"{session_id}.jsonl"
    legacy_path.write_text(
        json.dumps(
            {
                "type": "user",
                "uuid": "user-1",
                "message": {"role": "user", "content": [{"type": "text", "text": "legacy"}]},
                "sessionId": session_id,
                "cwd": str(subdir),
                "timestamp": "2026-06-24T00:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    engine = build_local_engine(
        cwd=subdir,
        config_home=config_home,
        model_provider=FakeModelProvider(["continued"]),
        session_id=session_id,
        resume=True,
        require_api_key=False,
    )

    assert engine.session_store.project_dir == config_home / "projects" / sanitize_path(repo.resolve())
    assert engine.session_store.transcript_path == legacy_path
    assert [message["uuid"] for message in engine.mutable_messages] == ["user-1"]
    assert list_local_sessions(subdir, config_home) == [session_id]
    assert latest_local_session_id(subdir, config_home) == session_id
