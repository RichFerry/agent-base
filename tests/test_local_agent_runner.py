"""Example local runner tests.

These tests keep the v0.2 runner slice in the examples layer: no real network,
no new public API, and no bypass permissions by default.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess
from urllib.error import HTTPError, URLError

import pytest

from agent_kernel.model_provider import FakeModelProvider, OpenAIChatModelProvider
from agent_kernel.query_engine import QueryEngine
import examples.local_agent as local_agent
from examples.local_agent import (
    WEB_FETCH_MAX_BYTES_ENV,
    WEB_FETCH_MAX_CHARS_ENV,
    WEB_FETCH_PROVIDER_ENV,
    WEB_FETCH_TIMEOUT_ENV,
    WEB_SEARCH_API_KEY_ENV,
    WEB_SEARCH_MODEL_ENV,
    WEB_SEARCH_PROVIDER_ENV,
    WEB_SEARCH_TIMEOUT_ENV,
    WEB_SEARCH_URL_ENV,
    DEFAULT_LOCAL_CONFIG_NAME,
    LocalConfigError,
    MCPFixtureConfigurationError,
    MissingCredentialsError,
    SkillsConfigurationError,
    WebFetchConfigurationError,
    WebSearchConfigurationError,
    build_local_engine,
    build_web_fetch_handler_from_env,
    build_web_search_handler_from_env,
    discover_local_skills,
    format_web_search_unavailable_message,
    latest_local_session_id,
    list_memory_files,
    list_local_sessions,
    load_local_runner_config,
    load_mcp_fixture,
    main,
    make_http_web_fetch_handler,
    make_stub_web_search_handler,
    read_memory_file,
    run_local_agent_once,
    validate_memory,
    validate_local_skills,
    write_memory_file,
)
from agent_kernel.skills import SkillTool


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    return repo


def _write_skill(skills_root: Path, name: str, body: str) -> Path:
    skill_dir = skills_root / name
    skill_dir.mkdir(parents=True)
    path = skill_dir / "SKILL.md"
    path.write_text(body, encoding="utf-8")
    return path


def _example_mcp_fixture() -> Path:
    return Path(__file__).parents[1] / "examples" / "mcp" / "echo-mcp.json"


class _FakeHTTPResponse:
    def __init__(
        self,
        payload: bytes,
        *,
        status: int = 200,
        reason: str = "OK",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.payload = payload
        self.status = status
        self.reason = reason
        self.headers = headers or {}

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def getcode(self) -> int:
        return self.status

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            return self.payload
        return self.payload[:size]


def test_local_agent_script_can_run_as_file_help() -> None:
    """The example script works when invoked as python3 examples/local_agent.py."""
    repo_root = Path(__file__).parents[1]

    result = subprocess.run(
        ["python3", "examples/local_agent.py", "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Run one prompt through the local Python Agent Kernel" in result.stdout
    assert result.stderr == ""


def test_cli_init_config_writes_starter_config_without_secrets(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """--init-config creates a local JSON config template without storing keys."""
    config_path = tmp_path / DEFAULT_LOCAL_CONFIG_NAME

    exit_code = main(["--cwd", str(tmp_path), "--init-config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert f"wrote {config_path}" in captured.out
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload["provider"]["type"] == "anthropic"
    assert "api" not in json.dumps(payload).lower()
    assert captured.err == ""


def test_cli_doctor_reads_config_without_model_credentials(capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--doctor reports effective local config without calling a model or network."""
    monkeypatch.delenv("AGENT_KERNEL_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    skills_root = tmp_path / "skills"
    _write_skill(skills_root, "echo", "Echo locally.")
    config_path = tmp_path / "agent-kernel.json"
    config_path.write_text(
        json.dumps(
            {
                "provider": {"type": "openai-chat", "model": "gpt-test", "baseUrl": "https://example.invalid"},
                "runner": {"permissionMode": "bypass", "maxTurns": 3},
                "webSearch": {"enabled": True, "provider": "stub"},
                "webFetch": {"enabled": False, "provider": "http"},
                "skills": {"dir": "skills"},
                "mcp": {},
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["--cwd", str(tmp_path), "--agent-config", str(config_path), "--doctor"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert f"config_path={config_path}" in captured.out
    assert "provider=openai-chat" in captured.out
    assert "model=gpt-test" in captured.out
    assert "credentials=missing" in captured.out
    assert "permission_mode=bypass" in captured.out
    assert "max_turns=3" in captured.out
    assert f"skills_dir={skills_root} status=ok type=dir" in captured.out
    assert captured.err == ""


def test_config_skills_dir_feeds_list_skills(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Config file defaults can drive local inspection commands before provider setup."""
    skills_root = tmp_path / "skills"
    _write_skill(
        skills_root,
        "echo",
        """---
name: echo
description: Echo from config
---
Echo.
""",
    )
    config_path = tmp_path / "agent-kernel.json"
    config_path.write_text(json.dumps({"skills": {"dir": "skills"}}), encoding="utf-8")

    exit_code = main(["--cwd", str(tmp_path), "--agent-config", str(config_path), "--list-skills"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "echo\tEcho from config" in captured.out
    assert captured.err == ""


def test_cli_flags_override_config_for_doctor(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Explicit CLI flags override persisted local config defaults."""
    config_path = tmp_path / "agent-kernel.json"
    config_path.write_text(json.dumps({"runner": {"permissionMode": "bypass", "maxTurns": 3}}), encoding="utf-8")

    exit_code = main(
        [
            "--cwd",
            str(tmp_path),
            "--agent-config",
            str(config_path),
            "--permission-mode",
            "ask",
            "--max-turns",
            "7",
            "--doctor",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "permission_mode=ask" in captured.out
    assert "max_turns=7" in captured.out


def test_settings_json_project_user_explicit_precedence_and_env_override(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """settings.json layers merge deterministically and AGENT_KERNEL_* env wins over settings."""
    repo = _repo(tmp_path)
    config_home = tmp_path / ".claude"
    config_home.mkdir()
    (config_home / DEFAULT_LOCAL_CONFIG_NAME).write_text(
        json.dumps({"provider": {"type": "anthropic", "model": "user-model"}, "runner": {"maxTurns": 2}}),
        encoding="utf-8",
    )
    (repo / DEFAULT_LOCAL_CONFIG_NAME).write_text(
        json.dumps({"provider": {"type": "openai-chat", "model": "project-model"}, "runner": {"permissionMode": "bypass"}}),
        encoding="utf-8",
    )
    explicit = tmp_path / "explicit-settings.json"
    explicit.write_text(json.dumps({"runner": {"maxTurns": 5}}), encoding="utf-8")
    monkeypatch.setenv("AGENT_KERNEL_MODEL", "env-model")

    exit_code = main(["--cwd", str(repo), "--config-home", str(config_home), "--agent-config", str(explicit), "--print-effective-config"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["provider"]["type"] == "openai-chat"
    assert payload["provider"]["model"] == "env-model"
    assert payload["runner"]["permissionMode"] == "bypass"
    assert payload["runner"]["maxTurns"] == 5
    assert [Path(path).name for path in payload["configPaths"]] == [DEFAULT_LOCAL_CONFIG_NAME, DEFAULT_LOCAL_CONFIG_NAME, "explicit-settings.json"]
    assert captured.err == ""


def test_settings_json_rejects_secret_like_supported_fields(tmp_path: Path) -> None:
    """Runner settings reject API keys in supported sections while ignoring unrelated TS settings sections."""
    repo = _repo(tmp_path)
    config_home = tmp_path / ".claude"
    config_home.mkdir()
    (config_home / DEFAULT_LOCAL_CONFIG_NAME).write_text(json.dumps({"env": {"ANTHROPIC_AUTH_TOKEN": "sk-not-for-this-runner-123456789"}}), encoding="utf-8")
    assert load_local_runner_config(cwd=repo, config_home=config_home).paths == ((config_home / DEFAULT_LOCAL_CONFIG_NAME),)

    bad = repo / DEFAULT_LOCAL_CONFIG_NAME
    bad.write_text(json.dumps({"provider": {"apiKey": "sk-should-not-be-here-123456789"}}), encoding="utf-8")
    with pytest.raises(LocalConfigError, match="looks like a secret"):
        load_local_runner_config(cwd=repo, config_home=config_home)


def test_settings_json_rejects_unsafe_memory_default_path(tmp_path: Path) -> None:
    """memory.defaultPath is config, but still follows memory path confinement."""
    repo = _repo(tmp_path)
    (repo / DEFAULT_LOCAL_CONFIG_NAME).write_text(json.dumps({"memory": {"defaultPath": "../outside.md"}}), encoding="utf-8")

    with pytest.raises(LocalConfigError, match="memory.defaultPath must be a relative path"):
        load_local_runner_config(cwd=repo, config_home=tmp_path / ".claude")


def test_settings_memory_defaults_drive_cli_and_engine(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """memory.enabled/defaultPath in settings affect local CLI memory defaults and kernel config."""
    repo = _repo(tmp_path)
    config_home = tmp_path / ".claude"
    (repo / DEFAULT_LOCAL_CONFIG_NAME).write_text(
        json.dumps({"memory": {"enabled": False, "defaultPath": "NOTES.md"}}),
        encoding="utf-8",
    )
    engine = build_local_engine(
        cwd=repo,
        config_home=config_home,
        model_provider=FakeModelProvider(["ok"]),
        memory_enabled=False,
        require_api_key=False,
    )

    status_code = main(["memory", "status", "--cwd", str(repo), "--config-home", str(config_home)])
    status = capsys.readouterr()
    write_code = main(["--cwd", str(repo), "--config-home", str(config_home), "--memory-write", "NOTES.md", "--memory-text", "custom index"])
    capsys.readouterr()
    read_code = main(["memory", "read", "--cwd", str(repo), "--config-home", str(config_home)])
    read = capsys.readouterr()

    assert engine.config.auto_memory_enabled is False
    assert status_code == 0
    assert "enabled=false" in status.out
    assert "entrypoint_exists=false" in status.out
    assert "NOTES.md" in status.out
    assert write_code == 0
    assert read_code == 0
    assert read.out == "custom index"


def test_settings_debug_defaults_drive_debug_outputs(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """debug.provider in settings behaves like --debug-provider without requiring credentials."""
    repo = _repo(tmp_path)
    (repo / DEFAULT_LOCAL_CONFIG_NAME).write_text(
        json.dumps({"provider": {"type": "openai-chat", "model": "gpt-test"}, "debug": {"provider": True}}),
        encoding="utf-8",
    )

    exit_code = main(["--cwd", str(repo), "--config-home", str(tmp_path / ".claude")])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["type"] == "openai-chat"
    assert payload["model"] == "gpt-test"
    assert payload["credentials"] == "missing"


def test_settings_mcp_timeouts_pass_to_config_loader(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """mcp.startupTimeout/toolTimeout in settings are passed to local stdio config loading."""
    repo = _repo(tmp_path)
    mcp_config = repo / "mcp.json"
    mcp_config.write_text(json.dumps({"mcpServers": {"fake": {"command": "python3", "args": ["server.py"]}}}), encoding="utf-8")
    (repo / DEFAULT_LOCAL_CONFIG_NAME).write_text(
        json.dumps({"mcp": {"configs": ["mcp.json"], "startupTimeout": 1.25, "toolTimeout": 2.5}}),
        encoding="utf-8",
    )
    captured_calls: list[dict[str, object]] = []

    def fake_load_mcp_config(path, *, cwd=None, startup_timeout_seconds=None, tool_timeout_seconds=None):
        captured_calls.append(
            {
                "path": Path(path),
                "cwd": cwd,
                "startup_timeout_seconds": startup_timeout_seconds,
                "tool_timeout_seconds": tool_timeout_seconds,
            }
        )
        return ()

    monkeypatch.setattr(local_agent, "load_mcp_config", fake_load_mcp_config)

    exit_code = main(["mcp", "doctor", "--cwd", str(repo), "--config-home", str(tmp_path / ".claude")])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "mcp ok (0 servers)" in captured.out
    assert captured_calls == [
        {
            "path": mcp_config,
            "cwd": repo,
            "startup_timeout_seconds": 1.25,
            "tool_timeout_seconds": 2.5,
        }
    ]


def test_cli_management_aliases_for_config_doctor_and_effective_config(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Subcommand-style aliases map to the same non-mutating config diagnostics."""
    repo = _repo(tmp_path)
    (repo / DEFAULT_LOCAL_CONFIG_NAME).write_text(json.dumps({"runner": {"permissionMode": "bypass"}}), encoding="utf-8")

    doctor_code = main(["config", "doctor", "--cwd", str(repo)])
    doctor = capsys.readouterr()
    effective_code = main(["config", "effective", "--cwd", str(repo)])
    effective = capsys.readouterr()

    assert doctor_code == 0
    assert "permission_mode=bypass" in doctor.out
    assert effective_code == 0
    assert json.loads(effective.out)["runner"]["permissionMode"] == "bypass"


def test_build_local_engine_uses_provider_env_without_global_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Runner config/env can select a provider without mutating process env."""
    monkeypatch.delenv("AGENT_KERNEL_PROVIDER", raising=False)
    provider_env = {
        "AGENT_KERNEL_PROVIDER": "openai-chat",
        "AGENT_KERNEL_API_KEY": "test-key",
        "AGENT_KERNEL_MODEL": "gpt-test",
    }

    engine = build_local_engine(
        cwd=_repo(tmp_path),
        config_home=tmp_path / ".claude",
        provider_env=provider_env,
    )

    assert isinstance(engine.model_provider, OpenAIChatModelProvider)
    assert engine.model_provider.model == "gpt-test"
    assert os.environ.get("AGENT_KERNEL_PROVIDER") is None


def test_build_local_engine_uses_query_engine_and_safe_default_permissions(tmp_path: Path) -> None:
    """The example helper assembles QueryEngine without switching to bypass."""
    provider = FakeModelProvider(["ok"])
    repo = _repo(tmp_path)
    _write_skill(
        repo / ".claude" / "skills",
        "ambient",
        """---
name: ambient
description: Ambient project skill
---
This should not load unless passed through skills_dir.
""",
    )
    engine = build_local_engine(
        cwd=repo,
        config_home=tmp_path / ".claude",
        model_provider=provider,
        session_id="local-session",
        require_api_key=False,
    )

    assert isinstance(engine, QueryEngine)
    assert engine.session_id == "local-session"
    assert engine.model_provider is provider
    assert engine.tool_use_context.app_state.tool_permission_context.mode == "ask"
    assert engine.tool_use_context.web_search_handler is None
    assert engine.skills == []
    assert not any(isinstance(tool, SkillTool) for tool in engine.tools)
    assert engine.config.mcp_clients == ()
    assert not any(tool.name.startswith("mcp__") for tool in engine.tools)
    assert {"Read", "Write"}.issubset({tool.name for tool in engine.tools})


def test_build_local_engine_injects_web_search_handler_and_bypass_mode(tmp_path: Path) -> None:
    """WebSearch injection stays in the example layer and bypass is only pass-through."""
    provider = FakeModelProvider(["ok"])
    search_handler = make_stub_web_search_handler()
    engine = build_local_engine(
        cwd=_repo(tmp_path),
        config_home=tmp_path / ".claude",
        model_provider=provider,
        web_search_handler=search_handler,
        permission_mode="bypass",
        require_api_key=False,
    )

    assert engine.tool_use_context.web_search_handler is search_handler
    assert engine.tool_use_context.app_state.tool_permission_context.mode == "bypass"


def test_build_web_search_handler_from_env_uses_stub_results_file(tmp_path: Path) -> None:
    """The example env adapter is deterministic and never performs network I/O."""
    results_path = tmp_path / "search-results.json"
    results_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "content": [
                            {
                                "title": "Python Downloads",
                                "url": "https://www.python.org/downloads/",
                            }
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    handler = build_web_search_handler_from_env(
        {
            WEB_SEARCH_PROVIDER_ENV: "stub",
            "AGENT_KERNEL_WEB_SEARCH_STUB_RESULTS": str(results_path),
        }
    )

    assert handler is not None
    result = handler({"query": "latest Python release"})
    assert result["query"] == "latest Python release"
    assert result["results"][0]["content"][0]["title"] == "Python Downloads"


def test_build_web_search_handler_from_env_uses_http_json_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """The opt-in http-json adapter maps provider JSON into WebSearch results."""
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeHTTPResponse(
            json.dumps(
                {
                    "items": [
                        {
                            "title": "Adapter Result",
                            "url": "https://example.invalid/search",
                            "snippet": "Adapter result summary.",
                        }
                    ]
                }
            ).encode("utf-8")
        )

    monkeypatch.setattr(local_agent, "urlopen", fake_urlopen)

    handler = build_web_search_handler_from_env(
        {
            WEB_SEARCH_PROVIDER_ENV: "http-json",
            WEB_SEARCH_URL_ENV: "https://search.example.invalid/query",
            WEB_SEARCH_API_KEY_ENV: "fake-search-key",
            WEB_SEARCH_TIMEOUT_ENV: "3.5",
        }
    )

    assert handler is not None
    result = handler({"query": "adapter smoke", "allowed_domains": ["example.invalid"]})

    assert captured["url"] == "https://search.example.invalid/query"
    assert captured["timeout"] == 3.5
    assert captured["body"] == {"query": "adapter smoke", "allowed_domains": ["example.invalid"]}
    assert captured["headers"]["Authorization"] == "Bearer fake-search-key"
    assert result["query"] == "adapter smoke"
    assert result["results"][0]["content"][0]["title"] == "Adapter Result"
    assert result["results"][0]["content"][0]["url"] == "https://example.invalid/search"


def test_web_search_http_json_normalizes_common_backend_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    """HTTP JSON adapter accepts common result containers and field aliases."""
    payloads = [
        {"results": [{"title": "Result A", "url": "https://example.invalid/a", "snippet": "Snippet A"}]},
        {"items": [{"name": "Result B", "link": "https://example.invalid/b", "content": "Snippet B"}]},
        {"data": {"results": [{"title": "Result C", "href": "https://example.invalid/c", "summary": "Snippet C"}]}},
        [{"name": "Result D", "href": "https://example.invalid/d", "text": "Snippet D"}],
    ]

    def fake_urlopen(request, timeout):
        return _FakeHTTPResponse(json.dumps(payloads.pop(0)).encode("utf-8"))

    monkeypatch.setattr(local_agent, "urlopen", fake_urlopen)
    handler = build_web_search_handler_from_env(
        {
            WEB_SEARCH_PROVIDER_ENV: "http-json",
            WEB_SEARCH_URL_ENV: "https://search.example.invalid/query",
        }
    )

    assert handler is not None
    results = [handler({"query": f"shape {index}"}) for index in range(4)]

    assert results[0]["results"][0]["content"][0] == {
        "title": "Result A",
        "url": "https://example.invalid/a",
        "snippet": "Snippet A",
    }
    assert results[1]["results"][0]["content"][0] == {
        "title": "Result B",
        "url": "https://example.invalid/b",
        "snippet": "Snippet B",
    }
    assert results[2]["results"][0]["content"][0] == {
        "title": "Result C",
        "url": "https://example.invalid/c",
        "snippet": "Snippet C",
    }
    assert results[3]["results"][0]["content"][0] == {
        "title": "Result D",
        "url": "https://example.invalid/d",
        "snippet": "Snippet D",
    }


def test_http_json_web_search_provider_requires_url() -> None:
    """http-json is explicit and fails clearly without a configured endpoint."""
    with pytest.raises(WebSearchConfigurationError, match=WEB_SEARCH_URL_ENV):
        build_web_search_handler_from_env({WEB_SEARCH_PROVIDER_ENV: "http-json"})


def test_http_json_web_search_provider_errors_are_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provider HTTP and JSON failures become readable tool errors."""

    def fake_http_error(request, timeout):
        raise HTTPError(request.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr(local_agent, "urlopen", fake_http_error)
    handler = build_web_search_handler_from_env(
        {
            WEB_SEARCH_PROVIDER_ENV: "http-json",
            WEB_SEARCH_URL_ENV: "https://search.example.invalid/query",
        }
    )
    assert handler is not None
    with pytest.raises(RuntimeError, match="HTTP 401"):
        handler({"query": "adapter smoke"})

    monkeypatch.setattr(local_agent, "urlopen", lambda request, timeout: _FakeHTTPResponse(b"{not-json"))
    with pytest.raises(RuntimeError, match="invalid JSON"):
        handler({"query": "adapter smoke"})


def test_build_web_search_handler_from_env_uses_anthropic_compatible_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """The opt-in anthropic-compatible adapter maps search blocks into WebSearch results."""
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeHTTPResponse(
            json.dumps(
                {
                    "id": "msg_search",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Found one result."},
                        {
                            "type": "web_search_tool_result",
                            "tool_use_id": "srvu_search",
                            "content": [
                                {
                                    "title": "Search Adapter Result",
                                    "url": "https://example.invalid/adapter",
                                }
                            ],
                        },
                    ],
                }
            ).encode("utf-8")
        )

    monkeypatch.setattr(local_agent, "urlopen", fake_urlopen)

    handler = build_web_search_handler_from_env(
        {
            WEB_SEARCH_PROVIDER_ENV: "anthropic-compatible",
            WEB_SEARCH_URL_ENV: "https://search.example.invalid/anthropic",
            WEB_SEARCH_API_KEY_ENV: "fake-search-key",
            WEB_SEARCH_MODEL_ENV: "search-model",
            WEB_SEARCH_TIMEOUT_ENV: "4",
        }
    )

    assert handler is not None
    result = handler({"query": "adapter smoke", "blocked_domains": ["blocked.example"]})

    body = captured["body"]
    headers = captured["headers"]
    assert captured["url"] == "https://search.example.invalid/anthropic/v1/messages"
    assert captured["timeout"] == 4.0
    assert headers["Authorization"] == "Bearer fake-search-key"
    assert body["model"] == "search-model"
    assert body["stream"] is False
    assert body["messages"][0]["role"] == "user"
    assert "adapter smoke" in body["messages"][0]["content"][0]["text"]
    assert body["tools"][0]["type"] == "web_search_20250305"
    assert body["tools"][0]["blocked_domains"] == ["blocked.example"]
    assert result["query"] == "adapter smoke"
    assert result["results"][0]["content"][0]["title"] == "Search Adapter Result"
    assert result["results"][0]["content"][0]["url"] == "https://example.invalid/adapter"
    assert result["durationSeconds"] >= 0


def test_anthropic_compatible_web_search_provider_requires_url_key_and_model() -> None:
    """anthropic-compatible stays explicit and fails clearly before network I/O."""
    with pytest.raises(WebSearchConfigurationError, match=WEB_SEARCH_URL_ENV):
        build_web_search_handler_from_env(
            {
                WEB_SEARCH_PROVIDER_ENV: "anthropic-compatible",
                WEB_SEARCH_API_KEY_ENV: "fake-search-key",
                WEB_SEARCH_MODEL_ENV: "search-model",
            }
        )
    with pytest.raises(WebSearchConfigurationError, match=WEB_SEARCH_API_KEY_ENV):
        build_web_search_handler_from_env(
            {
                WEB_SEARCH_PROVIDER_ENV: "anthropic-compatible",
                WEB_SEARCH_URL_ENV: "https://search.example.invalid/anthropic",
                WEB_SEARCH_MODEL_ENV: "search-model",
            }
        )
    with pytest.raises(WebSearchConfigurationError, match=WEB_SEARCH_MODEL_ENV):
        build_web_search_handler_from_env(
            {
                WEB_SEARCH_PROVIDER_ENV: "anthropic-compatible",
                WEB_SEARCH_URL_ENV: "https://search.example.invalid/anthropic",
                WEB_SEARCH_API_KEY_ENV: "fake-search-key",
            }
        )


def test_anthropic_compatible_web_search_provider_synthetic_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """Natural-language provider output is wrapped into a bounded synthetic result."""

    def fake_urlopen(request, timeout):
        return _FakeHTTPResponse(
            json.dumps(
                {
                    "id": "msg_search",
                    "content": [
                        {
                            "type": "text",
                            "text": "This is a natural language answer from the search-capable backend.",
                        }
                    ],
                }
            ).encode("utf-8")
        )

    monkeypatch.setattr(local_agent, "urlopen", fake_urlopen)
    handler = build_web_search_handler_from_env(
        {
            WEB_SEARCH_PROVIDER_ENV: "anthropic-compatible",
            WEB_SEARCH_URL_ENV: "https://search.example.invalid/anthropic/v1",
            WEB_SEARCH_API_KEY_ENV: "fake-search-key",
            WEB_SEARCH_MODEL_ENV: "search-model",
        }
    )
    assert handler is not None

    result = handler({"query": "adapter smoke"})

    synthetic = result["results"][0]["content"][0]
    assert synthetic["title"] == "Anthropic-compatible search result"
    assert synthetic["url"] == ""
    assert "natural language answer" in synthetic["snippet"]


def test_anthropic_compatible_web_search_provider_error_does_not_leak_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """HTTP failures are readable and do not include the configured API key."""
    secret = "fake-search-key"

    def fake_http_error(request, timeout):
        raise HTTPError(request.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr(local_agent, "urlopen", fake_http_error)
    handler = build_web_search_handler_from_env(
        {
            WEB_SEARCH_PROVIDER_ENV: "anthropic-compatible",
            WEB_SEARCH_URL_ENV: "https://search.example.invalid/anthropic",
            WEB_SEARCH_API_KEY_ENV: secret,
            WEB_SEARCH_MODEL_ENV: "search-model",
        }
    )
    assert handler is not None

    with pytest.raises(RuntimeError) as excinfo:
        handler({"query": "adapter smoke"})

    assert "HTTP 401" in str(excinfo.value)
    assert secret not in str(excinfo.value)


def test_run_local_agent_once_uses_anthropic_compatible_web_search_handler(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The adapter is invoked through WebSearchTool and enters normal tool_result events."""
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeHTTPResponse(
            json.dumps(
                {
                    "id": "msg_search",
                    "content": [
                        {
                            "type": "web_search_tool_result",
                            "tool_use_id": "srvu_search",
                            "content": [
                                {
                                    "title": "Kernel Search Result",
                                    "url": "https://example.invalid/kernel",
                                }
                            ],
                        }
                    ],
                }
            ).encode("utf-8")
        )

    monkeypatch.setattr(local_agent, "urlopen", fake_urlopen)
    search_handler = build_web_search_handler_from_env(
        {
            WEB_SEARCH_PROVIDER_ENV: "anthropic-compatible",
            WEB_SEARCH_URL_ENV: "https://search.example.invalid/anthropic",
            WEB_SEARCH_API_KEY_ENV: "fake-search-key",
            WEB_SEARCH_MODEL_ENV: "search-model",
        }
    )
    provider = FakeModelProvider(
        [
            [
                {
                    "type": "tool_use",
                    "id": "toolu_search",
                    "name": "WebSearch",
                    "input": {"query": "kernel search"},
                }
            ],
            "Search completed.",
        ]
    )
    engine = build_local_engine(
        cwd=_repo(tmp_path),
        config_home=tmp_path / ".claude",
        model_provider=provider,
        web_search_handler=search_handler,
        permission_mode="bypass",
        require_api_key=False,
    )

    result = asyncio.run(run_local_agent_once("search", engine=engine, max_turns=3))
    tool_result = next(
        event["message"]["content"][0]
        for event in result.events
        if event.get("type") == "user" and event["message"]["content"][0].get("type") == "tool_result"
    )

    assert captured["body"]["model"] == "search-model"
    assert result.final_response == "Search completed."
    assert "Kernel Search Result" in tool_result["content"]
    assert "https://example.invalid/kernel" in tool_result["content"]
    assert any("[tool_use] WebSearch" in line for line in result.logs)
    assert any("[tool_result:ok] toolu_search" in line for line in result.logs)


def test_build_local_engine_defaults_web_fetch_to_no_network(tmp_path: Path) -> None:
    """The local runner disables WebFetch network access unless explicitly configured."""
    engine = build_local_engine(
        cwd=_repo(tmp_path),
        config_home=tmp_path / ".claude",
        model_provider=FakeModelProvider(["ok"]),
        require_api_key=False,
    )

    assert engine.tool_use_context.web_fetch_handler is not None
    with pytest.raises(RuntimeError, match="WebFetch is not configured"):
        engine.tool_use_context.web_fetch_handler("https://docs.python.org/3/")


def test_build_web_fetch_handler_from_env_uses_http_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """The opt-in WebFetch HTTP adapter uses standard-library GET with limits."""
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.header_items())
        return _FakeHTTPResponse(
            b"# Python docs\nHello",
            headers={"content-type": "text/markdown; charset=utf-8"},
        )

    monkeypatch.setattr(local_agent, "urlopen", fake_urlopen)
    handler = build_web_fetch_handler_from_env(
        {
            WEB_FETCH_PROVIDER_ENV: "http",
            WEB_FETCH_TIMEOUT_ENV: "2.5",
            WEB_FETCH_MAX_BYTES_ENV: "100",
            WEB_FETCH_MAX_CHARS_ENV: "100",
        }
    )

    assert handler is not None
    result = handler("https://docs.python.org/3/")

    assert captured["url"] == "https://docs.python.org/3/"
    assert captured["timeout"] == 2.5
    assert captured["headers"]["User-agent"] == "agent-kernel-local-runner/0.3"
    assert result["bytes"] == len(b"# Python docs\nHello")
    assert result["content"] == "# Python docs\nHello"
    assert result["contentType"] == "text/markdown; charset=utf-8"
    assert result["code"] == 200


def test_web_fetch_http_provider_errors_are_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid URL, timeout, and oversized responses fail with readable messages."""
    handler = make_http_web_fetch_handler(timeout_seconds=1, max_bytes=4, max_chars=20)

    with pytest.raises(RuntimeError, match="Only http and https"):
        handler("ftp://example.invalid/file")

    monkeypatch.setattr(local_agent, "urlopen", lambda request, timeout: (_ for _ in ()).throw(TimeoutError()))
    with pytest.raises(RuntimeError, match="timed out"):
        handler("https://docs.python.org/3/")

    monkeypatch.setattr(local_agent, "urlopen", lambda request, timeout: _FakeHTTPResponse(b"12345"))
    with pytest.raises(RuntimeError, match="byte limit"):
        handler("https://docs.python.org/3/")

    monkeypatch.setattr(local_agent, "urlopen", lambda request, timeout: _FakeHTTPResponse("abcdef".encode("utf-8")))
    char_limited = make_http_web_fetch_handler(timeout_seconds=1, max_bytes=100, max_chars=3)
    with pytest.raises(RuntimeError, match="character limit"):
        char_limited("https://docs.python.org/3/")

    monkeypatch.setattr(local_agent, "urlopen", lambda request, timeout: (_ for _ in ()).throw(URLError("offline")))
    with pytest.raises(RuntimeError, match="offline"):
        make_http_web_fetch_handler()("https://docs.python.org/3/")


def test_run_local_agent_once_injects_web_fetch_handler_without_preflight(tmp_path: Path) -> None:
    """A WebFetch tool_use can call an injected handler and return a normal tool_result."""
    calls: list[str] = []

    def fetch_handler(url: str) -> dict[str, object]:
        calls.append(url)
        return {
            "bytes": 18,
            "code": 200,
            "codeText": "OK",
            "content": "# Python docs\nHello",
            "contentType": "text/markdown",
            "url": url,
        }

    provider = FakeModelProvider(
        [
            [
                {
                    "type": "tool_use",
                    "id": "toolu_fetch",
                    "name": "WebFetch",
                    "input": {"url": "https://docs.python.org/3/", "prompt": "summarize"},
                }
            ],
            "Fetched docs.",
        ]
    )
    engine = build_local_engine(
        cwd=_repo(tmp_path),
        config_home=tmp_path / ".claude",
        model_provider=provider,
        web_fetch_handler=fetch_handler,
        permission_mode="bypass",
        require_api_key=False,
    )

    result = asyncio.run(run_local_agent_once("fetch docs", engine=engine, max_turns=3))
    rows = [json.loads(line) for line in result.transcript_path.read_text(encoding="utf-8").splitlines()]
    tool_result = next(
        event["message"]["content"][0]
        for event in result.events
        if event.get("type") == "user" and event["message"]["content"][0].get("type") == "tool_result"
    )

    assert calls == ["https://docs.python.org/3/"]
    assert tool_result["content"] == "# Python docs\nHello"
    assert result.final_response == "Fetched docs."
    assert any("[tool_use] WebFetch" in line for line in result.logs)
    assert any("[tool_result:ok] toolu_fetch" in line for line in result.logs)
    assert not any("preflight" in json.dumps(event, ensure_ascii=False).lower() for event in result.events)
    assert not any("preflight" in json.dumps(row, ensure_ascii=False).lower() for row in rows)


def test_web_fetch_without_provider_returns_clear_unavailable_message(tmp_path: Path) -> None:
    """Default local runner WebFetch path fails closed instead of performing network I/O."""
    provider = FakeModelProvider(
        [
            [
                {
                    "type": "tool_use",
                    "id": "toolu_fetch",
                    "name": "WebFetch",
                    "input": {"url": "https://docs.python.org/3/", "prompt": "summarize"},
                }
            ],
            "Fetch unavailable.",
        ]
    )
    engine = build_local_engine(
        cwd=_repo(tmp_path),
        config_home=tmp_path / ".claude",
        model_provider=provider,
        permission_mode="bypass",
        require_api_key=False,
    )

    result = asyncio.run(run_local_agent_once("fetch docs", engine=engine, max_turns=3))
    tool_result = next(
        event["message"]["content"][0]
        for event in result.events
        if event.get("type") == "user" and event["message"]["content"][0].get("type") == "tool_result"
    )

    assert tool_result["is_error"] is True
    assert "WebFetch is not configured" in tool_result["content"]


def test_build_local_engine_loads_skills_dir(tmp_path: Path) -> None:
    """The runner can opt in to local-only skills via skills_dir."""
    skills_root = tmp_path / "skills"
    skill_path = _write_skill(
        skills_root,
        "echo",
        """---
name: echo
description: Echo text locally
---
Echo the arguments exactly: $ARGUMENTS
""",
    )
    engine = build_local_engine(
        cwd=_repo(tmp_path),
        config_home=tmp_path / ".claude",
        model_provider=FakeModelProvider(["ok"]),
        skills_dir=skills_root,
        require_api_key=False,
    )

    assert [skill.name for skill in discover_local_skills(skills_root)] == ["echo"]
    assert engine.skills[0].name == "echo"
    assert engine.skills[0].base_dir == skill_path.parent
    assert engine.config.skill_discovery_mode == "explicit"
    assert any(isinstance(tool, SkillTool) for tool in engine.tools)
    assert engine.tool_use_context.app_state.tool_permission_context.mode == "ask"


def test_discover_local_skills_reports_duplicate_names(tmp_path: Path) -> None:
    """Duplicate local skill names fail before they can collide in the registry."""
    skills_root = tmp_path / "skills"
    _write_skill(
        skills_root,
        "first",
        """---
name: echo
description: First echo
---
Echo first.
""",
    )
    _write_skill(
        skills_root,
        "second",
        """---
name: echo
description: Second echo
---
Echo second.
""",
    )

    with pytest.raises(SkillsConfigurationError, match="Duplicate skill name 'echo'"):
        discover_local_skills(skills_root)


def test_skills_multi_dir_validation_json_and_info_aliases(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """v0.5 skills commands support multiple dirs, JSON list output, validation, and info lookup."""
    first = tmp_path / "skills-a"
    second = tmp_path / "skills-b"
    _write_skill(
        first,
        "echo",
        """---
name: echo
description: Echo skill
---
Echo.
""",
    )
    _write_skill(
        second,
        "summarize",
        """---
name: summarize
description: Summarize skill
---
Summarize.
""",
    )

    report = validate_local_skills((first, second))
    assert [skill.name for skill in report.skills] == ["echo", "summarize"]

    list_code = main(["skills", "list", "--skills-dir", str(first), "--skills-dir", str(second), "--json"])
    listed = capsys.readouterr()
    validate_code = main(["skills", "validate", "--skills-dir", str(first), "--skills-dir", str(second)])
    validated = capsys.readouterr()
    info_code = main(["skills", "info", "summarize", "--skills-dir", str(first), "--skills-dir", str(second)])
    info = capsys.readouterr()

    assert list_code == 0
    assert [skill["name"] for skill in json.loads(listed.out)["skills"]] == ["echo", "summarize"]
    assert validate_code == 0
    assert "skills ok (2)" in validated.out
    assert info_code == 0
    assert json.loads(info.out)["name"] == "summarize"


def test_skills_strict_validation_reports_fork_and_unknown_frontmatter(tmp_path: Path) -> None:
    """Strict skill validation turns unsupported fork/extra frontmatter into clear errors."""
    skills_root = tmp_path / "skills"
    _write_skill(
        skills_root,
        "forked",
        """---
name: forked
description: Forked
context: fork
extra-key: value
---
Forked.
""",
    )

    report = validate_local_skills((skills_root,))
    assert any("forked skills are not implemented" in warning for warning in report.warnings)
    with pytest.raises(SkillsConfigurationError, match="context=fork"):
        validate_local_skills((skills_root,), strict=True)


def test_build_local_engine_loads_mcp_fixture(tmp_path: Path) -> None:
    """The runner can opt in to local-only MCP tools via mcp_fixture."""
    engine = build_local_engine(
        cwd=_repo(tmp_path),
        config_home=tmp_path / ".claude",
        model_provider=FakeModelProvider(["ok"]),
        mcp_fixture=_example_mcp_fixture(),
        require_api_key=False,
    )
    init = engine.get_system_init_message()

    assert engine.config.mcp_clients[0].name == "local-echo"
    assert "mcp__local-echo__echo" in {tool.name for tool in engine.tools}
    assert "ListMcpResourcesTool" in {tool.name for tool in engine.tools}
    assert "ReadMcpResourceTool" in {tool.name for tool in engine.tools}
    assert init["mcp_servers"] == [{"name": "local-echo", "status": "connected"}]
    assert engine.tool_use_context.app_state.tool_permission_context.mode == "ask"


def test_build_local_engine_skills_dir_errors_are_clear(tmp_path: Path) -> None:
    """Missing or invalid local skills directories fail before model setup."""
    missing = tmp_path / "missing-skills"
    invalid = tmp_path / "invalid-skills"
    (invalid / "broken").mkdir(parents=True)

    with pytest.raises(SkillsConfigurationError, match="does not exist"):
        build_local_engine(
            cwd=_repo(tmp_path),
            config_home=tmp_path / ".claude",
            model_provider=FakeModelProvider(["ok"]),
            skills_dir=missing,
            require_api_key=False,
        )

    with pytest.raises(SkillsConfigurationError, match="No valid skills found"):
        build_local_engine(
            cwd=_repo(tmp_path),
            config_home=tmp_path / ".claude-invalid",
            model_provider=FakeModelProvider(["ok"]),
            skills_dir=invalid,
            require_api_key=False,
        )


def test_cli_skills_dir_missing_returns_clear_error(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """CLI reports bad skills_dir without requiring model credentials."""
    exit_code = main(["--skills-dir", str(tmp_path / "missing"), "hello"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "Skills directory does not exist" in captured.err
    assert captured.out == ""


def test_cli_list_skills_does_not_require_model_credentials(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """--list-skills inspects local skills and exits before provider setup."""
    skills_root = tmp_path / "skills"
    _write_skill(
        skills_root,
        "echo",
        """---
name: echo
description: Echo text locally
---
Echo the arguments exactly.
""",
    )

    exit_code = main(["--skills-dir", str(skills_root), "--list-skills"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "echo\tEcho text locally" in captured.out
    assert captured.err == ""


def test_cli_list_skills_without_dir_is_deterministic(capsys: pytest.CaptureFixture[str]) -> None:
    """Without --skills-dir, --list-skills reports that no explicit skills are loaded."""
    exit_code = main(["--list-skills"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "No skills loaded. Pass --skills-dir PATH" in captured.out
    assert captured.err == ""


def test_list_sessions_and_continue_are_deterministic(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Session listing is read-only and --continue can target the latest transcript."""
    repo = _repo(tmp_path)
    config_home = tmp_path / ".claude"
    first = build_local_engine(
        cwd=repo,
        config_home=config_home,
        model_provider=FakeModelProvider(["first"]),
        session_id="session-a",
        require_api_key=False,
    )
    second = build_local_engine(
        cwd=repo,
        config_home=config_home,
        model_provider=FakeModelProvider(["second"]),
        session_id="session-b",
        require_api_key=False,
    )
    asyncio.run(run_local_agent_once("first prompt", engine=first))
    asyncio.run(run_local_agent_once("second prompt", engine=second))
    os.utime(first.session_store.transcript_path, (1, 1))
    os.utime(second.session_store.transcript_path, (2, 2))

    assert list_local_sessions(repo, config_home) == ["session-a", "session-b"]
    assert latest_local_session_id(repo, config_home) == "session-b"

    exit_code = main(["--cwd", str(repo), "--config-home", str(config_home), "--list-sessions"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out.splitlines() == ["session-a", "session-b"]
    assert captured.err == ""


def test_build_local_engine_resume_loads_existing_transcript(tmp_path: Path) -> None:
    """Runner resume flag uses the existing QueryEngine transcript restore path."""
    repo = _repo(tmp_path)
    config_home = tmp_path / ".claude"
    first = build_local_engine(
        cwd=repo,
        config_home=config_home,
        model_provider=FakeModelProvider(["first response"]),
        session_id="resume-cli",
        require_api_key=False,
    )
    asyncio.run(run_local_agent_once("first prompt", engine=first))

    resumed = build_local_engine(
        cwd=repo,
        config_home=config_home,
        model_provider=FakeModelProvider(["second response"]),
        session_id="resume-cli",
        resume=True,
        require_api_key=False,
    )
    assert [message["type"] for message in resumed.mutable_messages] == ["user", "assistant"]

    result = asyncio.run(run_local_agent_once("second prompt", engine=resumed))
    rows = [json.loads(line) for line in result.transcript_path.read_text(encoding="utf-8").splitlines()]

    assert [row["type"] for row in rows] == ["user", "assistant", "user", "assistant"]
    assert rows[0]["sessionId"] == "resume-cli"
    assert rows[-1]["sessionId"] == "resume-cli"


def test_memory_cli_read_write_status_and_path_safety(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Memory commands are explicit, local, and confined to the project memory dir."""
    repo = _repo(tmp_path)
    config_home = tmp_path / ".claude"

    status_code = main(["--cwd", str(repo), "--config-home", str(config_home), "--memory-status"])
    status = capsys.readouterr()
    assert status_code == 0
    assert "memory_dir=" in status.out
    assert "entrypoint_exists=false" in status.out

    write_code = main(
        [
            "--cwd",
            str(repo),
            "--config-home",
            str(config_home),
            "--memory-write",
            "notes/preference.md",
            "--memory-text",
            "Use concise answers.",
        ]
    )
    written = capsys.readouterr()
    assert write_code == 0
    assert "wrote " in written.out

    assert read_memory_file("notes/preference.md", cwd=repo, config_home=config_home) == "Use concise answers."
    target = write_memory_file("MEMORY.md", "- [Preference](notes/preference.md) - concise\n", cwd=repo, config_home=config_home)
    assert target.name == "MEMORY.md"

    read_code = main(["--cwd", str(repo), "--config-home", str(config_home), "--memory-read"])
    read = capsys.readouterr()
    assert read_code == 0
    assert "- [Preference](notes/preference.md)" in read.out

    unsafe_code = main(
        [
            "--cwd",
            str(repo),
            "--config-home",
            str(config_home),
            "--memory-write",
            "../escape.md",
            "--memory-text",
            "nope",
        ]
    )
    unsafe = capsys.readouterr()
    assert unsafe_code == 2
    assert "Memory path must stay inside the project memory directory" in unsafe.err


def test_memory_list_skips_symlink_escape(tmp_path: Path) -> None:
    """Memory listing follows the same project-directory confinement as read/write."""
    repo = _repo(tmp_path)
    config_home = tmp_path / ".claude"
    safe_file = write_memory_file("inside.md", "safe", cwd=repo, config_home=config_home)
    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    symlink_path = safe_file.parent / "linked-outside.md"
    try:
        symlink_path.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink creation is not available: {exc}")

    files = list_memory_files(cwd=repo, config_home=config_home)
    issues = validate_memory(cwd=repo, config_home=config_home)

    assert {file["path"] for file in files} == {"inside.md"}
    assert any("linked-outside.md" in issue and "escapes memory directory" in issue for issue in issues)


def test_session_management_info_export_delete_aliases(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Session management aliases expose metadata/export/delete without rewriting transcript rows."""
    repo = _repo(tmp_path)
    config_home = tmp_path / ".claude"
    engine = build_local_engine(
        cwd=repo,
        config_home=config_home,
        model_provider=FakeModelProvider(["session response"]),
        session_id="managed-session",
        require_api_key=False,
    )
    asyncio.run(run_local_agent_once("hello session", engine=engine))

    info_code = main(["sessions", "info", "managed-session", "--cwd", str(repo), "--config-home", str(config_home), "--json"])
    info = capsys.readouterr()
    export_code = main(["sessions", "export", "managed-session", "--cwd", str(repo), "--config-home", str(config_home)])
    exported = capsys.readouterr()
    delete_without_yes = main(["sessions", "delete", "managed-session", "--cwd", str(repo), "--config-home", str(config_home)])
    denied = capsys.readouterr()
    delete_code = main(["sessions", "delete", "managed-session", "--cwd", str(repo), "--config-home", str(config_home), "--yes"])
    deleted = capsys.readouterr()

    payload = json.loads(info.out)
    assert info_code == 0
    assert payload["sessionId"] == "managed-session"
    assert payload["messageCount"] == 2
    assert payload["hasToolResultPairingIssues"] is False
    assert export_code == 0
    assert '"sessionId":"managed-session"' in exported.out
    assert delete_without_yes == 2
    assert "requires --yes" in denied.err
    assert delete_code == 0
    assert "deleted " in deleted.out


def test_memory_management_list_remember_append_validate_and_delete_aliases(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Memory management commands make explicit memory workflows usable."""
    repo = _repo(tmp_path)
    config_home = tmp_path / ".claude"

    remember_code = main(
        [
            "memory",
            "remember",
            "--cwd",
            str(repo),
            "--config-home",
            str(config_home),
            "--memory-type",
            "feedback",
            "--memory-name",
            "Terse Replies",
            "--memory-text",
            "Prefer terse engineering summaries.",
        ]
    )
    remembered = capsys.readouterr()
    list_code = main(["memory", "list", "--cwd", str(repo), "--config-home", str(config_home), "--json"])
    listed = capsys.readouterr()
    append_code = main(
        [
            "memory",
            "append",
            "feedback/terse-replies.md",
            "--cwd",
            str(repo),
            "--config-home",
            str(config_home),
            "--memory-text",
            "\nExtra note.",
        ]
    )
    appended = capsys.readouterr()
    validate_code = main(["memory", "validate", "--cwd", str(repo), "--config-home", str(config_home)])
    validated = capsys.readouterr()
    delete_code = main(["memory", "forget", "feedback/terse-replies.md", "--cwd", str(repo), "--config-home", str(config_home), "--yes"])
    deleted = capsys.readouterr()

    assert remember_code == 0
    assert "remembered " in remembered.out
    assert list_code == 0
    listed_paths = [item["path"] for item in json.loads(listed.out)]
    assert "MEMORY.md" in listed_paths
    assert "feedback/terse-replies.md" in listed_paths
    assert append_code == 0
    assert "appended " in appended.out
    assert validate_code == 0
    assert "memory ok" in validated.out
    assert delete_code == 0
    assert "deleted " in deleted.out


def test_build_local_engine_mcp_fixture_errors_are_clear(tmp_path: Path) -> None:
    """Missing or invalid MCP fixtures fail before model setup."""
    missing = tmp_path / "missing-mcp.json"
    invalid_json = tmp_path / "invalid-mcp.json"
    invalid_json.write_text("{not json", encoding="utf-8")
    invalid_shape = tmp_path / "invalid-shape.json"
    invalid_shape.write_text(json.dumps({"name": "broken", "tools": []}), encoding="utf-8")

    with pytest.raises(MCPFixtureConfigurationError, match="does not exist"):
        build_local_engine(
            cwd=_repo(tmp_path),
            config_home=tmp_path / ".claude-missing-mcp",
            model_provider=FakeModelProvider(["ok"]),
            mcp_fixture=missing,
            require_api_key=False,
        )

    with pytest.raises(MCPFixtureConfigurationError, match="not valid JSON"):
        build_local_engine(
            cwd=_repo(tmp_path),
            config_home=tmp_path / ".claude-invalid-json-mcp",
            model_provider=FakeModelProvider(["ok"]),
            mcp_fixture=invalid_json,
            require_api_key=False,
        )

    with pytest.raises(MCPFixtureConfigurationError, match="non-empty 'tools' array"):
        build_local_engine(
            cwd=_repo(tmp_path),
            config_home=tmp_path / ".claude-invalid-shape-mcp",
            model_provider=FakeModelProvider(["ok"]),
            mcp_fixture=invalid_shape,
            require_api_key=False,
        )


def test_cli_mcp_fixture_missing_returns_clear_error(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """CLI reports bad mcp_fixture without requiring model credentials."""
    exit_code = main(["--mcp-fixture", str(tmp_path / "missing.json"), "hello"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "MCP fixture does not exist" in captured.err
    assert captured.out == ""


def test_mcp_management_list_and_doctor_aliases(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """MCP management aliases inspect explicit local fixtures/configs without a model call."""
    fixture = _example_mcp_fixture()

    list_code = main(["mcp", "list", "--mcp-fixture", str(fixture), "--json"])
    listed = capsys.readouterr()
    doctor_code = main(["mcp", "doctor", "--mcp-fixture", str(fixture)])
    doctored = capsys.readouterr()

    assert list_code == 0
    payload = json.loads(listed.out)
    assert payload["servers"][0]["name"] == "local-echo"
    assert "mcp__local-echo__echo" in payload["servers"][0]["tools"]
    assert doctor_code == 0
    assert "mcp ok (1 servers)" in doctored.out


def test_build_local_engine_rejects_duplicate_mcp_clients(tmp_path: Path) -> None:
    """Multiple MCP capability sources fail clearly if normalized server names collide."""
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    payload = {"name": "same", "tools": [{"name": "echo", "result": "ok"}]}
    first.write_text(json.dumps(payload), encoding="utf-8")
    second.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(Exception, match="collides"):
        build_local_engine(
            cwd=_repo(tmp_path),
            config_home=tmp_path / ".claude",
            model_provider=FakeModelProvider(["ok"]),
            mcp_fixtures=(first, second),
            require_api_key=False,
        )


def test_run_local_agent_once_sends_user_input_through_query_loop(tmp_path: Path) -> None:
    """A single prompt reaches the fake provider and returns assistant final text."""
    provider = FakeModelProvider(["hello from model"])
    engine = build_local_engine(
        cwd=_repo(tmp_path),
        config_home=tmp_path / ".claude",
        model_provider=provider,
        require_api_key=False,
    )

    result = asyncio.run(run_local_agent_once("say hello", engine=engine, max_turns=1))

    serialized_messages = json.dumps(provider.calls[0]["messages"], ensure_ascii=False)
    assert "say hello" in serialized_messages
    assert result.final_response == "hello from model"
    assert result.transcript_path.exists()
    assert any("[sdk:init]" in line for line in result.logs)
    assert any("[sdk:result] success" in line for line in result.logs)


def test_run_local_agent_once_logs_tool_use_and_tool_result(tmp_path: Path) -> None:
    """A fake model tool_use is executed and returned as a tool_result."""
    repo = _repo(tmp_path)
    target = repo / "sample.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")
    provider = FakeModelProvider(
        [
            [
                {
                    "type": "tool_use",
                    "id": "toolu_read",
                    "name": "Read",
                    "input": {"file_path": str(target)},
                }
            ],
            "I read the file.",
        ]
    )
    engine = build_local_engine(
        cwd=repo,
        config_home=tmp_path / ".claude",
        model_provider=provider,
        require_api_key=False,
    )

    result = asyncio.run(run_local_agent_once("read sample", engine=engine, max_turns=3))
    tool_result = next(
        event["message"]["content"][0]
        for event in result.events
        if event.get("type") == "user" and event["message"]["content"][0].get("type") == "tool_result"
    )

    assert result.final_response == "I read the file."
    assert "alpha" in tool_result["content"]
    assert any("[tool_use] Read" in line for line in result.logs)
    assert any("[tool_result:ok] toolu_read" in line for line in result.logs)
    assert len(provider.calls) == 2


def test_run_local_agent_once_invokes_local_skill(tmp_path: Path) -> None:
    """A fake model can invoke a locally loaded inline skill through Skill tool."""
    skills_root = tmp_path / "skills"
    skill_path = _write_skill(
        skills_root,
        "echo",
        """---
name: echo
description: Echo text locally
---
Echo the arguments exactly.
Arguments: $ARGUMENTS
Skill dir: ${CLAUDE_SKILL_DIR}
Session: ${CLAUDE_SESSION_ID}
""",
    )
    provider = FakeModelProvider(
        [
            [{"type": "tool_use", "id": "toolu_skill", "name": "Skill", "input": {"skill": "echo", "args": "hello"}}],
            "hello",
        ]
    )
    engine = build_local_engine(
        cwd=_repo(tmp_path),
        config_home=tmp_path / ".claude",
        model_provider=provider,
        session_id="local-skill-session",
        skills_dir=skills_root,
        require_api_key=False,
    )

    result = asyncio.run(run_local_agent_once("use echo", engine=engine, max_turns=3))
    tool_result = next(
        event["message"]["content"][0]
        for event in result.events
        if event.get("type") == "user" and event["message"]["content"][0].get("type") == "tool_result"
    )
    second_call = json.dumps(provider.calls[1]["messages"], ensure_ascii=False)

    assert result.final_response == "hello"
    assert tool_result["content"] == "Launching skill: echo"
    assert "<command-name>echo</command-name>" in second_call
    assert "<command-args>hello</command-args>" in second_call
    assert f"Base directory for this skill: {skill_path.parent}" in second_call
    assert "Session: local-skill-session" in second_call
    assert any("[tool_use] Skill" in line for line in result.logs)
    assert any("[tool_result:ok] toolu_skill Launching skill: echo" in line for line in result.logs)


def test_run_local_agent_once_invokes_mcp_fixture_tool(tmp_path: Path) -> None:
    """A fake model can invoke a local MCP fixture tool through QueryEngine."""
    provider = FakeModelProvider(
        [
            [
                {
                    "type": "tool_use",
                    "id": "toolu_mcp",
                    "name": "mcp__local-echo__echo",
                    "input": {"text": "hello"},
                }
            ],
            "MCP echoed hello.",
        ]
    )
    engine = build_local_engine(
        cwd=_repo(tmp_path),
        config_home=tmp_path / ".claude",
        model_provider=provider,
        mcp_fixture=_example_mcp_fixture(),
        permission_mode="bypass",
        require_api_key=False,
    )
    call_handler = engine.config.mcp_clients[0].call_tool_handler

    result = asyncio.run(run_local_agent_once("call mcp echo", engine=engine, max_turns=3))
    tool_result = next(
        event["message"]["content"][0]
        for event in result.events
        if event.get("type") == "user" and event["message"]["content"][0].get("type") == "tool_result"
    )
    transcript = result.transcript_path.read_text(encoding="utf-8")

    assert getattr(call_handler, "calls") == [{"tool_name": "echo", "args": {"text": "hello"}}]
    assert tool_result["content"] == '{"echo":"hello","source":"local-mcp-fixture"}'
    assert "MCP echoed hello." == result.final_response
    assert '\\"echo\\":\\"hello\\"' in transcript
    assert any("[tool_use] mcp__local-echo__echo" in line for line in result.logs)
    assert any("[tool_result:ok] toolu_mcp" in line for line in result.logs)
    assert any("mcp=local-echo/echo status=completed" in line for line in result.logs)


def test_mcp_fixture_default_ask_mode_denies_before_handler_call(tmp_path: Path) -> None:
    """Default ask mode stays safe and does not call the MCP fixture handler."""
    provider = FakeModelProvider(
        [
            [
                {
                    "type": "tool_use",
                    "id": "toolu_mcp",
                    "name": "mcp__local-echo__echo",
                    "input": {"text": "hello"},
                }
            ],
            "Denied safely.",
        ]
    )
    engine = build_local_engine(
        cwd=_repo(tmp_path),
        config_home=tmp_path / ".claude",
        model_provider=provider,
        mcp_fixture=_example_mcp_fixture(),
        require_api_key=False,
    )
    call_handler = engine.config.mcp_clients[0].call_tool_handler

    result = asyncio.run(run_local_agent_once("call mcp echo", engine=engine, max_turns=3))
    tool_result = next(
        event["message"]["content"][0]
        for event in result.events
        if event.get("type") == "user" and event["message"]["content"][0].get("type") == "tool_result"
    )

    assert getattr(call_handler, "calls") == []
    assert engine.tool_use_context.app_state.tool_permission_context.mode == "ask"
    assert tool_result["is_error"] is True
    assert "MCPTool requires permission" in tool_result["content"]


def test_run_local_agent_once_injects_web_search_handler(tmp_path: Path) -> None:
    """A WebSearch tool_use can call an injected fake handler and return results."""
    repo = _repo(tmp_path)
    calls = []

    def search_handler(args):
        calls.append(dict(args))
        return [
            {
                "title": "Python Downloads",
                "url": "https://www.python.org/downloads/",
            }
        ]

    provider = FakeModelProvider(
        [
            [
                {
                    "type": "tool_use",
                    "id": "toolu_search",
                    "name": "WebSearch",
                    "input": {"query": "latest Python release"},
                }
            ],
            "Python release info found.",
        ]
    )
    engine = build_local_engine(
        cwd=repo,
        config_home=tmp_path / ".claude",
        model_provider=provider,
        web_search_handler=search_handler,
        permission_mode="bypass",
        require_api_key=False,
    )

    result = asyncio.run(run_local_agent_once("search latest Python release", engine=engine, max_turns=3))
    tool_result = next(
        event["message"]["content"][0]
        for event in result.events
        if event.get("type") == "user" and event["message"]["content"][0].get("type") == "tool_result"
    )

    assert calls == [{"query": "latest Python release"}]
    assert result.final_response == "Python release info found."
    assert "Python Downloads" in tool_result["content"]
    assert "https://www.python.org/downloads/" in tool_result["content"]
    assert any("[tool_use] WebSearch" in line for line in result.logs)
    assert any("[tool_result:ok] toolu_search" in line for line in result.logs)


def test_web_search_without_handler_returns_clear_unavailable_message(tmp_path: Path) -> None:
    """Missing WebSearch provider is surfaced as a readable tool_result error."""
    provider = FakeModelProvider(
        [
            [
                {
                    "type": "tool_use",
                    "id": "toolu_search",
                    "name": "WebSearch",
                    "input": {"query": "latest Python release"},
                }
            ],
            "Search unavailable.",
        ]
    )
    engine = build_local_engine(
        cwd=_repo(tmp_path),
        config_home=tmp_path / ".claude",
        model_provider=provider,
        permission_mode="bypass",
        require_api_key=False,
    )

    result = asyncio.run(run_local_agent_once("search latest Python release", engine=engine, max_turns=3))
    tool_result = next(
        event["message"]["content"][0]
        for event in result.events
        if event.get("type") == "user" and event["message"]["content"][0].get("type") == "tool_result"
    )

    assert tool_result["is_error"] is True
    assert format_web_search_unavailable_message() in tool_result["content"]
    assert any("[tool_result:error] toolu_search" in line for line in result.logs)


def test_web_search_default_ask_mode_denies_before_handler_call(tmp_path: Path) -> None:
    """Default ask mode stays safe and does not call the configured search handler."""
    calls = []
    provider = FakeModelProvider(
        [
            [
                {
                    "type": "tool_use",
                    "id": "toolu_search",
                    "name": "WebSearch",
                    "input": {"query": "latest Python release"},
                }
            ],
            "Denied safely.",
        ]
    )
    engine = build_local_engine(
        cwd=_repo(tmp_path),
        config_home=tmp_path / ".claude",
        model_provider=provider,
        web_search_handler=lambda args: calls.append(dict(args)) or [],
        require_api_key=False,
    )

    result = asyncio.run(run_local_agent_once("search latest Python release", engine=engine, max_turns=3))
    tool_result = next(
        event["message"]["content"][0]
        for event in result.events
        if event.get("type") == "user" and event["message"]["content"][0].get("type") == "tool_result"
    )

    assert calls == []
    assert engine.tool_use_context.app_state.tool_permission_context.mode == "ask"
    assert tool_result["is_error"] is True
    assert "Permission denied" in tool_result["content"]


def test_permission_deny_is_logged_and_transcript_remains_valid(tmp_path: Path) -> None:
    """Default ask mode denies writes without breaking the transcript chain."""
    repo = _repo(tmp_path)
    target = repo / "created.txt"
    provider = FakeModelProvider(
        [
            [
                {
                    "type": "tool_use",
                    "id": "toolu_write",
                    "name": "Write",
                    "input": {"file_path": str(target), "content": "hello"},
                }
            ],
            "Write was denied safely.",
        ]
    )
    engine = build_local_engine(
        cwd=repo,
        config_home=tmp_path / ".claude",
        model_provider=provider,
        require_api_key=False,
    )

    result = asyncio.run(run_local_agent_once("write a file", engine=engine, max_turns=3))
    rows = [json.loads(line) for line in result.transcript_path.read_text(encoding="utf-8").splitlines()]

    assert result.final_response == "Write was denied safely."
    assert not target.exists()
    assert any("[permission] denied" in line for line in result.logs)
    assert any(
        row.get("type") == "user"
        and row.get("message", {}).get("content", [{}])[0].get("type") == "tool_result"
        and row["message"]["content"][0].get("is_error") is True
        for row in rows
    )
    assert any("Permission denied" in line for line in result.transcript_path.read_text(encoding="utf-8").splitlines())


def test_missing_api_key_error_is_clear(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Real-provider runner mode fails early with a readable credentials message."""
    monkeypatch.delenv("AGENT_KERNEL_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(MissingCredentialsError, match="ANTHROPIC_AUTH_TOKEN.*ANTHROPIC_API_KEY"):
        build_local_engine(cwd=_repo(tmp_path), config_home=tmp_path / ".claude")


def test_missing_api_key_does_not_start_mcp_config_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Credential failure happens before local stdio MCP config loading can start a process."""
    monkeypatch.delenv("AGENT_KERNEL_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config_path = tmp_path / "mcp.json"
    config_path.write_text(json.dumps({"mcpServers": {"echo": {"command": "python3", "args": ["server.py"]}}}), encoding="utf-8")
    calls: list[Path] = []

    def fail_if_called(path, *, cwd=None):
        calls.append(Path(path))
        raise AssertionError("MCP config loader should not run without model credentials")

    monkeypatch.setattr(local_agent, "load_mcp_config", fail_if_called)

    with pytest.raises(MissingCredentialsError):
        build_local_engine(cwd=_repo(tmp_path), config_home=tmp_path / ".claude", mcp_config=config_path)

    assert calls == []


def test_cli_missing_api_key_returns_error_status(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """The script entry point reports missing credentials without a traceback."""
    monkeypatch.delenv("AGENT_KERNEL_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    exit_code = main(["hello"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "Missing Anthropic-compatible API credentials" in captured.err
    assert captured.out == ""


def test_cli_enable_web_search_without_provider_is_clear(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Enabling WebSearch without provider config fails before any model/network call."""
    monkeypatch.delenv(WEB_SEARCH_PROVIDER_ENV, raising=False)
    monkeypatch.delenv("AGENT_KERNEL_WEB_SEARCH_STUB_RESULTS", raising=False)

    exit_code = main(["--enable-web-search", "hello"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert format_web_search_unavailable_message() in captured.err
    assert captured.out == ""


def test_cli_enable_web_fetch_without_provider_is_clear(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Enabling WebFetch without provider config fails before any model/network call."""
    monkeypatch.delenv(WEB_FETCH_PROVIDER_ENV, raising=False)

    exit_code = main(["--enable-web-fetch", "hello"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "WebFetch is not configured" in captured.err
    assert captured.out == ""
