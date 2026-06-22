"""Example local runner tests.

These tests keep the v0.2 runner slice in the examples layer: no real network,
no new public API, and no bypass permissions by default.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import subprocess
from urllib.error import HTTPError, URLError

import pytest

from agent_kernel.model_provider import FakeModelProvider
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
    load_mcp_fixture,
    main,
    make_http_web_fetch_handler,
    make_stub_web_search_handler,
    run_local_agent_once,
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
    assert any(isinstance(tool, SkillTool) for tool in engine.tools)
    assert engine.tool_use_context.app_state.tool_permission_context.mode == "ask"


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
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(MissingCredentialsError, match="ANTHROPIC_AUTH_TOKEN.*ANTHROPIC_API_KEY"):
        build_local_engine(cwd=_repo(tmp_path), config_home=tmp_path / ".claude")


def test_cli_missing_api_key_returns_error_status(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """The script entry point reports missing credentials without a traceback."""
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
