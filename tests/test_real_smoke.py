"""Opt-in real smoke tests.

These tests are skipped by default. Real model checks require
AGENT_KERNEL_RUN_REAL_SMOKE=1 plus Anthropic-compatible credentials. Real
WebSearch adapter checks require AGENT_KERNEL_RUN_REAL_SMOKE=1 plus explicit
WebSearch endpoint configuration.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from agent_kernel import AnthropicModelProvider, KernelConfig, QueryEngine
from agent_kernel.model_provider import FakeModelProvider
from examples.local_agent import (
    WEB_SEARCH_API_KEY_ENV,
    WEB_SEARCH_MODEL_ENV,
    WEB_SEARCH_PROVIDER_ENV,
    WEB_SEARCH_URL_ENV,
    build_web_search_handler_from_env,
    make_stub_web_search_handler,
    run_local_agent_once,
)


RUN_REAL_SMOKE = os.environ.get("AGENT_KERNEL_RUN_REAL_SMOKE") == "1"
HAS_CREDENTIALS = bool(os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY"))
RUN_REAL_HTTP_JSON_WEB_SEARCH = (
    RUN_REAL_SMOKE
    and (os.environ.get(WEB_SEARCH_PROVIDER_ENV) or "").strip().lower() == "http-json"
    and bool(os.environ.get(WEB_SEARCH_URL_ENV))
)
RUN_REAL_ANTHROPIC_COMPATIBLE_WEB_SEARCH = (
    RUN_REAL_SMOKE
    and (os.environ.get(WEB_SEARCH_PROVIDER_ENV) or "").strip().lower() == "anthropic-compatible"
    and bool(os.environ.get(WEB_SEARCH_URL_ENV))
    and bool(os.environ.get(WEB_SEARCH_API_KEY_ENV))
    and bool(os.environ.get(WEB_SEARCH_MODEL_ENV))
)

requires_real_model = pytest.mark.skipif(
    not (RUN_REAL_SMOKE and HAS_CREDENTIALS),
    reason="real smoke requires AGENT_KERNEL_RUN_REAL_SMOKE=1 and Anthropic-compatible credentials",
)
requires_real_web_search = pytest.mark.skipif(
    not RUN_REAL_HTTP_JSON_WEB_SEARCH,
    reason=f"real WebSearch smoke requires AGENT_KERNEL_RUN_REAL_SMOKE=1, {WEB_SEARCH_PROVIDER_ENV}=http-json, and {WEB_SEARCH_URL_ENV}",
)
requires_real_anthropic_compatible_web_search = pytest.mark.skipif(
    not RUN_REAL_ANTHROPIC_COMPATIBLE_WEB_SEARCH,
    reason=(
        "real Anthropic-compatible WebSearch smoke requires AGENT_KERNEL_RUN_REAL_SMOKE=1, "
        f"{WEB_SEARCH_PROVIDER_ENV}=anthropic-compatible, {WEB_SEARCH_URL_ENV}, "
        f"{WEB_SEARCH_API_KEY_ENV}, and {WEB_SEARCH_MODEL_ENV}"
    ),
)


async def _collect(iterator):
    return [event async for event in iterator]


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    return repo


def _examples_dir() -> Path:
    return Path(__file__).parents[1] / "examples"


def _transcript_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@requires_real_model
def test_real_model_provider_submit_message_smoke(tmp_path: Path) -> None:
    """A real provider can complete one minimal QueryEngine turn."""
    config = KernelConfig(cwd=_repo(tmp_path), config_home=tmp_path / ".claude")
    engine = QueryEngine(
        model_provider=AnthropicModelProvider.from_env(),
        config=config,
        session_id="real-provider-smoke",
    )

    events = asyncio.run(
        _collect(
            engine.submit_message(
                "Reply with one short sentence: agent kernel real provider smoke test.",
                max_turns=1,
                sdk_events=True,
            )
        )
    )
    rows = _transcript_rows(engine.session_store.transcript_path)

    assert events[0]["type"] == "system"
    assert events[0]["subtype"] == "init"
    assert events[-1]["type"] == "result"
    assert events[-1]["is_error"] is False
    assert str(events[-1].get("result") or "").strip()
    assert engine.session_store.transcript_path.exists()
    assert any(row.get("type") == "assistant" for row in rows)


@requires_real_model
def test_real_local_runner_with_opt_in_capabilities_smoke(tmp_path: Path) -> None:
    """The local runner can call a real model while optional capabilities are registered."""
    result = asyncio.run(
        run_local_agent_once(
            "Reply with one short sentence: local runner real smoke test. Do not use tools.",
            cwd=_repo(tmp_path),
            config_home=tmp_path / ".claude",
            model_provider=AnthropicModelProvider.from_env(),
            session_id="real-local-runner-smoke",
            web_search_handler=make_stub_web_search_handler(),
            skills_dir=_examples_dir() / "skills",
            mcp_fixture=_examples_dir() / "mcp" / "echo-mcp.json",
            permission_mode="bypass",
            max_turns=2,
            require_api_key=False,
        )
    )
    init = result.events[0]

    assert init["type"] == "system"
    assert init["subtype"] == "init"
    assert "WebSearch" in init["tools"]
    assert "Skill" in init["tools"]
    assert "mcp__local-echo__echo" in init["tools"]
    assert init["skills"] == ["echo"]
    assert init["mcp_servers"] == [{"name": "local-echo", "status": "connected"}]
    assert result.events[-1]["type"] == "result"
    assert result.events[-1]["is_error"] is False
    assert result.final_response.strip()
    assert result.transcript_path.exists()


@requires_real_web_search
def test_real_http_json_web_search_adapter_smoke(tmp_path: Path) -> None:
    """The opt-in http-json WebSearch adapter can feed a real provider result through tool_result."""
    real_handler = build_web_search_handler_from_env(os.environ)
    calls: list[dict] = []

    def recording_handler(args: dict) -> dict:
        calls.append(dict(args))
        return real_handler(args)  # type: ignore[misc]

    provider = FakeModelProvider(
        [
            [
                {
                    "type": "tool_use",
                    "id": "toolu_real_search",
                    "name": "WebSearch",
                    "input": {"query": "agent kernel real websearch smoke"},
                }
            ],
            "Real WebSearch adapter smoke completed.",
        ]
    )
    result = asyncio.run(
        run_local_agent_once(
            "Use WebSearch once for a real adapter smoke.",
            cwd=_repo(tmp_path),
            config_home=tmp_path / ".claude",
            model_provider=provider,
            session_id="real-websearch-adapter-smoke",
            web_search_handler=recording_handler,
            permission_mode="bypass",
            max_turns=3,
            require_api_key=False,
        )
    )
    tool_results = [
        block
        for event in result.events
        if event.get("type") == "user"
        for block in event.get("message", {}).get("content", [])
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    rows = _transcript_rows(result.transcript_path)

    assert calls == [{"query": "agent kernel real websearch smoke"}]
    assert result.events[0]["type"] == "system"
    assert result.events[0]["subtype"] == "init"
    assert result.events[-1]["type"] == "result"
    assert result.events[-1]["is_error"] is False
    assert len(tool_results) == 1
    assert tool_results[0].get("is_error") is not True
    assert str(tool_results[0].get("content") or "").strip()
    assert result.final_response.strip()
    assert result.transcript_path.exists()
    assert any(
        row.get("type") == "user"
        and row.get("message", {}).get("content", [{}])[0].get("tool_use_id") == "toolu_real_search"
        for row in rows
    )


@requires_real_anthropic_compatible_web_search
def test_real_anthropic_compatible_web_search_adapter_smoke(tmp_path: Path) -> None:
    """The opt-in Anthropic-compatible search adapter stays on the WebSearch tool_result path."""
    real_handler = build_web_search_handler_from_env(os.environ)
    calls: list[dict] = []

    def recording_handler(args: dict) -> dict:
        calls.append(dict(args))
        return real_handler(args)  # type: ignore[misc]

    provider = FakeModelProvider(
        [
            [
                {
                    "type": "tool_use",
                    "id": "toolu_anthropic_search",
                    "name": "WebSearch",
                    "input": {"query": "agent kernel anthropic-compatible search smoke"},
                }
            ],
            "Anthropic-compatible WebSearch adapter smoke completed.",
        ]
    )
    result = asyncio.run(
        run_local_agent_once(
            "Use WebSearch once for an Anthropic-compatible adapter smoke.",
            cwd=_repo(tmp_path),
            config_home=tmp_path / ".claude",
            model_provider=provider,
            session_id="real-anthropic-compatible-websearch-smoke",
            web_search_handler=recording_handler,
            permission_mode="bypass",
            max_turns=3,
            require_api_key=False,
        )
    )
    tool_results = [
        block
        for event in result.events
        if event.get("type") == "user"
        for block in event.get("message", {}).get("content", [])
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    rows = _transcript_rows(result.transcript_path)

    assert calls == [{"query": "agent kernel anthropic-compatible search smoke"}]
    assert result.events[0]["type"] == "system"
    assert result.events[0]["subtype"] == "init"
    assert result.events[-1]["type"] == "result"
    assert result.events[-1]["is_error"] is False
    assert len(tool_results) == 1
    assert tool_results[0].get("is_error") is not True
    assert str(tool_results[0].get("content") or "").strip()
    assert result.final_response.strip()
    assert result.transcript_path.exists()
    assert any(
        row.get("type") == "user"
        and row.get("message", {}).get("content", [{}])[0].get("tool_use_id") == "toolu_anthropic_search"
        for row in rows
    )
