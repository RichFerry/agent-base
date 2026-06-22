"""Opt-in real local runner E2E smoke tests.

These tests are skipped by default. They use a real Anthropic-compatible model
only when AGENT_KERNEL_RUN_REAL_E2E=1 and model credentials are present. All
optional capabilities stay local or deterministic so the smoke does not depend
on real search, real fetch, or a real MCP server.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest

from agent_kernel import AnthropicModelProvider
from examples.local_agent import make_stub_web_search_handler, run_local_agent_once


RUN_REAL_E2E = os.environ.get("AGENT_KERNEL_RUN_REAL_E2E") == "1"
HAS_CREDENTIALS = bool(os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY"))

pytestmark = pytest.mark.skipif(
    not (RUN_REAL_E2E and HAS_CREDENTIALS),
    reason="real runner E2E requires AGENT_KERNEL_RUN_REAL_E2E=1 and Anthropic-compatible credentials",
)


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    return repo


def _examples_dir() -> Path:
    return Path(__file__).parents[1] / "examples"


def _transcript_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _tool_use_names(events: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for event in events:
        if event.get("type") != "assistant":
            continue
        for block in event.get("message", {}).get("content", []):
            if isinstance(block, dict) and block.get("type") == "tool_use":
                names.append(str(block.get("name") or ""))
    return names


def _tool_result_blocks(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for event in events:
        if event.get("type") != "user":
            continue
        for block in event.get("message", {}).get("content", []):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                blocks.append(block)
    return blocks


def test_real_local_runner_e2e_with_opt_in_capabilities(tmp_path: Path) -> None:
    """A real model can enter the local runner with all optional capabilities registered."""
    fetch_calls: list[str] = []

    def local_fetch_handler(url: str) -> dict[str, Any]:
        fetch_calls.append(url)
        return {
            "bytes": 35,
            "code": 200,
            "codeText": "OK",
            "content": "# Local WebFetch Stub\nE2E content.",
            "contentType": "text/markdown",
            "url": url,
        }

    result = asyncio.run(
        run_local_agent_once(
            (
                "Use available tools briefly if helpful: search the stub result, "
                "use the echo skill with hello, and call the local echo MCP tool. "
                "Do not use Bash, file editing, or filesystem tools. "
                "If you do not use every tool, reply with one short sentence."
            ),
            cwd=_repo(tmp_path),
            config_home=tmp_path / ".claude",
            model_provider=AnthropicModelProvider.from_env(),
            session_id="real-runner-e2e-smoke",
            web_search_handler=make_stub_web_search_handler(),
            web_fetch_handler=local_fetch_handler,
            skills_dir=_examples_dir() / "skills",
            mcp_fixture=_examples_dir() / "mcp" / "echo-mcp.json",
            permission_mode="ask",
            max_turns=4,
            require_api_key=False,
        )
    )
    init = result.events[0]
    rows = _transcript_rows(result.transcript_path)
    tool_use_names = _tool_use_names(result.events)
    tool_results = _tool_result_blocks(result.events)

    assert init["type"] == "system"
    assert init["subtype"] == "init"
    assert init["permissionMode"] == "ask"
    assert "WebSearch" in init["tools"]
    assert "WebFetch" in init["tools"]
    assert "Skill" in init["tools"]
    assert "mcp__local-echo__echo" in init["tools"]
    assert init["skills"] == ["echo"]
    assert init["mcp_servers"] == [{"name": "local-echo", "status": "connected"}]
    assert result.events[-1]["type"] == "result"
    assert result.events[-1]["is_error"] is False
    assert result.final_response.strip()
    assert result.transcript_path.exists()
    assert any(row.get("type") == "assistant" for row in rows)
    assert any(event.get("type") == "result" for event in result.events)
    assert isinstance(tool_use_names, list)
    assert isinstance(tool_results, list)
    assert isinstance(fetch_calls, list)
