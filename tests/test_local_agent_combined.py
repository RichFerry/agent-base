"""Combined local runner capability tests.

These tests verify that WebSearch, local Skills, and local MCP fixtures can
coexist in the same QueryEngine without using network, real models, or a real
MCP server.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from agent_kernel.model_provider import FakeModelProvider
from agent_kernel.skills import SkillTool
from examples.local_agent import build_local_engine, run_local_agent_once


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    return repo


def _examples_dir() -> Path:
    return Path(__file__).parents[1] / "examples"


def _tool_result_blocks(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for event in events:
        if event.get("type") != "user":
            continue
        content = event.get("message", {}).get("content", [])
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                blocks.append(block)
    return blocks


def _transcript_tool_result_blocks(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return _tool_result_blocks(rows)


def test_build_local_engine_loads_web_search_skills_and_mcp_together(tmp_path: Path) -> None:
    """Optional local capabilities can be enabled in one runner engine."""
    search_handler = lambda args: [{"title": "Combined", "url": "https://example.invalid/combined"}]
    engine = build_local_engine(
        cwd=_repo(tmp_path),
        config_home=tmp_path / ".claude",
        model_provider=FakeModelProvider(["ok"]),
        web_search_handler=search_handler,
        skills_dir=_examples_dir() / "skills",
        mcp_fixture=_examples_dir() / "mcp" / "echo-mcp.json",
        permission_mode="bypass",
        require_api_key=False,
    )
    tool_names = [tool.name for tool in engine.tools]
    init = engine.get_system_init_message()

    assert engine.tool_use_context.web_search_handler is search_handler
    assert [skill.name for skill in engine.skills] == ["echo"]
    assert any(isinstance(tool, SkillTool) for tool in engine.tools)
    assert engine.config.mcp_clients[0].name == "local-echo"
    assert "WebSearch" in tool_names
    assert "Skill" in tool_names
    assert "mcp__local-echo__echo" in tool_names
    assert len(tool_names) == len(set(tool_names))
    assert init["permissionMode"] == "bypass"
    assert init["mcp_servers"] == [{"name": "local-echo", "status": "connected"}]


def test_combined_flow_runs_web_search_skill_and_mcp(tmp_path: Path) -> None:
    """A fake model can call WebSearch, Skill, and MCP in one local run."""
    search_calls: list[dict[str, Any]] = []

    def search_handler(args: dict[str, Any]) -> list[dict[str, str]]:
        search_calls.append(dict(args))
        return [
            {
                "title": "Combined Search Result",
                "url": "https://example.invalid/combined",
                "snippet": "local deterministic search result",
            }
        ]

    provider = FakeModelProvider(
        [
            [
                {
                    "type": "tool_use",
                    "id": "toolu_search",
                    "name": "WebSearch",
                    "input": {"query": "combined capability smoke"},
                },
                {
                    "type": "tool_use",
                    "id": "toolu_skill",
                    "name": "Skill",
                    "input": {"skill": "echo", "args": "from skill"},
                },
                {
                    "type": "tool_use",
                    "id": "toolu_mcp",
                    "name": "mcp__local-echo__echo",
                    "input": {"text": "from mcp"},
                },
            ],
            "Combined flow complete.",
        ]
    )
    engine = build_local_engine(
        cwd=_repo(tmp_path),
        config_home=tmp_path / ".claude",
        model_provider=provider,
        session_id="combined-session",
        web_search_handler=search_handler,
        skills_dir=_examples_dir() / "skills",
        mcp_fixture=_examples_dir() / "mcp" / "echo-mcp.json",
        permission_mode="bypass",
        require_api_key=False,
    )
    call_handler = engine.config.mcp_clients[0].call_tool_handler

    result = asyncio.run(
        run_local_agent_once(
            "search something, use echo skill, then call echo MCP",
            engine=engine,
            max_turns=4,
        )
    )
    event_results = {block["tool_use_id"]: block for block in _tool_result_blocks(result.events)}
    transcript_results = {block["tool_use_id"]: block for block in _transcript_tool_result_blocks(result.transcript_path)}
    second_model_call = json.dumps(provider.calls[1]["messages"], ensure_ascii=False)

    assert result.final_response == "Combined flow complete."
    assert result.events[0]["type"] == "system"
    assert result.events[0]["subtype"] == "init"
    assert result.events[-1]["type"] == "result"
    assert result.events[-1]["subtype"] == "success"
    assert search_calls == [{"query": "combined capability smoke"}]
    assert getattr(call_handler, "calls") == [{"tool_name": "echo", "args": {"text": "from mcp"}}]
    assert engine.tool_use_context.invoked_skills["echo"]["name"] == "echo"
    assert set(event_results) == {"toolu_search", "toolu_skill", "toolu_mcp"}
    assert set(transcript_results) == {"toolu_search", "toolu_skill", "toolu_mcp"}
    assert "Combined Search Result" in event_results["toolu_search"]["content"]
    assert event_results["toolu_skill"]["content"] == "Launching skill: echo"
    assert event_results["toolu_mcp"]["content"] == '{"echo":"from mcp","source":"local-mcp-fixture"}'
    assert "<command-name>echo</command-name>" in second_model_call
    assert "<command-args>from skill</command-args>" in second_model_call
    assert "Combined Search Result" in second_model_call
    assert '\\"echo\\":\\"from mcp\\"' in second_model_call
    assert any("[tool_use] WebSearch" in line for line in result.logs)
    assert any("[tool_use] Skill" in line for line in result.logs)
    assert any("[tool_use] mcp__local-echo__echo" in line for line in result.logs)
    assert any("[tool_result:ok] toolu_search" in line for line in result.logs)
    assert any("[tool_result:ok] toolu_skill" in line for line in result.logs)
    assert any("[tool_result:ok] toolu_mcp" in line for line in result.logs)
    assert any("mcp=local-echo/echo status=completed" in line for line in result.logs)


def test_build_local_engine_without_optional_capability_args_loads_none(tmp_path: Path) -> None:
    """Absent local-runner flags leave optional adapters unloaded."""
    engine = build_local_engine(
        cwd=_repo(tmp_path),
        config_home=tmp_path / ".claude",
        model_provider=FakeModelProvider(["ok"]),
        require_api_key=False,
    )

    assert engine.tool_use_context.web_search_handler is None
    assert engine.skills == []
    assert engine.config.mcp_clients == ()
    assert not any(isinstance(tool, SkillTool) for tool in engine.tools)
    assert not any(tool.name.startswith("mcp__") for tool in engine.tools)
    assert engine.tool_use_context.app_state.tool_permission_context.mode == "ask"
