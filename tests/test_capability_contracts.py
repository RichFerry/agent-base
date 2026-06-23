"""v0.2 local runner capability contract tests.

The v0.2 runner keeps WebSearch, Skills, and MCP as explicit opt-in
capabilities. These tests lock the extension boundaries without adding product
features, network access, real model calls, or a real MCP server.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from agent_kernel.model_provider import FakeModelProvider
from agent_kernel.skills import SkillTool
from examples.local_agent import (
    MCPFixtureConfigurationError,
    WEB_SEARCH_PROVIDER_ENV,
    WebSearchConfigurationError,
    build_local_engine,
    build_web_search_handler_from_env,
    load_mcp_fixture,
    main,
    run_local_agent_once,
)


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    return repo


def _examples_dir() -> Path:
    return Path(__file__).parents[1] / "examples"


def _write_skill(skills_root: Path, name: str, body: str) -> Path:
    skill_dir = skills_root / name
    skill_dir.mkdir(parents=True)
    path = skill_dir / "SKILL.md"
    path.write_text(body, encoding="utf-8")
    return path


def _tool_names(engine) -> list[str]:
    return [tool.name for tool in engine.tools]


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


def _transcript_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _transcript_kinds(rows: list[dict[str, Any]]) -> list[str]:
    kinds: list[str] = []
    for row in rows:
        content = row.get("message", {}).get("content", [])
        first = content[0] if content else {}
        if row.get("type") == "assistant" and any(block.get("type") == "tool_use" for block in content if isinstance(block, dict)):
            kinds.append("assistant:tool_use")
        elif row.get("type") == "assistant":
            kinds.append("assistant:text")
        elif row.get("type") == "user" and isinstance(first, dict) and first.get("type") == "tool_result":
            kinds.append(f"user:tool_result:{first.get('tool_use_id')}")
        elif row.get("type") == "user" and row.get("isMeta"):
            kinds.append("user:skill_prompt")
        elif row.get("type") == "user":
            kinds.append("user:prompt")
        else:
            kinds.append(str(row.get("type")))
    return kinds


def test_runner_capabilities_are_explicit_opt_in_and_isolated(tmp_path: Path) -> None:
    """Default runner state stays minimal; opt-ins do not contaminate each other."""
    default_engine = build_local_engine(
        cwd=_repo(tmp_path),
        config_home=tmp_path / ".claude-default",
        model_provider=FakeModelProvider(["ok"]),
        require_api_key=False,
    )
    web_engine = build_local_engine(
        cwd=_repo(tmp_path),
        config_home=tmp_path / ".claude-web",
        model_provider=FakeModelProvider(["ok"]),
        web_search_handler=lambda args: [],
        require_api_key=False,
    )
    skills_engine = build_local_engine(
        cwd=_repo(tmp_path),
        config_home=tmp_path / ".claude-skills",
        model_provider=FakeModelProvider(["ok"]),
        skills_dir=_examples_dir() / "skills",
        require_api_key=False,
    )
    mcp_engine = build_local_engine(
        cwd=_repo(tmp_path),
        config_home=tmp_path / ".claude-mcp",
        model_provider=FakeModelProvider(["ok"]),
        mcp_fixture=_examples_dir() / "mcp" / "echo-mcp.json",
        require_api_key=False,
    )

    assert "WebSearch" in _tool_names(default_engine)
    assert default_engine.tool_use_context.web_search_handler is None
    assert default_engine.tool_use_context.web_fetch_handler is not None
    assert default_engine.skills == []
    assert default_engine.config.mcp_clients == ()
    assert not any(isinstance(tool, SkillTool) for tool in default_engine.tools)
    assert not any(name.startswith("mcp__") for name in _tool_names(default_engine))

    assert web_engine.tool_use_context.web_search_handler is not None
    assert web_engine.tool_use_context.web_fetch_handler is not None
    assert web_engine.skills == []
    assert web_engine.config.mcp_clients == ()
    assert not any(isinstance(tool, SkillTool) for tool in web_engine.tools)
    assert not any(name.startswith("mcp__") for name in _tool_names(web_engine))

    assert skills_engine.tool_use_context.web_search_handler is None
    assert skills_engine.tool_use_context.web_fetch_handler is not None
    assert [skill.name for skill in skills_engine.skills] == ["echo"]
    assert skills_engine.config.mcp_clients == ()
    assert "Skill" in _tool_names(skills_engine)
    assert not any(name.startswith("mcp__") for name in _tool_names(skills_engine))

    assert mcp_engine.tool_use_context.web_search_handler is None
    assert mcp_engine.tool_use_context.web_fetch_handler is not None
    assert mcp_engine.skills == []
    assert mcp_engine.config.mcp_clients[0].name == "local-echo"
    assert not any(isinstance(tool, SkillTool) for tool in mcp_engine.tools)
    assert "mcp__local-echo__echo" in _tool_names(mcp_engine)


def test_combined_registry_and_sdk_init_contract(tmp_path: Path) -> None:
    """Combined opt-in keeps unique tool names and stable SDK init capability fields."""
    engine = build_local_engine(
        cwd=_repo(tmp_path),
        config_home=tmp_path / ".claude",
        model_provider=FakeModelProvider(["ok"]),
        web_search_handler=lambda args: [],
        skills_dir=_examples_dir() / "skills",
        mcp_fixture=_examples_dir() / "mcp" / "echo-mcp.json",
        session_id="capability-contract",
        require_api_key=False,
    )
    tool_names = _tool_names(engine)
    init = engine.get_system_init_message()

    assert len(tool_names) == len(set(tool_names))
    assert {"WebSearch", "Skill", "mcp__local-echo__echo", "ListMcpResourcesTool", "ReadMcpResourceTool"}.issubset(tool_names)
    assert init["type"] == "system"
    assert init["subtype"] == "init"
    assert init["session_id"] == "capability-contract"
    assert init["permissionMode"] == "ask"
    assert init["skills"] == ["echo"]
    assert init["mcp_servers"] == [{"name": "local-echo", "status": "connected"}]
    assert {"WebSearch", "Skill", "mcp__local-echo__echo"}.issubset(set(init["tools"]))


def test_combined_tool_event_shape_and_transcript_order_contract(tmp_path: Path) -> None:
    """Combined tool_use/tool_result events and transcript ordering stay stable."""
    search_calls: list[dict[str, Any]] = []

    def search_handler(args: dict[str, Any]) -> list[dict[str, str]]:
        search_calls.append(dict(args))
        return [{"title": "Contract Search", "url": "https://example.invalid/contract"}]

    provider = FakeModelProvider(
        [
            [
                {"type": "tool_use", "id": "toolu_search", "name": "WebSearch", "input": {"query": "contract"}},
                {"type": "tool_use", "id": "toolu_skill", "name": "Skill", "input": {"skill": "echo", "args": "contract skill"}},
                {"type": "tool_use", "id": "toolu_mcp", "name": "mcp__local-echo__echo", "input": {"text": "contract mcp"}},
            ],
            "contract complete",
        ]
    )
    engine = build_local_engine(
        cwd=_repo(tmp_path),
        config_home=tmp_path / ".claude",
        model_provider=provider,
        web_search_handler=search_handler,
        skills_dir=_examples_dir() / "skills",
        mcp_fixture=_examples_dir() / "mcp" / "echo-mcp.json",
        permission_mode="bypass",
        require_api_key=False,
    )
    call_handler = engine.config.mcp_clients[0].call_tool_handler

    result = asyncio.run(run_local_agent_once("combined contract", engine=engine, max_turns=4))
    assistant_event = next(event for event in result.events if event.get("type") == "assistant" and event["message"]["content"][0]["type"] == "tool_use")
    tool_uses = [block for block in assistant_event["message"]["content"] if block.get("type") == "tool_use"]
    tool_results = _tool_result_blocks(result.events)
    rows = _transcript_rows(result.transcript_path)

    assert [block["name"] for block in tool_uses] == ["WebSearch", "Skill", "mcp__local-echo__echo"]
    assert [block["id"] for block in tool_uses] == ["toolu_search", "toolu_skill", "toolu_mcp"]
    assert [block["tool_use_id"] for block in tool_results] == ["toolu_search", "toolu_skill", "toolu_mcp"]
    assert all(block["type"] == "tool_result" and "content" in block and block.get("is_error") is not True for block in tool_results)
    assert "Contract Search" in tool_results[0]["content"]
    assert tool_results[1]["content"] == "Launching skill: echo"
    assert tool_results[2]["content"] == '{"echo":"contract mcp","source":"local-mcp-fixture"}'
    assert search_calls == [{"query": "contract"}]
    assert getattr(call_handler, "calls") == [{"tool_name": "echo", "args": {"text": "contract mcp"}}]
    assert result.events[0]["subtype"] == "init"
    assert result.events[-1]["subtype"] == "success"
    assert _transcript_kinds(rows) == [
        "user:prompt",
        "assistant:tool_use",
        "user:tool_result:toolu_search",
        "user:tool_result:toolu_skill",
        "user:skill_prompt",
        "user:tool_result:toolu_mcp",
        "assistant:text",
    ]


def test_permission_mode_contract_defaults_to_ask_and_bypass_is_pass_through(tmp_path: Path) -> None:
    """Runner permission mode remains a two-value pass-through."""
    ask_engine = build_local_engine(
        cwd=_repo(tmp_path),
        config_home=tmp_path / ".claude-ask",
        model_provider=FakeModelProvider(["ok"]),
        require_api_key=False,
    )
    bypass_engine = build_local_engine(
        cwd=_repo(tmp_path),
        config_home=tmp_path / ".claude-bypass",
        model_provider=FakeModelProvider(["ok"]),
        permission_mode="bypass",
        require_api_key=False,
    )

    assert ask_engine.tool_use_context.app_state.tool_permission_context.mode == "ask"
    assert ask_engine.get_system_init_message()["permissionMode"] == "ask"
    assert bypass_engine.tool_use_context.app_state.tool_permission_context.mode == "bypass"
    assert bypass_engine.get_system_init_message()["permissionMode"] == "bypass"
    with pytest.raises(ValueError, match="permission_mode must be 'ask' or 'bypass'"):
        build_local_engine(
            cwd=_repo(tmp_path),
            config_home=tmp_path / ".claude-invalid-permission",
            model_provider=FakeModelProvider(["ok"]),
            permission_mode="plan",
            require_api_key=False,
        )


def test_mcp_fixture_invalid_and_name_collision_contract(tmp_path: Path) -> None:
    """Invalid fixtures fail clearly; registry name collisions are de-duplicated."""
    invalid = tmp_path / "invalid-mcp.json"
    invalid.write_text(json.dumps({"name": "broken", "tools": []}), encoding="utf-8")
    collision = tmp_path / "collision-mcp.json"
    collision.write_text(
        json.dumps(
            {
                "name": "collision",
                "tools": [
                    {
                        "name": "WebSearch",
                        "description": "Collides with the default WebSearch tool when skipPrefix is true.",
                        "skipPrefix": True,
                        "result": "should not replace the default WebSearch tool",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(MCPFixtureConfigurationError, match="non-empty 'tools' array"):
        load_mcp_fixture(invalid)

    engine = build_local_engine(
        cwd=_repo(tmp_path),
        config_home=tmp_path / ".claude-collision",
        model_provider=FakeModelProvider(["ok"]),
        mcp_fixture=collision,
        require_api_key=False,
    )
    tool_names = _tool_names(engine)

    assert tool_names.count("WebSearch") == 1
    assert not any(name.startswith("mcp__collision__") for name in tool_names)
    assert engine.get_system_init_message()["mcp_servers"] == [{"name": "collision", "status": "connected"}]


def test_unsupported_scopes_remain_explicit_contract(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Unsupported network search and forked skills stay explicit; MCP config fails clearly."""
    with pytest.raises(WebSearchConfigurationError, match="Unsupported WebSearch provider"):
        build_web_search_handler_from_env({WEB_SEARCH_PROVIDER_ENV: "real-network"})

    exit_code = main(["--mcp-config", "server.json", "hello"])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "MCP config does not exist: server.json" in captured.err

    skills_root = tmp_path / "skills"
    _write_skill(
        skills_root,
        "forked-echo",
        """---
name: forked-echo
description: Forked echo
context: fork
---
Echo from a forked skill.
""",
    )
    provider = FakeModelProvider(
        [
            [{"type": "tool_use", "id": "toolu_fork", "name": "Skill", "input": {"skill": "forked-echo"}}],
            "done",
        ]
    )
    engine = build_local_engine(
        cwd=_repo(tmp_path),
        config_home=tmp_path / ".claude-forked",
        model_provider=provider,
        skills_dir=skills_root,
        permission_mode="bypass",
        require_api_key=False,
    )

    result = asyncio.run(run_local_agent_once("run forked", engine=engine, max_turns=2))
    block = _tool_result_blocks(result.events)[0]

    assert block["is_error"] is True
    assert "Forked skill execution is not implemented in this Python kernel." in block["content"]
