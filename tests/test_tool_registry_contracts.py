"""Tool registry and capability collision contracts for v0.2.

These tests pin registry behavior around built-in tools, local runner
capabilities, MCP collisions, and transcript ordering. They do not add product
features or start real providers.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from agent_kernel.config import KernelConfig, MCPClientConfig
from agent_kernel.model_provider import FakeModelProvider
from agent_kernel.query_engine import QueryEngine
from agent_kernel.skills import SkillTool
from examples.local_agent import build_local_engine, run_local_agent_once


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


def _tool_names(engine: QueryEngine) -> list[str]:
    return [tool.name for tool in engine.tools]


def _tool_result_blocks(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for event in events:
        if event.get("type") != "user":
            continue
        for block in event.get("message", {}).get("content", []):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                blocks.append(block)
    return blocks


def _transcript_kinds(path: Path) -> list[str]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
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


def _mcp_client(
    name: str,
    *,
    tool_name: str = "search",
    resources: tuple[dict[str, Any], ...] = (),
    client_type: str = "connected",
    skip_prefix: bool = False,
    calls: list[tuple[str, dict[str, Any]]] | None = None,
) -> MCPClientConfig:
    def handler(tool: str, args: dict[str, Any]) -> dict[str, Any]:
        if calls is not None:
            calls.append((tool, dict(args)))
        return {"structuredContent": {"tool": tool, "args": args}}

    return MCPClientConfig(
        name=name,
        type=client_type,
        tools=(
            {
                "name": tool_name,
                "description": f"{name} {tool_name}",
                "inputSchema": {"type": "object", "properties": {}, "additionalProperties": True},
                "annotations": {"readOnlyHint": True},
                **({"skipPrefix": True} if skip_prefix else {}),
            },
        ),
        resources=resources,
        call_tool_handler=handler,
    )


def test_combined_registry_order_and_uniqueness_contract(tmp_path: Path) -> None:
    """Built-ins, WebSearch, Skills, and MCP tools keep one stable registry order."""
    engine = build_local_engine(
        cwd=_repo(tmp_path),
        config_home=tmp_path / ".claude",
        model_provider=FakeModelProvider(["ok"]),
        web_search_handler=lambda args: [],
        skills_dir=_examples_dir() / "skills",
        mcp_fixture=_examples_dir() / "mcp" / "echo-mcp.json",
        require_api_key=False,
    )
    tool_names = _tool_names(engine)

    expected_tool_names = [
        "Bash",
        "Glob",
        "Grep",
        "LS",
        "Read",
        "Write",
        "Edit",
        "MultiEdit",
        "NotebookEdit",
        "TodoWrite",
        "Agent",
        "WebSearch",
        "WebFetch",
        "mcp__local-echo__echo",
        "ListMcpResourcesTool",
        "ReadMcpResourceTool",
        "Skill",
    ]
    expected_sdk_tool_names = ["Task" if name == "Agent" else name for name in expected_tool_names]

    assert tool_names == expected_tool_names
    assert len(tool_names) == len(set(tool_names))
    assert engine.get_system_init_message()["tools"] == expected_sdk_tool_names


def test_mcp_multi_client_collisions_and_failed_clients_are_stable(tmp_path: Path) -> None:
    """Duplicate MCP tool names are de-duplicated; failed clients do not add tools."""
    repo = _repo(tmp_path)
    duplicate_a = _mcp_client("duplicate", calls=[])
    duplicate_b = _mcp_client("duplicate", calls=[])
    alpha = _mcp_client("alpha", resources=({"uri": "fixture://alpha/readme", "name": "Alpha"},))
    beta = _mcp_client("beta", resources=({"uri": "fixture://beta/readme", "name": "Beta"},))
    failed = _mcp_client("failed", tool_name="ghost", client_type="failed")
    config = KernelConfig(
        cwd=repo,
        config_home=tmp_path / ".claude",
        mcp_clients=(duplicate_a, duplicate_b, alpha, beta, failed),
    )
    engine = QueryEngine(model_provider=FakeModelProvider(["ok"]), config=config, session_id="mcp-collisions")
    tool_names = _tool_names(engine)
    init = engine.get_system_init_message()

    assert tool_names.count("mcp__duplicate__search") == 1
    assert "mcp__alpha__search" in tool_names
    assert "mcp__beta__search" in tool_names
    assert "mcp__failed__ghost" not in tool_names
    assert tool_names.count("ListMcpResourcesTool") == 1
    assert tool_names.count("ReadMcpResourceTool") == 1
    assert init["mcp_servers"] == [
        {"name": "duplicate", "status": "connected"},
        {"name": "duplicate", "status": "connected"},
        {"name": "alpha", "status": "connected"},
        {"name": "beta", "status": "connected"},
        {"name": "failed", "status": "failed"},
    ]


def test_mcp_resource_tools_and_normal_mcp_tools_do_not_pollute_each_other(tmp_path: Path) -> None:
    """Resource helper tools stay global; normal MCP tools keep their MCP prefix."""
    client = _mcp_client(
        "docs",
        tool_name="ListMcpResourcesTool",
        resources=({"uri": "fixture://docs/readme", "name": "Docs"},),
    )
    config = KernelConfig(cwd=_repo(tmp_path), config_home=tmp_path / ".claude", mcp_clients=(client,))
    engine = QueryEngine(model_provider=FakeModelProvider(["ok"]), config=config)
    tool_names = _tool_names(engine)

    assert "mcp__docs__ListMcpResourcesTool" in tool_names
    assert "ListMcpResourcesTool" in tool_names
    assert "ReadMcpResourceTool" in tool_names
    assert tool_names.count("ListMcpResourcesTool") == 1
    assert tool_names.count("mcp__docs__ListMcpResourcesTool") == 1


def test_skill_names_do_not_override_builtin_or_mcp_tools(tmp_path: Path) -> None:
    """A skill can share a display name with another capability without replacing tools."""
    skills_root = tmp_path / "skills"
    _write_skill(
        skills_root,
        "WebSearch",
        """---
name: WebSearch
description: Skill named like a built-in tool
---
This is a skill, not the built-in WebSearch tool.
""",
    )
    config = KernelConfig(
        cwd=_repo(tmp_path),
        config_home=tmp_path / ".claude",
        skill_paths=(skills_root,),
        mcp_clients=(_mcp_client("skills", tool_name="Skill"),),
    )
    setattr(config, "_agent_kernel_skill_paths_only", True)
    engine = QueryEngine(model_provider=FakeModelProvider(["ok"]), config=config)
    tool_names = _tool_names(engine)

    assert [skill.name for skill in engine.skills] == ["WebSearch"]
    assert tool_names.count("WebSearch") == 1
    assert tool_names.count("Skill") == 1
    assert "mcp__skills__Skill" in tool_names
    assert any(isinstance(tool, SkillTool) for tool in engine.tools)


def test_websearch_collision_executes_builtin_not_mcp_or_skill(tmp_path: Path) -> None:
    """MCP/Skill WebSearch name collisions do not replace the built-in WebSearch."""
    skills_root = tmp_path / "skills"
    _write_skill(
        skills_root,
        "WebSearch",
        """---
name: WebSearch
description: Skill named WebSearch
---
Skill prompt.
""",
    )
    mcp_calls: list[tuple[str, dict[str, Any]]] = []
    search_calls: list[dict[str, Any]] = []
    collision_client = _mcp_client("collision", tool_name="WebSearch", skip_prefix=True, calls=mcp_calls)
    provider = FakeModelProvider(
        [
            [{"type": "tool_use", "id": "toolu_search", "name": "WebSearch", "input": {"query": "collision"}}],
            "builtin search won",
        ]
    )

    config = KernelConfig(
        cwd=_repo(tmp_path),
        config_home=tmp_path / ".claude",
        skill_paths=(skills_root,),
        mcp_clients=(collision_client,),
    )
    setattr(config, "_agent_kernel_skill_paths_only", True)
    engine = QueryEngine(model_provider=provider, config=config)
    engine.tool_use_context.web_search_handler = lambda args: search_calls.append(dict(args)) or [{"title": "Built-in", "url": "https://example.invalid"}]
    engine.tool_use_context.app_state.tool_permission_context.mode = "bypass"

    result = asyncio.run(run_local_agent_once("search collision", engine=engine, max_turns=3))
    tool_result = _tool_result_blocks(result.events)[0]

    assert _tool_names(engine).count("WebSearch") == 1
    assert mcp_calls == []
    assert search_calls == [{"query": "collision"}]
    assert "Built-in" in tool_result["content"]
    assert result.final_response == "builtin search won"


def test_transcript_ordering_survives_collision_case(tmp_path: Path) -> None:
    """Collision cases still preserve tool_use and tool_result transcript ordering."""
    skills_root = tmp_path / "skills"
    _write_skill(
        skills_root,
        "WebSearch",
        """---
name: WebSearch
description: Skill named WebSearch
---
Skill prompt: $ARGUMENTS
""",
    )
    collision_client = _mcp_client("collision", tool_name="WebSearch", skip_prefix=True)
    provider = FakeModelProvider(
        [
            [
                {"type": "tool_use", "id": "toolu_search", "name": "WebSearch", "input": {"query": "collision order"}},
                {"type": "tool_use", "id": "toolu_skill", "name": "Skill", "input": {"skill": "WebSearch", "args": "skill order"}},
            ],
            "ordered",
        ]
    )
    config = KernelConfig(
        cwd=_repo(tmp_path),
        config_home=tmp_path / ".claude",
        skill_paths=(skills_root,),
        mcp_clients=(collision_client,),
    )
    setattr(config, "_agent_kernel_skill_paths_only", True)
    engine = QueryEngine(model_provider=provider, config=config)
    engine.tool_use_context.web_search_handler = lambda args: [{"title": "Ordered", "url": "https://example.invalid/order"}]
    engine.tool_use_context.app_state.tool_permission_context.mode = "bypass"

    result = asyncio.run(run_local_agent_once("collision order", engine=engine, max_turns=3))

    assert _transcript_kinds(result.transcript_path) == [
        "user:prompt",
        "assistant:tool_use",
        "user:tool_result:toolu_search",
        "user:tool_result:toolu_skill",
        "user:skill_prompt",
        "assistant:text",
    ]
