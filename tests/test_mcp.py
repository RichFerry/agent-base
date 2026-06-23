"""MCP 命名、动态 schema、权限、tool call 与 resource 协议测试。

测试使用注入 handler 代替真实 MCP 连接，验证动态工具仍完整经过普通 Tool 管线，并
展示 text/image/resource 返回值如何映射为模型可接受的 tool_result。
"""

from __future__ import annotations

from pathlib import Path
import asyncio
import json

import pytest

from agent_kernel.config import KernelConfig, MCPClientConfig
from agent_kernel.mcp import (
    MCPConfigurationError,
    build_mcp_tool_name,
    close_mcp_clients,
    get_mcp_display_name,
    load_mcp_config,
    mcp_info_from_string,
    normalize_name_for_mcp,
)
from agent_kernel.model_provider import AnthropicModelProvider, FakeModelProvider
from agent_kernel.query_engine import QueryEngine
from examples.local_agent import build_local_engine, load_mcp_fixture


async def _collect(iterator):
    """消费异步生成器并把全部事件收集为列表，便于同步断言。"""
    return [event async for event in iterator]


def make_config(tmp_path: Path, *clients: MCPClientConfig) -> KernelConfig:
    """为当前测试创建隔离 cwd 和 config_home 的最小 KernelConfig。"""
    repo = tmp_path / "repo"
    repo.mkdir()
    return KernelConfig(cwd=repo, config_home=tmp_path / ".claude", mcp_clients=clients)


def test_mcp_name_helpers_follow_source_shape() -> None:
    """验证 ``mcp name helpers follow source shape`` 场景的行为、消息形状和关键不变量。"""
    assert normalize_name_for_mcp("github.com server") == "github_com_server"
    assert normalize_name_for_mcp("claude.ai Slack") == "claude_ai_Slack"
    assert build_mcp_tool_name("github.com server", "add comment") == "mcp__github_com_server__add_comment"
    assert mcp_info_from_string("mcp__github__issue__comment") == {
        "serverName": "github",
        "toolName": "issue__comment",
    }
    assert get_mcp_display_name("mcp__github__issue_search", "github") == "issue_search"


def test_query_engine_registers_mcp_tools_and_sdk_init_lists_server(tmp_path: Path) -> None:
    """验证 ``query engine registers mcp tools and sdk init lists server`` 场景的行为、消息形状和关键不变量。"""
    client = MCPClientConfig(
        name="github.com server",
        instructions="Use issue numbers exactly.",
        tools=(
            {
                "name": "search issues",
                "description": "Search GitHub issues.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
                "annotations": {"readOnlyHint": True, "title": "Search Issues"},
                "_meta": {"anthropic/searchHint": "search\nissues", "anthropic/alwaysLoad": True},
            },
        ),
    )
    engine = QueryEngine(model_provider=FakeModelProvider(["ok"]), config=make_config(tmp_path, client), session_id="session-1")

    tool = next(tool for tool in engine.tools if tool.name == "mcp__github_com_server__search_issues")
    init = engine.get_system_init_message()
    system_prompt = engine.prompt_composer.get_system_prompt(tools=engine.tools, model=engine.model)

    assert tool.user_facing_name({}) == "github.com server - Search Issues (MCP)"
    assert tool.search_hint == "search issues"
    assert tool.always_load is True
    assert tool.is_read_only({}) is True
    assert init["mcp_servers"] == [{"name": "github.com server", "status": "connected"}]
    assert "Use issue numbers exactly." in "\n\n".join(system_prompt)


def test_local_mcp_fixture_loader_builds_connected_client() -> None:
    """验证 local runner MCP fixture 复用 MCPClientConfig 注入形态。"""
    fixture_path = Path(__file__).parents[1] / "examples" / "mcp" / "echo-mcp.json"

    client = load_mcp_fixture(fixture_path)
    result = client.call_tool_handler("echo", {"text": "hello"}) if client.call_tool_handler else None

    assert client.name == "local-echo"
    assert client.type == "connected"
    assert client.tools[0]["name"] == "echo"
    assert client.resources[0]["uri"] == "fixture://local-echo/readme"
    assert result == {"structuredContent": {"echo": "hello", "source": "local-mcp-fixture"}}


def test_mcp_stdio_config_loader_registers_tools_and_closes_process(tmp_path: Path) -> None:
    """Local stdio MCP config can register tools without leaving a process behind."""
    repo_root = Path(__file__).parents[1]
    config_path = repo_root / "examples" / "mcp" / "stdio-config.json"

    clients = load_mcp_config(config_path, cwd=repo_root)
    try:
        client_config = clients[0]
        stdio_client = client_config.client
        result = client_config.call_tool_handler("echo", {"text": "hello"}) if client_config.call_tool_handler else None

        assert client_config.name == "stdio-echo"
        assert client_config.type == "connected"
        assert client_config.tools[0]["name"] == "echo"
        assert result["structuredContent"] == {"echo": "hello", "source": "stdio-mcp-smoke"}
        assert getattr(stdio_client, "calls") == [{"tool_name": "echo", "args": {"text": "hello"}}]
    finally:
        close_mcp_clients(clients)

    process = getattr(stdio_client, "process")
    assert process is not None
    assert process.poll() == 0


def test_mcp_stdio_config_tool_use_reaches_transcript(tmp_path: Path) -> None:
    """Runner-style MCP config injection stays on the normal QueryEngine tool path."""
    repo_root = Path(__file__).parents[1]
    config_path = repo_root / "examples" / "mcp" / "stdio-config.json"
    expected_tool_name = build_mcp_tool_name("stdio-echo", "echo")
    provider = FakeModelProvider(
        [
            [
                {
                    "type": "tool_use",
                    "id": "toolu_stdio_config",
                    "name": expected_tool_name,
                    "input": {"text": "hello"},
                }
            ],
            "done",
        ]
    )

    engine = build_local_engine(
        cwd=repo_root,
        config_home=tmp_path / ".claude",
        model_provider=provider,
        mcp_config=config_path,
        permission_mode="bypass",
        require_api_key=False,
    )
    try:
        result = asyncio.run(_collect(engine.submit_message("call stdio mcp", max_turns=3, sdk_events=True)))
    finally:
        close_mcp_clients(getattr(engine, "_agent_kernel_owned_mcp_clients", ()))

    tool_result = next(
        block
        for event in result
        if event.get("type") == "user"
        for block in event.get("message", {}).get("content", [])
        if isinstance(block, dict) and block.get("type") == "tool_result"
    )

    assert result[0]["type"] == "system"
    assert expected_tool_name in result[0]["tools"]
    assert result[0]["mcp_servers"] == [{"name": "stdio-echo", "status": "connected"}]
    assert tool_result["tool_use_id"] == "toolu_stdio_config"
    assert tool_result["content"] == '{"echo":"hello","source":"stdio-mcp-smoke"}'
    assert result[-1]["type"] == "result"
    assert result[-1]["is_error"] is False
    rows = [json.loads(line) for line in engine.session_store.transcript_path.read_text(encoding="utf-8").splitlines()]
    assert any(
        row.get("type") == "user"
        and row.get("message", {}).get("content", [{}])[0].get("tool_use_id") == "toolu_stdio_config"
        for row in rows
    )


def test_mcp_stdio_config_invalid_path_and_type_are_clear(tmp_path: Path) -> None:
    """Invalid MCP config paths or unsupported server types fail before registering tools."""
    with pytest.raises(MCPConfigurationError, match="does not exist"):
        load_mcp_config(tmp_path / "missing.json", cwd=tmp_path)

    empty = tmp_path / "empty-mcp.json"
    empty.write_text(json.dumps({"mcpServers": {}}), encoding="utf-8")
    with pytest.raises(MCPConfigurationError, match="at least one server"):
        load_mcp_config(empty, cwd=tmp_path)

    invalid_entry = tmp_path / "invalid-entry-mcp.json"
    invalid_entry.write_text(json.dumps({"mcpServers": {"broken": "not-an-object"}}), encoding="utf-8")
    with pytest.raises(MCPConfigurationError, match='MCP server "broken" config must be an object'):
        load_mcp_config(invalid_entry, cwd=tmp_path)

    invalid = tmp_path / "mcp.json"
    invalid.write_text(json.dumps({"mcpServers": {"remote": {"type": "http", "command": "server"}}}), encoding="utf-8")

    with pytest.raises(MCPConfigurationError, match="Only local stdio is supported"):
        load_mcp_config(invalid, cwd=tmp_path)


def test_mcp_tool_use_executes_handler_and_returns_tool_result(tmp_path: Path) -> None:
    """验证 ``mcp tool use executes handler and returns tool result`` 场景的行为、消息形状和关键不变量。"""
    calls = []

    def call_tool(tool_name: str, args: dict):
        """为当前测试提供 ``call_tool`` 辅助行为。"""
        calls.append((tool_name, args))
        return {"content": [{"type": "text", "text": f"result for {args['query']}"}]}

    client = MCPClientConfig(
        name="github",
        tools=(
            {
                "name": "search",
                "description": "Search GitHub.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        ),
        call_tool_handler=call_tool,
    )
    provider = FakeModelProvider(
        [
            [
                {
                    "type": "tool_use",
                    "id": "toolu_mcp",
                    "name": "mcp__github__search",
                    "input": {"query": "bugs"},
                }
            ],
            "done",
        ]
    )
    engine = QueryEngine(model_provider=provider, config=make_config(tmp_path, client))
    engine.tool_use_context.app_state.tool_permission_context.mode = "bypass"

    events = asyncio.run(_collect(engine.submit_message("search", max_turns=3)))

    tool_message = next(event for event in events if event.get("type") == "user")
    assert calls == [("search", {"query": "bugs"})]
    assert tool_message["message"]["content"][0]["content"] == [{"type": "text", "text": "result for bugs"}]
    assert any(event.get("type") == "tool_progress" and event["progress"]["type"] == "mcp_progress" for event in events)
    assert events[-1]["terminal"]["reason"] == "completed"


def test_mcp_tool_ask_mode_denies_without_permission_callback(tmp_path: Path) -> None:
    """验证 ``mcp tool ask mode denies without permission callback`` 场景的行为、消息形状和关键不变量。"""
    client = MCPClientConfig(
        name="slack",
        tools=({"name": "post", "description": "Post a message.", "inputSchema": {"type": "object", "properties": {}}},),
        call_tool_handler=lambda tool_name, args: "posted",
    )
    provider = FakeModelProvider(
        [
            [
                {
                    "type": "tool_use",
                    "id": "toolu_mcp",
                    "name": "mcp__slack__post",
                    "input": {},
                }
            ],
            "done",
        ]
    )
    engine = QueryEngine(model_provider=provider, config=make_config(tmp_path, client))

    events = asyncio.run(_collect(engine.submit_message("post", max_turns=2)))
    block = next(event for event in events if event.get("type") == "user")["message"]["content"][0]

    assert block["is_error"] is True
    assert "MCPTool requires permission" in block["content"]


def test_mcp_resource_tools_list_and_read_resources(tmp_path: Path) -> None:
    """验证 ``mcp resource tools list and read resources`` 场景的行为、消息形状和关键不变量。"""
    client = MCPClientConfig(
        name="docs",
        resources=(
            {
                "uri": "file://guide",
                "name": "Guide",
                "mimeType": "text/plain",
                "text": "hello docs",
            },
        ),
    )
    provider = FakeModelProvider(
        [
            [{"type": "tool_use", "id": "toolu_list", "name": "ListMcpResourcesTool", "input": {"server": "docs"}}],
            [{"type": "tool_use", "id": "toolu_read", "name": "ReadMcpResourceTool", "input": {"server": "docs", "uri": "file://guide"}}],
            "done",
        ]
    )
    engine = QueryEngine(model_provider=provider, config=make_config(tmp_path, client))

    events = asyncio.run(_collect(engine.submit_message("resources", max_turns=4)))
    tool_messages = [event for event in events if event.get("type") == "user"]

    assert len(tool_messages) == 2
    assert '"server":"docs"' in tool_messages[0]["message"]["content"][0]["content"]
    assert "hello docs" in tool_messages[1]["message"]["content"][0]["content"]


def test_anthropic_provider_uses_mcp_input_json_schema(tmp_path: Path) -> None:
    """验证 ``anthropic provider uses mcp input json schema`` 场景的行为、消息形状和关键不变量。"""
    captured = {}

    def transport(url, headers, body, timeout):
        """模拟模型 HTTP transport，并捕获请求体供断言。"""
        captured["body"] = body
        return {"id": "msg_text", "content": [{"type": "text", "text": "ok"}]}

    client = MCPClientConfig(
        name="github",
        tools=(
            {
                "name": "search",
                "description": "Search GitHub.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "Search query"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        ),
        call_tool_handler=lambda tool_name, args: "ok",
    )
    engine = QueryEngine(
        model_provider=AnthropicModelProvider(
            base_url="https://api.anthropic.com",
            auth_token="secret-token",
            transport=transport,
        ),
        config=make_config(tmp_path, client),
    )

    asyncio.run(_collect(engine.submit_message("hi", max_turns=1)))
    mcp_spec = next(tool for tool in captured["body"]["tools"] if tool["name"] == "mcp__github__search")

    assert mcp_spec["input_schema"]["properties"]["query"]["description"] == "Search query"
    assert mcp_spec["input_schema"]["required"] == ["query"]
