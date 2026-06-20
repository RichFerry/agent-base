"""MCP 命名、动态 schema、权限、tool call 与 resource 协议测试。

测试使用注入 handler 代替真实 MCP 连接，验证动态工具仍完整经过普通 Tool 管线，并
展示 text/image/resource 返回值如何映射为模型可接受的 tool_result。
"""

from __future__ import annotations

from pathlib import Path
import asyncio

from agent_kernel.config import KernelConfig, MCPClientConfig
from agent_kernel.mcp import (
    build_mcp_tool_name,
    get_mcp_display_name,
    mcp_info_from_string,
    normalize_name_for_mcp,
)
from agent_kernel.model_provider import AnthropicModelProvider, FakeModelProvider
from agent_kernel.query_engine import QueryEngine


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
