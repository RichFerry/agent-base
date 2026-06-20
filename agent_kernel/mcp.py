"""Model Context Protocol 动态工具和资源的内核适配层。

每个 MCP tool 使用 ``mcp__<server>__<tool>`` 稳定命名，原始 inputSchema 被保留并用于
校验/API tool spec。MCPTool 继承普通 Tool，因此自动获得 Pre/Post hook、ask/bypass
权限、并发调度、取消和 tool_result 回灌。server 可标记 read_only/destructive，影响
权限建议与并发安全。

``transform_mcp_result`` 把 SDK/MCP 返回的 text、image、embedded resource 和普通 JSON
归一为 Anthropic tool_result content。ListMcpResourcesTool/ReadMcpResourceTool 提供资源
协议入口。实际连接和认证不在这里维护：MCPClientConfig 注入 client 或 handler，
``mcp_tools_from_clients`` 只根据当前 connected 配置生成工具对象。

名称解析 helper 同时服务权限显示与 SDK init，修改命名规则会影响 transcript 兼容性。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from .config import MCPClientConfig
from .messages import AssistantMessage, ToolResultBlock
from .permissions import PermissionDecision
from .tools.base import Tool, ToolResult, ToolUseContext, ValidationResult


MAX_MCP_DESCRIPTION_LENGTH = 2_000


def normalize_name_for_mcp(name: str) -> str:
    """规范化name for MCP，供MCP 适配流程使用。"""
    normalized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    if name.startswith("claude.ai "):
        normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def get_mcp_prefix(server_name: str) -> str:
    """获取MCP prefix，供MCP 适配流程使用。"""
    return f"mcp__{normalize_name_for_mcp(server_name)}__"


def build_mcp_tool_name(server_name: str, tool_name: str) -> str:
    """构造MCP 工具 name，供MCP 适配流程使用。"""
    return f"{get_mcp_prefix(server_name)}{normalize_name_for_mcp(tool_name)}"


def mcp_info_from_string(tool_string: str) -> dict[str, str | None] | None:
    """完成 ``mcp_info_from_string`` 对应的MCP 适配内部步骤。"""
    parts = tool_string.split("__")
    if len(parts) < 2 or parts[0] != "mcp" or not parts[1]:
        return None
    return {
        "serverName": parts[1],
        "toolName": "__".join(parts[2:]) if len(parts) > 2 else None,
    }


def get_mcp_display_name(full_name: str, server_name: str) -> str:
    """获取MCP display name，供MCP 适配流程使用。"""
    return full_name.replace(get_mcp_prefix(server_name), "")


def is_mcp_tool_name(name: str) -> bool:
    """判断MCP 工具 name，供MCP 适配流程使用。"""
    return name.startswith("mcp__")


def get_tool_name_for_permission_check(tool: Tool) -> str:
    """获取工具 name for 权限 check，供MCP 适配流程使用。"""
    mcp_info = getattr(tool, "mcp_info", None)
    if isinstance(mcp_info, dict):
        return build_mcp_tool_name(mcp_info["serverName"], mcp_info["toolName"])
    return tool.name


def find_mcp_client_config(configs: tuple[MCPClientConfig, ...], server_name: str) -> MCPClientConfig | None:
    """查找MCP client 配置，供MCP 适配流程使用。"""
    normalized = normalize_name_for_mcp(server_name)
    for client in configs:
        if normalize_name_for_mcp(client.name) == normalized:
            return client
    return None


def transform_mcp_result(result: Any) -> str | list[dict[str, Any]]:
    """把 MCP text/image/resource content 转成 Anthropic tool_result content。"""
    # Handler 可以直接复用内核 ToolResult，也可以返回原始 MCP SDK 形态。
    if isinstance(result, ToolResult):
        result = result.data
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        if result.get("isError"):
            # MCP 协议把远端错误放在正常响应体中；转换为异常交给工具管线封装。
            content = result.get("content")
            if isinstance(content, list) and content:
                first = content[0]
                if isinstance(first, dict) and "text" in first:
                    raise RuntimeError(str(first["text"]))
            raise RuntimeError(str(result.get("error") or "MCP tool returned error"))
        if "toolResult" in result:
            return str(result["toolResult"])
        if result.get("structuredContent") is not None:
            # structuredContent 没有 Anthropic 原生 block，对其做稳定 JSON 序列化。
            return json.dumps(result["structuredContent"], ensure_ascii=False, separators=(",", ":"))
        if isinstance(result.get("content"), list):
            return _transform_mcp_content_array(result["content"])
    if isinstance(result, list):
        return _transform_mcp_content_array(result)
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


def _transform_mcp_content_array(content: list[Any]) -> list[dict[str, Any]]:
    """转换MCP 内容 array，供MCP 适配流程使用。"""
    blocks: list[dict[str, Any]] = []
    for item in content:
        if isinstance(item, str):
            blocks.append({"type": "text", "text": item})
            continue
        if not isinstance(item, dict):
            blocks.append({"type": "text", "text": json.dumps(item, ensure_ascii=False, separators=(",", ":"))})
            continue
        item_type = item.get("type")
        if item_type == "text":
            blocks.append({"type": "text", "text": str(item.get("text", ""))})
        elif item_type == "image":
            # MCP mimeType 对应 Anthropic source.media_type。
            block = {"type": "image", "source": item.get("source")}
            if item.get("mimeType") and isinstance(block["source"], dict):
                block["source"].setdefault("media_type", item["mimeType"])
            blocks.append(block)
        elif item_type == "resource":
            blocks.append({"type": "text", "text": json.dumps(item.get("resource", item), ensure_ascii=False, separators=(",", ":"))})
        else:
            blocks.append({"type": "text", "text": json.dumps(item, ensure_ascii=False, separators=(",", ":"))})
    return blocks


def _schema_required(input_json_schema: dict[str, Any] | None) -> list[str]:
    """完成 ``_schema_required`` 对应的MCP 适配内部步骤。"""
    if not isinstance(input_json_schema, dict):
        return []
    required = input_json_schema.get("required")
    return [str(item) for item in required] if isinstance(required, list) else []


@dataclass
class MCPTool(Tool):
    """一个 server tool 的动态 Tool 包装器，保留原始 JSON Schema。"""
    server_name: str
    tool_name: str
    description_text: str = ""
    input_json_schema: dict[str, Any] | None = None
    annotations: dict[str, Any] | None = None
    meta: dict[str, Any] | None = None
    client_config: MCPClientConfig | None = None
    skip_prefix: bool = False

    max_result_size_chars: int = 100_000

    def __post_init__(self) -> None:
        """完成 dataclass 创建后的派生字段初始化与规范化。"""
        # 默认前缀防止不同 server 的同名工具冲突；显式 skipPrefix 保留兼容入口。
        self.name = self.tool_name if self.skip_prefix else build_mcp_tool_name(self.server_name, self.tool_name)
        self.mcp_info = {"serverName": self.server_name, "toolName": self.tool_name}
        self.search_hint = self._search_hint()
        self.always_load = bool((self.meta or {}).get("anthropic/alwaysLoad"))

    async def description(self, input: dict | None = None) -> str:
        """返回提供给模型和调用方展示的工具简短说明。"""
        return self.description_text or ""

    async def prompt(self) -> str:
        """返回该工具按源码保留的模型使用提示词。"""
        description = self.description_text or ""
        if len(description) > MAX_MCP_DESCRIPTION_LENGTH:
            return description[:MAX_MCP_DESCRIPTION_LENGTH] + "... [truncated]"
        return description

    def validate_schema(self, input: dict) -> ValidationResult:
        """校验输入对象的字段、必填项和基础类型。"""
        if not isinstance(input, dict):
            return ValidationResult(False, "Input must be an object.")
        for field_name in _schema_required(self.input_json_schema):
            if field_name not in input:
                return ValidationResult(False, f"Missing required field: {field_name}")
        return ValidationResult(True)

    async def check_permissions(self, input: dict, context: ToolUseContext) -> PermissionDecision:
        """返回该工具针对当前输入的初步权限建议。"""
        return PermissionDecision.ask("MCPTool requires permission.")

    def is_concurrency_safe(self, input: dict) -> bool:
        """判断当前输入是否可以与相邻安全工具并发执行。"""
        return bool((self.annotations or {}).get("readOnlyHint"))

    def is_read_only(self, input: dict) -> bool:
        """判断当前输入是否只读取外部状态而不产生修改。"""
        return bool((self.annotations or {}).get("readOnlyHint"))

    def is_destructive(self, input: dict) -> bool:
        """判断当前输入是否可能产生破坏性副作用。"""
        return bool((self.annotations or {}).get("destructiveHint"))

    def user_facing_name(self, input: dict | None = None) -> str:
        """根据当前输入返回适合界面展示的工具名称。"""
        display_name = (self.annotations or {}).get("title") or self.tool_name
        return f"{self.server_name} - {display_name} (MCP)"

    async def call(self, args: dict, context: ToolUseContext, can_use_tool, parent_message: AssistantMessage, on_progress=None) -> ToolResult:
        """执行工具核心逻辑，并返回标准 ToolResult。"""
        if on_progress:
            on_progress(
                {
                    "type": "mcp_progress",
                    "status": "started",
                    "serverName": self.server_name,
                    "toolName": self.tool_name,
                }
            )
        # Tool 可持有创建时 client，也可在调用时从最新 config 重新查找。
        client_config = self.client_config or find_mcp_client_config(context.config.mcp_clients, self.server_name)
        if client_config is None:
            raise RuntimeError(f'MCP server "{self.server_name}" not found.')
        if client_config.type != "connected":
            raise RuntimeError(f'MCP server "{self.server_name}" is not connected.')
        result = await _call_mcp_tool(client_config, self.tool_name, args)
        transformed = transform_mcp_result(result)
        if on_progress:
            on_progress(
                {
                    "type": "mcp_progress",
                    "status": "completed",
                    "serverName": self.server_name,
                    "toolName": self.tool_name,
                }
            )
        return ToolResult(transformed)

    def map_tool_result_to_tool_result_block_param(self, content: Any, tool_use_id: str) -> ToolResultBlock:
        """把工具内部结果映射为 Anthropic tool_result block。"""
        return {"tool_use_id": tool_use_id, "type": "tool_result", "content": content}

    def to_api_spec(self) -> dict[str, Any]:
        """构造发送给模型 API 的工具名称、说明和 JSON Schema。"""
        schema = self.input_json_schema or {"type": "object", "properties": {}, "additionalProperties": True}
        return {
            "name": self.name,
            "description": self.description_text or "",
            "input_schema": schema,
        }

    def _search_hint(self) -> str | None:
        """完成 ``_search_hint`` 对应的MCP 适配内部步骤。"""
        hint = (self.meta or {}).get("anthropic/searchHint")
        if not isinstance(hint, str):
            return None
        collapsed = re.sub(r"\s+", " ", hint).strip()
        return collapsed or None


class ListMcpResourcesTool(Tool):
    """实现 ``ListMcpResourcesTool``，并接入统一 Tool 生命周期。"""
    name = "ListMcpResourcesTool"
    search_hint = "list resources from connected MCP servers"
    input_schema = {"server": str}
    max_result_size_chars = 100_000

    def is_read_only(self, input: dict) -> bool:
        """判断当前输入是否只读取外部状态而不产生修改。"""
        return True

    def is_concurrency_safe(self, input: dict) -> bool:
        """判断当前输入是否可以与相邻安全工具并发执行。"""
        return True

    async def check_permissions(self, input: dict, context: ToolUseContext) -> PermissionDecision:
        """返回该工具针对当前输入的初步权限建议。"""
        return PermissionDecision.allow()

    async def description(self, input: dict | None = None) -> str:
        """返回提供给模型和调用方展示的工具简短说明。"""
        return "List MCP resources from connected servers."

    async def prompt(self) -> str:
        """返回该工具按源码保留的模型使用提示词。"""
        return "Lists resources exposed by connected MCP servers. Optionally pass a server name to filter."

    async def call(self, args: dict, context: ToolUseContext, can_use_tool, parent_message: AssistantMessage, on_progress=None) -> ToolResult:
        """执行工具核心逻辑，并返回标准 ToolResult。"""
        target_server = args.get("server")
        clients = [client for client in context.config.mcp_clients if client.type == "connected"]
        if target_server:
            clients = [client for client in clients if client.name == target_server]
            if not clients:
                available = ", ".join(client.name for client in context.config.mcp_clients)
                raise RuntimeError(f'Server "{target_server}" not found. Available servers: {available}')
        resources: list[dict[str, Any]] = []
        for client in clients:
            for resource in client.resources:
                resources.append({**resource, "server": resource.get("server", client.name)})
        return ToolResult(resources)

    def map_tool_result_to_tool_result_block_param(self, content: Any, tool_use_id: str) -> ToolResultBlock:
        """把工具内部结果映射为 Anthropic tool_result block。"""
        if not content:
            text = "No resources found. MCP servers may still provide tools even if they have no resources."
        else:
            text = json.dumps(content, ensure_ascii=False, separators=(",", ":"))
        return {"tool_use_id": tool_use_id, "type": "tool_result", "content": text}


class ReadMcpResourceTool(Tool):
    """实现 ``ReadMcpResourceTool``，并接入统一 Tool 生命周期。"""
    name = "ReadMcpResourceTool"
    search_hint = "read a specific MCP resource by URI"
    input_schema = {"server": str, "uri": str}
    required_fields = ("server", "uri")
    max_result_size_chars = 100_000

    def is_read_only(self, input: dict) -> bool:
        """判断当前输入是否只读取外部状态而不产生修改。"""
        return True

    def is_concurrency_safe(self, input: dict) -> bool:
        """判断当前输入是否可以与相邻安全工具并发执行。"""
        return True

    async def check_permissions(self, input: dict, context: ToolUseContext) -> PermissionDecision:
        """返回该工具针对当前输入的初步权限建议。"""
        return PermissionDecision.allow()

    async def description(self, input: dict | None = None) -> str:
        """返回提供给模型和调用方展示的工具简短说明。"""
        return "Read an MCP resource by server and URI."

    async def prompt(self) -> str:
        """返回该工具按源码保留的模型使用提示词。"""
        return "Reads a resource exposed by a connected MCP server."

    async def call(self, args: dict, context: ToolUseContext, can_use_tool, parent_message: AssistantMessage, on_progress=None) -> ToolResult:
        """执行工具核心逻辑，并返回标准 ToolResult。"""
        server_name = args["server"]
        uri = args["uri"]
        client = next((client for client in context.config.mcp_clients if client.name == server_name), None)
        if client is None:
            available = ", ".join(client.name for client in context.config.mcp_clients)
            raise RuntimeError(f'Server "{server_name}" not found. Available servers: {available}')
        if client.type != "connected":
            raise RuntimeError(f'Server "{server_name}" is not connected')
        if client.read_resource_handler is not None:
            result = client.read_resource_handler(uri)
            if hasattr(result, "__await__"):
                result = await result
            return ToolResult(result)
        resource = next((resource for resource in client.resources if resource.get("uri") == uri), None)
        if resource is None:
            raise RuntimeError(f'Resource "{uri}" not found on server "{server_name}"')
        return ToolResult({"contents": [resource]})

    def map_tool_result_to_tool_result_block_param(self, content: Any, tool_use_id: str) -> ToolResultBlock:
        """把工具内部结果映射为 Anthropic tool_result block。"""
        return {"tool_use_id": tool_use_id, "type": "tool_result", "content": json.dumps(content, ensure_ascii=False, separators=(",", ":"))}


def mcp_tools_from_clients(clients: tuple[MCPClientConfig, ...], *, include_resource_tools: bool = True) -> list[Tool]:
    """从已连接 client 构造稳定命名的工具列表和可选资源工具。"""
    tools: list[Tool] = []
    # 资源 list/read 是全局入口，只需注册一次，不随 server 重复。
    added_resource_tools = False
    for client in clients:
        if client.type != "connected":
            continue
        for tool_def in client.tools:
            tools.append(
                MCPTool(
                    server_name=client.name,
                    tool_name=str(tool_def["name"]),
                    description_text=str(tool_def.get("description") or ""),
                    input_json_schema=tool_def.get("inputSchema") or tool_def.get("input_schema") or {"type": "object", "properties": {}, "additionalProperties": True},
                    annotations=tool_def.get("annotations") or {},
                    meta=tool_def.get("_meta") or {},
                    client_config=client,
                    skip_prefix=bool(tool_def.get("skipPrefix", False)),
                )
            )
        if include_resource_tools and client.resources and not added_resource_tools:
            tools.extend([ListMcpResourcesTool(), ReadMcpResourceTool()])
            added_resource_tools = True
    return tools


async def _call_mcp_tool(client_config: MCPClientConfig, tool_name: str, args: dict[str, Any]) -> Any:
    """完成 ``_call_mcp_tool`` 对应的MCP 适配内部步骤。"""
    # 优先使用显式 handler，便于测试和不依赖特定 MCP SDK 的嵌入方式。
    if client_config.call_tool_handler is not None:
        result = client_config.call_tool_handler(tool_name, args)
        return await result if hasattr(result, "__await__") else result
    client = client_config.client
    if client is None:
        raise RuntimeError(f'MCP server "{client_config.name}" has no call handler.')
    # 兼容 Python 风格、JS 风格和底层 request 三种常见 client surface。
    if hasattr(client, "call_tool"):
        result = client.call_tool(tool_name, args)
        return await result if hasattr(result, "__await__") else result
    if hasattr(client, "callTool"):
        result = client.callTool({"name": tool_name, "arguments": args})
        return await result if hasattr(result, "__await__") else result
    if hasattr(client, "request"):
        result = client.request({"method": "tools/call", "params": {"name": tool_name, "arguments": args}})
        return await result if hasattr(result, "__await__") else result
    raise RuntimeError(f'MCP server "{client_config.name}" has no supported tool call interface.')
