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
import os
import re
import select
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import MCPClientConfig
from .messages import AssistantMessage, ToolResultBlock
from .permissions import PermissionDecision
from .tools.base import Tool, ToolResult, ToolUseContext, ValidationResult


MAX_MCP_DESCRIPTION_LENGTH = 2_000
MCP_CONFIG_ENV = "AGENT_KERNEL_MCP_CONFIG"
MAX_MCP_STDERR_TAIL_CHARS = 4_000
_SECRET_ENV_KEY_RE = re.compile(r"(api[_-]?key|token|secret|password|auth)", re.IGNORECASE)
_SECRET_VALUE_RE = re.compile(r"\b(?:sk|ak|pk|rk)-[A-Za-z0-9_-]{8,}\b")


class MCPConfigurationError(RuntimeError):
    """Raised when an MCP config cannot be loaded into connected clients."""


def _redact_mcp_diagnostic(text: str, env: dict[str, str] | None) -> str:
    """Redact obvious credentials before surfacing MCP process diagnostics."""
    redacted = text
    if env:
        for key, value in env.items():
            if not value or len(value) < 4:
                continue
            if _SECRET_ENV_KEY_RE.search(str(key)):
                redacted = redacted.replace(value, "[REDACTED]")
    return _SECRET_VALUE_RE.sub("[REDACTED]", redacted)


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


class StdioMCPClient:
    """Minimal stdio JSON-RPC client for local-only MCP server configs."""

    def __init__(
        self,
        *,
        name: str,
        command: list[str],
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        timeout_seconds: float = 5.0,
        startup_timeout_seconds: float | None = None,
    ) -> None:
        self.name = name
        self.command = command
        self.cwd = cwd
        self.env = env
        self.timeout_seconds = timeout_seconds
        self.startup_timeout_seconds = startup_timeout_seconds if startup_timeout_seconds is not None else timeout_seconds
        self.process: subprocess.Popen[str] | None = None
        self.next_id = 1
        self.calls: list[dict[str, Any]] = []
        self.stderr_tail = ""

    def start(self) -> None:
        """Start the configured local stdio process."""
        if not self.command or not self.command[0]:
            raise MCPConfigurationError(f'MCP server "{self.name}" command is empty.')
        self.process = subprocess.Popen(
            self.command,
            cwd=str(self.cwd) if self.cwd is not None else None,
            env=self.env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def request(self, method: str, params: dict[str, Any] | None = None, *, timeout_seconds: float | None = None) -> Any:
        """Send one JSON-RPC request and wait for the matching response."""
        if self.process is None:
            self.start()
        if self.process is None or self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError(f'MCP server "{self.name}" is not connected.')
        if self.process.poll() is not None:
            raise RuntimeError(f'MCP server "{self.name}" exited before {method}.')
        message_id = self.next_id
        self.next_id += 1
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": message_id, "method": method}
        if params is not None:
            message["params"] = params
        try:
            self.process.stdin.write(json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n")
            self.process.stdin.flush()
        except BrokenPipeError as exc:
            raise RuntimeError(f'MCP server "{self.name}" stdin closed.') from exc
        timeout = timeout_seconds if timeout_seconds is not None else self.timeout_seconds
        while True:
            readable, _, _ = select.select([self.process.stdout], [], [], timeout)
            if not readable:
                self._capture_stderr_tail()
                suffix = f" stderr={self.stderr_tail}" if self.stderr_tail else ""
                raise TimeoutError(f'MCP server "{self.name}" did not answer {method}.{suffix}')
            line = self.process.stdout.readline()
            if not line:
                self._capture_stderr_tail()
                suffix = f" stderr={self.stderr_tail}" if self.stderr_tail else ""
                raise RuntimeError(f'MCP server "{self.name}" exited before answering {method}.{suffix}')
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                continue
            if response.get("id") != message_id:
                continue
            if "error" in response:
                error = response["error"]
                if isinstance(error, dict):
                    raise RuntimeError(str(error.get("message") or error))
                raise RuntimeError(str(error))
            return response.get("result")

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        """Send one JSON-RPC notification if the process is still alive."""
        process = self.process
        if process is None or process.stdin is None or process.poll() is not None:
            return
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        try:
            process.stdin.write(json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError):
            return

    def initialize(self) -> dict[str, Any]:
        """Initialize the MCP server and send the initialized notification."""
        result = self.request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "agent-kernel-local", "version": "0.4.0"},
            },
            timeout_seconds=self.startup_timeout_seconds,
        )
        self.notify("notifications/initialized")
        return result if isinstance(result, dict) else {}

    def list_tools(self) -> list[dict[str, Any]]:
        """Return tools exposed by the server."""
        result = self.request("tools/list", {}, timeout_seconds=self.startup_timeout_seconds)
        tools = result.get("tools") if isinstance(result, dict) else None
        if not isinstance(tools, list):
            raise RuntimeError(f'MCP server "{self.name}" returned no tools.')
        return [tool for tool in tools if isinstance(tool, dict)]

    def list_resources(self) -> list[dict[str, Any]]:
        """Return resources exposed by the server, or an empty list when unsupported."""
        try:
            result = self.request("resources/list", {}, timeout_seconds=self.startup_timeout_seconds)
        except RuntimeError:
            return []
        resources = result.get("resources") if isinstance(result, dict) else None
        return [resource for resource in resources if isinstance(resource, dict)] if isinstance(resources, list) else []

    def call_tool(self, tool_name: str, args: dict[str, Any]) -> Any:
        """Call a server tool through the JSON-RPC connection."""
        self.calls.append({"tool_name": tool_name, "args": dict(args)})
        return self.request("tools/call", {"name": tool_name, "arguments": args})

    def read_resource(self, uri: str) -> Any:
        """Read a server resource through the JSON-RPC connection."""
        return self.request("resources/read", {"uri": uri})

    def close(self) -> None:
        """Shutdown the process and prevent stdio MCP background leaks."""
        process = self.process
        if process is None:
            return
        if process.poll() is None:
            try:
                self.request("shutdown", {})
            except Exception:
                pass
            self.notify("exit")
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)

    def _capture_stderr_tail(self) -> None:
        process = self.process
        if process is None or process.stderr is None:
            return
        fd = process.stderr.fileno()
        try:
            was_blocking = os.get_blocking(fd)
        except OSError:
            return
        try:
            os.set_blocking(fd, False)
            chunks: list[str] = []
            while True:
                readable, _, _ = select.select([fd], [], [], 0)
                if not readable:
                    break
                try:
                    chunk = os.read(fd, 4096)
                except BlockingIOError:
                    break
                if not chunk:
                    break
                chunks.append(chunk.decode("utf-8", errors="replace"))
        except Exception:
            return
        finally:
            try:
                os.set_blocking(fd, was_blocking)
            except OSError:
                pass
        if chunks:
            diagnostic = _redact_mcp_diagnostic("".join(chunks), self.env)
            self.stderr_tail = (self.stderr_tail + diagnostic)[-MAX_MCP_STDERR_TAIL_CHARS:]


def _resolve_stdio_command(command: str, args: list[str]) -> list[str]:
    executable = sys.executable if command == "python3" else command
    return [executable, *args]


def _load_json_config(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise MCPConfigurationError(f"Unable to read MCP config: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise MCPConfigurationError(f"MCP config is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise MCPConfigurationError("MCP config must be a JSON object.")
    return payload


def _server_entries_from_config(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_servers = payload.get("mcpServers")
    if isinstance(raw_servers, dict):
        if not raw_servers:
            raise MCPConfigurationError("MCP config mcpServers must include at least one server.")
        servers: dict[str, dict[str, Any]] = {}
        normalized_names: dict[str, str] = {}
        for name, config in raw_servers.items():
            server_name = str(name)
            if not isinstance(config, dict):
                raise MCPConfigurationError(f'MCP server "{server_name}" config must be an object.')
            normalized = normalize_name_for_mcp(server_name)
            if normalized in normalized_names:
                raise MCPConfigurationError(
                    f'MCP server "{server_name}" collides with "{normalized_names[normalized]}" after name normalization.'
                )
            normalized_names[normalized] = server_name
            servers[server_name] = config
        return servers
    # Backward-compatible local smoke shape: {"name": "...", "command": "...", "args": [...]}
    if payload.get("command"):
        name = str(payload.get("name") or payload.get("server") or "local")
        return {name: payload}
    raise MCPConfigurationError("MCP config must include an mcpServers object.")


def _positive_timeout(value: Any, *, label: str, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise MCPConfigurationError(f"{label} must be a positive number.")
    return float(value)


def _validate_mcp_tool_and_resource_names(server_name: str, tools: list[dict[str, Any]], resources: list[dict[str, Any]]) -> None:
    seen_tools: dict[str, str] = {}
    for tool in tools:
        name = str(tool.get("name") or "").strip()
        if not name:
            raise MCPConfigurationError(f'MCP server "{server_name}" returned a tool without a name.')
        normalized = normalize_name_for_mcp(name)
        if normalized in seen_tools:
            raise MCPConfigurationError(
                f'MCP server "{server_name}" tool "{name}" collides with "{seen_tools[normalized]}" after name normalization.'
            )
        seen_tools[normalized] = name
    seen_resources: set[str] = set()
    for resource in resources:
        uri = str(resource.get("uri") or "").strip()
        if not uri:
            raise MCPConfigurationError(f'MCP server "{server_name}" returned a resource without a uri.')
        if uri in seen_resources:
            raise MCPConfigurationError(f'MCP server "{server_name}" returned duplicate resource uri "{uri}".')
        seen_resources.add(uri)


def load_mcp_config(
    path: str | Path,
    *,
    cwd: str | Path | None = None,
    startup_timeout_seconds: float | None = None,
    tool_timeout_seconds: float | None = None,
) -> tuple[MCPClientConfig, ...]:
    """Load local-only stdio MCP servers into connected MCPClientConfig objects."""
    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise MCPConfigurationError(f"MCP config does not exist: {config_path}")
    if not config_path.is_file():
        raise MCPConfigurationError(f"MCP config path is not a file: {config_path}")
    payload = _load_json_config(config_path)
    base_cwd = Path(cwd).expanduser() if cwd is not None else Path.cwd()
    clients: list[MCPClientConfig] = []
    started_clients: list[StdioMCPClient] = []
    try:
        for server_name, server_config in _server_entries_from_config(payload).items():
            if server_config.get("disabled") is True:
                continue
            server_type = str(server_config.get("type") or "stdio")
            if server_type != "stdio":
                raise MCPConfigurationError(f'MCP server "{server_name}" uses unsupported type "{server_type}". Only local stdio is supported.')
            command = str(server_config.get("command") or "").strip()
            if not command:
                raise MCPConfigurationError(f'MCP server "{server_name}" must include command.')
            raw_args = server_config.get("args") or []
            if not isinstance(raw_args, list):
                raise MCPConfigurationError(f'MCP server "{server_name}" args must be an array.')
            raw_env = server_config.get("env") or {}
            if not isinstance(raw_env, dict):
                raise MCPConfigurationError(f'MCP server "{server_name}" env must be an object.')
            server_cwd = Path(server_config["cwd"]).expanduser() if server_config.get("cwd") else base_cwd
            default_startup_timeout = startup_timeout_seconds if startup_timeout_seconds is not None else tool_timeout_seconds
            startup_timeout = _positive_timeout(
                server_config.get("startupTimeout", server_config.get("toolTimeout", default_startup_timeout)),
                label=f'MCP server "{server_name}" startup timeout',
                default=5.0,
            )
            tool_timeout = _positive_timeout(
                server_config.get("toolTimeout", tool_timeout_seconds),
                label=f'MCP server "{server_name}" timeout',
                default=startup_timeout,
            )
            env = {**os.environ, **{str(key): str(value) for key, value in raw_env.items()}}
            client = StdioMCPClient(
                name=server_name,
                command=_resolve_stdio_command(command, [str(arg) for arg in raw_args]),
                cwd=server_cwd,
                env=env,
                timeout_seconds=tool_timeout,
                startup_timeout_seconds=startup_timeout,
            )
            client.start()
            started_clients.append(client)
            client.initialize()
            tools = client.list_tools()
            resources = client.list_resources()
            _validate_mcp_tool_and_resource_names(server_name, tools, resources)
            clients.append(
                MCPClientConfig(
                    name=server_name,
                    instructions=str(server_config.get("instructions") or server_config.get("description") or ""),
                    type="connected",
                    tools=tuple(tools),
                    resources=tuple(resources),
                    client=client,
                    call_tool_handler=client.call_tool,
                    read_resource_handler=client.read_resource if resources else None,
                )
            )
    except Exception:
        for client in started_clients:
            client.close()
        raise
    return tuple(clients)


def close_mcp_clients(clients: tuple[MCPClientConfig, ...]) -> None:
    """Close any stdio MCP clients attached to connected config objects."""
    for client_config in clients:
        client = client_config.client
        if hasattr(client, "close"):
            client.close()


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
