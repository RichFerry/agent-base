"""内部消息与 SDK 外围事件之间的纯映射层。

输入是 kernel Message 或 SDK-shaped dict，输出始终是新 dict；这里不访问模型、工具、
session 文件或全局状态。主要能力包括：
- user/assistant 消息及 synthetic/meta 标记双向映射。
- compact boundary metadata 的 camelCase/snake_case 兼容。
- system/init、status、result、error 等 SDK 生命周期事件构造。
- 从最终 assistant messages 提取文本 result。

``build_system_init_message`` 的工具顺序来自 QueryEngine 当前注册表，agents/skills/MCP
只作为描述字段出现。SDK 事件是 opt-in 外围层，核心 ``query()`` 从不依赖本模块，
因此 subagent 和内部测试可直接消费原始事件。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .config import KernelConfig
from .messages import Message
from .tools.base import Tool


EMPTY_USAGE: dict[str, int] = {
    "input_tokens": 0,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
    "output_tokens": 0,
    "server_tool_use": 0,
    "service_tier": 0,
}


def _now_iso() -> str:
    """完成 ``_now_iso`` 对应的SDK 映射内部步骤。"""
    return datetime.now(timezone.utc).isoformat()


def sdk_compat_tool_name(name: str) -> str:
    # Keep emitting the legacy SDK wire name for the Agent tool.
    """完成 ``sdk_compat_tool_name`` 对应的SDK 映射内部步骤。"""
    return "Task" if name == "Agent" else name


def _session_id(message: dict[str, Any], fallback: str) -> str:
    """完成 ``_session_id`` 对应的SDK 映射内部步骤。"""
    return str(message.get("session_id") or message.get("sessionId") or fallback)


def _snake_to_camel_segment(segment: dict[str, Any]) -> dict[str, Any]:
    """完成 ``_snake_to_camel_segment`` 对应的SDK 映射内部步骤。"""
    return {
        "headUuid": segment.get("headUuid") or segment.get("head_uuid"),
        "anchorUuid": segment.get("anchorUuid") or segment.get("anchor_uuid"),
        "tailUuid": segment.get("tailUuid") or segment.get("tail_uuid"),
    }


def _camel_to_snake_segment(segment: dict[str, Any]) -> dict[str, Any]:
    """完成 ``_camel_to_snake_segment`` 对应的SDK 映射内部步骤。"""
    return {
        "head_uuid": segment.get("head_uuid") or segment.get("headUuid"),
        "anchor_uuid": segment.get("anchor_uuid") or segment.get("anchorUuid"),
        "tail_uuid": segment.get("tail_uuid") or segment.get("tailUuid"),
    }


def to_sdk_compact_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """完成 ``to_sdk_compact_metadata`` 对应的SDK 映射内部步骤。"""
    sdk_meta: dict[str, Any] = {
        "trigger": meta.get("trigger"),
        "pre_tokens": meta.get("preTokens", meta.get("pre_tokens")),
    }
    # 同时接受内核 camelCase 与旧 SDK snake_case，便于 replay 老 transcript。
    segment = meta.get("preservedSegment") or meta.get("preserved_segment")
    if isinstance(segment, dict):
        sdk_meta["preserved_segment"] = _camel_to_snake_segment(segment)
    return sdk_meta


def from_sdk_compact_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """完成 ``from_sdk_compact_metadata`` 对应的SDK 映射内部步骤。"""
    internal_meta: dict[str, Any] = dict(meta)
    internal_meta["trigger"] = meta.get("trigger")
    internal_meta["preTokens"] = meta.get("pre_tokens", meta.get("preTokens"))
    internal_meta.pop("pre_tokens", None)
    segment = meta.get("preserved_segment") or meta.get("preservedSegment")
    if isinstance(segment, dict):
        internal_meta["preservedSegment"] = _snake_to_camel_segment(segment)
        internal_meta.pop("preserved_segment", None)
    return internal_meta


def to_sdk_messages(messages: list[Message] | list[dict[str, Any]], *, session_id: str) -> list[dict[str, Any]]:
    """把内部 user/assistant/compact boundary 映射为 SDK 消息。"""
    sdk_messages: list[dict[str, Any]] = []
    for message in messages:
        message_type = message.get("type")
        if message_type == "assistant":
            # SDK 外层补 session/uuid，Anthropic message payload 保持原形。
            sdk_messages.append(
                {
                    "type": "assistant",
                    "message": message.get("message"),
                    "session_id": _session_id(message, session_id),
                    "parent_tool_use_id": None,
                    "uuid": message.get("uuid"),
                    **({"error": message.get("error")} if message.get("error") is not None else {}),
                }
            )
        elif message_type == "user":
            # meta/transcript-only user 在 SDK 中统一表现为 synthetic。
            sdk_message = {
                "type": "user",
                "message": message.get("message"),
                "session_id": _session_id(message, session_id),
                "parent_tool_use_id": None,
                "uuid": message.get("uuid"),
                "timestamp": message.get("timestamp"),
                "isSynthetic": bool(message.get("isMeta") or message.get("isVisibleInTranscriptOnly")),
            }
            if message.get("toolUseResult") is not None:
                sdk_message["tool_use_result"] = message.get("toolUseResult")
            elif message.get("tool_use_result") is not None:
                sdk_message["tool_use_result"] = message.get("tool_use_result")
            sdk_messages.append(sdk_message)
        elif message_type == "system" and message.get("subtype") == "compact_boundary":
            # 普通 system 信息不进入 SDK history，compact boundary 是必要例外。
            metadata = message.get("compactMetadata") or message.get("compact_metadata")
            if isinstance(metadata, dict):
                sdk_messages.append(
                    {
                        "type": "system",
                        "subtype": "compact_boundary",
                        "session_id": _session_id(message, session_id),
                        "uuid": message.get("uuid"),
                        "compact_metadata": to_sdk_compact_metadata(metadata),
                    }
                )
    return sdk_messages


def to_internal_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """读取 SDK/replay 输入，恢复内核可消费的消息字段。"""
    internal_messages: list[dict[str, Any]] = []
    for message in messages:
        message_type = message.get("type")
        if message_type == "assistant":
            internal_messages.append(
                {
                    "type": "assistant",
                    "message": message.get("message"),
                    "uuid": message.get("uuid") or str(uuid4()),
                    "requestId": message.get("requestId"),
                    "timestamp": message.get("timestamp", _now_iso()),
                }
            )
        elif message_type == "user":
            internal = {
                "type": "user",
                "message": message.get("message"),
                "uuid": message.get("uuid") or str(uuid4()),
                "timestamp": message.get("timestamp", _now_iso()),
            }
            if message.get("isSynthetic"):
                internal["isMeta"] = True
            if message.get("tool_use_result") is not None:
                internal["toolUseResult"] = message.get("tool_use_result")
            internal_messages.append(internal)
        elif message_type == "system" and message.get("subtype") == "compact_boundary":
            metadata = message.get("compact_metadata") or message.get("compactMetadata")
            if isinstance(metadata, dict):
                internal_messages.append(
                    {
                        "type": "system",
                        "content": "Conversation compacted",
                        "level": "info",
                        "subtype": "compact_boundary",
                        "compactMetadata": from_sdk_compact_metadata(metadata),
                        "uuid": message.get("uuid") or str(uuid4()),
                        "timestamp": _now_iso(),
                    }
                )
    return internal_messages


def build_system_init_message(
    *,
    config: KernelConfig,
    session_id: str,
    tools: list[Tool],
    model: str,
    permission_mode: str,
    commands: list[dict[str, Any]] | None = None,
    agents: list[dict[str, Any]] | None = None,
    skills: list[dict[str, Any]] | None = None,
    plugins: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """构造 SDK 会话首事件，描述工具、模型、权限和扩展清单。"""
    output_style = config.output_style.name if config.output_style is not None else "default"
    # 可选列表统一成空列表，避免 SDK 调用方反复处理 null。
    commands = commands or []
    agents = agents or []
    skills = skills or []
    plugins = plugins or []
    workspace = config.workspace_runtime
    return {
        "type": "system",
        "subtype": "init",
        "cwd": str(config.cwd),
        "workspace": {
            "root": str(workspace.workspace_root),
            "rootSource": workspace.workspace_root_source,
            "sessionsDir": str(workspace.sessions_dir),
            "memoryScope": workspace.memory_scope,
            "memoryDir": str(workspace.memory_dir) if workspace.memory_dir is not None else None,
            "artifactsDir": str(workspace.artifacts_dir),
            "allowedWorkingDirectories": [str(path) for path in workspace.allowed_working_directories],
        },
        "session_id": session_id,
        "tools": [sdk_compat_tool_name(tool.name) for tool in tools],
        "mcp_servers": [{"name": client.name, "status": client.type} for client in config.mcp_clients],
        "model": model,
        "permissionMode": permission_mode,
        "slash_commands": [command["name"] for command in commands if command.get("userInvocable", True)],
        "apiKeySource": "none",
        "betas": [],
        "claude_code_version": "0.1.0-python-port",
        "output_style": output_style,
        "agents": [agent["agentType"] for agent in agents if "agentType" in agent],
        "skills": [skill["name"] for skill in skills if skill.get("userInvocable", True)],
        "plugins": [
            {"name": plugin["name"], "path": plugin.get("path", ""), "source": plugin.get("source", "")}
            for plugin in plugins
            if "name" in plugin
        ],
        "uuid": str(uuid4()),
    }


def build_sdk_status_message(
    *,
    session_id: str,
    status: dict[str, Any],
    permission_mode: str | None = None,
) -> dict[str, Any]:
    """构造SDK status 消息，供SDK 映射流程使用。"""
    # success/error 共用统计骨架，再根据 is_error 附加 errors 或 result。
    message = {
        "type": "system",
        "subtype": "status",
        "status": status,
        "uuid": str(uuid4()),
        "session_id": session_id,
    }
    if permission_mode is not None:
        message["permissionMode"] = permission_mode
    return message


def build_result_message(
    *,
    session_id: str,
    subtype: str,
    is_error: bool,
    duration_ms: int,
    num_turns: int,
    result: str = "",
    stop_reason: str | None = None,
    errors: list[str] | None = None,
    permission_denials: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """构造结果 消息，供SDK 映射流程使用。"""
    message = {
        "type": "result",
        "subtype": subtype,
        "duration_ms": duration_ms,
        "duration_api_ms": 0,
        "is_error": is_error,
        "num_turns": num_turns,
        "stop_reason": stop_reason,
        "session_id": session_id,
        "total_cost_usd": 0.0,
        "usage": dict(EMPTY_USAGE),
        "modelUsage": {},
        "permission_denials": permission_denials or [],
        "uuid": str(uuid4()),
    }
    if is_error:
        message["errors"] = errors or []
    else:
        message["result"] = result
    return message


def build_error_message(*, session_id: str, error: str) -> dict[str, Any]:
    """构造error 消息，供SDK 映射流程使用。"""
    return {
        "type": "system",
        "subtype": "error",
        "error": error,
        "uuid": str(uuid4()),
        "session_id": session_id,
    }


def extract_text_result(messages: list[Message] | list[dict[str, Any]]) -> str:
    """提取文本 结果，供SDK 映射流程使用。"""
    for message in reversed(messages):
        if message.get("type") != "assistant":
            continue
        payload = message.get("message")
        content = payload.get("content") if isinstance(payload, dict) else None
        if not isinstance(content, list):
            continue
        for block in reversed(content):
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                return block["text"]
    return ""
