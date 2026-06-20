"""内核消息协议、构造函数与模型 API 边界归一化。

消息分为两层：
- 内部层：user、assistant、system、attachment、tombstone，携带 uuid、meta、
  compact metadata、sourceToolAssistantUUID 等 transcript 信息。
- API 层：只保留 role=user/assistant 及 Anthropic content blocks。

公开构造函数保证每条消息都有稳定 uuid；tool_result 必须记录其来源 assistant uuid，
这样 JSONL parent 链才能在 resume 时复原。``normalize_messages_for_api`` 会保留
attachment 内容、合并连续 user turn、合并同 message id 的 assistant 流式分片，并
清除当前模型不支持的 tool-search 字段。随后 ``ensure_tool_result_pairing`` 处理异常
中断、compact 或旧 transcript 造成的配对损坏。

重要不变量：每个 tool_use id 全局唯一，并在紧随的 user turn 中恰好有一个对应
tool_result。内部 transcript 不为迎合 API 而原地修改；修复只发生在发送副本上。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, NotRequired, TypedDict
from uuid import uuid4


class TextBlock(TypedDict):
    """描述 ``TextBlock`` 的静态消息字段结构。"""
    type: Literal["text"]
    text: str


class ToolUseBlock(TypedDict):
    """描述 ``ToolUseBlock`` 的静态消息字段结构。"""
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any]


class ToolResultBlock(TypedDict, total=False):
    """描述 ``ToolResultBlock`` 的静态消息字段结构。"""
    type: Literal["tool_result"]
    tool_use_id: str
    content: str | list[dict[str, Any]]
    is_error: NotRequired[bool]


ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock


class UserPayload(TypedDict):
    """描述 ``UserPayload`` 的静态消息字段结构。"""
    role: Literal["user"]
    content: list[ContentBlock]


class AssistantPayload(TypedDict):
    """描述 ``AssistantPayload`` 的静态消息字段结构。"""
    id: str
    role: Literal["assistant"]
    content: list[ContentBlock]


class UserMessage(TypedDict):
    """描述 ``UserMessage`` 的静态消息字段结构。"""
    type: Literal["user"]
    uuid: str
    message: UserPayload
    isMeta: NotRequired[bool]
    isCompactSummary: NotRequired[bool]
    isVisibleInTranscriptOnly: NotRequired[bool]
    sourceToolAssistantUUID: NotRequired[str]


class AssistantMessage(TypedDict):
    """描述 ``AssistantMessage`` 的静态消息字段结构。"""
    type: Literal["assistant"]
    uuid: str
    message: AssistantPayload


class SystemMessage(TypedDict):
    """描述 ``SystemMessage`` 的静态消息字段结构。"""
    type: Literal["system"]
    uuid: str
    content: str
    subtype: NotRequired[str]
    isMeta: NotRequired[bool]
    timestamp: NotRequired[str]
    level: NotRequired[str]
    compactMetadata: NotRequired[dict[str, Any]]
    microcompactMetadata: NotRequired[dict[str, Any]]
    logicalParentUuid: NotRequired[str]
    error: NotRequired[str]


class AttachmentMessage(TypedDict, total=False):
    """描述 ``AttachmentMessage`` 的静态消息字段结构。"""
    type: Literal["attachment"]
    uuid: str
    message: dict[str, Any]


class TombstoneMessage(TypedDict):
    """描述 ``TombstoneMessage`` 的静态消息字段结构。"""
    type: Literal["tombstone"]
    message: AssistantMessage


Message = UserMessage | AssistantMessage | SystemMessage | AttachmentMessage | TombstoneMessage

# 这些固定文本属于中断和配对修复协议，可能被 UI、resume 或测试识别。
INTERRUPT_MESSAGE = "[Request interrupted by user]"
INTERRUPT_MESSAGE_FOR_TOOL_USE = "[Request interrupted by user for tool use]"
SYNTHETIC_TOOL_RESULT_PLACEHOLDER = "[Tool result missing due to internal error]"
ORPHANED_TOOL_RESULT_PLACEHOLDER = "[Orphaned tool result removed due to conversation resume]"
NO_CONTENT_MESSAGE = "(no content)"


def text_block(text: str) -> TextBlock:
    """完成 ``text_block`` 对应的消息协议内部步骤。"""
    return {"type": "text", "text": text}


def create_user_message(
    content: str | list[ContentBlock],
    *,
    uuid: str | None = None,
    is_meta: bool = False,
    is_compact_summary: bool = False,
    is_visible_in_transcript_only: bool = False,
) -> UserMessage:
    """创建用户 消息，供消息协议流程使用。"""
    blocks: list[ContentBlock]
    if isinstance(content, str):
        blocks = [text_block(content)]
    else:
        blocks = content
    msg: UserMessage = {
        "type": "user",
        "uuid": uuid or str(uuid4()),
        "message": {"role": "user", "content": blocks},
    }
    if is_meta:
        msg["isMeta"] = True
    if is_compact_summary:
        msg["isCompactSummary"] = True
    if is_visible_in_transcript_only:
        msg["isVisibleInTranscriptOnly"] = True
    return msg


def create_assistant_message(
    content: str | list[ContentBlock],
    *,
    uuid: str | None = None,
    message_id: str | None = None,
) -> AssistantMessage:
    """创建assistant 消息，供消息协议流程使用。"""
    blocks: list[ContentBlock]
    if isinstance(content, str):
        blocks = [text_block(content)]
    else:
        blocks = content
    return {
        "type": "assistant",
        "uuid": uuid or str(uuid4()),
        "message": {
            "id": message_id or f"msg_{uuid4().hex}",
            "role": "assistant",
            "content": blocks,
        },
    }


def create_tool_result_message(
    block: ToolResultBlock,
    *,
    uuid: str | None = None,
    source_tool_assistant_uuid: str | None = None,
) -> UserMessage:
    """创建工具 结果 消息，供消息协议流程使用。"""
    message = create_user_message([block], uuid=uuid)
    if source_tool_assistant_uuid:
        message["sourceToolAssistantUUID"] = source_tool_assistant_uuid
    return message


def create_user_interruption_message(*, tool_use: bool = False) -> UserMessage:
    """创建用户 interruption 消息，供消息协议流程使用。"""
    return create_user_message(INTERRUPT_MESSAGE_FOR_TOOL_USE if tool_use else INTERRUPT_MESSAGE)


def normalize_messages_for_api(messages: list[Message]) -> list[Message]:
    """生成 API 可接受的 user/assistant 序列，但不修改 transcript 原对象。

    这里会合并连续 user turn、合并相同 message id 的 assistant 流式分片、把
    attachment 作为 user context 保留下来，并移除未开启 tool-search beta 时 API
    不接受的 caller/tool_reference 字段。
    """
    # 始终构造新列表，不能为了 API 兼容修改 transcript 中的原始消息。
    normalized: list[Message] = []
    for message in messages:
        payload = message.get("message")
        role = payload.get("role") if isinstance(payload, dict) else None
        content = payload.get("content") if isinstance(payload, dict) else None
        if role not in {"user", "assistant"} or not isinstance(content, list):
            continue

        api_message: Message
        if message.get("type") == role:
            api_message = message
        else:
            # attachment 的 payload 本身是 user role；这里只投影类型，不丢正文。
            api_message = {
                "type": role,
                "uuid": message.get("uuid") or str(uuid4()),
                "message": payload,
            }

        if role == "user":
            # tool_reference 依赖特定 beta；默认 API 请求必须删除它。
            normalized_content: list[dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result" or not isinstance(block.get("content"), list):
                    normalized_content.append(block)
                    continue
                inner_content = [
                    inner
                    for inner in block["content"]
                    if not (isinstance(inner, dict) and inner.get("type") == "tool_reference")
                ]
                normalized_content.append(
                    {
                        **block,
                        "content": inner_content
                        or [{"type": "text", "text": "[Tool references removed - tool search not enabled]"}],
                    }
                )
            if normalized_content != content:
                api_message = {**api_message, "message": {**payload, "content": normalized_content}}
            if normalized and normalized[-1].get("type") == "user":
                # Bedrock 等后端不接受连续 user turn，提前合并也便于 pairing 扫描。
                previous = normalized[-1]
                previous_payload = previous.get("message")
                previous_content = previous_payload.get("content") if isinstance(previous_payload, dict) else None
                if isinstance(previous_content, list):
                    normalized[-1] = {
                        **previous,
                        "message": {
                            **previous_payload,
                            "content": [*previous_content, *normalized_content],
                        },
                    }
                    continue
            normalized.append(api_message)
            continue

        # caller 等派生字段不能回传给未开启 tool-search beta 的模型。
        normalized_assistant_content: list[dict[str, Any]] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                normalized_assistant_content.append(
                    {
                        "type": "tool_use",
                        "id": block.get("id"),
                        "name": block.get("name"),
                        "input": block.get("input") if isinstance(block.get("input"), dict) else {},
                    }
                )
            else:
                normalized_assistant_content.append(block)
        if normalized_assistant_content != content:
            api_message = {**api_message, "message": {**payload, "content": normalized_assistant_content}}

        message_id = payload.get("id")
        merged = False
        # 同一 API response 的流式 blocks 可能被多条内部消息承载，只按 message id 合并。
        for previous_index in range(len(normalized) - 1, -1, -1):
            previous = normalized[previous_index]
            if previous.get("type") == "assistant":
                previous_payload = previous.get("message")
                if isinstance(previous_payload, dict) and message_id and previous_payload.get("id") == message_id:
                    previous_content = previous_payload.get("content")
                    if isinstance(previous_content, list):
                        normalized[previous_index] = {
                            **previous,
                            "message": {
                                **previous_payload,
                                "content": [*previous_content, *normalized_assistant_content],
                            },
                        }
                        merged = True
                    break
                continue
            previous_payload = previous.get("message")
            previous_content = previous_payload.get("content") if isinstance(previous_payload, dict) else None
            is_tool_result = isinstance(previous_content, list) and any(
                isinstance(block, dict) and block.get("type") == "tool_result" for block in previous_content
            )
            if not is_tool_result:
                break
        if not merged:
            normalized.append(api_message)
    return normalized


def ensure_tool_result_pairing(messages: list[Message]) -> list[Message]:
    """防御性修复 tool_use/tool_result 的双向配对关系。

    正向缺失会插入 synthetic error result；反向孤儿和重复 result 会被删除；重复
    tool_use id 也只保留第一次。正常消息不会被复制或改写，因此不会破坏 prompt
    cache 的字节稳定性。
    """
    result: list[Message] = []
    # 跨 assistant message 去重，修复旧 transcript 重复写入相同 tool_use id 的情况。
    all_seen_tool_use_ids: set[str] = set()
    index = 0

    while index < len(messages):
        message = messages[index]
        if message.get("type") != "assistant":
            # 开头或连续 user turn 中的 tool_result 没有可配对 assistant，应剥离。
            if message.get("type") == "user" and (not result or result[-1].get("type") != "assistant"):
                payload = message.get("message")
                content = payload.get("content") if isinstance(payload, dict) else None
                if isinstance(content, list):
                    stripped = [block for block in content if not (isinstance(block, dict) and block.get("type") == "tool_result")]
                    if len(stripped) != len(content):
                        if stripped:
                            result.append({**message, "message": {**payload, "content": stripped}})
                        elif not result:
                            result.append(
                                {
                                    **message,
                                    "message": {
                                        **payload,
                                        "content": [text_block(ORPHANED_TOOL_RESULT_PLACEHOLDER)],
                                    },
                                }
                            )
                        index += 1
                        continue
            result.append(message)
            index += 1
            continue

        payload = message.get("message")
        content = payload.get("content") if isinstance(payload, dict) else None
        if not isinstance(content, list):
            result.append(message)
            index += 1
            continue

        # server-side tool use 的结果与 use block 位于同一 assistant 内容数组。
        server_result_ids = {
            str(block["tool_use_id"])
            for block in content
            if isinstance(block, dict) and isinstance(block.get("tool_use_id"), str)
        }
        seen_tool_use_ids: list[str] = []
        seen_tool_use_id_set: set[str] = set()
        final_content: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                final_content.append(block)
                continue
            if block.get("type") == "tool_use":
                tool_use_id = block.get("id")
                if isinstance(tool_use_id, str):
                    if tool_use_id in all_seen_tool_use_ids:
                        continue
                    all_seen_tool_use_ids.add(tool_use_id)
                    seen_tool_use_ids.append(tool_use_id)
                    seen_tool_use_id_set.add(tool_use_id)
            if block.get("type") in {"server_tool_use", "mcp_tool_use"} and block.get("id") not in server_result_ids:
                continue
            final_content.append(block)

        if not final_content:
            final_content = [text_block("[Tool use interrupted]")]
        assistant_message: Message = (
            message
            if len(final_content) == len(content)
            else {**message, "message": {**payload, "content": final_content}}
        )
        result.append(assistant_message)

        next_message = messages[index + 1] if index + 1 < len(messages) else None
        next_payload = next_message.get("message") if isinstance(next_message, dict) else None
        next_content = next_payload.get("content") if isinstance(next_payload, dict) else None
        existing_result_ids: set[str] = set()
        duplicate_result_ids: set[str] = set()
        if next_message is not None and next_message.get("type") == "user" and isinstance(next_content, list):
            for block in next_content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                tool_use_id = block.get("tool_use_id")
                if not isinstance(tool_use_id, str):
                    continue
                if tool_use_id in existing_result_ids:
                    duplicate_result_ids.add(tool_use_id)
                existing_result_ids.add(tool_use_id)

        # 正向补缺失，反向删孤儿；两种错误都常见于中断和旧版 resume。
        missing_ids = [tool_use_id for tool_use_id in seen_tool_use_ids if tool_use_id not in existing_result_ids]
        orphaned_ids = existing_result_ids - seen_tool_use_id_set
        if not missing_ids and not orphaned_ids and not duplicate_result_ids:
            index += 1
            continue

        # 占位结果只保证协议可继续，不伪装成工具真实输出。
        synthetic_blocks: list[ToolResultBlock] = [
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": SYNTHETIC_TOOL_RESULT_PLACEHOLDER,
                "is_error": True,
            }
            for tool_use_id in missing_ids
        ]
        if next_message is not None and next_message.get("type") == "user" and isinstance(next_content, list):
            filtered_content: list[dict[str, Any]] = []
            seen_result_ids: set[str] = set()
            for block in next_content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tool_use_id = block.get("tool_use_id")
                    if tool_use_id in orphaned_ids or tool_use_id in seen_result_ids:
                        continue
                    if isinstance(tool_use_id, str):
                        seen_result_ids.add(tool_use_id)
                filtered_content.append(block)
            patched_content = [*synthetic_blocks, *filtered_content]
            result.append(
                {
                    **next_message,
                    "message": {
                        **next_payload,
                        "content": patched_content or [text_block(NO_CONTENT_MESSAGE)],
                    },
                }
            )
            index += 2
            continue
        if synthetic_blocks:
            result.append(create_user_message(synthetic_blocks, is_meta=True))
        index += 1

    return result


@dataclass(frozen=True)
class Terminal:
    """封装 ``Terminal`` 对应的消息协议状态与行为。"""
    reason: Literal["completed", "max_turns", "aborted", "error", "hook_stopped"]
    turns: int
    message: str | None = None


def create_system_message(
    content: str,
    *,
    subtype: str = "informational",
    level: str = "info",
    uuid: str | None = None,
    **extra: Any,
) -> SystemMessage:
    """创建system 消息，供消息协议流程使用。"""
    from datetime import datetime, timezone

    return {
        "type": "system",
        "uuid": uuid or str(uuid4()),
        "content": content,
        "subtype": subtype,
        "level": level,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "isMeta": False,
        **extra,
    }


def create_attachment_message(
    text: str,
    *,
    attachment_type: str = "text",
    uuid: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> AttachmentMessage:
    """创建attachment 消息，供消息协议流程使用。"""
    return {
        "type": "attachment",
        "uuid": uuid or str(uuid4()),
        "message": {
            "role": "user",
            "content": [text_block(text)],
            "attachment": {
                "type": attachment_type,
                **(metadata or {}),
            },
        },
    }
