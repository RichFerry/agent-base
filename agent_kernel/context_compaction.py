"""上下文预算、摘要压缩、prompt-too-long 恢复和旧工具结果清理。

本模块提供三种层级：
- auto/full compact：调用模型生成结构化摘要，替换较老历史。
- partial compact：摘要旧段，同时原样保留安全的最近消息段。
- microcompact：不调用模型，只清空较旧 tool_result 的大正文。

完整流程会估算 token、选择不切断 tool_use/tool_result 的 split、剥离摘要请求中的
图片、对 compact 自身的 413 做多轮头部裁剪，并生成 compact boundary metadata。
compact 后根据 ReadFileStateEntry 和文件 mtime 恢复仍有效的文件上下文，且受文件数、
字符数和 token budget 三重限制。

返回值同时包含新消息、summary、preserved segment、统计量和 restored_file_state；
query 负责采用结果，SessionStore 负责持久化边界。提示词常量是源码协议正文，不应
因注释或重构改写。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import Any
from uuid import uuid4
import copy
import json
import os
import re

from .config import ContextCompactionConfig
from .messages import AssistantMessage, AttachmentMessage, Message, SystemMessage, UserMessage, create_attachment_message, create_system_message, create_user_message
from .model_provider import ModelProvider


# 摘要请求的输出上限和 auto compact 预留预算来自源码常量。
COMPACT_MAX_OUTPUT_TOKENS = 20_000
MAX_OUTPUT_TOKENS_FOR_SUMMARY = 20_000
AUTOCOMPACT_BUFFER_TOKENS = 13_000
# microcompact 使用稳定占位符，resume 时可识别已清理结果。
TIME_BASED_MC_CLEARED_MESSAGE = "[Old tool result content cleared]"
# PTL retry 删除头部后插入标记，提醒摘要模型历史并非完整起点。
PTL_RETRY_MARKER = "[earlier conversation truncated for compaction retry]"
# 文件恢复有独立上限，防止 compact 后附件再次撑满窗口。
POST_COMPACT_MAX_CHARS_PER_FILE = 80_000
POST_COMPACT_TOKEN_BUDGET = 40_000

NO_TOOLS_PREAMBLE = """CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.

- Do NOT use Read, Bash, Grep, Glob, Edit, Write, or ANY other tool.
- You already have all the context you need in the conversation above.
- Tool calls will be REJECTED and will waste your only turn — you will fail the task.
- Your entire response must be plain text: an <analysis> block followed by a <summary> block.

"""

DETAILED_ANALYSIS_INSTRUCTION_BASE = """Before providing your final summary, wrap your analysis in <analysis> tags to organize your thoughts and ensure you've covered all necessary points. In your analysis process:

1. Chronologically analyze each message and section of the conversation. For each section thoroughly identify:
   - The user's explicit requests and intents
   - Your approach to addressing the user's requests
   - Key decisions, technical concepts and code patterns
   - Specific details like:
     - file names
     - full code snippets
     - function signatures
     - file edits
   - Errors that you ran into and how you fixed them
   - Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
2. Double-check for technical accuracy and completeness, addressing each required element thoroughly."""

BASE_COMPACT_PROMPT = f"""Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions.
This summary should be thorough in capturing technical details, code patterns, and architectural decisions that would be essential for continuing development work without losing context.

{DETAILED_ANALYSIS_INSTRUCTION_BASE}

Your summary should include the following sections:

1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail
2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks discussed.
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. Pay special attention to the most recent messages and include full code snippets where applicable and include a summary of why this file read or edit is important.
4. Errors and fixes: List all errors that you ran into, and how you fixed them. Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
5. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.
6. All user messages: List ALL user messages that are not tool results. These are critical for understanding the users' feedback and changing intent.
7. Pending Tasks: Outline any pending tasks that you have explicitly been asked to work on.
8. Current Work: Describe in detail precisely what was being worked on immediately before this summary request, paying special attention to the most recent messages from both user and assistant. Include file names and code snippets where applicable.
9. Optional Next Step: List the next step that you will take that is related to the most recent work you were doing. IMPORTANT: ensure that this step is DIRECTLY in line with the user's most recent explicit requests, and the task you were working on immediately before this summary request. If your last task was concluded, then only list next steps if they are explicitly in line with the users request. Do not start on tangential requests or really old requests that were already completed without confirming with the user first.
                       If there is a next step, include direct quotes from the most recent conversation showing exactly what task you were working on and where you left off. This should be verbatim to ensure there's no drift in task interpretation.

Here's an example of how your output should be structured:

<example>
<analysis>
[Your thought process, ensuring all points are covered thoroughly and accurately]
</analysis>

<summary>
1. Primary Request and Intent:
   [Detailed description]

2. Key Technical Concepts:
   - [Concept 1]
   - [Concept 2]
   - [...]

3. Files and Code Sections:
   - [File Name 1]
      - [Summary of why this file is important]
      - [Summary of the changes made to this file, if any]
      - [Important Code Snippet]
   - [File Name 2]
      - [Important Code Snippet]
   - [...]

4. Errors and fixes:
    - [Detailed description of error 1]:
      - [How you fixed the error]
      - [User feedback on the error if any]
    - [...]

5. Problem Solving:
   [Description of solved problems and ongoing troubleshooting]

6. All user messages: 
    - [Detailed non tool use user message]
    - [...]

7. Pending Tasks:
   - [Task 1]
   - [Task 2]
   - [...]

8. Current Work:
   [Precise description of current work]

9. Optional Next Step:
   [Optional Next step to take]

</summary>
</example>

Please provide your summary based on the conversation so far, following this structure and ensuring precision and thoroughness in your response. 

There may be additional summarization instructions provided in the included context. If so, remember to follow these instructions when creating the above summary. Examples of instructions include:
<example>
## Compact Instructions
When summarizing the conversation focus on typescript code changes and also remember the mistakes you made and how you fixed them.
</example>

<example>
# Summary instructions
When you are using compact - please focus on test output and code changes. Include file reads verbatim.
</example>
"""

NO_TOOLS_TRAILER = (
    "\n\nREMINDER: Do NOT call any tools. Respond with plain text only — "
    "an <analysis> block followed by a <summary> block. "
    "Tool calls will be rejected and you will fail the task."
)

COMPACT_SYSTEM_PROMPT = "You are a helpful AI assistant tasked with summarizing conversations."


class CompactionError(RuntimeError):
    """表示上下文压缩阶段的专用错误。"""
    pass


@dataclass(frozen=True)
class CompactionResult:
    """一次完整 compact 的新消息、统计量和恢复状态。"""
    boundary_marker: SystemMessage
    summary_messages: list[UserMessage]
    attachments: list[AttachmentMessage]
    messages_to_keep: list[Message]
    messages: list[Message]
    pre_compact_token_count: int
    post_compact_token_count: int
    restored_file_state: dict[str, Any]
    prompt_too_long_retries: int = 0


@dataclass(frozen=True)
class MicrocompactResult:
    """封装 ``MicrocompactResult`` 产生的结构化结果。"""
    messages: list[Message]
    boundary_marker: SystemMessage | None
    compacted_tool_ids: list[str]
    tokens_saved: int


def is_env_truthy(value: str | None) -> bool:
    """判断env truthy，供上下文压缩流程使用。"""
    return value is not None and value.lower() in {"1", "true", "yes", "on"}


def get_auto_compact_threshold(config: ContextCompactionConfig) -> int:
    """获取auto 压缩 threshold，供上下文压缩流程使用。"""
    if config.threshold_tokens is not None:
        return config.threshold_tokens
    reserved = min(config.max_output_tokens, MAX_OUTPUT_TOKENS_FOR_SUMMARY)
    return max(1, config.context_window_tokens - reserved - config.auto_compact_buffer_tokens)


def rough_token_count_estimation(text: str) -> int:
    """完成 ``rough_token_count_estimation`` 对应的上下文压缩内部步骤。"""
    if not text:
        return 0
    return ceil(len(text) / 4)


def _json_token_text(value: Any) -> str:
    """完成 ``_json_token_text`` 对应的上下文压缩内部步骤。"""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def estimate_message_tokens(messages: list[Message]) -> int:
    """用稳定启发式估算 token；仅用于阈值，不替代服务端 usage。"""
    total = 0
    for message in messages:
        if message.get("type") not in {"user", "assistant"}:
            continue
        payload = message.get("message")
        if not isinstance(payload, dict):
            continue
        content = payload.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                total += rough_token_count_estimation(str(block))
                continue
            block_type = block.get("type")
            if block_type == "text":
                total += rough_token_count_estimation(str(block.get("text", "")))
            elif block_type == "tool_result":
                block_content = block.get("content", "")
                if isinstance(block_content, str):
                    total += rough_token_count_estimation(block_content)
                else:
                    total += rough_token_count_estimation(_json_token_text(block_content))
            elif block_type == "tool_use":
                total += rough_token_count_estimation(str(block.get("name", "")) + _json_token_text(block.get("input", {})))
            elif block_type in {"image", "document"}:
                # 无法从 base64 字节稳定估算视觉 token，采用保守固定成本。
                total += 2_000
            elif block_type == "thinking":
                total += rough_token_count_estimation(str(block.get("thinking", "")))
            elif block_type == "redacted_thinking":
                total += rough_token_count_estimation(str(block.get("data", "")))
            else:
                total += rough_token_count_estimation(_json_token_text(block))
    return ceil(total * (4 / 3))


def is_compact_boundary_message(message: dict) -> bool:
    """判断压缩 边界 消息，供上下文压缩流程使用。"""
    return message.get("type") == "system" and message.get("subtype") == "compact_boundary"


def find_last_compact_boundary_index(messages: list[Message]) -> int:
    """查找last 压缩 边界 index，供上下文压缩流程使用。"""
    for index in range(len(messages) - 1, -1, -1):
        if is_compact_boundary_message(messages[index]):
            return index
    return -1


def get_messages_after_compact_boundary(messages: list[Message]) -> list[Message]:
    """获取消息集合 after 压缩 边界，供上下文压缩流程使用。"""
    boundary_index = find_last_compact_boundary_index(messages)
    return messages if boundary_index == -1 else messages[boundary_index:]


def strip_images_from_messages(messages: list[Message]) -> list[Message]:
    """移除images from 消息集合，供上下文压缩流程使用。"""
    stripped: list[Message] = []
    for message in messages:
        if message.get("type") != "user":
            stripped.append(message)
            continue
        payload = message.get("message")
        if not isinstance(payload, dict) or not isinstance(payload.get("content"), list):
            stripped.append(message)
            continue
        has_media_block = False
        new_content: list[dict[str, Any]] = []
        for block in payload["content"]:
            if not isinstance(block, dict):
                new_content.append(block)
                continue
            if block.get("type") == "image":
                has_media_block = True
                new_content.append({"type": "text", "text": "[image]"})
                continue
            if block.get("type") == "document":
                has_media_block = True
                new_content.append({"type": "text", "text": "[document]"})
                continue
            if block.get("type") == "tool_result" and isinstance(block.get("content"), list):
                tool_has_media = False
                tool_content = []
                for item in block["content"]:
                    if isinstance(item, dict) and item.get("type") == "image":
                        tool_has_media = True
                        tool_content.append({"type": "text", "text": "[image]"})
                    elif isinstance(item, dict) and item.get("type") == "document":
                        tool_has_media = True
                        tool_content.append({"type": "text", "text": "[document]"})
                    else:
                        tool_content.append(item)
                if tool_has_media:
                    has_media_block = True
                    new_content.append({**block, "content": tool_content})
                else:
                    new_content.append(block)
                continue
            new_content.append(block)
        if not has_media_block:
            stripped.append(message)
            continue
        new_payload = {**payload, "content": new_content}
        stripped.append({**message, "message": new_payload})  # type: ignore[arg-type]
    return stripped


def get_assistant_message_text(message: AssistantMessage) -> str | None:
    """获取assistant 消息 文本，供上下文压缩流程使用。"""
    content = message.get("message", {}).get("content")
    if not isinstance(content, list):
        return None
    parts = [block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"]
    text = "".join(parts)
    return text if text else None


def get_compact_prompt(custom_instructions: str | None = None) -> str:
    """获取压缩 提示词，供上下文压缩流程使用。"""
    prompt = NO_TOOLS_PREAMBLE + BASE_COMPACT_PROMPT
    if custom_instructions and custom_instructions.strip() != "":
        prompt += f"\n\nAdditional Instructions:\n{custom_instructions}"
    prompt += NO_TOOLS_TRAILER
    return prompt


def format_compact_summary(summary: str) -> str:
    """格式化压缩 summary，供上下文压缩流程使用。"""
    formatted_summary = summary
    formatted_summary = re.sub(r"<analysis>[\s\S]*?</analysis>", "", formatted_summary, count=1)
    summary_match = re.search(r"<summary>([\s\S]*?)</summary>", formatted_summary)
    if summary_match:
        content = summary_match.group(1) or ""
        formatted_summary = re.sub(
            r"<summary>[\s\S]*?</summary>",
            f"Summary:\n{content.strip()}",
            formatted_summary,
            count=1,
        )
    formatted_summary = re.sub(r"\n\n+", "\n\n", formatted_summary)
    return formatted_summary.strip()


def get_compact_user_summary_message(
    summary: str,
    suppress_follow_up_questions: bool = False,
    transcript_path: str | None = None,
    recent_messages_preserved: bool = False,
) -> str:
    """获取压缩 用户 summary 消息，供上下文压缩流程使用。"""
    formatted_summary = format_compact_summary(summary)
    base_summary = f"""This session is being continued from a previous conversation that ran out of context. The summary below covers the earlier portion of the conversation.

{formatted_summary}"""
    if transcript_path:
        base_summary += f"\n\nIf you need specific details from before compaction (like exact code snippets, error messages, or content you generated), read the full transcript at: {transcript_path}"
    if recent_messages_preserved:
        base_summary += "\n\nRecent messages are preserved verbatim."
    if suppress_follow_up_questions:
        return f"""{base_summary}
Continue the conversation from where it left off without asking the user any further questions. Resume directly — do not acknowledge the summary, do not recap what was happening, do not preface with "I'll continue" or similar. Pick up the last task as if the break never happened."""
    return base_summary


def create_compact_boundary_message(
    trigger: str,
    pre_tokens: int,
    last_pre_compact_message_uuid: str | None = None,
    user_context: str | None = None,
    messages_summarized: int | None = None,
    preserved_segment: dict[str, str] | None = None,
) -> SystemMessage:
    """创建压缩 边界 消息，供上下文压缩流程使用。"""
    compact_metadata: dict[str, Any] = {
        "trigger": trigger,
        "preTokens": pre_tokens,
        "userContext": user_context,
        "messagesSummarized": messages_summarized,
    }
    if preserved_segment:
        compact_metadata["preservedSegment"] = preserved_segment
    message: SystemMessage = {
        "type": "system",
        "subtype": "compact_boundary",
        "content": "Conversation compacted",
        "isMeta": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uuid": str(uuid4()),
        "level": "info",
        "compactMetadata": compact_metadata,
    }
    if last_pre_compact_message_uuid:
        message["logicalParentUuid"] = last_pre_compact_message_uuid
    return message


def create_microcompact_boundary_message(
    pre_tokens: int,
    tokens_saved: int,
    compacted_tool_ids: list[str],
) -> SystemMessage:
    """创建microcompact 边界 消息，供上下文压缩流程使用。"""
    return create_system_message(
        "Context microcompacted",
        subtype="microcompact_boundary",
        level="info",
        microcompactMetadata={
            "trigger": "auto",
            "preTokens": pre_tokens,
            "tokensSaved": tokens_saved,
            "compactedToolIds": compacted_tool_ids,
            "clearedAttachmentUUIDs": [],
        },
    )


def should_auto_compact(
    messages: list[Message],
    config: ContextCompactionConfig,
    *,
    query_source: str | None = None,
    snip_tokens_freed: int = 0,
) -> bool:
    """完成 ``should_auto_compact`` 对应的上下文压缩内部步骤。"""
    if query_source in {"session_memory", "compact"}:
        return False
    if is_env_truthy(os.environ.get("DISABLE_COMPACT")) or is_env_truthy(os.environ.get("DISABLE_AUTO_COMPACT")):
        return False
    if not config.enabled:
        return False
    if not messages:
        return False
    token_count = estimate_message_tokens(messages) - snip_tokens_freed
    return token_count >= get_auto_compact_threshold(config)


def _is_prompt_too_long_summary(summary: str) -> bool:
    """判断提示词 too long summary，供上下文压缩流程使用。"""
    lowered = summary.lower()
    return (
        lowered.startswith("prompt is too long")
        or lowered.startswith("conversation too long")
        or "prompt is too long" in lowered[:300]
        or "maximum context" in lowered[:300]
    )


def is_prompt_too_long_error(error: BaseException) -> bool:
    """判断提示词 too long error，供上下文压缩流程使用。"""
    message = str(error).lower()
    error_type = type(error).__name__.lower()
    return (
        "prompt is too long" in message
        or "conversation too long" in message
        or "maximum context" in message
        or "context length" in message
        or "request too large" in message
        or "413" in message
        or "prompt_too_long" in message
        or "prompttoolong" in error_type
    )


def _message_tool_use_ids(message: Message) -> set[str]:
    """完成 ``_message_tool_use_ids`` 对应的上下文压缩内部步骤。"""
    if message.get("type") != "assistant":
        return set()
    content = message.get("message", {}).get("content")
    if not isinstance(content, list):
        return set()
    return {
        str(block["id"])
        for block in content
        if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id")
    }


def _message_tool_result_ids(message: Message) -> set[str]:
    """完成 ``_message_tool_result_ids`` 对应的上下文压缩内部步骤。"""
    if message.get("type") != "user":
        return set()
    content = message.get("message", {}).get("content")
    if not isinstance(content, list):
        return set()
    return {
        str(block["tool_use_id"])
        for block in content
        if isinstance(block, dict) and block.get("type") == "tool_result" and block.get("tool_use_id")
    }


def _tool_use_index(messages: list[Message]) -> dict[str, int]:
    """完成 ``_tool_use_index`` 对应的上下文压缩内部步骤。"""
    index: dict[str, int] = {}
    for message_index, message in enumerate(messages):
        for tool_use_id in _message_tool_use_ids(message):
            index[tool_use_id] = message_index
    return index


def truncate_head_for_prompt_too_long_retry(messages: list[Message]) -> list[Message] | None:
    """compact 自身过长时裁掉最老头部，同时避免拆散工具调用对。"""
    if len(messages) < 2:
        return None
    input_messages = messages
    first = input_messages[0]
    if first.get("type") == "user" and first.get("isMeta"):
        content = first.get("message", {}).get("content", [])
        if (
            isinstance(content, list)
            and content
            and isinstance(content[0], dict)
            and content[0].get("type") == "text"
            and content[0].get("text") == PTL_RETRY_MARKER
        ):
            input_messages = input_messages[1:]
    if len(input_messages) < 2:
        return None
    # 每次约丢弃最老五分之一，逐步给 compact retry 腾出输入空间。
    drop_count = min(max(1, len(input_messages) // 5), len(input_messages) - 1)
    split_index = _safe_head_truncation_index(input_messages, drop_count)
    if split_index <= 0 or split_index >= len(input_messages):
        return None
    sliced = input_messages[split_index:]
    if sliced and sliced[0].get("type") == "assistant":
        return [create_user_message(PTL_RETRY_MARKER, is_meta=True), *sliced]
    return sliced


def _safe_partial_split_index(messages: list[Message], requested_split: int) -> int:
    """完成 ``_safe_partial_split_index`` 对应的上下文压缩内部步骤。"""
    if requested_split <= 0:
        return 0
    if requested_split >= len(messages):
        return len(messages)
    split_index = requested_split
    tool_uses = _tool_use_index(messages)
    changed = True
    # split 左移到所有跨边界 tool id 都闭合的位置。
    while changed:
        changed = False
        kept_result_ids: set[str] = set()
        for message in messages[split_index:]:
            kept_result_ids.update(_message_tool_result_ids(message))
        for tool_use_id in kept_result_ids:
            use_index = tool_uses.get(tool_use_id)
            if use_index is not None and use_index < split_index:
                split_index = use_index
                changed = True
                break
    return split_index


def _safe_head_truncation_index(messages: list[Message], requested_split: int) -> int:
    """完成 ``_safe_head_truncation_index`` 对应的上下文压缩内部步骤。"""
    if requested_split <= 0:
        return 0
    if requested_split >= len(messages):
        return len(messages)
    split_index = requested_split
    tool_uses = _tool_use_index(messages)
    while split_index < len(messages):
        result_ids = _message_tool_result_ids(messages[split_index])
        dangling = [
            tool_use_id
            for tool_use_id in result_ids
            if (tool_uses.get(tool_use_id) is not None and tool_uses[tool_use_id] < split_index)
        ]
        if not dangling:
            break
        split_index += 1
    return split_index


def _split_partial_compact_messages(messages: list[Message], keep_recent: int) -> tuple[list[Message], list[Message]]:
    """完成 ``_split_partial_compact_messages`` 对应的上下文压缩内部步骤。"""
    if keep_recent <= 0 or len(messages) <= keep_recent + 1:
        return messages, []
    split_index = _safe_partial_split_index(messages, len(messages) - keep_recent)
    if split_index <= 0:
        return messages, []
    return messages[:split_index], messages[split_index:]


def _normalize_path_key(path: str) -> str:
    """规范化路径 key，供上下文压缩流程使用。"""
    try:
        return str(Path(path).expanduser().resolve())
    except OSError:
        return str(Path(path).expanduser())


def _collect_preserved_read_paths(messages: list[Message]) -> set[str]:
    """收集preserved 读取 路径集合，供上下文压缩流程使用。"""
    paths: set[str] = set()
    for message in messages:
        if message.get("type") != "assistant":
            continue
        content = message.get("message", {}).get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use" or block.get("name") != "Read":
                continue
            input_value = block.get("input")
            if isinstance(input_value, dict) and isinstance(input_value.get("file_path"), str):
                paths.add(_normalize_path_key(input_value["file_path"]))
    return paths


def _file_mtime_ms(path: Path) -> int:
    """完成 ``_file_mtime_ms`` 对应的上下文压缩内部步骤。"""
    return int(path.stat().st_mtime * 1000)


def _read_restored_file_state(file_path: str, state: Any) -> tuple[str, Any] | None:
    """读取restored 文件 状态，供上下文压缩流程使用。"""
    from .tools.base import ReadFileStateEntry

    original_content = getattr(state, "content", "")
    if not isinstance(original_content, str):
        return None
    path = Path(file_path).expanduser()
    content = original_content
    timestamp = int(getattr(state, "timestamp", 0) or 0)
    try:
        if path.exists() and path.is_file():
            raw_bytes = path.read_bytes()
            if b"\x00" in raw_bytes[:8192]:
                return None
            content = raw_bytes.decode("utf-8", errors="replace").replace("\r\n", "\n")
            timestamp = _file_mtime_ms(path)
    except OSError:
        content = original_content
    truncated = content[:POST_COMPACT_MAX_CHARS_PER_FILE]
    is_partial = bool(getattr(state, "is_partial_view", False)) or len(content) > len(truncated)
    restored = ReadFileStateEntry(
        content=truncated if is_partial else content,
        timestamp=timestamp,
        offset=getattr(state, "offset", None),
        limit=getattr(state, "limit", None),
        is_partial_view=is_partial,
    )
    return truncated, restored


def create_post_compact_file_attachments(
    read_file_state: dict[str, Any] | None,
    max_files: int,
    preserved_messages: list[Message] | None = None,
) -> tuple[list[AttachmentMessage], dict[str, Any]]:
    """创建post 压缩 文件 attachments，供上下文压缩流程使用。"""
    if not read_file_state or max_files <= 0:
        return [], {}
    attachments: list[AttachmentMessage] = []
    restored_file_state: dict[str, Any] = {}
    # preserved tail 已经含有的 Read 不再额外生成附件，避免重复上下文。
    preserved_paths = _collect_preserved_read_paths(preserved_messages or [])
    for file_path, state in read_file_state.items():
        if _normalize_path_key(file_path) not in preserved_paths:
            continue
        restored = _read_restored_file_state(file_path, state)
        if restored is not None:
            _, restored_state = restored
            restored_file_state[file_path] = restored_state
    candidates = [
        (file_path, state)
        for file_path, state in read_file_state.items()
        if _normalize_path_key(file_path) not in preserved_paths
    ]
    candidates.sort(key=lambda item: int(getattr(item[1], "timestamp", 0) or 0), reverse=True)
    used_tokens = 0
    for file_path, state in candidates[:max_files]:
        # 重新读取磁盘而非盲信旧 content，保留用户在 compact 期间的外部修改。
        restored = _read_restored_file_state(file_path, state)
        if restored is None:
            continue
        restored_content, restored_state = restored
        suffix = "\n[...truncated...]" if restored_state.is_partial_view else ""
        attachment = create_attachment_message(
            f"Post-compact restored file context for `{file_path}`:\n\n{restored_content}{suffix}",
            attachment_type="file",
            metadata={
                "filePath": file_path,
                "restoredAfterCompact": True,
                "isPartialView": restored_state.is_partial_view,
            },
        )
        attachment_tokens = rough_token_count_estimation(_json_token_text(attachment))
        if used_tokens + attachment_tokens > POST_COMPACT_TOKEN_BUDGET:
            continue
        used_tokens += attachment_tokens
        attachments.append(attachment)
        restored_file_state[file_path] = restored_state
    return attachments, restored_file_state


async def _stream_compact_summary(
    *,
    messages: list[Message],
    summary_request: UserMessage,
    model_provider: ModelProvider,
    model: str,
    config: ContextCompactionConfig,
    abort_signal: Any | None = None,
) -> AssistantMessage | None:
    """完成 ``_stream_compact_summary`` 对应的上下文压缩内部步骤。"""
    if abort_signal is not None and hasattr(abort_signal, "throw_if_aborted"):
        abort_signal.throw_if_aborted()
    summary_response: AssistantMessage | None = None
    async for response in model_provider.stream(
        messages=strip_images_from_messages([*get_messages_after_compact_boundary(messages), summary_request]),
        system_prompt=[COMPACT_SYSTEM_PROMPT],
        tools=[],
        options={
            "model": model,
            "querySource": "compact",
            "max_tokens": min(config.max_output_tokens, COMPACT_MAX_OUTPUT_TOKENS),
            "abortSignal": abort_signal,
        },
    ):
        if abort_signal is not None and hasattr(abort_signal, "throw_if_aborted"):
            abort_signal.throw_if_aborted()
        summary_response = response
    return summary_response


async def compact_conversation(
    messages: list[Message],
    *,
    model_provider: ModelProvider,
    model: str,
    config: ContextCompactionConfig,
    transcript_path: str | None = None,
    is_auto_compact: bool = True,
    read_file_state: dict[str, Any] | None = None,
    abort_signal: Any | None = None,
) -> CompactionResult:
    """执行摘要请求、prompt-too-long retry、partial preserve 和文件恢复。"""
    if abort_signal is not None and hasattr(abort_signal, "throw_if_aborted"):
        abort_signal.throw_if_aborted()
    if not messages:
        raise CompactionError("Not enough messages to compact.")

    pre_compact_token_count = estimate_message_tokens(messages)
    messages_to_summarize, messages_to_keep = _split_partial_compact_messages(messages, config.partial_keep_recent_messages)
    summary_request = create_user_message(get_compact_prompt(config.custom_instructions))
    summary_response: AssistantMessage | None = None
    ptl_attempts = 0
    # compact 模型也可能 prompt-too-long；每次重试进一步裁剪最老头部。
    while True:
        try:
            summary_response = await _stream_compact_summary(
                messages=messages_to_summarize,
                summary_request=summary_request,
                model_provider=model_provider,
                model=model,
                config=config,
                abort_signal=abort_signal,
            )
        except Exception as exc:
            if not is_prompt_too_long_error(exc):
                raise
            ptl_attempts += 1
            truncated = (
                truncate_head_for_prompt_too_long_retry(messages_to_summarize)
                if ptl_attempts <= config.max_prompt_too_long_retries
                else None
            )
            if truncated is None:
                raise CompactionError("Conversation too long. Press esc twice to go up a few messages and try again.") from exc
            messages_to_summarize = truncated
            continue
        if summary_response is None:
            raise CompactionError("Compaction interrupted · This may be due to network issues — please try again.")
        summary = get_assistant_message_text(summary_response)
        if not summary:
            raise CompactionError("Failed to generate conversation summary - response did not contain valid text content")
        if not _is_prompt_too_long_summary(summary):
            break
        ptl_attempts += 1
        truncated = (
            truncate_head_for_prompt_too_long_retry(messages_to_summarize)
            if ptl_attempts <= config.max_prompt_too_long_retries
            else None
        )
        if truncated is None:
            raise CompactionError("Conversation too long. Press esc twice to go up a few messages and try again.")
        messages_to_summarize = truncated

    if summary_response is None:
        raise CompactionError("Compaction interrupted · This may be due to network issues — please try again.")

    summary = get_assistant_message_text(summary_response)
    if not summary:
        raise CompactionError("Failed to generate conversation summary - response did not contain valid text content")
    if summary.startswith("API Error"):
        raise CompactionError(summary)

    # 摘要作为 synthetic user 回灌，保持下一条 assistant 的角色交替。
    summary_message = create_user_message(
        get_compact_user_summary_message(
            summary,
            config.suppress_follow_up_questions,
            transcript_path,
            recent_messages_preserved=bool(messages_to_keep),
        ),
        is_compact_summary=True,
        is_visible_in_transcript_only=True,
    )
    preserved_segment = None
    if messages_to_keep:
        preserved_segment = {
            "headUuid": messages_to_keep[0]["uuid"],
            "anchorUuid": summary_message["uuid"],
            "tailUuid": messages_to_keep[-1]["uuid"],
        }
    boundary_marker = create_compact_boundary_message(
        "auto" if is_auto_compact else "manual",
        pre_compact_token_count,
        messages_to_summarize[-1].get("uuid"),
        messages_summarized=len(messages_to_summarize),
        preserved_segment=preserved_segment,
    )
    summary_messages = [summary_message]
    attachments, restored_file_state = create_post_compact_file_attachments(
        read_file_state,
        config.post_compact_max_files_to_restore,
        messages_to_keep,
    )
    compacted_messages: list[Message] = [boundary_marker, *summary_messages, *messages_to_keep, *attachments]
    return CompactionResult(
        boundary_marker=boundary_marker,
        summary_messages=summary_messages,
        attachments=attachments,
        messages_to_keep=messages_to_keep,
        messages=compacted_messages,
        pre_compact_token_count=pre_compact_token_count,
        post_compact_token_count=estimate_message_tokens(compacted_messages),
        restored_file_state=restored_file_state,
        prompt_too_long_retries=ptl_attempts,
    )


def _iter_tool_result_blocks(messages: list[Message]) -> list[tuple[int, int, dict[str, Any]]]:
    """遍历工具 结果 blocks，供上下文压缩流程使用。"""
    blocks: list[tuple[int, int, dict[str, Any]]] = []
    for message_index, message in enumerate(messages):
        if message.get("type") != "user":
            continue
        content = message.get("message", {}).get("content")
        if not isinstance(content, list):
            continue
        for block_index, block in enumerate(content):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                blocks.append((message_index, block_index, block))
    return blocks


def microcompact_messages(messages: list[Message], keep_recent: int = 3) -> MicrocompactResult:
    """清旧 tool_result 正文，并生成可持久化的 microcompact boundary。"""
    tool_result_blocks = _iter_tool_result_blocks(messages)
    # 最近结果最可能仍被模型引用，只处理 keep_recent 之前的候选。
    if len(tool_result_blocks) <= keep_recent:
        return MicrocompactResult(messages, None, [], 0)
    pre_tokens = estimate_message_tokens(messages)
    compacted = copy.deepcopy(messages)
    compacted_tool_ids: list[str] = []
    for message_index, block_index, block in tool_result_blocks[: max(0, len(tool_result_blocks) - keep_recent)]:
        if block.get("content") == TIME_BASED_MC_CLEARED_MESSAGE:
            continue
        target = compacted[message_index]["message"]["content"][block_index]  # type: ignore[index]
        original_content = target.get("content", "")
        target["content"] = TIME_BASED_MC_CLEARED_MESSAGE
        target["is_error"] = bool(target.get("is_error", False))
        compacted_tool_ids.append(str(target.get("tool_use_id", "")))
        if isinstance(original_content, str):
            target["clearedContentTokens"] = rough_token_count_estimation(original_content)
        else:
            target["clearedContentTokens"] = rough_token_count_estimation(_json_token_text(original_content))
    if not compacted_tool_ids:
        return MicrocompactResult(messages, None, [], 0)
    post_tokens = estimate_message_tokens(compacted)
    tokens_saved = max(0, pre_tokens - post_tokens)
    boundary = create_microcompact_boundary_message(pre_tokens, tokens_saved, compacted_tool_ids)
    return MicrocompactResult([boundary, *compacted], boundary, compacted_tool_ids, tokens_saved)
