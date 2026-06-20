"""核心 agent loop，也是内核行为语义最集中的模块。

一轮循环的固定阶段：
1. 检查 abort，并按配置执行 microcompact/auto compact。
2. 拼接动态 system/user context，yield ``stream_request_start``。
3. 调用 ModelProvider，收集本轮所有 assistant message 和 tool_use block。
4. 若无 tool_use，运行 Stop hook 并返回 completed terminal。
5. 若有 tool_use，调用 ``run_tools``，把 user/tool_result 回灌历史后进入下一轮。

恢复路径也保持消息协议：prompt-too-long 会先 compact 再重试；模型在发出 tool_use 后
异常或取消时，会生成缺失的 error tool_result；工具中断会补结果和源码同文案的用户
interruption message。这样 transcript 的每个中间状态都能继续发送给模型。

本模块不写文件、不决定权限、不构造完整 prompt，也不知道 HTTP。它只协调依赖并按
发生顺序 yield message/progress/context/terminal 事件。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator

from .config import ContextCompactionConfig
from .context_compaction import compact_conversation, is_prompt_too_long_error, microcompact_messages, should_auto_compact
from .hooks import HookResult, hook_blocking_message, run_hook_event, stop_failure_hook_input, stop_hook_input
from .messages import (
    AssistantMessage,
    Message,
    Terminal,
    create_attachment_message,
    create_system_message,
    create_tool_result_message,
    create_user_interruption_message,
    create_user_message,
)
from .model_provider import ModelProvider
from .prompt_composer import append_system_context, prepend_user_context
from .tool_execution import run_tools
from .tools.base import ToolUseContext


def _hook_additional_context_message(result: HookResult, event: str) -> dict | None:
    """完成 ``_hook_additional_context_message`` 对应的agent loop内部步骤。"""
    if not result.additional_context:
        return None
    contexts = result.additional_context if isinstance(result.additional_context, list) else [result.additional_context]
    return create_attachment_message(
        "\n".join(str(item) for item in contexts),
        attachment_type="hook_additional_context",
        metadata={"hookName": event, "hookEvent": event},
    )


def _hook_stopped_message(event: str, reason: str | None) -> dict:
    """完成 ``_hook_stopped_message`` 对应的agent loop内部步骤。"""
    message = reason or f"{event} hook prevented continuation"
    return create_attachment_message(
        message,
        attachment_type="hook_stopped_continuation",
        metadata={"message": message, "hookName": event, "hookEvent": event},
    )


def _is_hook_stopped_message(message: dict) -> bool:
    """判断hook stopped 消息，供agent loop流程使用。"""
    payload = message.get("message")
    attachment = payload.get("attachment") if isinstance(payload, dict) else None
    return isinstance(attachment, dict) and attachment.get("type") == "hook_stopped_continuation"


def _abort_reason(params: "QueryParams") -> str:
    """完成 ``_abort_reason`` 对应的agent loop内部步骤。"""
    reason = params.tool_use_context.abort_controller.signal.reason
    return str(reason or "Request was aborted.")


def _raise_if_aborted(params: "QueryParams") -> None:
    """完成 ``_raise_if_aborted`` 对应的agent loop内部步骤。"""
    params.tool_use_context.abort_controller.signal.throw_if_aborted()


def _aborted_terminal(params: "QueryParams", turn_count: int) -> dict:
    """完成 ``_aborted_terminal`` 对应的agent loop内部步骤。"""
    return {"type": "terminal", "terminal": Terminal("aborted", turn_count, _abort_reason(params)).__dict__}


def _tool_result_ids(messages: list[dict]) -> set[str]:
    """完成 ``_tool_result_ids`` 对应的agent loop内部步骤。"""
    result: set[str] = set()
    for message in messages:
        payload = message.get("message")
        content = payload.get("content") if isinstance(payload, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result" and isinstance(block.get("tool_use_id"), str):
                result.add(block["tool_use_id"])
    return result


def _missing_tool_result_messages(
    assistant_messages: list[AssistantMessage],
    error_message: str,
    *,
    existing_messages: list[dict] | None = None,
) -> list[dict]:
    """完成 ``_missing_tool_result_messages`` 对应的agent loop内部步骤。"""
    existing_ids = _tool_result_ids(existing_messages or [])
    results: list[dict] = []
    for assistant_message in assistant_messages:
        for block in assistant_message["message"]["content"]:
            if block.get("type") != "tool_use" or block.get("id") in existing_ids:
                continue
            tool_use_id = block.get("id")
            if not isinstance(tool_use_id, str):
                continue
            results.append(
                create_tool_result_message(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": error_message,
                        "is_error": True,
                    },
                    source_tool_assistant_uuid=assistant_message["uuid"],
                )
            )
            existing_ids.add(tool_use_id)
    return results


def _interruption_message(params: "QueryParams", *, tool_use: bool) -> dict | None:
    """完成 ``_interruption_message`` 对应的agent loop内部步骤。"""
    if params.tool_use_context.abort_controller.signal.reason == "interrupt":
        return None
    return create_user_interruption_message(tool_use=tool_use)


@dataclass
class QueryParams:
    """一次 query 运行所需的全部依赖与初始状态。"""
    # 三种上下文分开传递，避免动态信息改变稳定 system prompt 的缓存边界。
    messages: list[Message]
    system_prompt: list[str]
    user_context: dict[str, str]
    system_context: dict[str, str]
    # 执行依赖由调用方注入，使同一 loop 可复用于主 agent 和 subagent。
    tool_use_context: ToolUseContext
    model_provider: ModelProvider
    # 下列字段只控制本次运行，不属于跨 session 的消息状态。
    query_source: str = "python-port"
    max_turns: int | None = None
    model: str = "fake-model"
    context_compaction: ContextCompactionConfig | None = None
    transcript_path: str | None = None


async def query(params: QueryParams) -> AsyncIterator[dict]:
    """运行 agent loop，并按发生顺序产出消息、进度和 terminal 事件。"""
    # 使用浅拷贝维护本次 loop 的工作历史；QueryEngine 根据 yield 事件同步持久状态。
    messages: list[Message] = list(params.messages)
    # turn_count 统计模型轮次；执行一批工具后才进入下一轮。
    turn_count = 1
    # 每次 submit 最多主动 compact 一次，防止阈值判断形成压缩循环。
    compacted_this_query = False
    # 主请求的 prompt-too-long 恢复同样是 single-shot，失败后应暴露真实错误。
    prompt_too_long_recovery_attempted = False
    # Stop hook 返回 blocking message 后置位，供下一轮 hook 判断是否正在重入。
    stop_hook_active = False
    while True:
        # 每轮入口先检查取消，避免取消后又发起 compact 或模型请求。
        try:
            _raise_if_aborted(params)
        except asyncio.CancelledError:
            yield _aborted_terminal(params, turn_count)
            return
        if params.context_compaction is not None and params.context_compaction.microcompact_enabled:
            # microcompact 只清旧 tool_result 内容，不生成整段会话摘要。
            microcompact_result = microcompact_messages(
                messages,
                keep_recent=params.context_compaction.microcompact_keep_recent_tool_results,
            )
            if microcompact_result.boundary_marker is not None:
                messages = list(microcompact_result.messages)
                yield {
                    "type": "context_microcompacted",
                    "messages": microcompact_result.messages,
                    "boundary": microcompact_result.boundary_marker,
                    "compactedToolIds": microcompact_result.compacted_tool_ids,
                    "tokensSaved": microcompact_result.tokens_saved,
                }
        if (
            not compacted_this_query
            and params.context_compaction is not None
            and should_auto_compact(messages, params.context_compaction, query_source=params.query_source)
        ):
            # auto compact 在模型请求前运行；失败时是否沿用原上下文由配置决定。
            try:
                compaction_result = await compact_conversation(
                    messages,
                    model_provider=params.model_provider,
                    model=params.model,
                    config=params.context_compaction,
                    transcript_path=params.transcript_path,
                    is_auto_compact=True,
                    read_file_state=params.tool_use_context.read_file_state,
                    abort_signal=params.tool_use_context.abort_controller.signal,
                )
            except asyncio.CancelledError:
                params.tool_use_context.abort_controller.abort(_abort_reason(params))
                yield _aborted_terminal(params, turn_count)
                return
            except Exception as exc:
                yield {"type": "context_compaction_failed", "error": str(exc)}
                if not params.context_compaction.fallback_to_original_on_error:
                    yield {
                        "type": "terminal",
                        "terminal": Terminal("error", turn_count, str(exc)).__dict__,
                    }
                    return
                compacted_this_query = True
            else:
                messages = list(compaction_result.messages)
                params.tool_use_context.read_file_state.clear()
                params.tool_use_context.read_file_state.update(compaction_result.restored_file_state)
                compacted_this_query = True
                yield {
                    "type": "context_compacted",
                    "messages": compaction_result.messages,
                    "boundary": compaction_result.boundary_marker,
                    "summary_messages": compaction_result.summary_messages,
                    "attachments": compaction_result.attachments,
                    "messagesToKeep": compaction_result.messages_to_keep,
                    "preCompactTokenCount": compaction_result.pre_compact_token_count,
                    "postCompactTokenCount": compaction_result.post_compact_token_count,
                    "promptTooLongRetries": compaction_result.prompt_too_long_retries,
                    "restoredFilePaths": list(compaction_result.restored_file_state),
                }
        # 从这个事件开始，调用方可以把 UI 状态切换到“模型请求中”。
        yield {"type": "stream_request_start"}
        full_system_prompt = append_system_context(params.system_prompt, params.system_context)
        model_messages = prepend_user_context(messages, params.user_context)
        params.tool_use_context.messages = list(messages)
        params.tool_use_context.rendered_system_prompt = list(full_system_prompt)
        params.tool_use_context.user_context = dict(params.user_context)
        params.tool_use_context.system_context = dict(params.system_context)
        assistant_messages: list[AssistantMessage] = []
        tool_uses = []
        # provider 可以 yield 多个 assistant 分片；所有 tool_use 都属于本轮。
        try:
            async for message in params.model_provider.stream(
                messages=model_messages,
                system_prompt=full_system_prompt,
                tools=params.tool_use_context.tools,
                options={
                    "model": params.model,
                    "querySource": params.query_source,
                    "abortSignal": params.tool_use_context.abort_controller.signal,
                },
            ):
                assistant_messages.append(message)
                yield message
                for block in message["message"]["content"]:
                    if block["type"] == "tool_use":
                        tool_uses.append(block)
        except asyncio.CancelledError:
            params.tool_use_context.abort_controller.abort(_abort_reason(params))
            for tool_result in _missing_tool_result_messages(assistant_messages, "Interrupted by user"):
                yield tool_result
            interruption = _interruption_message(params, tool_use=False)
            if interruption is not None:
                yield interruption
            yield _aborted_terminal(params, turn_count)
            return
        except Exception as exc:
            if (
                is_prompt_too_long_error(exc)
                and params.context_compaction is not None
                and params.context_compaction.enabled
                and params.query_source != "compact"
                and not prompt_too_long_recovery_attempted
            ):
                prompt_too_long_recovery_attempted = True
                try:
                    compaction_result = await compact_conversation(
                        messages,
                        model_provider=params.model_provider,
                        model=params.model,
                        config=params.context_compaction,
                        transcript_path=params.transcript_path,
                        is_auto_compact=True,
                        read_file_state=params.tool_use_context.read_file_state,
                        abort_signal=params.tool_use_context.abort_controller.signal,
                    )
                except asyncio.CancelledError:
                    params.tool_use_context.abort_controller.abort(_abort_reason(params))
                    yield _aborted_terminal(params, turn_count)
                    return
                except Exception as compact_exc:
                    yield {
                        "type": "context_compaction_failed",
                        "error": str(compact_exc),
                        "recoveringFrom": "prompt_too_long",
                    }
                else:
                    messages = list(compaction_result.messages)
                    params.tool_use_context.read_file_state.clear()
                    params.tool_use_context.read_file_state.update(compaction_result.restored_file_state)
                    compacted_this_query = True
                    yield {
                        "type": "context_compacted",
                        "messages": compaction_result.messages,
                        "boundary": compaction_result.boundary_marker,
                        "summary_messages": compaction_result.summary_messages,
                        "attachments": compaction_result.attachments,
                        "messagesToKeep": compaction_result.messages_to_keep,
                        "preCompactTokenCount": compaction_result.pre_compact_token_count,
                        "postCompactTokenCount": compaction_result.post_compact_token_count,
                        "promptTooLongRetries": compaction_result.prompt_too_long_retries,
                        "restoredFilePaths": list(compaction_result.restored_file_state),
                        "recoveringFrom": "prompt_too_long",
                    }
                    continue
            for tool_result in _missing_tool_result_messages(assistant_messages, str(exc)):
                yield tool_result
            async for result in run_hook_event(
                params.tool_use_context,
                stop_failure_hook_input(context=params.tool_use_context, error=str(exc), last_assistant_message=None),
            ):
                if result.message:
                    yield result.message
                if result.system_message:
                    yield result.system_message
                additional = _hook_additional_context_message(result, "StopFailure")
                if additional:
                    yield additional
            error_message = create_system_message(
                f"API Error: {type(exc).__name__}: {exc}",
                subtype="api_error",
                level="error",
                error=str(exc),
            )
            yield error_message
            yield {"type": "terminal", "terminal": Terminal("error", turn_count, str(exc)).__dict__}
            return
        # assistant 原消息先进入历史，随后产生的 tool_result 才能正确配对。
        messages.extend(assistant_messages)
        params.tool_use_context.messages = list(messages)
        try:
            _raise_if_aborted(params)
        except asyncio.CancelledError:
            for tool_result in _missing_tool_result_messages(assistant_messages, "Interrupted by user"):
                yield tool_result
            interruption = _interruption_message(params, tool_use=False)
            if interruption is not None:
                yield interruption
            yield _aborted_terminal(params, turn_count)
            return

        if not tool_uses:
            # 没有工具意味着模型准备结束；Stop hook 仍可阻止或要求继续。
            blocking_messages: list[Message] = []
            prevent_continuation = False
            stop_reason: str | None = None
            async for result in run_hook_event(
                params.tool_use_context,
                stop_hook_input(
                    context=params.tool_use_context,
                    messages=messages,
                    stop_hook_active=stop_hook_active,
                ),
            ):
                if result.message:
                    yield result.message
                if result.system_message:
                    yield result.system_message
                additional = _hook_additional_context_message(result, "Stop")
                if additional:
                    yield additional
                if result.blocking_error:
                    blocking = create_user_message(hook_blocking_message(result, "Stop"), is_meta=True)
                    blocking_messages.append(blocking)
                    yield blocking
                if result.prevent_continuation:
                    prevent_continuation = True
                    stop_reason = result.stop_reason or stop_reason
                    stopped = _hook_stopped_message("Stop", stop_reason)
                    yield stopped
            try:
                _raise_if_aborted(params)
            except asyncio.CancelledError:
                interruption = _interruption_message(params, tool_use=False)
                if interruption is not None:
                    yield interruption
                yield _aborted_terminal(params, turn_count)
                return
            if prevent_continuation:
                yield {"type": "terminal", "terminal": Terminal("hook_stopped", turn_count, stop_reason).__dict__}
                return
            if blocking_messages:
                messages.extend(blocking_messages)
                stop_hook_active = True
                continue
            yield {"type": "terminal", "terminal": Terminal("completed", turn_count).__dict__}
            return

        # 工具执行事件实时向外转发，只有可回灌消息会进入下一轮上下文。
        tool_results = []
        hook_stopped = False
        try:
            async for tool_message in run_tools(tool_uses, assistant_messages, params.tool_use_context):
                yield tool_message
                if tool_message.get("type") in {"user", "attachment", "system"}:
                    tool_results.append(tool_message)
                if _is_hook_stopped_message(tool_message):
                    hook_stopped = True
        except asyncio.CancelledError:
            params.tool_use_context.abort_controller.abort(_abort_reason(params))
            for tool_result in _missing_tool_result_messages(
                assistant_messages,
                "Interrupted by user",
                existing_messages=tool_results,
            ):
                yield tool_result
            interruption = _interruption_message(params, tool_use=True)
            if interruption is not None:
                yield interruption
            yield _aborted_terminal(params, turn_count)
            return
        messages.extend(tool_results)
        if hook_stopped:
            yield {"type": "terminal", "terminal": Terminal("hook_stopped", turn_count, "Execution stopped by hook.").__dict__}
            return
        next_turn_count = turn_count + 1
        if params.max_turns is not None and next_turn_count > params.max_turns:
            attachment = create_attachment_message(
                f"Reached maximum number of turns ({params.max_turns})",
                attachment_type="max_turns_reached",
                metadata={"maxTurns": params.max_turns, "turnCount": next_turn_count},
            )
            messages.append(attachment)
            yield attachment
            yield {"type": "terminal", "terminal": Terminal("max_turns", next_turn_count).__dict__}
            return
        turn_count = next_turn_count
