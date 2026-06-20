"""工具执行生命周期、错误封装、progress 转发与并发批次调度。

单调用顺序固定为：工具查找 -> schema 校验 -> ``validate_input`` -> PreToolUse hook ->
权限解析 -> ``Tool.call`` -> 结果映射 -> PostToolUse hook。任一普通异常都尽量转成带
原 tool_use_id 的 error tool_result；CancelledError 保持控制流语义，交给 query 生成
中断结果。PostToolUseFailure hook 也被保护，不能吞掉原始失败。

``run_tools`` 按 tool_use 原顺序切批：连续 ``is_concurrency_safe`` 调用并发执行，写入或
未知调用单独串行。并发结果可按完成时间 yield progress，但每个结果仍记录正确来源
assistant uuid。最大并发数来自环境变量，默认 10。

ToolUseContext 的 abort signal 会在校验、hook、权限、call 和等待循环之间检查；取消时
task 被显式 cancel 并 await，给 Bash 等工具运行 finally 清理的机会。
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import AsyncIterator

from .hooks import HookResult, hook_blocking_message, run_hook_event, tool_hook_input
from .messages import AssistantMessage, ToolUseBlock, create_attachment_message, create_tool_result_message, create_user_message
from .permissions import PermissionDecision, has_permissions_to_use_tool
from .tools.base import ToolUseContext, find_tool_by_name


def _tool_use_error(message: str) -> str:
    """完成 ``_tool_use_error`` 对应的工具执行内部步骤。"""
    return f"<tool_use_error>{message}</tool_use_error>"


def _error_tool_result(tool_use_id: str, message: str) -> dict:
    """完成 ``_error_tool_result`` 对应的工具执行内部步骤。"""
    return {
        "tool_use_id": tool_use_id,
        "type": "tool_result",
        "content": message,
        "is_error": True,
    }


def _exception_text(prefix: str, exc: BaseException) -> str:
    """完成 ``_exception_text`` 对应的工具执行内部步骤。"""
    return _tool_use_error(f"{prefix}: {type(exc).__name__}: {exc}")


def _abort_if_requested(context: ToolUseContext) -> None:
    """完成 ``_abort_if_requested`` 对应的工具执行内部步骤。"""
    context.abort_controller.signal.throw_if_aborted()


def _hook_name(event: str, tool_name: str) -> str:
    """完成 ``_hook_name`` 对应的工具执行内部步骤。"""
    return f"{event}:{tool_name}" if event not in {"Stop", "StopFailure"} else event


def _hook_additional_context_message(result: HookResult, event: str, tool_name: str, tool_use_id: str) -> dict | None:
    """完成 ``_hook_additional_context_message`` 对应的工具执行内部步骤。"""
    if not result.additional_context:
        return None
    contexts = result.additional_context if isinstance(result.additional_context, list) else [result.additional_context]
    return create_attachment_message(
        "\n".join(str(item) for item in contexts),
        attachment_type="hook_additional_context",
        metadata={
            "hookName": _hook_name(event, tool_name),
            "toolUseID": tool_use_id,
            "hookEvent": event,
        },
    )


def _hook_blocking_attachment(result: HookResult, event: str, tool_name: str, tool_use_id: str) -> dict:
    """完成 ``_hook_blocking_attachment`` 对应的工具执行内部步骤。"""
    message = hook_blocking_message(result, event)
    return create_attachment_message(
        message,
        attachment_type="hook_blocking_error",
        metadata={
            "hookName": _hook_name(event, tool_name),
            "toolUseID": tool_use_id,
            "hookEvent": event,
            "blockingError": message,
        },
    )


def _hook_stopped_attachment(event: str, tool_name: str, tool_use_id: str, reason: str | None) -> dict:
    """完成 ``_hook_stopped_attachment`` 对应的工具执行内部步骤。"""
    message = reason or "Execution stopped by hook"
    return create_attachment_message(
        message,
        attachment_type="hook_stopped_continuation",
        metadata={
            "message": message,
            "hookName": _hook_name(event, tool_name),
            "toolUseID": tool_use_id,
            "hookEvent": event,
        },
    )


def _hook_permission_decision(result: HookResult, fallback_message: str | None = None) -> PermissionDecision | None:
    """完成 ``_hook_permission_decision`` 对应的工具执行内部步骤。"""
    if result.permission_behavior is None or result.permission_behavior == "passthrough":
        return None
    if result.permission_behavior == "allow":
        return PermissionDecision.allow(updated_input=result.updated_input)
    if result.permission_behavior == "ask":
        return PermissionDecision(
            "ask",
            result.hook_permission_decision_reason or fallback_message,
            updated_input=result.updated_input,
        )
    return PermissionDecision.deny(result.hook_permission_decision_reason or fallback_message or "Permission denied by hook.")


async def _run_pre_tool_hooks(tool, input: dict, tool_use_id: str, context: ToolUseContext) -> tuple[list[dict], dict, PermissionDecision | None, bool, str | None]:
    """执行pre 工具 hook 集合，供工具执行流程使用。"""
    updates: list[dict] = []
    # 只保留最后一个显式权限覆盖，但所有 hook 消息按产生顺序保留。
    force_permission: PermissionDecision | None = None
    prevent_continuation = False
    stop_reason: str | None = None
    hook_input = tool_hook_input(
        event="PreToolUse",
        tool_name=tool.name,
        tool_input=input,
        tool_use_id=tool_use_id,
        context=context,
    )
    async for result in run_hook_event(context, hook_input):
        if result.message:
            updates.append(result.message)
        if result.system_message:
            updates.append(result.system_message)
        additional = _hook_additional_context_message(result, "PreToolUse", tool.name, tool_use_id)
        if additional:
            updates.append(additional)
        if result.blocking_error:
            updates.append(_hook_blocking_attachment(result, "PreToolUse", tool.name, tool_use_id))
            force_permission = PermissionDecision.deny(hook_blocking_message(result, "PreToolUse"))
        if result.prevent_continuation:
            prevent_continuation = True
            stop_reason = result.stop_reason or stop_reason
        permission_decision = _hook_permission_decision(result)
        if permission_decision is not None:
            force_permission = permission_decision
        if result.updated_input is not None and result.permission_behavior in {None, "passthrough"}:
            # 后续 hook 应观察前一个 hook 更新后的 input。
            input = result.updated_input
            hook_input["tool_input"] = input
    return updates, input, force_permission, prevent_continuation, stop_reason


async def _safe_post_tool_failure_hooks(tool, input: dict, tool_use_id: str, context: ToolUseContext, error: str, *, is_interrupt: bool = False) -> list[dict]:
    """完成 ``_safe_post_tool_failure_hooks`` 对应的工具执行内部步骤。"""
    try:
        return await _run_post_tool_failure_hooks(tool, input, tool_use_id, context, error, is_interrupt=is_interrupt)
    except asyncio.CancelledError:
        # 取消属于 query 控制流，不能包装成普通 hook_execution_error。
        raise
    except Exception as exc:
        return [
            create_attachment_message(
                f"PostToolUseFailure hook failed: {type(exc).__name__}: {exc}",
                attachment_type="hook_execution_error",
                metadata={
                    "hookName": _hook_name("PostToolUseFailure", tool.name),
                    "toolUseID": tool_use_id,
                    "hookEvent": "PostToolUseFailure",
                },
            )
        ]


async def _failure_updates(tool, input: dict, tool_use_id: str, parent_message: AssistantMessage, context: ToolUseContext, error: str) -> list[dict]:
    """完成 ``_failure_updates`` 对应的工具执行内部步骤。"""
    # 先产生配对 tool_result，再附加 failure hook 消息，保证 API 邻接关系。
    updates = [
        create_tool_result_message(
            _error_tool_result(tool_use_id, error),
            source_tool_assistant_uuid=parent_message["uuid"],
        )
    ]
    updates.extend(await _safe_post_tool_failure_hooks(tool, input, tool_use_id, context, error))
    return updates


async def _run_permission_denied_hooks(tool, input: dict, tool_use_id: str, context: ToolUseContext, reason: str) -> list[dict]:
    """执行权限 denied hook 集合，供工具执行流程使用。"""
    updates: list[dict] = []
    hook_input = tool_hook_input(
        event="PermissionDenied",
        tool_name=tool.name,
        tool_input=input,
        tool_use_id=tool_use_id,
        context=context,
        reason=reason,
    )
    async for result in run_hook_event(context, hook_input):
        if result.message:
            updates.append(result.message)
        if result.system_message:
            updates.append(result.system_message)
        additional = _hook_additional_context_message(result, "PermissionDenied", tool.name, tool_use_id)
        if additional:
            updates.append(additional)
        if result.retry:
            updates.append(
                create_user_message(
                    "The PermissionDenied hook indicated this command is now approved. You may retry it if you would like.",
                    is_meta=True,
                )
            )
    return updates


async def _run_post_tool_hooks(tool, input: dict, tool_use_id: str, context: ToolUseContext, tool_response) -> list[dict]:
    """执行post 工具 hook 集合，供工具执行流程使用。"""
    updates: list[dict] = []
    hook_input = tool_hook_input(
        event="PostToolUse",
        tool_name=tool.name,
        tool_input=input,
        tool_use_id=tool_use_id,
        context=context,
        tool_response=tool_response,
    )
    async for result in run_hook_event(context, hook_input):
        if result.message:
            updates.append(result.message)
        if result.system_message:
            updates.append(result.system_message)
        additional = _hook_additional_context_message(result, "PostToolUse", tool.name, tool_use_id)
        if additional:
            updates.append(additional)
        if result.blocking_error:
            updates.append(_hook_blocking_attachment(result, "PostToolUse", tool.name, tool_use_id))
        if result.prevent_continuation:
            updates.append(_hook_stopped_attachment("PostToolUse", tool.name, tool_use_id, result.stop_reason))
    return updates


async def _run_post_tool_failure_hooks(tool, input: dict, tool_use_id: str, context: ToolUseContext, error: str, *, is_interrupt: bool = False) -> list[dict]:
    """执行post 工具 failure hook 集合，供工具执行流程使用。"""
    updates: list[dict] = []
    hook_input = tool_hook_input(
        event="PostToolUseFailure",
        tool_name=tool.name,
        tool_input=input,
        tool_use_id=tool_use_id,
        context=context,
        error=error,
        is_interrupt=is_interrupt,
    )
    async for result in run_hook_event(context, hook_input):
        if result.message:
            updates.append(result.message)
        if result.system_message:
            updates.append(result.system_message)
        additional = _hook_additional_context_message(result, "PostToolUseFailure", tool.name, tool_use_id)
        if additional:
            updates.append(additional)
        if result.blocking_error:
            updates.append(_hook_blocking_attachment(result, "PostToolUseFailure", tool.name, tool_use_id))
    return updates


async def run_tool_use(
    tool_use: ToolUseBlock,
    parent_message: AssistantMessage,
    context: ToolUseContext,
    on_progress=None,
) -> list[dict]:
    """执行一个 tool_use block，并返回可直接 yield/持久化的消息列表。"""
    _abort_if_requested(context)
    tool = find_tool_by_name(context.tools, tool_use["name"])
    tool_use_id = tool_use["id"]
    if tool is None:
        # 未知工具也返回标准 error result，模型可以在下一轮自行纠正名称。
        return [
            create_tool_result_message(
                _error_tool_result(tool_use_id, _tool_use_error(f"Tool {tool_use['name']} does not exist.")),
                source_tool_assistant_uuid=parent_message["uuid"],
            )
        ]

    # schema 校验只看字段形状，validate_input 再执行路径和业务不变量校验。
    input = dict(tool_use.get("input") or {})
    # hook 可以更新 input、直接给出权限决策，或阻止后续 continuation。
    try:
        _abort_if_requested(context)
        schema_result = tool.validate_schema(input)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return await _failure_updates(tool, input, tool_use_id, parent_message, context, _exception_text("InputValidationError", exc))
    if not schema_result.result:
        return [
            create_tool_result_message(
                _error_tool_result(tool_use_id, _tool_use_error(f"InputValidationError: {schema_result.message or 'Invalid tool input.'}")),
                source_tool_assistant_uuid=parent_message["uuid"],
            )
        ]

    # 权限解析是执行前最后一道门；deny 也必须返回标准 tool_result。
    try:
        _abort_if_requested(context)
        validation = await tool.validate_input(input, context)
        _abort_if_requested(context)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return await _failure_updates(tool, input, tool_use_id, parent_message, context, _exception_text("InputValidationError", exc))
    if not validation.result:
        return [
            create_tool_result_message(
                _error_tool_result(tool_use_id, _tool_use_error(validation.message or "Invalid tool input.")),
                source_tool_assistant_uuid=parent_message["uuid"],
            )
        ]

    # 工具自身 timeout 与 query 取消都通过 CancelledError 进入统一清理路径。
    try:
        _abort_if_requested(context)
        updates, input, hook_permission, should_prevent_continuation, stop_reason = await _run_pre_tool_hooks(tool, input, tool_use_id, context)
        _abort_if_requested(context)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return await _failure_updates(tool, input, tool_use_id, parent_message, context, _exception_text("PreToolUse hook failed", exc))
    try:
        _abort_if_requested(context)
        permission = await has_permissions_to_use_tool(tool, input, context, parent_message, tool_use_id, force_decision=hook_permission)
        _abort_if_requested(context)
        updates.extend(context.drain_hook_messages())
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        updates.extend(await _failure_updates(tool, input, tool_use_id, parent_message, context, _exception_text("Permission check failed", exc)))
        return updates
    if permission.behavior == "deny":
        # 权限拒绝是可恢复工具结果，不终止整个 agent loop。
        message = permission.message or "Permission denied."
        updates.append(
            create_tool_result_message(
                _error_tool_result(tool_use_id, message),
                source_tool_assistant_uuid=parent_message["uuid"],
            )
        )
        try:
            updates.extend(await _run_permission_denied_hooks(tool, input, tool_use_id, context, message))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            updates.append(
                create_attachment_message(
                    f"PermissionDenied hook failed: {type(exc).__name__}: {exc}",
                    attachment_type="hook_execution_error",
                    metadata={
                        "hookName": _hook_name("PermissionDenied", tool.name),
                        "toolUseID": tool_use_id,
                        "hookEvent": "PermissionDenied",
                    },
                )
            )
        return updates
    if permission.updated_input is not None:
        input = permission.updated_input

    try:
        _abort_if_requested(context)
        call = tool.call(input, context, has_permissions_to_use_tool, parent_message, on_progress=on_progress)
        timeout_seconds = context.tool_timeout_seconds
        if isinstance(input.get("timeout"), int):
            # 工具声明的毫秒 timeout 可能长于默认值，额外留 5 秒做进程清理。
            requested_timeout = max(0, int(input["timeout"])) / 1000
            timeout_seconds = max(timeout_seconds or 0, requested_timeout + 5)
        if timeout_seconds is not None:
            result = await asyncio.wait_for(call, timeout=timeout_seconds)
        else:
            result = await call
        _abort_if_requested(context)
        block = tool.map_tool_result_to_tool_result_block_param(result.data, tool_use_id)
    except asyncio.TimeoutError:
        error = _tool_use_error(f"Tool {tool.name} timed out.")
        block = _error_tool_result(tool_use_id, error)
        updates.append(create_tool_result_message(block, source_tool_assistant_uuid=parent_message["uuid"]))
        updates.extend(await _safe_post_tool_failure_hooks(tool, input, tool_use_id, context, error))
        return updates
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        error = _tool_use_error(f"{type(exc).__name__}: {exc}")
        block = _error_tool_result(tool_use_id, error)
        updates.append(create_tool_result_message(block, source_tool_assistant_uuid=parent_message["uuid"]))
        updates.extend(await _safe_post_tool_failure_hooks(tool, input, tool_use_id, context, error))
        return updates
    # result 必须先于工具产生的 new_messages，保持 tool_use/tool_result 紧邻。
    updates.append(create_tool_result_message(block, source_tool_assistant_uuid=parent_message["uuid"]))
    updates.extend(result.new_messages)
    try:
        _abort_if_requested(context)
        updates.extend(await _run_post_tool_hooks(tool, input, tool_use_id, context, result.data))
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        updates.append(
            create_attachment_message(
                f"PostToolUse hook failed: {type(exc).__name__}: {exc}",
                attachment_type="hook_execution_error",
                metadata={
                    "hookName": _hook_name("PostToolUse", tool.name),
                    "toolUseID": tool_use_id,
                    "hookEvent": "PostToolUse",
                },
            )
        )
    if should_prevent_continuation:
        updates.append(_hook_stopped_attachment("PreToolUse", tool.name, tool_use_id, stop_reason))
    return updates


def _parent_for_tool_use(tool_use: ToolUseBlock, parent_messages: AssistantMessage | list[AssistantMessage]) -> AssistantMessage:
    """完成 ``_parent_for_tool_use`` 对应的工具执行内部步骤。"""
    if isinstance(parent_messages, dict):
        return parent_messages
    for message in parent_messages:
        for block in message["message"]["content"]:
            if block["type"] == "tool_use" and block.get("id") == tool_use["id"]:
                return message
    return parent_messages[-1]


def _is_concurrency_safe(tool_use: ToolUseBlock, context: ToolUseContext) -> bool:
    """判断concurrency safe，供工具执行流程使用。"""
    tool = find_tool_by_name(context.tools, tool_use["name"])
    if tool is None:
        return False
    input = dict(tool_use.get("input") or {})
    if not tool.validate_schema(input).result:
        return False
    try:
        return bool(tool.is_concurrency_safe(input))
    except Exception:
        return False


def _partition_tool_calls(tool_uses: list[ToolUseBlock], context: ToolUseContext) -> list[tuple[bool, list[ToolUseBlock]]]:
    """完成 ``_partition_tool_calls`` 对应的工具执行内部步骤。"""
    batches: list[tuple[bool, list[ToolUseBlock]]] = []
    for tool_use in tool_uses:
        is_safe = _is_concurrency_safe(tool_use, context)
        # 只合并连续安全调用；遇到写工具后必须形成新的顺序边界。
        if is_safe and batches and batches[-1][0]:
            batches[-1][1].append(tool_use)
        else:
            batches.append((is_safe, [tool_use]))
    return batches


def _max_concurrency() -> int:
    """完成 ``_max_concurrency`` 对应的工具执行内部步骤。"""
    import os

    try:
        return max(1, int(os.environ.get("CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY", "10")))
    except ValueError:
        return 10


async def run_tools(
    tool_uses: list[ToolUseBlock],
    parent_messages: AssistantMessage | list[AssistantMessage],
    context: ToolUseContext,
) -> AsyncIterator[dict]:
    """按源码规则把连续并发安全调用分批，其余调用逐个串行执行。"""
    _abort_if_requested(context)
    for is_concurrency_safe, batch in _partition_tool_calls(tool_uses, context):
        _abort_if_requested(context)
        if is_concurrency_safe and len(batch) > 1:
            async for update in _run_tools_concurrently(batch, parent_messages, context):
                yield update
            continue
        for tool_use in batch:
            async for update in _run_tool_stream(tool_use, parent_messages, context):
                yield update


async def _run_tool_stream(
    tool_use: ToolUseBlock,
    parent_messages: AssistantMessage | list[AssistantMessage],
    context: ToolUseContext,
) -> AsyncIterator[dict]:
    """执行工具 流，供工具执行流程使用。"""
    progress_queue: asyncio.Queue[dict] = asyncio.Queue()

    def on_progress(progress):
        """把工具进度封装为事件并写入当前异步队列。"""
        progress_queue.put_nowait(
            {
                "type": "tool_progress",
                "tool_use_id": tool_use["id"],
                "tool_name": tool_use["name"],
                "progress": progress,
            }
        )

    parent_message = _parent_for_tool_use(tool_use, parent_messages)
    # 工具运行在 task 中，主生成器才能同时轮询 progress 和 abort。
    task = asyncio.create_task(run_tool_use(tool_use, parent_message, context, on_progress=on_progress))
    try:
        while True:
            if context.abort_controller.signal.aborted:
                task.cancel()
                raise asyncio.CancelledError(str(context.abort_controller.signal.reason or "Request was aborted."))
            done, _ = await asyncio.wait({task}, timeout=0.05)
            # progress 不进入模型上下文，只作为实时事件向调用方发送。
            while not progress_queue.empty():
                yield progress_queue.get_nowait()
            if done:
                try:
                    updates = task.result()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    updates = [
                        create_tool_result_message(
                            _error_tool_result(tool_use["id"], _tool_use_error(f"Tool execution failed: {type(exc).__name__}: {exc}")),
                            source_tool_assistant_uuid=parent_message["uuid"],
                        )
                    ]
                for update in updates:
                    yield update
                break
    finally:
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


async def _run_tools_concurrently(
    tool_uses: list[ToolUseBlock],
    parent_messages: AssistantMessage | list[AssistantMessage],
    context: ToolUseContext,
) -> AsyncIterator[dict]:
    """执行工具集合 concurrently，供工具执行流程使用。"""
    semaphore = asyncio.Semaphore(_max_concurrency())
    # 每个 tool_use 使用独立队列，避免并发 progress 丢失来源 ID。
    progress_queues: dict[str, asyncio.Queue[dict]] = {tool_use["id"]: asyncio.Queue() for tool_use in tool_uses}

    async def run_one(tool_use: ToolUseBlock):
        """在并发信号量保护下执行一个工具调用并收集更新。"""
        async with semaphore:
            parent_message = _parent_for_tool_use(tool_use, parent_messages)

            def on_progress(progress):
                """把工具进度封装为事件并写入当前异步队列。"""
                progress_queues[tool_use["id"]].put_nowait(
                    {
                        "type": "tool_progress",
                        "tool_use_id": tool_use["id"],
                        "tool_name": tool_use["name"],
                        "progress": progress,
                    }
                )

            return await run_tool_use(tool_use, parent_message, context, on_progress=on_progress)

    # task -> tool_use 映射用于完成时恢复正确 parent assistant 和错误 ID。
    tasks = {asyncio.create_task(run_one(tool_use)): tool_use for tool_use in tool_uses}
    try:
        while tasks:
            if context.abort_controller.signal.aborted:
                for task in tasks:
                    task.cancel()
                raise asyncio.CancelledError(str(context.abort_controller.signal.reason or "Request was aborted."))
            done, _ = await asyncio.wait(set(tasks), timeout=0.05, return_when=asyncio.FIRST_COMPLETED)
            for queue in progress_queues.values():
                while not queue.empty():
                    yield queue.get_nowait()
            for task in done:
                tool_use = tasks.pop(task)
                parent_message = _parent_for_tool_use(tool_use, parent_messages)
                try:
                    updates = task.result()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    updates = [
                        create_tool_result_message(
                            _error_tool_result(tool_use["id"], _tool_use_error(f"Tool execution failed: {type(exc).__name__}: {exc}")),
                            source_tool_assistant_uuid=parent_message["uuid"],
                        )
                    ]
                for update in updates:
                    yield update
        for queue in progress_queues.values():
            while not queue.empty():
                yield queue.get_nowait()
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
