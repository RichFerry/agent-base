"""生命周期 hook 的输入构造、注册匹配、执行和返回值归一化。

支持的事件覆盖 PreToolUse、PermissionRequest、PermissionDenied、PostToolUse、
PostToolUseFailure、Stop/StopFailure 和 SubagentStart/SubagentStop。每种输入都从共同的
session/cwd/permission 基础字段扩展，保证 hook runner 得到稳定形状。

HookRegistry 适合进程内 callback；HookRunner 是更通用的可注入执行器。两者都支持
同步值、awaitable、列表和 async iterator，``_normalize_hook_return`` 最终统一成
HookResult 流。HookResult 可附加消息、更新 input、覆盖权限、返回 blocking error、
阻止 continuation 或请求 retry。

本模块只描述和运行 hook，不解释结果；query.py 与 tool_execution.py 在各自生命周期
位置消费结果并决定消息回灌方式。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch
from inspect import signature
from typing import Any, AsyncIterator, Awaitable, Callable, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from .tools.base import ToolUseContext


HookEvent = Literal[
    "PreToolUse",
    "PermissionRequest",
    "PermissionDenied",
    "PostToolUse",
    "PostToolUseFailure",
    "Stop",
    "StopFailure",
    "SubagentStart",
    "SubagentStop",
]

HookOutcome = Literal["success", "blocking", "non_blocking_error", "cancelled"]
PermissionBehavior = Literal["allow", "deny", "ask", "passthrough"]


@dataclass
class HookResult:
    """一个 hook 对消息、权限、输入或控制流的结构化影响。"""
    message: dict[str, Any] | None = None
    system_message: dict[str, Any] | None = None
    blocking_error: str | dict[str, Any] | None = None
    outcome: HookOutcome = "success"
    prevent_continuation: bool = False
    stop_reason: str | None = None
    permission_behavior: PermissionBehavior | None = None
    hook_permission_decision_reason: str | None = None
    additional_context: str | list[str] | None = None
    initial_user_message: str | None = None
    updated_input: dict[str, Any] | None = None
    updated_mcp_tool_output: Any | None = None
    permission_request_result: dict[str, Any] | None = None
    retry: bool = False
    hook_source: str | None = None


HookCallback = Callable[[dict[str, Any]], HookResult | dict[str, Any] | list[Any] | Awaitable[Any]]
HookRunner = Callable[[dict[str, Any], "ToolUseContext"], HookResult | dict[str, Any] | list[Any] | Awaitable[Any] | AsyncIterator[Any]]


@dataclass
class HookMatcher:
    """封装 ``HookMatcher`` 对应的hook 生命周期状态与行为。"""
    event: HookEvent
    callback: HookCallback
    matcher: str | None = None
    name: str | None = None


@dataclass
class HookRegistry:
    """按 event 与可选 matcher 保存本地 hook。"""
    matchers: list[HookMatcher] = field(default_factory=list)

    def register(
        self,
        event: HookEvent,
        callback: HookCallback,
        *,
        matcher: str | None = None,
        name: str | None = None,
    ) -> None:
        """注册一个带可选 matcher 和名称的 hook callback。"""
        self.matchers.append(HookMatcher(event=event, callback=callback, matcher=matcher, name=name))

    def matching(self, hook_input: dict[str, Any]) -> list[HookMatcher]:
        """返回与当前 hook 输入事件和 matcher 匹配的注册项。"""
        event = hook_input.get("hook_event_name")
        # 不同 event 使用不同匹配文本：工具事件按 tool_name，失败事件按 error。
        match_query = _match_query(hook_input)
        matched = []
        for matcher in self.matchers:
            if matcher.event != event:
                continue
            if not matcher.matcher or _matches_pattern(match_query, matcher.matcher):
                matched.append(matcher)
        return matched

    async def run(self, hook_input: dict[str, Any]) -> AsyncIterator[HookResult]:
        """按注册顺序异步执行匹配项并产生规范化结果。"""
        for matcher in self.matching(hook_input):
            # 传入副本，避免某个 hook 原地修改共享输入影响后续 hook。
            result = matcher.callback(dict(hook_input))
            async for normalized in _normalize_hook_return(await _maybe_await(result)):
                if matcher.name and normalized.hook_source is None:
                    normalized.hook_source = matcher.name
                yield normalized


def create_base_hook_input(context: "ToolUseContext", permission_mode: str | None = None) -> dict[str, Any]:
    """创建base hook 输入，供hook 生命周期流程使用。"""
    config = context.config
    return {
        "session_id": getattr(context, "session_id", None),
        "transcript_path": getattr(context, "transcript_path", None),
        "cwd": str(config.cwd),
        "permission_mode": permission_mode or context.get_app_state().tool_permission_context.mode,
    }


def tool_hook_input(
    *,
    event: HookEvent,
    tool_name: str,
    tool_input: dict[str, Any],
    tool_use_id: str,
    context: "ToolUseContext",
    tool_response: Any | None = None,
    reason: str | None = None,
    permission_suggestions: list[dict[str, Any]] | None = None,
    error: str | None = None,
    is_interrupt: bool | None = None,
) -> dict[str, Any]:
    """完成 ``tool_hook_input`` 对应的hook 生命周期内部步骤。"""
    hook_input = {
        **create_base_hook_input(context),
        "hook_event_name": event,
        "tool_name": tool_name,
        "tool_input": tool_input,
        "tool_use_id": tool_use_id,
    }
    if tool_response is not None:
        hook_input["tool_response"] = tool_response
    if reason is not None:
        hook_input["reason"] = reason
    if permission_suggestions is not None:
        hook_input["permission_suggestions"] = permission_suggestions
    if error is not None:
        hook_input["error"] = error
    if is_interrupt is not None:
        hook_input["is_interrupt"] = is_interrupt
    return hook_input


def stop_hook_input(
    *,
    context: "ToolUseContext",
    messages: list[dict[str, Any]],
    stop_hook_active: bool = False,
) -> dict[str, Any]:
    """完成 ``stop_hook_input`` 对应的hook 生命周期内部步骤。"""
    last_assistant_text = ""
    for message in reversed(messages):
        if message.get("type") != "assistant":
            continue
        payload = message.get("message")
        content = payload.get("content") if isinstance(payload, dict) else None
        if not isinstance(content, list):
            continue
        text_parts = [block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"]
        last_assistant_text = "\n".join(part for part in text_parts if part).strip()
        break
    return {
        **create_base_hook_input(context),
        "hook_event_name": "Stop",
        "stop_hook_active": stop_hook_active,
        "last_assistant_message": last_assistant_text or None,
    }


def stop_failure_hook_input(
    *,
    context: "ToolUseContext",
    error: str,
    last_assistant_message: str | None = None,
) -> dict[str, Any]:
    """完成 ``stop_failure_hook_input`` 对应的hook 生命周期内部步骤。"""
    return {
        **create_base_hook_input(context),
        "hook_event_name": "StopFailure",
        "error": error or "unknown",
        "error_details": None,
        "last_assistant_message": last_assistant_message,
    }


def subagent_start_hook_input(
    *,
    context: "ToolUseContext",
    agent_id: str,
    agent_type: str,
    description: str | None = None,
) -> dict[str, Any]:
    """完成 ``subagent_start_hook_input`` 对应的hook 生命周期内部步骤。"""
    return {
        **create_base_hook_input(context),
        "hook_event_name": "SubagentStart",
        "agent_id": agent_id,
        "agent_type": agent_type,
        "description": description,
    }


def subagent_stop_hook_input(
    *,
    context: "ToolUseContext",
    agent_id: str,
    agent_type: str,
    status: str,
    result: str | None = None,
) -> dict[str, Any]:
    """完成 ``subagent_stop_hook_input`` 对应的hook 生命周期内部步骤。"""
    return {
        **create_base_hook_input(context),
        "hook_event_name": "SubagentStop",
        "agent_id": agent_id,
        "agent_type": agent_type,
        "status": status,
        "result": result,
    }


async def run_hook_event(context: "ToolUseContext", hook_input: dict[str, Any]) -> AsyncIterator[HookResult]:
    """先运行 registry，再运行可注入 runner，并保持结果产生顺序。"""
    # 进程内 registry 先执行，外部 runner 后执行，顺序与注册语义保持稳定。
    async for result in context.hook_registry.run(hook_input):
        yield result
    if context.hook_runner is None:
        return
    returned = _call_runner(context.hook_runner, hook_input, context)
    async for result in _normalize_hook_return(await _maybe_await(returned)):
        yield result


def hook_blocking_message(result: HookResult, hook_name: str) -> str:
    """完成 ``hook_blocking_message`` 对应的hook 生命周期内部步骤。"""
    if isinstance(result.blocking_error, dict):
        return str(result.blocking_error.get("blockingError") or result.blocking_error.get("blocking_error") or "Hook blocked execution")
    if result.blocking_error:
        return str(result.blocking_error)
    if result.stop_reason:
        return result.stop_reason
    if result.hook_permission_decision_reason:
        return result.hook_permission_decision_reason
    return f"Execution stopped by {hook_name} hook"


async def _maybe_await(value: Any) -> Any:
    """完成 ``_maybe_await`` 对应的hook 生命周期内部步骤。"""
    if hasattr(value, "__await__"):
        return await value
    return value


def _call_runner(runner: HookRunner, hook_input: dict[str, Any], context: "ToolUseContext") -> Any:
    """完成 ``_call_runner`` 对应的hook 生命周期内部步骤。"""
    # 兼容只接收 hook_input 的轻量 callback 和接收 context 的完整 runner。
    try:
        arity = len(signature(runner).parameters)
    except (TypeError, ValueError):
        arity = 2
    if arity <= 1:
        return runner(hook_input)  # type: ignore[misc]
    return runner(hook_input, context)


async def _normalize_hook_return(value: Any) -> AsyncIterator[HookResult]:
    """规范化hook return，供hook 生命周期流程使用。"""
    if value is None:
        return
    if hasattr(value, "__aiter__"):
        # 递归展开让列表、async iterator 和混合嵌套返回拥有同一消费方式。
        async for item in value:
            async for normalized in _normalize_hook_return(item):
                yield normalized
        return
    if isinstance(value, HookResult):
        yield value
        return
    if isinstance(value, list) or isinstance(value, tuple):
        for item in value:
            async for normalized in _normalize_hook_return(item):
                yield normalized
        return
    if isinstance(value, dict):
        yield _hook_result_from_dict(value)
        return
    raise TypeError(f"Unsupported hook result type: {type(value).__name__}")


def _hook_result_from_dict(value: dict[str, Any]) -> HookResult:
    """完成 ``_hook_result_from_dict`` 对应的hook 生命周期内部步骤。"""
    # 同时接受通用顶层字段和 Claude Code hookSpecificOutput 嵌套形态。
    hook_specific = value.get("hookSpecificOutput")
    permission_request_result = value.get("permissionRequestResult")
    permission_behavior = value.get("permissionBehavior")
    updated_input = value.get("updatedInput")
    additional_context = value.get("additionalContext")
    updated_mcp_tool_output = value.get("updatedMCPToolOutput")
    retry = bool(value.get("retry", False))
    if isinstance(hook_specific, dict):
        # 事件专用字段优先于顶层兼容字段。
        permission_behavior = hook_specific.get("permissionDecision", permission_behavior)
        updated_input = hook_specific.get("updatedInput", updated_input)
        additional_context = hook_specific.get("additionalContext", additional_context)
        updated_mcp_tool_output = hook_specific.get("updatedMCPToolOutput", updated_mcp_tool_output)
        retry = bool(hook_specific.get("retry", retry))
        permission_request_result = hook_specific.get("decision", permission_request_result)
    decision = value.get("decision")
    blocking_error = value.get("blockingError") or value.get("blocking_error")
    if decision == "block" and blocking_error is None:
        blocking_error = value.get("reason") or value.get("stopReason") or "Hook blocked execution"
    return HookResult(
        message=value.get("message"),
        system_message=value.get("systemMessage"),
        blocking_error=blocking_error,
        outcome=value.get("outcome", "success"),
        prevent_continuation=bool(value.get("preventContinuation") or value.get("continue") is False),
        stop_reason=value.get("stopReason"),
        permission_behavior=permission_behavior,
        hook_permission_decision_reason=value.get("hookPermissionDecisionReason") or value.get("reason"),
        additional_context=additional_context,
        initial_user_message=value.get("initialUserMessage"),
        updated_input=updated_input,
        updated_mcp_tool_output=updated_mcp_tool_output,
        permission_request_result=permission_request_result,
        retry=retry,
        hook_source=value.get("hookSource"),
    )


def _match_query(hook_input: dict[str, Any]) -> str:
    """完成 ``_match_query`` 对应的hook 生命周期内部步骤。"""
    event = hook_input.get("hook_event_name")
    if event in {"PreToolUse", "PostToolUse", "PostToolUseFailure", "PermissionRequest", "PermissionDenied"}:
        return str(hook_input.get("tool_name") or "")
    if event == "StopFailure":
        return str(hook_input.get("error") or "")
    return ""


def _matches_pattern(value: str, pattern: str) -> bool:
    """完成 ``_matches_pattern`` 对应的hook 生命周期内部步骤。"""
    return value == pattern or fnmatch(value, pattern)
