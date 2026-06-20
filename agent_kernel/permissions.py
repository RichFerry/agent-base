"""工具权限决策、源码模式别名和 ask/bypass 最终解析。

权限分两阶段：具体 Tool 根据输入返回 allow/ask/deny 建议；本模块结合 session mode、
PermissionRequest hook 与可注入 callback 得出最终决定。项目内安全只读操作通常 allow，
写入和危险 Bash 通常 ask；无 UI/callback 时 ask 安全降级为 deny。

内核公开目标模式只有 ask 与 bypass，但兼容 default、acceptEdits、bypassPermissions、
plan、dontAsk 等源码名称。``bypass_immune`` 表示敏感路径/结构安全约束，任何模式都不能
绕过。Hook 可修改 input 或直接给出权限行为，修改后的 input 会传给实际工具。

本模块不弹 UI；交互决策完全由 ``permission_callback`` 注入。拒绝仍由 tool_execution
转换为正常 tool_result，agent loop 不因权限异常断链。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable, Literal, Protocol

from .messages import create_attachment_message

PermissionMode = Literal["ask", "bypass", "default", "acceptEdits", "bypassPermissions", "plan", "dontAsk"]
PermissionBehavior = Literal["allow", "deny", "ask"]


@dataclass
class PermissionDecision:
    """权限管线中的结构化决策，可附带 hook 修正后的 input。"""
    behavior: PermissionBehavior
    message: str | None = None
    updated_input: dict | None = None
    bypass_immune: bool = False

    @classmethod
    def allow(cls, message: str | None = None, *, updated_input: dict | None = None) -> "PermissionDecision":
        """完成 ``allow`` 对应的权限解析内部步骤。"""
        return cls("allow", message, updated_input=updated_input)

    @classmethod
    def deny(cls, message: str | None = None) -> "PermissionDecision":
        """完成 ``deny`` 对应的权限解析内部步骤。"""
        return cls("deny", message)

    @classmethod
    def ask(cls, message: str | None = None, *, bypass_immune: bool = False) -> "PermissionDecision":
        """完成 ``ask`` 对应的权限解析内部步骤。"""
        return cls("ask", message, bypass_immune=bypass_immune)


@dataclass
class ToolPermissionContext:
    """封装 ``ToolPermissionContext`` 对应的权限解析状态与行为。"""
    mode: PermissionMode = "ask"
    additional_working_directories: dict[str, str] = field(default_factory=dict)


class PermissionTool(Protocol):
    """权限解析器所需的最小 Tool 结构协议。"""

    name: str

    def is_read_only(self, input: dict) -> bool:
        """判断当前输入是否只会读取状态。"""

        ...

    async def check_permissions(self, input: dict, context: object) -> PermissionDecision:
        """返回工具针对当前输入的初步权限建议。"""

        ...

    def prepare_permission_matcher(self, input: dict) -> Callable[[str], bool] | None:
        """可选地构造规则匹配函数，供权限规则比较输入。"""

        ...


PermissionCallback = Callable[[PermissionTool, dict, object, PermissionDecision], Awaitable[PermissionDecision] | PermissionDecision]


async def _maybe_await(decision):
    """完成 ``_maybe_await`` 对应的权限解析内部步骤。"""
    if hasattr(decision, "__await__"):
        return await decision
    return decision


async def resolve_ask(
    tool: PermissionTool,
    input: dict,
    context: object,
    decision: PermissionDecision,
    assistant_message: dict | None = None,
    tool_use_id: str | None = None,
) -> PermissionDecision:
    """把 ask 交给 hook/callback；无人决策时安全地 deny。"""
    # Hook 比 UI callback 更早运行，可用于组织级策略或非交互批准。
    hook_decision = await resolve_permission_request_hooks(tool, input, context, decision, assistant_message, tool_use_id)
    if hook_decision is not None:
        return hook_decision
    callback = getattr(context, "permission_callback", None)
    if callback is not None:
        return await _maybe_await(callback(tool, input, context, decision))
    # Kernel 不假装用户同意：没有交互决策器时 ask 必须降级为 deny。
    if decision.message:
        return PermissionDecision.deny(f"Permission denied: {decision.message}")
    return PermissionDecision.deny("Permission denied.")


async def resolve_permission_request_hooks(
    tool: PermissionTool,
    input: dict,
    context: object,
    decision: PermissionDecision,
    assistant_message: dict | None = None,
    tool_use_id: str | None = None,
) -> PermissionDecision | None:
    """解析并确定权限 请求 hook 集合，供权限解析流程使用。"""
    from .hooks import hook_blocking_message, run_hook_event, tool_hook_input

    if not hasattr(context, "hook_registry") and not hasattr(context, "hook_runner"):
        return None
    hook_input = tool_hook_input(
        event="PermissionRequest",
        tool_name=tool.name,
        tool_input=input,
        tool_use_id=tool_use_id or "",
        context=context,
        permission_suggestions=[],
    )
    async for result in run_hook_event(context, hook_input):
        if result.message:
            context.push_hook_message(result.message)
        if result.system_message:
            context.push_hook_message(result.system_message)
        if result.additional_context:
            # 附加上下文属于 transcript 消息，而不是权限 decision 的文本字段。
            contexts = result.additional_context if isinstance(result.additional_context, list) else [result.additional_context]
            context.push_hook_message(
                create_attachment_message(
                    "\n".join(str(item) for item in contexts),
                    attachment_type="hook_additional_context",
                    metadata={
                        "hookName": f"PermissionRequest:{tool.name}",
                        "toolUseID": tool_use_id,
                        "hookEvent": "PermissionRequest",
                    },
                )
            )
        if result.prevent_continuation or result.blocking_error:
            return PermissionDecision.deny(hook_blocking_message(result, "PermissionRequest"))
        request_result = result.permission_request_result
        if request_result is not None:
            behavior = request_result.get("behavior")
            if behavior == "allow":
                updated_input = request_result.get("updatedInput") or decision.updated_input
                # Hook 修改 input 后重新调用工具安全检查，防止批准绕过敏感路径。
                safe_decision = await tool.check_permissions(updated_input or input, context)
                if safe_decision.behavior == "deny":
                    return safe_decision
                context.push_hook_message(
                    create_attachment_message(
                        "PermissionRequest hook allowed tool use.",
                        attachment_type="hook_permission_decision",
                        metadata={"decision": "allow", "toolUseID": tool_use_id, "hookEvent": "PermissionRequest"},
                    )
                )
                return PermissionDecision.allow(updated_input=updated_input)
            if behavior == "deny":
                context.push_hook_message(
                    create_attachment_message(
                        "PermissionRequest hook denied tool use.",
                        attachment_type="hook_permission_decision",
                        metadata={"decision": "deny", "toolUseID": tool_use_id, "hookEvent": "PermissionRequest"},
                    )
                )
                return PermissionDecision.deny(request_result.get("message") or decision.message or "Permission denied by PermissionRequest hook.")
        if result.permission_behavior == "allow":
            updated_input = result.updated_input or decision.updated_input
            safe_decision = await tool.check_permissions(updated_input or input, context)
            if safe_decision.behavior == "deny":
                return safe_decision
            context.push_hook_message(
                create_attachment_message(
                    "PermissionRequest hook allowed tool use.",
                    attachment_type="hook_permission_decision",
                    metadata={"decision": "allow", "toolUseID": tool_use_id, "hookEvent": "PermissionRequest"},
                )
            )
            return PermissionDecision.allow(updated_input=updated_input)
        if result.permission_behavior == "deny":
            context.push_hook_message(
                create_attachment_message(
                    "PermissionRequest hook denied tool use.",
                    attachment_type="hook_permission_decision",
                    metadata={"decision": "deny", "toolUseID": tool_use_id, "hookEvent": "PermissionRequest"},
                )
            )
            return PermissionDecision.deny(result.hook_permission_decision_reason or decision.message or "Permission denied by PermissionRequest hook.")
    return None


async def has_permissions_to_use_tool(
    tool: PermissionTool,
    input: dict,
    context: object,
    assistant_message: dict | None = None,
    tool_use_id: str | None = None,
    force_decision: PermissionDecision | None = None,
) -> PermissionDecision:
    """按安全约束、模式、hook 与工具建议得到最终权限。"""
    permission_context: ToolPermissionContext = context.get_app_state().tool_permission_context

    # 工具自身的 deny 是硬约束，任何模式或 hook 都不能覆盖。
    tool_decision = await tool.check_permissions(input, context)
    if tool_decision.behavior == "deny":
        return tool_decision

    if force_decision is not None:
        # PreToolUse hook 可以收紧权限或修正 input，但修正后仍需重新安全检查。
        if force_decision.behavior == "deny":
            return force_decision
        if force_decision.updated_input is not None:
            input = force_decision.updated_input
            tool_decision = await tool.check_permissions(input, context)
            if tool_decision.behavior == "deny":
                return tool_decision
        if force_decision.behavior == "ask":
            tool_decision = PermissionDecision(
                "ask",
                force_decision.message or tool_decision.message,
                updated_input=force_decision.updated_input or tool_decision.updated_input,
                bypass_immune=force_decision.bypass_immune or tool_decision.bypass_immune,
            )
        elif force_decision.behavior == "allow":
            return PermissionDecision.allow(updated_input=force_decision.updated_input or tool_decision.updated_input)

    if tool_decision.behavior == "allow":
        return PermissionDecision.allow(updated_input=tool_decision.updated_input)

    if permission_context.mode in {"bypass", "bypassPermissions"} and not tool_decision.bypass_immune:
        # bypass 仅跳过人工确认，不跳过标记为 bypass_immune 的安全要求。
        return PermissionDecision.allow(updated_input=tool_decision.updated_input)

    if permission_context.mode == "acceptEdits" and tool.name in {"Edit", "Write", "MultiEdit", "NotebookEdit"} and not tool_decision.bypass_immune:
        return PermissionDecision.allow(updated_input=tool_decision.updated_input)

    if permission_context.mode == "plan" and not tool.is_read_only(input):
        # plan 模式允许继续探索，但禁止所有有副作用调用。
        return PermissionDecision.deny(tool_decision.message or f"Tool {tool.name} is not available in plan mode.")

    if permission_context.mode == "dontAsk":
        return PermissionDecision.deny(tool_decision.message or "Permission denied.")

    return await resolve_ask(tool, input, context, tool_decision, assistant_message, tool_use_id)
