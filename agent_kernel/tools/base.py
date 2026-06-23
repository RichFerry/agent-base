"""所有工具共享的数据模型、会话上下文与抽象协议。

数据对象：
- ValidationResult：schema/业务校验结果，不用异常表达预期输入错误。
- ToolResult：工具原始 data 及需要额外注入历史的新消息。
- ReadFileStateEntry：Edit 的 read-before-write 快照和 partial view 元数据。
- AppState：权限上下文及 Todo 等轻量 session 状态。
- ToolUseContext：贯穿主 loop/subagent 的执行依赖与可变状态。

Tool 基类把职责拆成 description/prompt、schema、validate_input、check_permissions、call、
result mapping。tool_execution.py 控制调用顺序，具体工具不应自行跳过权限管线。
``is_concurrency_safe`` 默认委托 ``is_read_only``；写工具必须显式保持 false。

ToolUseContext 还承载 hook registry、abort controller、model provider、web handlers、当前
rendered prompt 与消息快照。它是 session 内共享对象，不是全局单例；subagent 会创建
隔离副本。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..abort import AbortController
from ..config import KernelConfig
from ..hooks import HookRegistry, HookRunner
from ..messages import AssistantMessage, ToolResultBlock
from ..permissions import PermissionDecision, PermissionCallback, ToolPermissionContext


@dataclass
class ValidationResult:
    """封装 ``ValidationResult`` 产生的结构化结果。"""
    result: bool
    message: str | None = None
    error_code: int | None = None
    meta: dict[str, Any] | None = None


@dataclass
class ToolResult:
    """封装 ``ToolResult`` 产生的结构化结果。"""
    data: Any
    new_messages: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ReadFileStateEntry:
    """封装 ``ReadFileStateEntry`` 对应的工具协议状态与行为。"""
    content: str
    timestamp: int
    offset: int | None = None
    limit: int | None = None
    is_partial_view: bool = False


@dataclass
class AppState:
    """封装 ``AppState`` 对应的工具协议状态与行为。"""
    tool_permission_context: ToolPermissionContext = field(default_factory=ToolPermissionContext)
    todos: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


@dataclass
class ToolUseContext:
    """贯穿 query 和所有工具调用的共享状态容器。"""
    # 核心静态依赖和 session 可变状态。
    config: KernelConfig
    tools: list["Tool"]
    app_state: AppState = field(default_factory=AppState)
    # Read 快照由文件工具写入，并在 Edit/compact 恢复时读取。
    read_file_state: dict[str, ReadFileStateEntry] = field(default_factory=dict)
    # 权限与基础工具运行选项。
    permission_callback: PermissionCallback | None = None
    custom_system_prompt: str | None = None
    append_system_prompt: str | None = None
    is_non_interactive_session: bool = False
    tool_timeout_seconds: float | None = 300.0
    background_tasks: dict[str, Any] = field(default_factory=dict)
    # 模型/Web handler 都是注入边界，工具模块不直接依赖具体供应商。
    model_provider: Any | None = None
    web_fetch_model: str | None = None
    web_search_handler: Callable[[dict], Any] | None = None
    web_fetch_handler: Callable[[str], Any] | None = None
    web_fetch_apply_handler: Callable[[str, str, bool], Any] | None = None
    # Hook 消息先缓存，权限解析后再按确定顺序回灌。
    hook_registry: HookRegistry = field(default_factory=HookRegistry)
    hook_runner: HookRunner | None = None
    pending_hook_messages: list[dict[str, Any]] = field(default_factory=list)
    session_id: str | None = None
    transcript_path: str | None = None
    invoked_skills: dict[str, Any] = field(default_factory=dict)
    # 当前请求快照供 hook、skill 和 subagent 读取，不替代 QueryEngine 历史。
    messages: list[dict[str, Any]] = field(default_factory=list)
    rendered_system_prompt: list[str] = field(default_factory=list)
    user_context: dict[str, str] = field(default_factory=dict)
    system_context: dict[str, str] = field(default_factory=dict)
    agent_id: str | None = None
    agent_type: str | None = None
    # 同一个 controller 贯穿模型、compact 和本轮所有工具。
    abort_controller: AbortController = field(default_factory=AbortController)

    def get_app_state(self) -> AppState:
        """返回当前 ToolUseContext 持有的可变应用状态。"""
        return self.app_state

    def set_app_state(self, updater: Callable[[AppState], AppState]) -> None:
        """使用纯 updater 替换当前应用状态。"""
        self.app_state = updater(self.app_state)

    def push_hook_message(self, message: dict[str, Any]) -> None:
        """缓存 hook 产生、稍后随工具结果发送的消息。"""
        self.pending_hook_messages.append(message)

    def drain_hook_messages(self) -> list[dict[str, Any]]:
        """取出并清空当前等待发送的 hook 消息。"""
        messages = list(self.pending_hook_messages)
        self.pending_hook_messages.clear()
        return messages


class Tool:
    """Agent Base-style 工具协议的 Python 基类。"""
    name: str = ""
    aliases: tuple[str, ...] = ()
    search_hint: str | None = None
    max_result_size_chars: int = 100_000
    input_schema: dict[str, type | tuple[type, ...]] = {}
    required_fields: tuple[str, ...] = ()

    async def description(self, input: dict | None = None) -> str:
        """返回提供给模型和调用方展示的工具简短说明。"""
        return self.name

    async def prompt(self) -> str:
        """返回该工具按源码保留的模型使用提示词。"""
        return ""

    def user_facing_name(self, input: dict | None = None) -> str:
        """根据当前输入返回适合界面展示的工具名称。"""
        return self.name

    def get_path(self, input: dict) -> str:
        """从工具输入中提取用于权限检查的目标路径。"""
        return ""

    def is_concurrency_safe(self, input: dict) -> bool:
        """判断当前输入是否可以与相邻安全工具并发执行。"""
        return self.is_read_only(input)

    def is_read_only(self, input: dict) -> bool:
        """判断当前输入是否只读取外部状态而不产生修改。"""
        return False

    def is_destructive(self, input: dict) -> bool:
        """判断当前输入是否可能产生破坏性副作用。"""
        return False

    def prepare_permission_matcher(self, input: dict):
        """为当前输入构造可选的权限规则匹配函数。"""
        path = self.get_path(input)
        if path:
            from fnmatch import fnmatch

            return lambda pattern: fnmatch(path, pattern)
        return None

    def validate_schema(self, input: dict) -> ValidationResult:
        """执行无副作用的字段存在性与 Python 类型校验。"""
        if not isinstance(input, dict):
            return ValidationResult(False, "Input must be an object.")
        for field_name in self.required_fields:
            if field_name not in input:
                return ValidationResult(False, f"Missing required field: {field_name}")
        for field_name, field_type in self.input_schema.items():
            if field_name in input and input[field_name] is not None and not isinstance(input[field_name], field_type):
                return ValidationResult(False, f"Field {field_name} has invalid type.")
        return ValidationResult(True)

    async def validate_input(self, input: dict, context: ToolUseContext) -> ValidationResult:
        """执行依赖上下文的业务输入校验。"""
        return ValidationResult(True)

    async def check_permissions(self, input: dict, context: ToolUseContext) -> PermissionDecision:
        """返回该工具针对当前输入的初步权限建议。"""
        return PermissionDecision.ask()

    async def call(
        self,
        args: dict,
        context: ToolUseContext,
        can_use_tool,
        parent_message: AssistantMessage,
        on_progress=None,
    ) -> ToolResult:
        """执行工具；权限已经由 tool_execution 在调用前确认。"""
        raise NotImplementedError

    def map_tool_result_to_tool_result_block_param(self, content: Any, tool_use_id: str) -> ToolResultBlock:
        """把工具内部结果映射为 Anthropic tool_result block。"""
        return {"tool_use_id": tool_use_id, "type": "tool_result", "content": str(content)}


def find_tool_by_name(tools: list[Tool], name: str) -> Tool | None:
    """查找工具 by name，供工具协议流程使用。"""
    for tool in tools:
        if tool.name == name or name in tool.aliases:
            return tool
    return None


def ensure_absolute(path: str | Path) -> Path:
    """确保absolute，供工具协议流程使用。"""
    expanded = Path(path).expanduser()
    return expanded if expanded.is_absolute() else expanded.resolve()
