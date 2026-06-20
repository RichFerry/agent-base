"""面向调用方的 Agent Kernel 门面和 session 级依赖装配器。

``QueryEngine`` 是最常用入口。初始化时它完成：选择真实/Fake provider、加载 memory、
skill 和 agent、注册默认/MCP 工具、创建 SessionStore，以及构造共享 ToolUseContext。
``mutable_messages`` 保存当前会话的有效历史；``resume=True`` 时从 JSONL 恢复。

``submit_message`` 的职责不是实现 agent loop，而是：
1. 记录用户消息。
2. 请求 PromptComposer 生成 system/user/system-context 三部分。
3. 构造 QueryParams 并逐项转发 ``query()`` 事件。
4. 将可持久化消息同步回 mutable_messages 与 transcript。
5. 可选地在外层添加 SDK system/init、result 和 error 事件。

``cancel`` 只触发共享 AbortController；实际模型、compact、工具清理由下层完成。核心
query 保持不依赖 SDK，从而也能被 subagent 复用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import time
from typing import AsyncIterator
from uuid import uuid4

from .abort import AbortController
from .agents import AGENT_TOOL_NAME, AgentDefinition, AgentTool, load_agents
from .config import KernelConfig
from .memory import MemoryLoader
from .messages import Message, create_user_message
from .model_provider import AnthropicModelProvider, FakeModelProvider, ModelProvider
from .mcp import mcp_tools_from_clients
from .prompt_composer import PromptComposer
from .query import QueryParams, query
from .sdk import build_error_message, build_result_message, build_sdk_status_message, build_system_init_message, extract_text_result
from .session import SessionStore
from .skills import SKILL_TOOL_NAME, SkillDefinition, SkillTool, load_skills
from .tools import (
    BashTool,
    EditTool,
    FileReadTool,
    FileWriteTool,
    GlobTool,
    GrepTool,
    LSTool,
    MultiEditTool,
    NotebookEditTool,
    TodoWriteTool,
    Tool,
    ToolUseContext,
    WebFetchTool,
    WebSearchTool,
)


def default_tools() -> list[Tool]:
    """返回内核默认工具集；顺序也决定 system/init 中的稳定展示顺序。"""
    return [
        BashTool(),
        GlobTool(),
        GrepTool(),
        LSTool(),
        FileReadTool(),
        FileWriteTool(),
        EditTool(),
        MultiEditTool(),
        NotebookEditTool(),
        TodoWriteTool(),
        WebSearchTool(),
        WebFetchTool(),
    ]


DEFAULT_TOOL_NAMES = {tool.name for tool in default_tools()}


def _is_default_tool_list(tools: list[Tool]) -> bool:
    """判断default 工具 list，供会话入口流程使用。"""
    return {tool.name for tool in tools} == DEFAULT_TOOL_NAMES


def merge_mcp_tools(tools: list[Tool], config: KernelConfig) -> list[Tool]:
    """完成 ``merge_mcp_tools`` 对应的会话入口内部步骤。"""
    existing_names = {tool.name for tool in tools}
    merged = list(tools)
    for tool in mcp_tools_from_clients(config.mcp_clients):
        if tool.name not in existing_names:
            merged.append(tool)
            existing_names.add(tool.name)
    return merged


@dataclass
class QueryEngine:
    """一个可 resume、可连续 submit 的有状态 agent 会话。"""
    # provider/config/tools 是可替换依赖；默认值组成可直接运行的内核。
    model_provider: ModelProvider | None = None
    config: KernelConfig = field(default_factory=KernelConfig)
    tools: list[Tool] = field(default_factory=default_tools)
    # session_id 同时决定 transcript 文件名和 SDK 事件关联键。
    session_id: str = field(default_factory=lambda: str(uuid4()))
    model: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-6"))
    # 这是跨 submit 的有效消息历史，不包含 progress 等瞬时事件。
    mutable_messages: list[Message] = field(default_factory=list)
    # UI/应用层通过 callback 完成 ask；内核本身不实现交互界面。
    permission_callback: object | None = None
    resume: bool = False

    def __post_init__(self) -> None:
        """解析所有可插拔组件，并创建本 session 的 ToolUseContext。"""
        if self.model_provider is None:
            self.model_provider = (
                AnthropicModelProvider.from_env()
                if os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
                else FakeModelProvider()
            )
        # 以下组件共享同一 KernelConfig 和 session_id，保证路径及扩展视图一致。
        self.memory_loader = MemoryLoader(self.config)
        self.prompt_composer = PromptComposer(self.config, self.memory_loader)
        self.session_store = SessionStore(self.config, self.session_id)
        self.skills: list[SkillDefinition] = load_skills(self.config)
        self.agents: list[AgentDefinition] = load_agents(self.config)
        should_add_agent_tool = _is_default_tool_list(self.tools) or bool(self.config.agents or self.config.agent_paths)
        if self.agents and should_add_agent_tool and not any(tool.name == AGENT_TOOL_NAME or AGENT_TOOL_NAME in tool.aliases for tool in self.tools):
            web_index = next((index for index, tool in enumerate(self.tools) if tool.name == "WebSearch"), len(self.tools))
            self.tools.insert(web_index, AgentTool(self.agents, config=self.config))
        self.tools = merge_mcp_tools(self.tools, self.config)
        if self.skills and not any(tool.name == SKILL_TOOL_NAME for tool in self.tools):
            self.tools.append(SkillTool(self.skills))
        if self.resume and not self.mutable_messages:
            self.mutable_messages.extend(self.session_store.load_messages())
        self.tool_use_context = ToolUseContext(
            config=self.config,
            tools=self.tools,
            permission_callback=self.permission_callback,
            model_provider=self.model_provider,
            web_fetch_model=self.model,
            session_id=self.session_id,
            transcript_path=str(self.session_store.transcript_path),
        )

    def cancel(self, reason: str = "Request was aborted.") -> None:
        """触发当前 submit 共用的 AbortController。"""
        self.tool_use_context.abort_controller.abort(reason)

    def get_system_init_message(self) -> dict:
        """构造描述当前会话能力的 SDK system/init 消息。"""
        return build_system_init_message(
            config=self.config,
            session_id=self.session_id,
            tools=self.tools,
            model=self.model,
            permission_mode=self.tool_use_context.app_state.tool_permission_context.mode,
            agents=[agent.as_sdk_dict() for agent in self.agents],
            skills=[skill.as_sdk_dict() for skill in self.skills],
        )

    def get_sdk_status_message(self, status: dict | None = None) -> dict:
        """构造当前会话的 SDK status 消息。"""
        return build_sdk_status_message(
            session_id=self.session_id,
            status=status or {"status": "ready"},
            permission_mode=self.tool_use_context.app_state.tool_permission_context.mode,
        )

    async def submit_message(
        self,
        prompt: str,
        *,
        max_turns: int | None = None,
        custom_system_prompt: str | None = None,
        append_system_prompt: str | None = None,
        override_system_prompt: str | None = None,
        agent_system_prompt: str | None = None,
        sdk_events: bool = False,
    ) -> AsyncIterator[dict]:
        """提交一个用户 turn，并逐个 yield 内核事件或可选 SDK 事件。"""
        # 上一次请求可能已取消；新 turn 必须使用全新的 controller。
        if self.tool_use_context.abort_controller.signal.aborted:
            self.tool_use_context.abort_controller = AbortController()
        start_time = time.monotonic()
        yielded_messages: list[Message] = []
        terminal_event: dict | None = None
        error_text: str | None = None
        if sdk_events:
            yield self.get_system_init_message()
        # 用户消息先进入内存和 transcript，确保即使模型失败也可 resume。
        user_message = create_user_message(prompt)
        self.mutable_messages.append(user_message)
        self.session_store.record_transcript([user_message])
        system_prompt, user_context, system_context = self.prompt_composer.fetch_system_prompt_parts(
            tools=self.tools,
            model=self.model,
            custom_system_prompt=custom_system_prompt,
            append_system_prompt=append_system_prompt,
            override_system_prompt=override_system_prompt,
            agent_system_prompt=agent_system_prompt,
        )
        params = QueryParams(
            messages=self.mutable_messages,
            system_prompt=system_prompt,
            user_context=user_context,
            system_context=system_context,
            tool_use_context=self.tool_use_context,
            model_provider=self.model_provider,
            max_turns=max_turns,
            model=self.model,
            context_compaction=self.config.context_compaction,
            transcript_path=str(self.session_store.transcript_path),
        )
        # query 是唯一 agent loop；这里仅同步可持久化状态并转发事件。
        async for event in query(params):
            if event.get("type") == "context_compacted":
                self.mutable_messages[:] = event["messages"]
                self.session_store.record_transcript(event["messages"])
            elif event.get("type") == "context_microcompacted":
                self.mutable_messages[:] = event["messages"]
                self.session_store.record_transcript([event["boundary"]])
            elif event.get("type") in {"assistant", "user", "system", "attachment"}:
                self.mutable_messages.append(event)
                self.session_store.record_transcript([event])
                yielded_messages.append(event)
                if event.get("type") == "system" and event.get("level") == "error":
                    error_text = event.get("error") or event.get("content")
            elif event.get("type") == "terminal":
                terminal_event = event["terminal"]
            yield event
        if not sdk_events:
            return
        duration_ms = int((time.monotonic() - start_time) * 1000)
        turns = int((terminal_event or {}).get("turns") or 0)
        reason = (terminal_event or {}).get("reason")
        if reason == "completed":
            yield build_result_message(
                session_id=self.session_id,
                subtype="success",
                is_error=False,
                duration_ms=duration_ms,
                num_turns=turns,
                result=extract_text_result(yielded_messages),
                stop_reason=None,
            )
            return
        if error_text:
            yield build_error_message(session_id=self.session_id, error=error_text)
        yield build_result_message(
            session_id=self.session_id,
            subtype="error_max_turns" if reason == "max_turns" else "error_during_execution",
            is_error=True,
            duration_ms=duration_ms,
            num_turns=turns,
            stop_reason=reason,
            errors=[error_text or (terminal_event or {}).get("message") or reason or "unknown error"],
        )
