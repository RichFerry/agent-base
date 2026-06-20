"""Agent Kernel 的公共导出面。

职责：
- 汇总应用调用方需要的稳定类型和函数，隐藏包内文件布局。
- 暴露 QueryEngine/query、模型、工具、消息、session、SDK、扩展等边界。
- 让测试和外部程序统一从 ``agent_kernel`` 导入，而不是依赖内部模块路径。

本模块不创建单例、不读取配置，也不执行任何 agent 行为。导入它只会加载类型和
定义；真正的 session 状态由 QueryEngine 持有。阅读时可把 ``__all__`` 当作内核的
公开能力清单，新增内部 helper 通常不应放进这里。
"""

from .abort import AbortController, AbortSignal
from .agents import AgentDefinition, AgentTool, SidechainSessionStore, format_agent_line, get_agent_tool_prompt, load_agents, resolve_agent_tools, run_subagent
from .config import AgentConfig, CachedMicrocompactConfig, ContextCompactionConfig, FeatureFlags, KernelConfig, MCPClientConfig, OutputStyleConfig, SkillConfig
from .context_compaction import (
    CompactionResult,
    MicrocompactResult,
    compact_conversation,
    format_compact_summary,
    get_compact_prompt,
    get_compact_user_summary_message,
    microcompact_messages,
    should_auto_compact,
)
from .hooks import HookEvent, HookMatcher, HookRegistry, HookResult, run_hook_event
from .messages import (
    AssistantMessage,
    ContentBlock,
    Message,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    create_attachment_message,
    create_assistant_message,
    create_system_message,
    create_user_interruption_message,
    create_user_message,
    ensure_tool_result_pairing,
    normalize_messages_for_api,
)
from .model_provider import AnthropicAPIError, AnthropicModelProvider, FakeModelProvider, ModelProvider, normalize_anthropic_stream_events
from .mcp import (
    ListMcpResourcesTool,
    MCPTool,
    ReadMcpResourceTool,
    build_mcp_tool_name,
    get_mcp_display_name,
    get_mcp_prefix,
    mcp_info_from_string,
    mcp_tools_from_clients,
    normalize_name_for_mcp,
)
from .permissions import (
    PermissionDecision,
    ToolPermissionContext,
    has_permissions_to_use_tool,
)
from .prompt_composer import PromptComposer, build_effective_system_prompt
from .query import QueryParams, query
from .query_engine import QueryEngine
from .sdk import (
    build_error_message,
    build_result_message,
    build_sdk_status_message,
    build_system_init_message,
    from_sdk_compact_metadata,
    sdk_compat_tool_name,
    to_internal_messages,
    to_sdk_compact_metadata,
    to_sdk_messages,
)
from .session import SessionStore
from .skills import SkillDefinition, SkillTool, format_skills_within_budget, get_skill_system_reminder, load_skills
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

__all__ = [
    "AssistantMessage",
    "AgentConfig",
    "AgentDefinition",
    "AgentTool",
    "AnthropicAPIError",
    "AnthropicModelProvider",
    "AbortController",
    "AbortSignal",
    "BashTool",
    "CachedMicrocompactConfig",
    "CompactionResult",
    "ContextCompactionConfig",
    "ContentBlock",
    "EditTool",
    "FakeModelProvider",
    "FeatureFlags",
    "FileReadTool",
    "FileWriteTool",
    "GlobTool",
    "GrepTool",
    "HookEvent",
    "HookMatcher",
    "HookRegistry",
    "HookResult",
    "KernelConfig",
    "LSTool",
    "ListMcpResourcesTool",
    "MCPClientConfig",
    "MCPTool",
    "Message",
    "MicrocompactResult",
    "ModelProvider",
    "MultiEditTool",
    "NotebookEditTool",
    "OutputStyleConfig",
    "PermissionDecision",
    "PromptComposer",
    "QueryEngine",
    "QueryParams",
    "ReadMcpResourceTool",
    "SessionStore",
    "SkillConfig",
    "SkillDefinition",
    "SkillTool",
    "SidechainSessionStore",
    "Tool",
    "ToolPermissionContext",
    "ToolResultBlock",
    "TodoWriteTool",
    "ToolUseBlock",
    "ToolUseContext",
    "UserMessage",
    "WebFetchTool",
    "WebSearchTool",
    "build_effective_system_prompt",
    "build_error_message",
    "build_result_message",
    "build_sdk_status_message",
    "build_system_init_message",
    "build_mcp_tool_name",
    "compact_conversation",
    "create_attachment_message",
    "create_assistant_message",
    "create_system_message",
    "create_user_interruption_message",
    "create_user_message",
    "ensure_tool_result_pairing",
    "format_agent_line",
    "format_compact_summary",
    "format_skills_within_budget",
    "from_sdk_compact_metadata",
    "get_agent_tool_prompt",
    "get_compact_prompt",
    "get_compact_user_summary_message",
    "get_mcp_display_name",
    "get_mcp_prefix",
    "get_skill_system_reminder",
    "has_permissions_to_use_tool",
    "load_agents",
    "load_skills",
    "mcp_info_from_string",
    "mcp_tools_from_clients",
    "microcompact_messages",
    "normalize_name_for_mcp",
    "normalize_anthropic_stream_events",
    "normalize_messages_for_api",
    "query",
    "resolve_agent_tools",
    "run_subagent",
    "run_hook_event",
    "sdk_compat_tool_name",
    "should_auto_compact",
    "to_internal_messages",
    "to_sdk_compact_metadata",
    "to_sdk_messages",
]
