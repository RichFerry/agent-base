"""内核配置模型、环境默认值和扩展声明。

``KernelConfig`` 是根对象，持有 cwd/config_home、平台信息、功能开关、compact、MCP、
skill、agent 和 output style 配置。子配置使用 dataclass，便于测试构造最小场景，也
让 query loop 不直接散落环境变量读取。

默认值来源分三类：调用方显式传入、环境变量、当前机器状态。路径和日期在实例创建
时计算，避免模块导入时冻结。FeatureFlags 只开启可选 prompt/extension 分支；稳定
主路径不应依赖远端 gate。配置对象大多 frozen，KernelConfig 保持可变以支持测试和
应用层装配。

阅读重点：ContextCompactionConfig 决定上下文恢复策略；MCP/Skill/AgentConfig 是静态
声明，真正加载和执行分别在 mcp.py、skills.py、agents.py。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
import os
import platform


def get_config_home() -> Path:
    """获取配置 home，供内核配置流程使用。"""
    # 显式 override 便于 SDK、测试和多租户进程隔离持久化目录。
    override = os.environ.get("CLAUDE_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".claude"


def local_iso_date() -> str:
    """完成 ``local_iso_date`` 对应的内核配置内部步骤。"""
    override = os.environ.get("CLAUDE_CODE_OVERRIDE_DATE")
    if override:
        return override
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d")


def shell_name() -> str:
    """完成 ``shell_name`` 对应的内核配置内部步骤。"""
    shell = os.environ.get("SHELL", "unknown")
    if "zsh" in shell:
        return "zsh"
    if "bash" in shell:
        return "bash"
    return shell


def os_version() -> str:
    """完成 ``os_version`` 对应的内核配置内部步骤。"""
    if os.name == "nt":
        return platform.platform()
    return f"{platform.system()} {platform.release()}"


@dataclass(frozen=True)
class FeatureFlags:
    """影响 prompt 或外围能力的开关；默认值代表稳定主路径。"""
    proactive: bool = False
    kairos: bool = False
    kairos_brief: bool = False
    token_budget: bool = False
    cached_microcompact: bool = False
    experimental_skill_search: bool = False
    mcp_instructions_delta: bool = False
    fork_subagent: bool = False
    global_cache_scope: bool = True


@dataclass(frozen=True)
class CachedMicrocompactConfig:
    """保存 ``CachedMicrocompactConfig`` 对应的不可变配置字段。"""
    enabled: bool = False
    system_prompt_suggest_summaries: bool = False
    keep_recent: int = 3
    supported_models: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContextCompactionConfig:
    """控制 auto/partial/micro compact 的阈值与失败策略。"""
    enabled: bool = False
    # None 表示根据 context window、输出预留和 buffer 动态计算阈值。
    threshold_tokens: int | None = None
    context_window_tokens: int = 200_000
    auto_compact_buffer_tokens: int = 13_000
    max_output_tokens: int = 20_000
    suppress_follow_up_questions: bool = True
    custom_instructions: str | None = None
    fallback_to_original_on_error: bool = True
    max_prompt_too_long_retries: int = 3
    # 大于 0 时只摘要旧段，并原样保留最近若干条安全消息。
    partial_keep_recent_messages: int = 0
    post_compact_max_files_to_restore: int = 5
    microcompact_enabled: bool = False
    microcompact_keep_recent_tool_results: int = 3


@dataclass(frozen=True)
class MCPClientConfig:
    """保存 ``MCPClientConfig`` 对应的不可变配置字段。"""
    name: str
    instructions: str | None = None
    type: str = "connected"
    tools: tuple[dict[str, Any], ...] = ()
    resources: tuple[dict[str, Any], ...] = ()
    # client 与 handler 二选一，handler 让内核无需依赖具体 MCP SDK。
    client: Any | None = None
    call_tool_handler: Callable[[str, dict[str, Any]], Any] | None = None
    read_resource_handler: Callable[[str], Any] | None = None


@dataclass(frozen=True)
class SkillConfig:
    """保存 ``SkillConfig`` 对应的不可变配置字段。"""
    name: str
    description: str
    content: str
    when_to_use: str | None = None
    allowed_tools: tuple[str, ...] = ()
    argument_hint: str | None = None
    argument_names: tuple[str, ...] = ()
    version: str | None = None
    model: str | None = None
    disable_model_invocation: bool = False
    user_invocable: bool = True
    source: str = "skills"
    loaded_from: str = "skills"
    base_dir: Path | None = None
    context: str = "inline"
    hooks: dict[str, Any] | None = None
    paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentConfig:
    """保存 ``AgentConfig`` 对应的不可变配置字段。"""
    name: str
    description: str
    prompt: str
    tools: tuple[str, ...] | None = None
    disallowed_tools: tuple[str, ...] = ()
    skills: tuple[str, ...] = ()
    mcp_servers: tuple[Any, ...] = ()
    hooks: dict[str, Any] | None = None
    color: str | None = None
    model: str | None = None
    effort: str | int | None = None
    permission_mode: str | None = None
    max_turns: int | None = None
    source: str = "flagSettings"
    base_dir: Path | None = None
    background: bool = False
    initial_prompt: str | None = None
    memory: str | None = None
    isolation: str | None = None
    omit_claude_md: bool = False
    critical_system_reminder: str | None = None


@dataclass(frozen=True)
class OutputStyleConfig:
    """保存 ``OutputStyleConfig`` 对应的不可变配置字段。"""
    name: str
    prompt: str
    description: str = ""
    source: str = "built-in"
    keep_coding_instructions: bool | None = None


@dataclass
class KernelConfig:
    """一个 QueryEngine 实例使用的完整静态配置。"""
    # 路径/环境字段会进入 prompt、权限和 transcript，因此必须由同一实例共享。
    cwd: Path = field(default_factory=lambda: Path.cwd())
    config_home: Path = field(default_factory=get_config_home)
    workspace_root: Path | None = None
    settings_paths: tuple[Path, ...] = ()
    session_start_date: str = field(default_factory=local_iso_date)
    platform: str = field(default_factory=lambda: "win32" if os.name == "nt" else "darwin" if platform.system() == "Darwin" else "linux")
    shell: str = field(default_factory=shell_name)
    os_version: str = field(default_factory=os_version)
    user_type: str = field(default_factory=lambda: os.environ.get("USER_TYPE", "external"))
    language: str | None = None
    # feature flags 控制可选行为，不能改变默认核心消息协议。
    features: FeatureFlags = field(default_factory=FeatureFlags)
    simple_mode: bool = field(default_factory=lambda: os.environ.get("CLAUDE_CODE_SIMPLE", "").lower() in {"1", "true", "yes"})
    is_non_interactive_session: bool = False
    auto_memory_enabled: bool = True
    kairos_active: bool = False
    scratchpad_enabled: bool = False
    scratchpad_dir: Path | None = None
    cached_microcompact: CachedMicrocompactConfig = field(default_factory=CachedMicrocompactConfig)
    context_compaction: ContextCompactionConfig = field(default_factory=ContextCompactionConfig)
    # 扩展声明保持不可变 tuple；QueryEngine 初始化时解析成运行对象。
    mcp_clients: tuple[MCPClientConfig, ...] = ()
    mcp_config_paths: tuple[Path, ...] = ()
    mcp_fixture_paths: tuple[Path, ...] = ()
    skills: tuple[SkillConfig, ...] = ()
    skill_paths: tuple[Path, ...] = ()
    skill_discovery_mode: str = "ambient"
    agents: tuple[AgentConfig, ...] = ()
    agent_paths: tuple[Path, ...] = ()
    disable_builtin_agents: bool = False
    output_style: OutputStyleConfig | None = None

    @property
    def workspace_runtime(self):
        """Return resolved workspace storage and extension boundary facts."""
        from .workspace import build_workspace_runtime

        return build_workspace_runtime(
            cwd=self.cwd,
            config_home=self.config_home,
            workspace_root=self.workspace_root,
            settings_paths=self.settings_paths,
            skill_paths=self.skill_paths,
            mcp_config_paths=self.mcp_config_paths,
            mcp_fixture_paths=self.mcp_fixture_paths,
            mcp_server_names=(client.name for client in self.mcp_clients),
            memory_enabled=self.auto_memory_enabled,
        )
