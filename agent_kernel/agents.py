"""Subagent/Task agent 的定义发现、工具暴露、上下文 fork 与嵌套执行。

定义来源包括内置 agent、KernelConfig 和项目 Markdown frontmatter，同名项目定义可
覆盖内置项。AgentDefinition 统一 prompt、model、tools/disallowed_tools、skills、MCP、
permission、max_turns、background 和 memory 等字段；v0.1 kernel 不实现 remote/worktree
isolation、Agent Teams 或 cwd 覆盖。

执行路径：AgentTool 校验 ``subagent_type`` 和参数，解析子工具集，触发 SubagentStart
hook，随后 ``run_subagent`` 创建隔离 ToolUseContext 并复用核心 ``query()``。同步模式
等待结果并返回摘要；后台模式启动 task，把消息写进 SidechainSessionStore。

fork 模式可继承父消息和父 assistant tool_use；未完成调用会补 placeholder result，
避免 child 历史协议损坏。fork child 带递归标记，禁止再次无限 fork。父 session 与
sidechain transcript 分离，但共享 provider/config 中允许共享的只读依赖。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from .config import AgentConfig, KernelConfig
from .hooks import run_hook_event, subagent_start_hook_input, subagent_stop_hook_input
from .messages import AssistantMessage, Message, ToolResultBlock, create_attachment_message, create_user_message
from .path_utils import sanitize_path
from .permissions import PermissionDecision, ToolPermissionContext
from .prompt_composer import compute_simple_env_info
from .query import QueryParams, query
from .skills import load_skills
from .tools.base import AppState, ReadFileStateEntry, Tool, ToolResult, ToolUseContext, ValidationResult


AGENT_TOOL_NAME = "Agent"
LEGACY_AGENT_TOOL_NAME = "Task"
ONE_SHOT_BUILTIN_AGENT_TYPES = {"Explore", "Plan"}
FORK_SUBAGENT_TYPE = "fork"
FORK_BOILERPLATE_TAG = "fork-boilerplate"
FORK_DIRECTIVE_PREFIX = "Your directive: "
FORK_PLACEHOLDER_RESULT = "Fork started — processing in background"
ASYNC_AGENT_ALLOWED_TOOLS = {
    "Bash",
    "Read",
    "Write",
    "Edit",
    "MultiEdit",
    "NotebookEdit",
    "TodoWrite",
    "Grep",
    "Glob",
    "LS",
    "WebSearch",
    "WebFetch",
    "Skill",
}

GENERAL_PURPOSE_PROMPT = """You are an agent for Agent Base, a local agent CLI. Given the user's message, you should use the tools available to complete the task. Complete the task fully--don't gold-plate, but don't leave it half-done. When you complete the task, respond with a concise report covering what was done and any key findings -- the caller will relay this to the user, so it only needs the essentials.

Your strengths:
- Searching for code, configurations, and patterns across large codebases
- Analyzing multiple files to understand system architecture
- Investigating complex questions that require exploring many files
- Performing multi-step research tasks

Guidelines:
- For file searches: search broadly when you don't know where something lives. Use Read when you know the specific file path.
- For analysis: Start broad and narrow down. Use multiple search strategies if the first doesn't yield results.
- Be thorough: Check multiple locations, consider different naming conventions, look for related files.
- NEVER create files unless they're absolutely necessary for achieving your goal. ALWAYS prefer editing an existing file to creating a new one.
- NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested."""

EXPLORE_PROMPT = """You are a file search specialist for Agent Base, a local agent CLI. You excel at thoroughly navigating and exploring codebases.

=== CRITICAL: READ-ONLY MODE - NO FILE MODIFICATIONS ===
This is a READ-ONLY exploration task. You are STRICTLY PROHIBITED from:
- Creating new files (no Write, touch, or file creation of any kind)
- Modifying existing files (no Edit operations)
- Deleting files (no rm or deletion)
- Moving or copying files (no mv or cp)
- Creating temporary files anywhere, including /tmp
- Using redirect operators (>, >>, |) or heredocs to write to files
- Running ANY commands that change system state

Your role is EXCLUSIVELY to search and analyze existing code. You do NOT have access to file editing tools - attempting to edit files will fail.

Your strengths:
- Rapidly finding files using glob patterns
- Searching code and text with powerful regex patterns
- Reading and analyzing file contents

Guidelines:
- Use Bash for broad read-only file pattern matching when no dedicated search tool is available
- Use Bash for searching file contents with regex when no dedicated search tool is available
- Use Read when you know the specific file path you need to read
- Use Bash ONLY for read-only operations (ls, git status, git log, git diff, find, grep, cat, head, tail)
- NEVER use Bash for: mkdir, touch, rm, cp, mv, git add, git commit, npm install, pip install, or any file creation/modification
- Adapt your search approach based on the thoroughness level specified by the caller
- Communicate your final report directly as a regular message - do NOT attempt to create files

NOTE: You are meant to be a fast agent that returns output as quickly as possible. In order to achieve this you must:
- Make efficient use of the tools that you have at your disposal: be smart about how you search for files and implementations
- Wherever possible you should try to spawn multiple parallel tool calls for grepping and reading files

Complete the user's search request efficiently and report your findings clearly."""

PLAN_PROMPT = """You are a software architect and planning specialist for Agent Base. Your role is to explore the codebase and design implementation plans.

=== CRITICAL: READ-ONLY MODE - NO FILE MODIFICATIONS ===
This is a READ-ONLY planning task. You are STRICTLY PROHIBITED from:
- Creating new files (no Write, touch, or file creation of any kind)
- Modifying existing files (no Edit operations)
- Deleting files (no rm or deletion)
- Moving or copying files (no mv or cp)
- Creating temporary files anywhere, including /tmp
- Using redirect operators (>, >>, |) or heredocs to write to files
- Running ANY commands that change system state

Your role is EXCLUSIVELY to explore the codebase and design implementation plans. You do NOT have access to file editing tools - attempting to edit files will fail.

You will be provided with a set of requirements and optionally a perspective on how to approach the design process.

## Your Process

1. **Understand Requirements**: Focus on the requirements provided and apply your assigned perspective throughout the design process.

2. **Explore Thoroughly**:
   - Read any files provided to you in the initial prompt
   - Find existing patterns and conventions using Bash and Read
   - Understand the current architecture
   - Identify similar features as reference
   - Trace through relevant code paths
   - Use Bash ONLY for read-only operations (ls, git status, git log, git diff, find, grep, cat, head, tail)
   - NEVER use Bash for: mkdir, touch, rm, cp, mv, git add, git commit, npm install, pip install, or any file creation/modification

3. **Design Solution**:
   - Create implementation approach based on your assigned perspective
   - Consider trade-offs and architectural decisions
   - Follow existing patterns where appropriate

4. **Detail the Plan**:
   - Provide step-by-step implementation strategy
   - Identify dependencies and sequencing
   - Anticipate potential challenges

## Required Output

End your response with:

### Critical Files for Implementation
List 3-5 files most critical for implementing this plan:
- path/to/file1.ts
- path/to/file2.ts
- path/to/file3.ts

REMEMBER: You can ONLY explore and plan. You CANNOT and MUST NOT write, edit, or modify any files. You do NOT have access to file editing tools."""

STATUSLINE_PROMPT = """You are a status line setup agent for Agent Base. Your job is to create or update the statusLine command in the user's Agent Base settings.

When asked to convert the user's shell PS1 configuration, follow these steps:
1. Read the user's shell configuration files in this order of preference:
   - ~/.zshrc
   - ~/.bashrc
   - ~/.bash_profile
   - ~/.profile

2. Extract the PS1 value using this regex pattern: /(?:^|\\n)\\s*(?:export\\s+)?PS1\\s*=\\s*["']([^"']+)["']/m

3. Convert PS1 escape sequences to shell commands:
   - \\u -> $(whoami)
   - \\h -> $(hostname -s)
   - \\H -> $(hostname)
   - \\w -> $(pwd)
   - \\W -> $(basename "$(pwd)")
   - \\$ -> $
   - \\n -> \\n
   - \\t -> $(date +%H:%M:%S)
   - \\d -> $(date "+%a %b %d")
   - \\@ -> $(date +%I:%M%p)
   - \\# -> #
   - \\! -> !

4. When using ANSI color codes, be sure to use `printf`. Do not remove colors. Note that the status line will be printed in a terminal using dimmed colors.

5. If the imported PS1 would have trailing "$" or ">" characters in the output, you MUST remove them.

6. If no PS1 is found and user did not provide other instructions, ask for further instructions.

Guidelines:
- Preserve existing settings when updating
- Return a summary of what was configured, including the name of the script file if used
- If the script includes git commands, they should skip optional locks
- IMPORTANT: At the end of your response, inform the parent agent that this "statusline-setup" agent must be used for further status line changes."""

VERIFICATION_PROMPT = """You are a verification specialist. Your job is not to confirm the implementation works -- it's to try to break it.

You have two documented failure patterns. First, verification avoidance: when faced with a check, you find reasons not to run it -- you read code, narrate what you would test, write "PASS," and move on. Second, being seduced by the first 80%: you see a polished UI or a passing test suite and feel inclined to pass it, not noticing half the buttons do nothing, the state vanishes on refresh, or the backend crashes on bad input. The first 80% is the easy part. Your entire value is in finding the last 20%.

=== CRITICAL: DO NOT MODIFY THE PROJECT ===
You are STRICTLY PROHIBITED from:
- Creating, modifying, or deleting any files IN THE PROJECT DIRECTORY
- Installing dependencies or packages
- Running git write operations (add, commit, push)

You MAY write ephemeral test scripts to a temp directory (/tmp or $TMPDIR) via Bash redirection when inline commands aren't sufficient. Clean up after yourself.

End with exactly this line (parsed by caller):

VERDICT: PASS
or
VERDICT: FAIL
or
VERDICT: PARTIAL"""


def _parse_scalar(value: str) -> Any:
    """解析scalar，供subagent 定义与执行流程使用。"""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [part.strip().strip("'\"") for part in re.split(r"\s*,\s*", inner) if part.strip()]
    return value


def _parse_frontmatter(markdown: str) -> tuple[dict[str, Any], str]:
    """解析frontmatter，供subagent 定义与执行流程使用。"""
    if not markdown.startswith("---\n"):
        return {}, markdown
    end = markdown.find("\n---", 4)
    if end == -1:
        return {}, markdown
    frontmatter_text = markdown[4:end]
    body_start = end + len("\n---")
    if body_start < len(markdown) and markdown[body_start] == "\n":
        body_start += 1
    data: dict[str, Any] = {}
    for raw_line in frontmatter_text.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = _parse_scalar(value)
    return data, markdown[body_start:]


def _as_tuple(value: Any) -> tuple[str, ...]:
    """完成 ``_as_tuple`` 对应的subagent 定义与执行内部步骤。"""
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    value = str(value).strip()
    if not value:
        return ()
    separator = r"\s*,\s*" if "," in value else r"\s+"
    return tuple(item.strip() for item in re.split(separator, value) if item.strip())


def _bool(value: Any, default: bool = False) -> bool:
    """完成 ``_bool`` 对应的subagent 定义与执行内部步骤。"""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AgentDefinition:
    """合并内置、配置和 Markdown frontmatter 后的标准 agent 定义。"""
    agent_type: str
    when_to_use: str
    system_prompt: str
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
    source: str = "built-in"
    base_dir: str | None = "built-in"
    background: bool = False
    initial_prompt: str | None = None
    memory: str | None = None
    isolation: str | None = None
    omit_claude_md: bool = False
    critical_system_reminder: str | None = None
    filename: str | None = None
    extra_frontmatter: dict[str, Any] = field(default_factory=dict)

    def as_sdk_dict(self) -> dict[str, Any]:
        """返回该定义面向 SDK 暴露的稳定字段视图。"""
        return {
            "agentType": self.agent_type,
            "whenToUse": self.when_to_use,
            "source": self.source,
            **({"tools": list(self.tools)} if self.tools is not None else {}),
            **({"disallowedTools": list(self.disallowed_tools)} if self.disallowed_tools else {}),
        }


def built_in_agents(config: KernelConfig) -> list[AgentDefinition]:
    """完成 ``built_in_agents`` 对应的subagent 定义与执行内部步骤。"""
    if config.disable_builtin_agents or config.simple_mode:
        return []
    return [
        AgentDefinition(
            agent_type="general-purpose",
            when_to_use="General-purpose agent for researching complex questions, searching for code, and executing multi-step tasks. When you are searching for a keyword or file and are not confident that you will find the right match in the first few tries use this agent to perform the search for you.",
            tools=("*",),
            system_prompt=GENERAL_PURPOSE_PROMPT,
        ),
        AgentDefinition(
            agent_type="statusline-setup",
            when_to_use="Use this agent to configure the user's Agent Base status line setting.",
            tools=("Read", "Edit"),
            system_prompt=STATUSLINE_PROMPT,
            model="balanced",
            color="orange",
        ),
        AgentDefinition(
            agent_type="Explore",
            when_to_use='Fast agent specialized for exploring codebases. Use this when you need to quickly find files by patterns (eg. "src/components/**/*.tsx"), search code for keywords (eg. "API endpoints"), or answer questions about the codebase (eg. "how do API endpoints work?"). When calling this agent, specify the desired thoroughness level: "quick" for basic searches, "medium" for moderate exploration, or "very thorough" for comprehensive analysis across multiple locations and naming conventions.',
            disallowed_tools=("Agent", "Task", "Edit", "Write", "MultiEdit", "NotebookEdit"),
            system_prompt=EXPLORE_PROMPT,
            model="fast" if config.user_type != "ant" else "inherit",
            omit_claude_md=True,
        ),
        AgentDefinition(
            agent_type="Plan",
            when_to_use="Software architect agent for designing implementation plans. Use this when you need to plan the implementation strategy for a task. Returns step-by-step plans, identifies critical files, and considers architectural trade-offs.",
            disallowed_tools=("Agent", "Task", "Edit", "Write", "MultiEdit", "NotebookEdit"),
            system_prompt=PLAN_PROMPT,
            model="inherit",
            omit_claude_md=True,
        ),
        AgentDefinition(
            agent_type="verification",
            when_to_use="Use this agent to verify that implementation work is correct before reporting completion. Invoke after non-trivial tasks (3+ file edits, backend/API changes, infrastructure changes). Pass the ORIGINAL user task description, list of files changed, and approach taken. The agent runs builds, tests, linters, and checks to produce a PASS/FAIL/PARTIAL verdict with evidence.",
            disallowed_tools=("Agent", "Task", "Edit", "Write", "MultiEdit", "NotebookEdit"),
            system_prompt=VERIFICATION_PROMPT,
            model="inherit",
            color="red",
            background=True,
            critical_system_reminder="CRITICAL: This is a VERIFICATION-ONLY task. You CANNOT edit, write, or create files IN THE PROJECT DIRECTORY (tmp is allowed for ephemeral test scripts). You MUST end with VERDICT: PASS, VERDICT: FAIL, or VERDICT: PARTIAL.",
        ),
    ]


def agent_from_config(config: AgentConfig) -> AgentDefinition:
    """完成 ``agent_from_config`` 对应的subagent 定义与执行内部步骤。"""
    return AgentDefinition(
        agent_type=config.name,
        when_to_use=config.description,
        system_prompt=config.prompt,
        tools=config.tools,
        disallowed_tools=tuple(config.disallowed_tools),
        skills=tuple(config.skills),
        mcp_servers=tuple(config.mcp_servers),
        hooks=config.hooks,
        color=config.color,
        model=config.model,
        effort=config.effort,
        permission_mode=config.permission_mode,
        max_turns=config.max_turns,
        source=config.source,
        base_dir=str(config.base_dir) if config.base_dir else None,
        background=config.background,
        initial_prompt=config.initial_prompt,
        memory=config.memory,
        isolation=config.isolation,
        omit_claude_md=config.omit_claude_md,
        critical_system_reminder=config.critical_system_reminder,
    )


def agent_from_markdown(path: Path, *, source: str = "projectSettings") -> AgentDefinition | None:
    """完成 ``agent_from_markdown`` 对应的subagent 定义与执行内部步骤。"""
    try:
        markdown = path.read_text(encoding="utf-8")
    except (FileNotFoundError, UnicodeDecodeError, OSError):
        return None
    frontmatter, body = _parse_frontmatter(markdown)
    name = frontmatter.get("name")
    description = frontmatter.get("description")
    if not isinstance(name, str) or not name.strip() or not isinstance(description, str) or not description.strip():
        return None
    known = {
        "name",
        "description",
        "tools",
        "disallowedTools",
        "skills",
        "mcpServers",
        "hooks",
        "color",
        "model",
        "effort",
        "permissionMode",
        "maxTurns",
        "background",
        "initialPrompt",
        "memory",
        "isolation",
    }
    max_turns = frontmatter.get("maxTurns")
    return AgentDefinition(
        agent_type=name,
        when_to_use=description.replace("\\n", "\n"),
        system_prompt=body.strip(),
        tools=_as_tuple(frontmatter.get("tools")) or None,
        disallowed_tools=_as_tuple(frontmatter.get("disallowedTools")),
        skills=_as_tuple(frontmatter.get("skills")),
        mcp_servers=tuple(frontmatter.get("mcpServers") or ()) if isinstance(frontmatter.get("mcpServers"), list) else (),
        hooks=frontmatter.get("hooks") if isinstance(frontmatter.get("hooks"), dict) else None,
        color=str(frontmatter["color"]) if frontmatter.get("color") is not None else None,
        model=str(frontmatter["model"]) if frontmatter.get("model") is not None else None,
        effort=frontmatter.get("effort"),
        permission_mode=str(frontmatter["permissionMode"]) if frontmatter.get("permissionMode") is not None else None,
        max_turns=int(max_turns) if isinstance(max_turns, int) or (isinstance(max_turns, str) and max_turns.isdigit()) else None,
        source=source,
        base_dir=str(path.parent),
        background=_bool(frontmatter.get("background"), False),
        initial_prompt=str(frontmatter["initialPrompt"]) if frontmatter.get("initialPrompt") is not None else None,
        memory=str(frontmatter["memory"]) if frontmatter.get("memory") is not None else None,
        isolation=str(frontmatter["isolation"]) if frontmatter.get("isolation") is not None else None,
        filename=path.stem,
        extra_frontmatter={key: value for key, value in frontmatter.items() if key not in known},
    )


def _agent_search_dirs(config: KernelConfig) -> list[tuple[Path, str]]:
    """完成 ``_agent_search_dirs`` 对应的subagent 定义与执行内部步骤。"""
    dirs = [
        (Path(config.config_home) / "agents", "userSettings"),
        (Path(config.cwd) / ".claude" / "agents", "projectSettings"),
        *[(Path(path), "projectSettings") for path in config.agent_paths],
    ]
    result: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for directory, source in dirs:
        path = directory.expanduser()
        key = path.resolve() if path.exists() else path.absolute()
        if key not in seen:
            result.append((path, source))
            seen.add(key)
    return result


def get_active_agents_from_list(agents: Iterable[AgentDefinition]) -> list[AgentDefinition]:
    """获取active agent 集合 from list，供subagent 定义与执行流程使用。"""
    by_type: dict[str, AgentDefinition] = {}
    for agent in agents:
        by_type[agent.agent_type] = agent
    return list(by_type.values())


def load_agents(config: KernelConfig) -> list[AgentDefinition]:
    """按优先级加载 agent，并让同名项目定义覆盖内置定义。"""
    # 按 built-in -> 文件 -> 显式 config 加载，后出现的同名定义覆盖前者。
    agents: list[AgentDefinition] = built_in_agents(config)
    for directory, source in _agent_search_dirs(config):
        if not directory.exists() or not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.md"), key=lambda item: item.name):
            agent = agent_from_markdown(path, source=source)
            if agent is not None:
                agents.append(agent)
    agents.extend(agent_from_config(agent) for agent in config.agents)
    return get_active_agents_from_list(agents)


def _tool_base_name(spec: str) -> str:
    """完成 ``_tool_base_name`` 对应的subagent 定义与执行内部步骤。"""
    spec = spec.strip()
    if "(" in spec and spec.endswith(")"):
        return spec.split("(", 1)[0]
    return spec


def resolve_agent_tools(agent: AgentDefinition, available_tools: list[Tool], *, is_async: bool = False) -> list[Tool]:
    """应用 allowlist/disallowed_tools，并为后台 agent 排除不安全工具。"""
    # 子 agent 永远不能递归调用 Agent/Task；fork 有独立受控入口。
    disallowed = {_tool_base_name(spec) for spec in agent.disallowed_tools}
    disallowed.update({AGENT_TOOL_NAME, LEGACY_AGENT_TOOL_NAME})
    filtered = [tool for tool in available_tools if tool.name not in disallowed and LEGACY_AGENT_TOOL_NAME not in getattr(tool, "aliases", ())]
    if is_async:
        # 后台 agent 没有主线程交互能力，只保留明确允许的非阻塞工具。
        filtered = [
            tool
            for tool in filtered
            if tool.name.startswith("mcp__") or tool.name in ASYNC_AGENT_ALLOWED_TOOLS
        ]
    if agent.tools is None or agent.tools == ("*",):
        return filtered
    allowed = {_tool_base_name(spec) for spec in agent.tools}
    return [tool for tool in filtered if tool.name in allowed or any(alias in allowed for alias in getattr(tool, "aliases", ()))]


def format_agent_line(agent: AgentDefinition) -> str:
    """格式化agent 行，供subagent 定义与执行流程使用。"""
    if agent.tools and agent.disallowed_tools:
        effective = [tool for tool in agent.tools if tool not in set(agent.disallowed_tools)]
        tools_desc = ", ".join(effective) if effective else "None"
    elif agent.tools:
        tools_desc = ", ".join(agent.tools)
    elif agent.disallowed_tools:
        tools_desc = f"All tools except {', '.join(agent.disallowed_tools)}"
    else:
        tools_desc = "All tools"
    return f"- {agent.agent_type}: {agent.when_to_use} (Tools: {tools_desc})"


def get_agent_tool_prompt(agents: Iterable[AgentDefinition], *, fork_enabled: bool = False) -> str:
    """获取agent 工具 提示词，供subagent 定义与执行流程使用。"""
    agent_list = "\n".join(format_agent_line(agent) for agent in agents)
    if fork_enabled:
        return f"""Launch a new agent to handle complex, multi-step tasks autonomously.

The Agent tool launches specialized agents (subprocesses) that autonomously handle complex tasks. Each agent type has specific capabilities and tools available to it.

Available agent types and the tools they have access to:
{agent_list}

When using the Agent tool, specify a subagent_type to use a specialized agent, or omit it to fork yourself — a fork inherits your full conversation context.

Usage notes:
- Always include a short description (3-5 words) summarizing what the agent will do
- Launch multiple agents concurrently whenever possible, to maximize performance; to do that, use a single message with multiple tool uses
- When the agent is done, it will return a single message back to you. The result returned by the agent is not visible to the user. To show the user the result, you should send a text message back to the user with a concise summary of the result.
- To continue a previously spawned agent, use SendMessage with the agent's ID or name as the `to` field. The agent resumes with its full context preserved. Each fresh Agent invocation with a subagent_type starts without context — provide a complete task description.
- The agent's outputs should generally be trusted
- Clearly tell the agent whether you expect it to write code or just to do research (search, file reads, web fetches, etc.)
- If the agent description mentions that it should be used proactively, then you should try your best to use it without the user having to ask for it first. Use your judgement.
- If the user specifies that they want you to run agents "in parallel", you MUST send a single message with multiple Agent tool use content blocks. For example, if you need to launch both a build-validator agent and a test-runner agent in parallel, send a single message with both tool calls.

## When to fork

Fork yourself (omit `subagent_type`) when the intermediate tool output isn't worth keeping in your context. The criterion is qualitative — "will I need this output again" — not task size.
- **Research**: fork open-ended questions. If research can be broken into independent questions, launch parallel forks in one message. A fork beats a fresh subagent for this — it inherits context and shares your cache.
- **Implementation**: prefer to fork implementation work that requires more than a couple of edits. Do research before jumping to implementation.

Forks are cheap because they share your prompt cache. Don't set `model` on a fork — a different model can't reuse the parent's cache. Pass a short `name` (one or two words, lowercase) so the user can see the fork in background task tracking and steer it mid-run.

**Don't peek.** The tool result includes an `output_file` path — do not Read or tail it unless the user explicitly asks for a progress check. You get a completion notification; trust it. Reading the transcript mid-flight pulls the fork's tool noise into your context, which defeats the point of forking.

**Don't race.** After launching, you know nothing about what the fork found. Never fabricate or predict fork results in any format — not as prose, summary, or structured output. The notification arrives as a user-role message in a later turn; it is never something you write yourself. If the user asks a follow-up before the notification lands, tell them the fork is still running — give status, not a guess.

**Writing a fork prompt.** Since the fork inherits your context, the prompt is a *directive* — what to do, not what the situation is. Be specific about scope: what's in, what's out, what another agent is handling. Don't re-explain background.

## Writing the prompt

When spawning a fresh agent (with a `subagent_type`), it starts with zero context. Brief the agent like a smart colleague who just walked into the room — it hasn't seen this conversation, doesn't know what you've tried, doesn't understand why this task matters.
- Explain what you're trying to accomplish and why.
- Describe what you've already learned or ruled out.
- Give enough context about the surrounding problem that the agent can make judgment calls rather than just following a narrow instruction.
- If you need a short response, say so ("report in under 200 words").
- Lookups: hand over the exact command. Investigations: hand over the question — prescribed steps become dead weight when the premise is wrong.

For fresh agents, terse command-style prompts produce shallow, generic work.

**Never delegate understanding.** Don't write "based on your findings, fix the bug" or "based on the research, implement it." Those phrases push synthesis onto the agent instead of doing it yourself. Write prompts that prove you understood: include file paths, line numbers, what specifically to change.

Example usage:

<example>
user: "What's left on this branch before we can ship?"
assistant: <thinking>Forking this — it's a survey question. I want the punch list, not the git output in my context.</thinking>
Agent({{
  name: "ship-audit",
  description: "Branch ship-readiness audit",
  prompt: "Audit what's left before this branch can ship. Check: uncommitted changes, commits ahead of main, whether tests exist, whether the GrowthBook gate is wired up, whether CI-relevant files changed. Report a punch list — done vs. missing. Under 200 words."
}})
assistant: Ship-readiness audit running.
<commentary>
Turn ends here. The coordinator knows nothing about the findings yet. What follows is a SEPARATE turn — the notification arrives from outside, as a user-role message. It is not something the coordinator writes.
</commentary>
[later turn — notification arrives as user message]
assistant: Audit's back. Three blockers: no tests for the new prompt path, GrowthBook gate wired but not in build_flags.yaml, and one uncommitted file.
</example>

<example>
user: "so is the gate wired up or not"
<commentary>
User asks mid-wait. The audit fork was launched to answer exactly this, and it hasn't returned. The coordinator does not have this answer. Give status, not a fabricated result.
</commentary>
assistant: Still waiting on the audit — that's one of the things it's checking. Should land shortly.
</example>

<example>
user: "Can you get a second opinion on whether this migration is safe?"
assistant: <thinking>I'll ask the code-reviewer agent — it won't see my analysis, so it can give an independent read.</thinking>
<commentary>
A subagent_type is specified, so the agent starts fresh. It needs full context in the prompt. The briefing explains what to assess and why.
</commentary>
Agent({{
  name: "migration-review",
  description: "Independent migration review",
  subagent_type: "code-reviewer",
  prompt: "Review migration 0042_user_schema.sql for safety. Context: we're adding a NOT NULL column to a 50M-row table. Existing rows get a backfill default. I want a second opinion on whether the backfill approach is safe under concurrent writes — I've checked locking behavior but want independent verification. Report: is this safe, and if not, what specifically breaks?"
}})
</example>
"""
    return f"""Launch a new agent to handle complex, multi-step tasks autonomously.

The Agent tool launches specialized agents (subprocesses) that autonomously handle complex tasks. Each agent type has specific capabilities and tools available to it.

Available agent types and the tools they have access to:
{agent_list}

When using the Agent tool, specify a subagent_type parameter to select which agent type to use. If omitted, the general-purpose agent is used.

When NOT to use the Agent tool:
- If you want to read a specific file path, use the Read tool or the Glob tool instead of the Agent tool, to find the match more quickly
- If you are searching for a specific class definition like "class Foo", use the Glob tool instead, to find the match more quickly
- If you are searching for code within a specific file or set of 2-3 files, use the Read tool instead of the Agent tool, to find the match more quickly
- Other tasks that are not related to the agent descriptions above

Usage notes:
- Always include a short description (3-5 words) summarizing what the agent will do
- Launch multiple agents concurrently whenever possible, to maximize performance; to do that, use a single message with multiple tool uses
- When the agent is done, it will return a single message back to you. The result returned by the agent is not visible to the user. To show the user the result, you should send a text message back to the user with a concise summary of the result.
- You can optionally run agents in the background using the run_in_background parameter. When an agent runs in the background, you will be automatically notified when it completes -- do NOT sleep, poll, or proactively check on its progress. Continue with other work or respond to the user instead.
- To continue a previously spawned agent, use SendMessage with the agent's ID or name as the `to` field. The agent resumes with its full context preserved. Each Agent invocation starts fresh -- provide a complete task description.
- The agent's outputs should generally be trusted
- Clearly tell the agent whether you expect it to write code or just to do research (search, file reads, web fetches, etc.), since it is not aware of the user's intent
- If the agent description mentions that it should be used proactively, then you should try your best to use it without the user having to ask for it first. Use your judgement.
- If the user specifies that they want you to run agents "in parallel", you MUST send a single message with multiple Agent tool use content blocks. For example, if you need to launch both a build-validator agent and a test-runner agent in parallel, send a single message with both tool calls.

## Writing the prompt

Brief the agent like a smart colleague who just walked into the room -- it hasn't seen this conversation, doesn't know what you've tried, doesn't understand why this task matters.
- Explain what you're trying to accomplish and why.
- Describe what you've already learned or ruled out.
- Give enough context about the surrounding problem that the agent can make judgment calls rather than just following a narrow instruction.
- If you need a short response, say so ("report in under 200 words").
- Lookups: hand over the exact command. Investigations: hand over the question -- prescribed steps become dead weight when the premise is wrong.

Terse command-style prompts produce shallow, generic work.

**Never delegate understanding.** Don't write "based on your findings, fix the bug" or "based on the research, implement it." Those phrases push synthesis onto the agent instead of doing it yourself. Write prompts that prove you understood: include file paths, line numbers, what specifically to change.
"""


def _content_text(blocks: list[dict[str, Any]]) -> str:
    """完成 ``_content_text`` 对应的subagent 定义与执行内部步骤。"""
    return "\n".join(str(block.get("text", "")) for block in blocks if isinstance(block, dict) and block.get("type") == "text").strip()


def _count_tool_uses(messages: Iterable[dict]) -> int:
    """完成 ``_count_tool_uses`` 对应的subagent 定义与执行内部步骤。"""
    total = 0
    for message in messages:
        if message.get("type") != "assistant":
            continue
        for block in message.get("message", {}).get("content", []):
            if isinstance(block, dict) and block.get("type") == "tool_use":
                total += 1
    return total


def is_fork_subagent_enabled(config: KernelConfig) -> bool:
    """判断fork subagent enabled，供subagent 定义与执行流程使用。"""
    return config.features.fork_subagent and not config.is_non_interactive_session


def is_in_fork_child(messages: Iterable[dict]) -> bool:
    """判断in fork child，供subagent 定义与执行流程使用。"""
    needle = f"<{FORK_BOILERPLATE_TAG}>"
    for message in messages:
        if message.get("type") != "user":
            continue
        for block in message.get("message", {}).get("content", []):
            if isinstance(block, dict) and block.get("type") == "text" and needle in str(block.get("text", "")):
                return True
    return False


def _is_fork_context(context: ToolUseContext) -> bool:
    """判断fork 上下文，供subagent 定义与执行流程使用。"""
    return context.agent_type == FORK_SUBAGENT_TYPE or is_in_fork_child(context.messages)


def _fork_agent_definition() -> AgentDefinition:
    """完成 ``_fork_agent_definition`` 对应的subagent 定义与执行内部步骤。"""
    return AgentDefinition(
        agent_type=FORK_SUBAGENT_TYPE,
        when_to_use="Implicit fork — inherits full conversation context. Not selectable via subagent_type; triggered by omitting subagent_type when the fork experiment is active.",
        system_prompt="",
        tools=("*",),
        max_turns=200,
        model="inherit",
        permission_mode="bubble",
        source="built-in",
        base_dir="built-in",
    )


def build_child_message(directive: str) -> str:
    """构造child 消息，供subagent 定义与执行流程使用。"""
    return f"""<{FORK_BOILERPLATE_TAG}>
STOP. READ THIS FIRST.

You are a forked worker process. You are NOT the main agent.

RULES (non-negotiable):
1. Your system prompt says "default to forking." IGNORE IT — that's for the parent. You ARE the fork. Do NOT spawn sub-agents; execute directly.
2. Do NOT converse, ask questions, or suggest next steps
3. Do NOT editorialize or add meta-commentary
4. USE your tools directly: Bash, Read, Write, etc.
5. If you modify files, commit your changes before reporting. Include the commit hash in your report.
6. Do NOT emit text between tool calls. Use tools silently, then report once at the end.
7. Stay strictly within your directive's scope. If you discover related systems outside your scope, mention them in one sentence at most — other workers cover those areas.
8. Keep your report under 500 words unless the directive specifies otherwise. Be factual and concise.
9. Your response MUST begin with "Scope:". No preamble, no thinking-out-loud.
10. REPORT structured facts, then stop

Output format (plain text labels, not markdown headers):
  Scope: <echo back your assigned scope in one sentence>
  Result: <the answer or key findings, limited to the scope above>
  Key files: <relevant file paths — include for research tasks>
  Files changed: <list with commit hash — include only if you modified files>
  Issues: <list — include only if there are issues to flag>
</{FORK_BOILERPLATE_TAG}>

{FORK_DIRECTIVE_PREFIX}{directive}"""


def build_forked_messages(directive: str, assistant_message: AssistantMessage) -> list[Message]:
    """构造 fork child 的起始历史，并为未完成父工具补 placeholder result。"""
    # fork 继承当前 assistant turn；其中可能同时含有多个尚未执行的 tool_use。
    tool_use_blocks = [
        block
        for block in assistant_message.get("message", {}).get("content", [])
        if isinstance(block, dict) and block.get("type") == "tool_use"
    ]
    if not tool_use_blocks:
        return [create_user_message(build_child_message(directive))]
    # 使用新外层 uuid，避免 child transcript 与 parent 消息主键冲突。
    full_assistant_message: AssistantMessage = {
        **assistant_message,
        "uuid": str(uuid4()),
        "message": {
            **assistant_message["message"],
            "content": list(assistant_message["message"]["content"]),
        },
    }
    # 每个 inherited tool_use 都补 placeholder，保证 child 第一轮 API pairing 合法。
    tool_result_blocks = [
        {
            "type": "tool_result",
            "tool_use_id": block["id"],
            "content": [{"type": "text", "text": FORK_PLACEHOLDER_RESULT}],
        }
        for block in tool_use_blocks
    ]
    return [
        full_assistant_message,
        create_user_message(
            [
                *tool_result_blocks,
                {"type": "text", "text": build_child_message(directive)},
            ]
        ),
    ]


def _filter_incomplete_tool_calls(messages: Iterable[Message]) -> list[Message]:
    """完成 ``_filter_incomplete_tool_calls`` 对应的subagent 定义与执行内部步骤。"""
    tool_use_ids_with_results: set[str] = set()
    for message in messages:
        if message.get("type") != "user":
            continue
        for block in message.get("message", {}).get("content", []):
            if isinstance(block, dict) and block.get("type") == "tool_result" and block.get("tool_use_id"):
                tool_use_ids_with_results.add(str(block["tool_use_id"]))
    filtered: list[Message] = []
    for message in messages:
        if message.get("type") == "assistant":
            blocks = message.get("message", {}).get("content", [])
            has_incomplete = any(
                isinstance(block, dict)
                and block.get("type") == "tool_use"
                and block.get("id")
                and str(block["id"]) not in tool_use_ids_with_results
                for block in blocks
            )
            if has_incomplete:
                continue
        filtered.append(message)
    return filtered


class SidechainSessionStore:
    """把 subagent 消息写入独立 JSONL，避免污染主 session 链。"""
    def __init__(self, config: KernelConfig, session_id: str | None, agent_id: str):
        """初始化实例内部状态和后续处理所需的缓存。"""
        self.config = config
        self.session_id = session_id or ""
        self.agent_id = agent_id
        self._last_uuid: str | None = None

    @property
    def path(self) -> Path:
        """返回当前 sidechain session 的持久化文件路径。"""
        return self.config.config_home / "projects" / sanitize_path(self.config.cwd) / "subagents" / f"{self.agent_id}.jsonl"

    def record(self, messages: list[dict]) -> None:
        """把尚未记录的消息追加写入当前 sidechain transcript。"""
        entries = []
        parent_uuid = self._last_uuid
        for message in messages:
            if message.get("type") not in {"user", "assistant", "system", "attachment"} or not message.get("uuid"):
                continue
            entries.append(
                {
                    **message,
                    "parentUuid": parent_uuid,
                    "isSidechain": True,
                    "agentId": self.agent_id,
                    "sessionId": self.session_id,
                    "cwd": str(self.config.cwd),
                    "timestamp": message.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                    "version": "0.1.0-python-port",
                }
            )
            parent_uuid = message["uuid"]
        if not entries:
            return
        self._last_uuid = parent_uuid
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            for entry in entries:
                fh.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.path.chmod(0o600)


async def run_subagent(
    *,
    agent: AgentDefinition,
    prompt: str,
    description: str,
    parent_context: ToolUseContext,
    model_provider: Any,
    model: str,
    model_override: str | None = None,
    is_async: bool = False,
    on_progress=None,
    agent_id: str | None = None,
    prompt_messages: list[Message] | None = None,
    fork_context_messages: list[Message] | None = None,
    override_system_prompt: list[str] | None = None,
    use_exact_tools: bool = False,
) -> dict[str, Any]:
    """建立隔离 ToolUseContext，运行嵌套 query，并汇总文本与 usage 形状。"""
    start = time.monotonic()
    resolved_agent_id = agent_id or f"agent-{uuid4().hex[:12]}"
    if prompt_messages is not None:
        # fork 路径携带父上下文，但先剔除无法配对的半截工具调用。
        initial_messages = [
            *_filter_incomplete_tool_calls(fork_context_messages or []),
            *prompt_messages,
        ]
    else:
        initial_messages = []
        if agent.initial_prompt:
            initial_messages.append(create_user_message(agent.initial_prompt, is_meta=True))
        for skill_name in agent.skills:
            skill = next((item for item in load_skills(parent_context.config) if item.name == skill_name), None)
            if skill is not None:
                initial_messages.append(create_user_message(skill.render_prompt("", parent_context.session_id), is_meta=True))
        initial_messages.append(create_user_message(prompt))
    # SubagentStart hook 的消息属于 child 初始上下文，因此在 query 前写入 sidechain。
    async for hook_result in run_hook_event(
        parent_context,
        subagent_start_hook_input(
            context=parent_context,
            agent_id=resolved_agent_id,
            agent_type=agent.agent_type,
            description=description,
        ),
    ):
        if hook_result.message:
            initial_messages.append(hook_result.message)
        if hook_result.system_message:
            initial_messages.append(hook_result.system_message)
        if hook_result.additional_context:
            contexts = hook_result.additional_context if isinstance(hook_result.additional_context, list) else [hook_result.additional_context]
            initial_messages.append(
                create_attachment_message(
                    "\n".join(str(item) for item in contexts),
                    attachment_type="hook_additional_context",
                    metadata={
                        "hookName": "SubagentStart",
                        "hookEvent": "SubagentStart",
                        "agentId": resolved_agent_id,
                        "agentType": agent.agent_type,
                    },
                )
            )
    sidechain = SidechainSessionStore(parent_context.config, parent_context.session_id, resolved_agent_id)
    sidechain.record(initial_messages)
    # fork 可精确继承父工具；普通角色 agent 应用自身 allow/disallow 规则。
    child_tools = list(parent_context.tools) if use_exact_tools else resolve_agent_tools(agent, parent_context.tools, is_async=is_async)
    parent_permission = parent_context.get_app_state().tool_permission_context
    requested_mode = agent.permission_mode if agent.permission_mode in {"ask", "bypass"} else parent_permission.mode
    # 只有 fork/精确继承模式复制 Read 快照，普通 subagent 从干净文件视图开始。
    child_read_state = (
        {path: ReadFileStateEntry(**entry.__dict__) for path, entry in parent_context.read_file_state.items()}
        if use_exact_tools or agent.agent_type == FORK_SUBAGENT_TYPE
        else {}
    )
    child_context = ToolUseContext(
        config=parent_context.config,
        tools=child_tools,
        app_state=AppState(
            ToolPermissionContext(
                mode=requested_mode,
                additional_working_directories=dict(parent_permission.additional_working_directories),
            )
        ),
        read_file_state=child_read_state,
        permission_callback=parent_context.permission_callback,
        model_provider=model_provider,
        web_fetch_model=model,
        web_search_handler=parent_context.web_search_handler,
        web_fetch_handler=parent_context.web_fetch_handler,
        web_fetch_apply_handler=parent_context.web_fetch_apply_handler,
        hook_registry=parent_context.hook_registry,
        hook_runner=parent_context.hook_runner,
        session_id=parent_context.session_id,
        transcript_path=str(sidechain.path),
        messages=list(initial_messages),
        rendered_system_prompt=list(override_system_prompt or []),
        user_context=dict(parent_context.user_context),
        system_context=dict(parent_context.system_context),
        agent_id=resolved_agent_id,
        agent_type=agent.agent_type,
    )
    # 调用参数优先于定义；inherit 明确使用父模型。
    agent_model = str(model_override or (model if agent.model in {None, "inherit"} else agent.model))
    if override_system_prompt is not None:
        system_prompt = list(override_system_prompt)
    else:
        system_prompt = [
            agent.system_prompt,
            compute_simple_env_info(parent_context.config, agent_model),
        ]
        if agent.critical_system_reminder:
            system_prompt.append(agent.critical_system_reminder)
    user_context = dict(parent_context.user_context)
    if agent.omit_claude_md:
        # Explore 等只需代码事实的 agent 可避免重复项目指令和过期状态。
        user_context.pop("claudeMd", None)
    system_context = {} if override_system_prompt is not None else dict(parent_context.system_context)
    if agent.agent_type in {"Explore", "Plan"}:
        system_context.pop("gitStatus", None)
    agent_messages: list[dict] = []
    async for event in query(
        QueryParams(
            messages=initial_messages,
            system_prompt=system_prompt,
            user_context=user_context,
            system_context=system_context,
            tool_use_context=child_context,
            model_provider=model_provider,
            query_source=f"agent:{agent.source}:{agent.agent_type}",
            max_turns=agent.max_turns,
            model=agent_model,
            context_compaction=parent_context.config.context_compaction,
            transcript_path=str(sidechain.path),
        )
    ):
        if event.get("type") in {"assistant", "user", "system", "attachment"}:
            agent_messages.append(event)
            sidechain.record([event])
        if event.get("type") == "tool_progress" and on_progress:
            on_progress({"type": "agent_progress", "agentId": resolved_agent_id, "message": event})
    # ToolResult 返回给父 agent 时只提取 child 最后一段非空 assistant 文本。
    assistant_messages = [message for message in agent_messages if message.get("type") == "assistant"]
    text_blocks: list[dict[str, str]] = []
    for message in reversed(assistant_messages):
        text = _content_text(message.get("message", {}).get("content", []))
        if text:
            text_blocks = [{"type": "text", "text": text}]
            break
    if not text_blocks:
        text_blocks = [{"type": "text", "text": "(Subagent completed but returned no output.)"}]
    result = {
        "status": "completed",
        "prompt": prompt,
        "agentId": resolved_agent_id,
        "agentType": agent.agent_type,
        "content": text_blocks,
        "totalToolUseCount": _count_tool_uses(agent_messages),
        "totalDurationMs": int((time.monotonic() - start) * 1000),
        "totalTokens": 0,
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": None,
            "cache_read_input_tokens": None,
            "server_tool_use": None,
            "service_tier": None,
            "cache_creation": None,
        },
        "transcriptPath": str(sidechain.path),
    }
    final_result_text = "\n".join(block["text"] for block in text_blocks)
    async for hook_result in run_hook_event(
        parent_context,
        subagent_stop_hook_input(
            context=parent_context,
            agent_id=resolved_agent_id,
            agent_type=agent.agent_type,
            status="completed",
            result=final_result_text,
        ),
    ):
        if hook_result.additional_context and on_progress:
            on_progress({"type": "agent_progress", "agentId": resolved_agent_id, "message": hook_result.additional_context})
    return result


def _agent_output_path(config: KernelConfig, agent_id: str) -> Path:
    """完成 ``_agent_output_path`` 对应的subagent 定义与执行内部步骤。"""
    output_dir = config.workspace_runtime.agent_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{agent_id}.log"


class AgentTool(Tool):
    """模型侧 Task/Agent 工具；负责参数校验、权限和同步/后台调度。"""
    name = AGENT_TOOL_NAME
    aliases = (LEGACY_AGENT_TOOL_NAME,)
    search_hint = "delegate work to a subagent"
    max_result_size_chars = 100_000
    input_schema = {
        "description": str,
        "prompt": str,
        "subagent_type": str,
        "model": str,
        "run_in_background": bool,
        "name": str,
    }
    required_fields = ("description", "prompt")

    def __init__(self, agents: Iterable[AgentDefinition], *, config: KernelConfig | None = None):
        """初始化实例内部状态和后续处理所需的缓存。"""
        self.agents = {agent.agent_type: agent for agent in agents}
        self.config = config

    async def description(self, input: dict | None = None) -> str:
        """返回提供给模型和调用方展示的工具简短说明。"""
        return "Launch a new agent"

    async def prompt(self) -> str:
        """返回该工具按源码保留的模型使用提示词。"""
        return get_agent_tool_prompt(
            self.agents.values(),
            fork_enabled=is_fork_subagent_enabled(self.config) if self.config is not None else False,
        )

    def is_read_only(self, input: dict) -> bool:
        """判断当前输入是否只读取外部状态而不产生修改。"""
        return True

    def is_concurrency_safe(self, input: dict) -> bool:
        """判断当前输入是否可以与相邻安全工具并发执行。"""
        return True

    def user_facing_name(self, input: dict | None = None) -> str:
        """根据当前输入返回适合界面展示的工具名称。"""
        subagent_type = input.get("subagent_type") if input else None
        return f"Agent: {subagent_type or 'general-purpose'}"

    async def validate_input(self, input: dict, context: ToolUseContext) -> ValidationResult:
        """执行依赖上下文的业务输入校验。"""
        if not isinstance(input.get("description"), str) or not input["description"].strip():
            return ValidationResult(False, "description is required.", 1)
        if not isinstance(input.get("prompt"), str) or not input["prompt"].strip():
            return ValidationResult(False, "prompt is required.", 2)
        fork_enabled = is_fork_subagent_enabled(context.config)
        agent_type = input.get("subagent_type")
        if not agent_type and fork_enabled:
            if _is_fork_context(context):
                return ValidationResult(False, "Fork is not available inside a forked worker. Complete your task directly using your tools.", 3)
        else:
            agent_type = agent_type or "general-purpose"
            if agent_type not in self.agents:
                return ValidationResult(False, f"Agent type '{agent_type}' not found. Available agents: {', '.join(self.agents)}", 3)
        unsupported_fields = (
            ("team_name", "Agent Teams are not implemented in this Python kernel. Omit team_name to spawn a subagent."),
            ("mode", "Agent permission mode override is not implemented in this Python kernel. Omit mode to spawn a subagent."),
            ("isolation", "Agent isolation overrides are not implemented in this Python kernel. Omit isolation to spawn a subagent."),
            ("cwd", "Agent cwd override is not implemented in this Python kernel. Omit cwd to spawn a subagent."),
        )
        for offset, (field_name, message) in enumerate(unsupported_fields, start=4):
            if field_name in input and input[field_name] is not None:
                return ValidationResult(False, message, offset)
        return ValidationResult(True)

    async def check_permissions(self, input: dict, context: ToolUseContext) -> PermissionDecision:
        """返回该工具针对当前输入的初步权限建议。"""
        return PermissionDecision.allow(updated_input=input)

    async def call(self, args: dict, context: ToolUseContext, can_use_tool, parent_message: AssistantMessage, on_progress=None) -> ToolResult:
        """执行工具核心逻辑，并返回标准 ToolResult。"""
        fork_enabled = is_fork_subagent_enabled(context.config)
        is_fork_path = not args.get("subagent_type") and fork_enabled
        if is_fork_path:
            if _is_fork_context(context):
                raise RuntimeError("Fork is not available inside a forked worker. Complete your task directly using your tools.")
            agent = _fork_agent_definition()
        else:
            agent_type = args.get("subagent_type") or "general-purpose"
            agent = self.agents[agent_type]
        agent_type = agent.agent_type
        agent_id = f"agent-{uuid4().hex[:12]}"
        should_run_async = bool(args.get("run_in_background") or agent.background or fork_enabled)
        prompt_messages = build_forked_messages(args["prompt"], parent_message) if is_fork_path else None
        fork_context_messages = list(context.messages) if is_fork_path else None
        override_system_prompt = list(context.rendered_system_prompt) if is_fork_path else None
        use_exact_tools = is_fork_path
        requested_model = (context.web_fetch_model or "fake-model") if is_fork_path else (args.get("model") or context.web_fetch_model or "fake-model")
        if on_progress:
            on_progress({"type": "agent_progress", "agentId": agent_id, "message": f"Launching agent: {agent_type}"})
        if should_run_async:
            output_path = _agent_output_path(context.config, agent_id)

            async def _run_background() -> None:
                """在后台执行 subagent，并持久化 sidechain 结果和状态。"""
                try:
                    result = await run_subagent(
                        agent=agent,
                        prompt=args["prompt"],
                        description=args["description"],
                        parent_context=context,
                        model_provider=context.model_provider,
                        model=requested_model,
                        model_override=None if is_fork_path else args.get("model"),
                        is_async=True,
                        on_progress=on_progress,
                        agent_id=agent_id,
                        prompt_messages=prompt_messages,
                        fork_context_messages=fork_context_messages,
                        override_system_prompt=override_system_prompt,
                        use_exact_tools=use_exact_tools,
                    )
                    final_text = "\n".join(block["text"] for block in result["content"])
                    output_path.write_text(final_text, encoding="utf-8")
                except Exception as exc:
                    output_path.write_text(f"Agent failed: {type(exc).__name__}: {exc}", encoding="utf-8")

            task = asyncio.create_task(_run_background())
            context.background_tasks[agent_id] = {
                "type": "agent",
                "task": task,
                "agentType": agent_type,
                "description": args["description"],
                "prompt": args["prompt"],
                "outputPath": str(output_path),
                **({"name": args["name"]} if args.get("name") else {}),
            }
            return ToolResult(
                {
                    "isAsync": True,
                    "status": "async_launched",
                    "agentId": agent_id,
                    "description": args["description"],
                    "prompt": args["prompt"],
                    "outputFile": str(output_path),
                    "canReadOutputFile": any(tool.name in {"Read", "Bash"} for tool in context.tools),
                    **({"name": args["name"]} if args.get("name") else {}),
                }
            )
        result = await run_subagent(
            agent=agent,
            prompt=args["prompt"],
            description=args["description"],
            parent_context=context,
            model_provider=context.model_provider,
            model=requested_model,
            model_override=args.get("model"),
            is_async=False,
            on_progress=on_progress,
            agent_id=agent_id,
            prompt_messages=prompt_messages,
            fork_context_messages=fork_context_messages,
            override_system_prompt=override_system_prompt,
            use_exact_tools=use_exact_tools,
        )
        return ToolResult(result)

    def map_tool_result_to_tool_result_block_param(self, content: dict, tool_use_id: str) -> ToolResultBlock:
        """把工具内部结果映射为 Anthropic tool_result block。"""
        if content.get("status") == "async_launched":
            prefix = f"Async agent launched successfully.\nagentId: {content['agentId']} (internal ID - do not mention to user. Use SendMessage with to: '{content['agentId']}' to continue this agent.)\nThe agent is working in the background. You will be notified automatically when it completes."
            if content.get("canReadOutputFile"):
                instructions = f"Do not duplicate this agent's work -- avoid working with the same files or topics it is using. Work on non-overlapping tasks, or briefly tell the user what you launched and end your response.\noutput_file: {content['outputFile']}\nIf asked, you can check progress before completion by using Read or Bash tail on the output file."
            else:
                instructions = "Briefly tell the user what you launched and end your response. Do not generate any other text -- agent results will arrive in a subsequent message."
            return {
                "tool_use_id": tool_use_id,
                "type": "tool_result",
                "content": [{"type": "text", "text": f"{prefix}\n{instructions}"}],
            }
        if content.get("status") == "completed":
            blocks = content.get("content") or [{"type": "text", "text": "(Subagent completed but returned no output.)"}]
            if content.get("agentType") in ONE_SHOT_BUILTIN_AGENT_TYPES:
                return {"tool_use_id": tool_use_id, "type": "tool_result", "content": blocks}
            trailer = {
                "type": "text",
                "text": f"agentId: {content['agentId']} (use SendMessage with to: '{content['agentId']}' to continue this agent)\n<usage>total_tokens: {content.get('totalTokens', 0)}\ntool_uses: {content.get('totalToolUseCount', 0)}\nduration_ms: {content.get('totalDurationMs', 0)}</usage>",
            }
            return {"tool_use_id": tool_use_id, "type": "tool_result", "content": [*blocks, trailer]}
        return {"tool_use_id": tool_use_id, "type": "tool_result", "content": str(content), "is_error": True}

    async def to_api_spec(self) -> dict[str, Any]:
        """构造发送给模型 API 的工具名称、说明和 JSON Schema。"""
        return {
            "name": self.name,
            "description": "\n\n".join([await self.description(None), await self.prompt()]),
            "input_schema": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "A short (3-5 word) description of the task"},
                    "prompt": {"type": "string", "description": "The task for the agent to perform"},
                    "subagent_type": {"type": "string", "description": "The type of specialized agent to use for this task"},
                    "model": {"type": "string", "enum": ["balanced", "frontier", "fast"], "description": "Optional model override for this agent."},
                    "run_in_background": {"type": "boolean", "description": "Set to true to run this agent in the background."},
                    "name": {"type": "string", "description": "Name for the spawned agent. Makes it addressable via SendMessage({to: name}) while running."},
                },
                "required": ["description", "prompt"],
                "additionalProperties": False,
            },
        }
