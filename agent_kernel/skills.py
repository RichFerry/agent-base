"""Skill 的文件发现、frontmatter 解析、预算化展示和调用协议。

Skill 是按需展开的指令包，不是 subagent。定义可来自 KernelConfig 或项目 skill 目录，
包含 description/content/allowed-tools/arguments/model/context/hooks/paths 等元数据。

PromptComposer 只通过 ``format_skills_within_budget`` 把名称和简述放入 system reminder，
避免所有正文占满上下文。模型调用 SkillTool 后，才由 ``render_prompt`` 展开参数与正文，
注册 skill hooks，并把结果作为新消息送入下一模型轮。``disable_model_invocation`` 可禁止
模型主动调用；有副作用的 skill 仍经过 ask/bypass 权限。

SkillTool 返回的是“上下文扩展”而非独立 agent 执行结果；需要隔离 loop、模型或工具集
时应使用 AgentTool。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from pathlib import Path
from typing import Any, Iterable

from .config import KernelConfig, SkillConfig
from .hooks import HookResult
from .messages import AssistantMessage, ToolResultBlock, create_user_message
from .permissions import PermissionDecision
from .tools.base import Tool, ToolResult, ToolUseContext, ValidationResult


COMMAND_NAME_TAG = "command-name"
COMMAND_ARGS_TAG = "command-args"
SKILL_TOOL_NAME = "Skill"

SKILL_BUDGET_CONTEXT_PERCENT = 0.01
CHARS_PER_TOKEN = 4
DEFAULT_CHAR_BUDGET = 8_000
MAX_LISTING_DESC_CHARS = 250

SKILL_TOOL_PROMPT = """Execute a skill within the main conversation

When users ask you to perform tasks, check if any of the available skills match. Skills provide specialized capabilities and domain knowledge.

When users reference a "slash command" or "/<something>" (e.g., "/commit", "/review-pr"), they are referring to a skill. Use this tool to invoke it.

How to invoke:
- Use this tool with the skill name and optional arguments
- Examples:
  - `skill: "pdf"` - invoke the pdf skill
  - `skill: "commit", args: "-m 'Fix bug'"` - invoke with arguments
  - `skill: "review-pr", args: "123"` - invoke with arguments
  - `skill: "ms-office-suite:pdf"` - invoke using fully qualified name

Important:
- Available skills are listed in system-reminder messages in the conversation
- When a skill matches the user's request, this is a BLOCKING REQUIREMENT: invoke the relevant Skill tool BEFORE generating any other response about the task
- NEVER mention a skill without actually calling this tool
- Do not invoke a skill that is already running
- Do not use this tool for built-in CLI commands (like /help, /clear, etc.)
- If you see a <command-name> tag in the current conversation turn, the skill has ALREADY been loaded - follow the instructions directly instead of calling this tool again
"""


def _normalize_skill_name(name: str) -> str:
    """规范化skill name，供skill 扩展流程使用。"""
    trimmed = name.strip()
    return trimmed[1:] if trimmed.startswith("/") else trimmed


def _strip_quotes(value: str) -> str:
    """移除quotes，供skill 扩展流程使用。"""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _parse_scalar(value: str) -> Any:
    """解析scalar，供skill 扩展流程使用。"""
    value = _strip_quotes(value.strip())
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
        return [_strip_quotes(part) for part in re.split(r"\s*,\s*", inner) if part.strip()]
    return value


def _parse_frontmatter(markdown: str) -> tuple[dict[str, Any], str]:
    """解析frontmatter，供skill 扩展流程使用。"""
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
    current_key: str | None = None
    for raw_line in frontmatter_text.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line.startswith((" ", "\t")) and current_key is not None:
            previous = data.get(current_key)
            data[current_key] = f"{previous}\n{line.strip()}" if previous else line.strip()
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        current_key = key.strip()
        data[current_key] = _parse_scalar(value)
    return data, markdown[body_start:]


def _as_tuple(value: Any) -> tuple[str, ...]:
    """完成 ``_as_tuple`` 对应的skill 扩展内部步骤。"""
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
    """完成 ``_bool`` 对应的skill 扩展内部步骤。"""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _extract_description(markdown: str, fallback: str) -> str:
    """提取description，供skill 扩展流程使用。"""
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or fallback
        return stripped[:MAX_LISTING_DESC_CHARS] or fallback
    return fallback


def _skill_search_dirs(config: KernelConfig) -> list[Path]:
    """完成 ``_skill_search_dirs`` 对应的skill 扩展内部步骤。"""
    discovery_mode = getattr(config, "skill_discovery_mode", "ambient")
    if discovery_mode == "explicit":
        dirs = [Path(path) for path in config.skill_paths]
    elif discovery_mode == "ambient":
        dirs = [
            Path(config.cwd) / ".claude" / "skills",
            Path(config.config_home) / "skills",
            *[Path(path) for path in config.skill_paths],
        ]
    else:
        raise ValueError("skill_discovery_mode must be 'ambient' or 'explicit'.")
    result: list[Path] = []
    seen: set[Path] = set()
    for directory in dirs:
        path = directory.expanduser()
        key = path.resolve() if path.exists() else path.absolute()
        if key not in seen:
            result.append(path)
            seen.add(key)
    return result


@dataclass(frozen=True)
class SkillDefinition:
    """一个已解析 skill 的正文、元数据、工具约束和 hook 配置。"""
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
    frontmatter_keys: tuple[str, ...] = ()
    extra_frontmatter: dict[str, Any] = field(default_factory=dict)

    def display_description(self) -> str:
        """返回适合 skill 索引展示的说明文本。"""
        desc = f"{self.description} - {self.when_to_use}" if self.when_to_use else self.description
        if len(desc) > MAX_LISTING_DESC_CHARS:
            return desc[: MAX_LISTING_DESC_CHARS - 1] + "..."
        return desc

    def has_only_safe_properties(self) -> bool:
        """判断 skill 是否只声明无副作用属性。"""
        if self.hooks:
            return False
        if self.allowed_tools:
            return False
        if self.extra_frontmatter:
            return False
        return True

    def as_sdk_dict(self) -> dict[str, Any]:
        """返回该定义面向 SDK 暴露的稳定字段视图。"""
        return {
            "name": self.name,
            "description": self.description,
            "userInvocable": self.user_invocable,
            "source": self.source,
            "loadedFrom": self.loaded_from,
        }

    def render_prompt(self, args: str | None, session_id: str | None) -> str:
        """把 skill 正文、参数和 session 信息展开为调用提示词。"""
        rendered = self.content
        rendered = rendered.replace("$ARGUMENTS", args or "")
        rendered = rendered.replace("${CLAUDE_SESSION_ID}", session_id or "")
        if self.base_dir is not None:
            rendered = rendered.replace("${CLAUDE_SKILL_DIR}", str(self.base_dir))
            rendered = f"Base directory for this skill: {self.base_dir}\n\n{rendered}"
        tags = [f"<{COMMAND_NAME_TAG}>{self.name}</{COMMAND_NAME_TAG}>"]
        if args:
            tags.append(f"<{COMMAND_ARGS_TAG}>{args}</{COMMAND_ARGS_TAG}>")
        return "\n".join([*tags, rendered])


def skill_from_config(config: SkillConfig) -> SkillDefinition:
    """完成 ``skill_from_config`` 对应的skill 扩展内部步骤。"""
    return SkillDefinition(
        name=config.name,
        description=config.description,
        content=config.content,
        when_to_use=config.when_to_use,
        allowed_tools=tuple(config.allowed_tools),
        argument_hint=config.argument_hint,
        argument_names=tuple(config.argument_names),
        version=config.version,
        model=config.model,
        disable_model_invocation=config.disable_model_invocation,
        user_invocable=config.user_invocable,
        source=config.source,
        loaded_from=config.loaded_from,
        base_dir=config.base_dir,
        context=config.context,
        hooks=config.hooks,
        paths=tuple(config.paths),
    )


def skill_from_markdown(path: Path, *, name: str | None = None, loaded_from: str = "skills", source: str = "skills") -> SkillDefinition | None:
    """完成 ``skill_from_markdown`` 对应的skill 扩展内部步骤。"""
    try:
        markdown = path.read_text(encoding="utf-8")
    except (FileNotFoundError, UnicodeDecodeError, OSError):
        return None
    frontmatter, body = _parse_frontmatter(markdown)
    resolved_name = str(frontmatter.get("name") or name or path.parent.name)
    display_name = _normalize_skill_name(resolved_name)
    known_keys = {
        "name",
        "description",
        "allowed-tools",
        "argument-hint",
        "arguments",
        "when_to_use",
        "version",
        "model",
        "disable-model-invocation",
        "user-invocable",
        "hooks",
        "context",
        "agent",
        "effort",
        "paths",
        "shell",
    }
    extra = {key: value for key, value in frontmatter.items() if key not in known_keys}
    return SkillDefinition(
        name=display_name,
        description=str(frontmatter.get("description") or _extract_description(body, display_name)),
        content=body.strip(),
        when_to_use=str(frontmatter["when_to_use"]) if frontmatter.get("when_to_use") is not None else None,
        allowed_tools=_as_tuple(frontmatter.get("allowed-tools")),
        argument_hint=str(frontmatter["argument-hint"]) if frontmatter.get("argument-hint") is not None else None,
        argument_names=_as_tuple(frontmatter.get("arguments")),
        version=str(frontmatter["version"]) if frontmatter.get("version") is not None else None,
        model=str(frontmatter["model"]) if frontmatter.get("model") not in {None, "inherit"} else None,
        disable_model_invocation=_bool(frontmatter.get("disable-model-invocation"), False),
        user_invocable=_bool(frontmatter.get("user-invocable"), True),
        source=source,
        loaded_from=loaded_from,
        base_dir=path.parent,
        context="fork" if frontmatter.get("context") == "fork" else "inline",
        hooks=frontmatter.get("hooks") if isinstance(frontmatter.get("hooks"), dict) else None,
        paths=_as_tuple(frontmatter.get("paths")),
        frontmatter_keys=tuple(frontmatter.keys()),
        extra_frontmatter=extra,
    )


def load_skills(config: KernelConfig) -> list[SkillDefinition]:
    """加载配置与目录 skill，同名项按后加载来源覆盖。"""
    # 显式 config 优先，目录中同名 skill 不再覆盖调用方传入定义。
    result: list[SkillDefinition] = [skill_from_config(skill) for skill in config.skills]
    seen = {skill.name for skill in result}
    for directory in _skill_search_dirs(config):
        if not directory.exists() or not directory.is_dir():
            continue
        for child in sorted(directory.iterdir(), key=lambda path: path.name):
            if not child.is_dir():
                continue
            skill = skill_from_markdown(child / "SKILL.md", name=child.name)
            if skill is None or skill.name in seen:
                continue
            result.append(skill)
            seen.add(skill.name)
    return result


def format_skills_within_budget(skills: Iterable[SkillDefinition], context_window_tokens: int | None = None) -> str:
    """只把 skill 索引压入 system prompt，不提前展开正文。"""
    # system reminder 只展示模型可主动调用的 skill；用户专用命令不暴露给模型。
    visible = [skill for skill in skills if skill.user_invocable and not skill.disable_model_invocation]
    if not visible:
        return ""
    budget = int((context_window_tokens or 200_000) * CHARS_PER_TOKEN * SKILL_BUDGET_CONTEXT_PERCENT) or DEFAULT_CHAR_BUDGET
    entries = [f"- {skill.name}: {skill.display_description()}" for skill in visible]
    full = "\n".join(entries)
    if len(full) <= budget:
        return full
    # 超预算时先降级为名称列表，最后才做硬字符截断。
    names_only = "\n".join(f"- {skill.name}" for skill in visible)
    if len(names_only) <= budget:
        return names_only
    return names_only[:budget].rstrip()


def get_skill_system_reminder(skills: Iterable[SkillDefinition], context_window_tokens: int | None = None) -> str | None:
    """获取skill system reminder，供skill 扩展流程使用。"""
    listing = format_skills_within_budget(skills, context_window_tokens=context_window_tokens)
    if not listing:
        return None
    return f"""# User-invocable skills
{listing}"""


def _register_skill_hooks(skill: SkillDefinition, context: ToolUseContext) -> None:
    """完成 ``_register_skill_hooks`` 对应的skill 扩展内部步骤。"""
    if not skill.hooks:
        return
    for event, handlers in skill.hooks.items():
        # frontmatter 允许单 handler 或列表，注册时统一展开并保留稳定名称。
        handler_list = handlers if isinstance(handlers, list) else [handlers]
        for index, handler in enumerate(handler_list):
            if callable(handler):
                context.hook_registry.register(
                    str(event),
                    handler,
                    name=f"Skill:{skill.name}:{event}:{index}",
                )
            elif isinstance(handler, HookResult):
                async def _return_hook_result(_hook_input, result=handler):
                    """把 skill 声明中的静态 hook 配置转换为 HookResult。"""
                    return result

                context.hook_registry.register(
                    str(event),
                    _return_hook_result,
                    name=f"Skill:{skill.name}:{event}:{index}",
                )


class SkillTool(Tool):
    """按名称展开 skill，并把内容作为新消息注入下一轮。"""
    name = SKILL_TOOL_NAME
    search_hint = "invoke a slash-command skill"
    max_result_size_chars = 100_000
    input_schema = {"skill": str, "args": str}
    required_fields = ("skill",)

    def __init__(self, skills: Iterable[SkillDefinition]):
        """初始化实例内部状态和后续处理所需的缓存。"""
        self.skills = {skill.name: skill for skill in skills}

    async def description(self, input: dict | None = None) -> str:
        """返回提供给模型和调用方展示的工具简短说明。"""
        if input and isinstance(input.get("skill"), str):
            return f"Execute skill: {input['skill']}"
        return "Execute a skill within the main conversation"

    async def prompt(self) -> str:
        """返回该工具按源码保留的模型使用提示词。"""
        return SKILL_TOOL_PROMPT

    def is_concurrency_safe(self, input: dict) -> bool:
        """判断当前输入是否可以与相邻安全工具并发执行。"""
        return False

    def user_facing_name(self, input: dict | None = None) -> str:
        """根据当前输入返回适合界面展示的工具名称。"""
        if input and isinstance(input.get("skill"), str):
            return f"Skill: {_normalize_skill_name(input['skill'])}"
        return self.name

    def _find_skill(self, name: str) -> SkillDefinition | None:
        """查找skill，供skill 扩展流程使用。"""
        normalized = _normalize_skill_name(name)
        return self.skills.get(normalized)

    async def validate_input(self, input: dict, context: ToolUseContext) -> ValidationResult:
        """执行依赖上下文的业务输入校验。"""
        raw_skill = input.get("skill")
        if not isinstance(raw_skill, str) or not raw_skill.strip():
            return ValidationResult(False, f"Invalid skill format: {raw_skill}", 1)
        command_name = _normalize_skill_name(raw_skill)
        skill = self._find_skill(command_name)
        if skill is None:
            return ValidationResult(False, f"Unknown skill: {command_name}", 2)
        if skill.disable_model_invocation:
            return ValidationResult(False, f"Skill {command_name} cannot be used with {SKILL_TOOL_NAME} tool due to disable-model-invocation", 4)
        return ValidationResult(True)

    async def check_permissions(self, input: dict, context: ToolUseContext) -> PermissionDecision:
        """返回该工具针对当前输入的初步权限建议。"""
        command_name = _normalize_skill_name(str(input.get("skill", "")))
        skill = self._find_skill(command_name)
        updated_input = {"skill": command_name, **({"args": input["args"]} if "args" in input else {})}
        # 纯指令 skill 可自动执行；声明工具/路径/hook 的 skill 需要 ask。
        if skill is not None and skill.has_only_safe_properties():
            return PermissionDecision.allow(updated_input=updated_input)
        return PermissionDecision.ask(f"Execute skill: {command_name}")

    async def call(self, args: dict, context: ToolUseContext, can_use_tool, parent_message: AssistantMessage, on_progress=None) -> ToolResult:
        """执行工具核心逻辑，并返回标准 ToolResult。"""
        command_name = _normalize_skill_name(args["skill"])
        skill = self._find_skill(command_name)
        if skill is None:
            raise ValueError(f"Unknown skill: {command_name}")
        if on_progress:
            on_progress({"type": "skill_progress", "message": f"Launching skill: {command_name}"})
        if not hasattr(context, "invoked_skills"):
            context.invoked_skills = {}
        context.invoked_skills[command_name] = {
            "name": command_name,
            "loadedFrom": skill.loaded_from,
            "source": skill.source,
        }
        # hook 在正文消息进入下一轮前注册，确保 skill 后续工具调用能触发。
        _register_skill_hooks(skill, context)
        if skill.context == "fork":
            return ToolResult(
                {
                    "success": False,
                    "commandName": command_name,
                    "status": "forked",
                    "agentId": None,
                    "result": "Forked skill execution is not implemented in this Python kernel.",
                }
            )
        # inline skill 不创建新 loop，而是追加 synthetic user prompt 驱动主 loop 继续。
        rendered = skill.render_prompt(args.get("args"), context.session_id)
        new_message = create_user_message(rendered, is_meta=True)
        new_message["sourceToolAssistantUUID"] = parent_message["uuid"]
        return ToolResult(
            {
                "success": True,
                "commandName": command_name,
                "allowedTools": list(skill.allowed_tools) or None,
                "model": skill.model,
                "status": "inline",
                "skillPrompt": rendered,
            },
            new_messages=[new_message],
        )

    def map_tool_result_to_tool_result_block_param(self, content: dict, tool_use_id: str) -> ToolResultBlock:
        """把工具内部结果映射为 Anthropic tool_result block。"""
        if content.get("status") == "forked":
            return {
                "tool_use_id": tool_use_id,
                "type": "tool_result",
                "content": f"Skill \"{content['commandName']}\" completed (forked execution).\n\nResult:\n{content.get('result', '')}",
                "is_error": not bool(content.get("success")),
            }
        payload = {key: value for key, value in content.items() if key != "skillPrompt" and value is not None}
        return {
            "tool_use_id": tool_use_id,
            "type": "tool_result",
            "content": f"Launching skill: {payload['commandName']}",
        }

    async def to_api_spec(self) -> dict[str, Any]:
        """构造发送给模型 API 的工具名称、说明和 JSON Schema。"""
        return {
            "name": self.name,
            "description": "\n\n".join([await self.description(None), await self.prompt()]),
            "input_schema": {
                "type": "object",
                "properties": {
                    "skill": {"type": "string", "description": 'The skill name. E.g., "commit", "review-pr", or "pdf"'},
                    "args": {"type": "string", "description": "Optional arguments for the skill"},
                },
                "required": ["skill"],
                "additionalProperties": False,
            },
        }
