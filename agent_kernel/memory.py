"""项目级长期 memory 的路径发现、内容加载和 prompt 包装。

Memory 与 transcript 不同：transcript 保存逐轮消息，memory 保存跨 session 可复用的
项目知识。目录键优先基于 git root，否则使用 cwd，经 sanitize 后落在
``<config_home>/projects/<project>/memory``。

``MEMORY.md`` 是入口索引，显式内容包装时会限制体积并包装成源码同形 system section；
memory 目录不存在时可按需创建，但读取不会凭空写内容。Kairos/daily log 使用日期路径，
可由 feature/config 开启；v0.1 默认 agent prompt 只加载 memory 路径和使用规则，不自动
内联入口文件内容。

本模块只做文件发现与文本包装，不负责从对话自动提取 memory，也不参与 compact。
PromptComposer 决定最终是否把返回内容放进 system prompt。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import KernelConfig
from .path_utils import find_git_root, sanitize_path

ENTRYPOINT_NAME = "MEMORY.md"
MAX_ENTRYPOINT_LINES = 200
MAX_ENTRYPOINT_BYTES = 25_000
AUTO_MEM_DISPLAY_NAME = "auto memory"
DIR_EXISTS_GUIDANCE = "This directory already exists — write to it directly with the Write tool (do not run mkdir or check for its existence)."

MEMORY_FRONTMATTER_EXAMPLE = [
    "```markdown",
    "---",
    "name: {{memory name}}",
    "description: {{one-line description — used to decide relevance in future conversations, so be specific}}",
    "type: {{user, feedback, project, reference}}",
    "---",
    "",
    "{{memory content — for feedback/project types, structure as: rule/fact, then **Why:** and **How to apply:** lines}}",
    "```",
]

TYPES_SECTION_INDIVIDUAL = [
    "## Types of memory",
    "",
    "There are several discrete types of memory that you can store in your memory system:",
    "",
    "<types>",
    "<type>",
    "    <name>user</name>",
    "    <description>Contain information about the user's role, goals, responsibilities, and knowledge. Great user memories help you tailor your future behavior to the user's preferences and perspective. Your goal in reading and writing these memories is to build up an understanding of who the user is and how you can be most helpful to them specifically. For example, you should collaborate with a senior software engineer differently than a student who is coding for the very first time. Keep in mind, that the aim here is to be helpful to the user. Avoid writing memories about the user that could be viewed as a negative judgement or that are not relevant to the work you're trying to accomplish together.</description>",
    "    <when_to_save>When you learn any details about the user's role, preferences, responsibilities, or knowledge</when_to_save>",
    "    <how_to_use>When your work should be informed by the user's profile or perspective. For example, if the user is asking you to explain a part of the code, you should answer that question in a way that is tailored to the specific details that they will find most valuable or that helps them build their mental model in relation to domain knowledge they already have.</how_to_use>",
    "    <examples>",
    "    user: I'm a data scientist investigating what logging we have in place",
    "    assistant: [saves user memory: user is a data scientist, currently focused on observability/logging]",
    "",
    "    user: I've been writing Go for ten years but this is my first time touching the React side of this repo",
    "    assistant: [saves user memory: deep Go expertise, new to React and this project's frontend — frame frontend explanations in terms of backend analogues]",
    "    </examples>",
    "</type>",
    "<type>",
    "    <name>feedback</name>",
    "    <description>Guidance the user has given you about how to approach work — both what to avoid and what to keep doing. These are a very important type of memory to read and write as they allow you to remain coherent and responsive to the way you should approach work in the project. Record from failure AND success: if you only save corrections, you will avoid past mistakes but drift away from approaches the user has already validated, and may grow overly cautious.</description>",
    "    <when_to_save>Any time the user corrects your approach (\"no not that\", \"don't\", \"stop doing X\") OR confirms a non-obvious approach worked (\"yes exactly\", \"perfect, keep doing that\", accepting an unusual choice without pushback). Corrections are easy to notice; confirmations are quieter — watch for them. In both cases, save what is applicable to future conversations, especially if surprising or not obvious from the code. Include *why* so you can judge edge cases later.</when_to_save>",
    "    <how_to_use>Let these memories guide your behavior so that the user does not need to offer the same guidance twice.</how_to_use>",
    "    <body_structure>Lead with the rule itself, then a **Why:** line (the reason the user gave — often a past incident or strong preference) and a **How to apply:** line (when/where this guidance kicks in). Knowing *why* lets you judge edge cases instead of blindly following the rule.</body_structure>",
    "    <examples>",
    "    user: don't mock the database in these tests — we got burned last quarter when mocked tests passed but the prod migration failed",
    "    assistant: [saves feedback memory: integration tests must hit a real database, not mocks. Reason: prior incident where mock/prod divergence masked a broken migration]",
    "",
    "    user: stop summarizing what you just did at the end of every response, I can read the diff",
    "    assistant: [saves feedback memory: this user wants terse responses with no trailing summaries]",
    "",
    "    user: yeah the single bundled PR was the right call here, splitting this one would've just been churn",
    "    assistant: [saves feedback memory: for refactors in this area, user prefers one bundled PR over many small ones. Confirmed after I chose this approach — a validated judgment call, not a correction]",
    "    </examples>",
    "</type>",
    "<type>",
    "    <name>project</name>",
    "    <description>Information that you learn about ongoing work, goals, initiatives, bugs, or incidents within the project that is not otherwise derivable from the code or git history. Project memories help you understand the broader context and motivation behind the work the user is doing within this working directory.</description>",
    "    <when_to_save>When you learn who is doing what, why, or by when. These states change relatively quickly so try to keep your understanding of this up to date. Always convert relative dates in user messages to absolute dates when saving (e.g., \"Thursday\" → \"2026-03-05\"), so the memory remains interpretable after time passes.</when_to_save>",
    "    <how_to_use>Use these memories to more fully understand the details and nuance behind the user's request and make better informed suggestions.</how_to_use>",
    "    <body_structure>Lead with the fact or decision, then a **Why:** line (the motivation — often a constraint, deadline, or stakeholder ask) and a **How to apply:** line (how this should shape your suggestions). Project memories decay fast, so the why helps future-you judge whether the memory is still load-bearing.</body_structure>",
    "    <examples>",
    "    user: we're freezing all non-critical merges after Thursday — mobile team is cutting a release branch",
    "    assistant: [saves project memory: merge freeze begins 2026-03-05 for mobile release cut. Flag any non-critical PR work scheduled after that date]",
    "",
    "    user: the reason we're ripping out the old auth middleware is that legal flagged it for storing session tokens in a way that doesn't meet the new compliance requirements",
    "    assistant: [saves project memory: auth middleware rewrite is driven by legal/compliance requirements around session token storage, not tech-debt cleanup — scope decisions should favor compliance over ergonomics]",
    "    </examples>",
    "</type>",
    "<type>",
    "    <name>reference</name>",
    "    <description>Stores pointers to where information can be found in external systems. These memories allow you to remember where to look to find up-to-date information outside of the project directory.</description>",
    "    <when_to_save>When you learn about resources in external systems and their purpose. For example, that bugs are tracked in a specific project in Linear or that feedback can be found in a specific Slack channel.</when_to_save>",
    "    <how_to_use>When the user references an external system or information that may be in an external system.</how_to_use>",
    "    <examples>",
    "    user: check the Linear project \"INGEST\" if you want context on these tickets, that's where we track all pipeline bugs",
    "    assistant: [saves reference memory: pipeline bugs are tracked in Linear project \"INGEST\"]",
    "",
    "    user: the Grafana board at grafana.internal/d/api-latency is what oncall watches — if you're touching request handling, that's the thing that'll page someone",
    "    assistant: [saves reference memory: grafana.internal/d/api-latency is the oncall latency dashboard — check it when editing request-path code]",
    "    </examples>",
    "</type>",
    "</types>",
    "",
]

WHAT_NOT_TO_SAVE_SECTION = [
    "## What NOT to save in memory",
    "",
    "- Code patterns, conventions, architecture, file paths, or project structure — these can be derived by reading the current project state.",
    "- Git history, recent changes, or who-changed-what — `git log` / `git blame` are authoritative.",
    "- Debugging solutions or fix recipes — the fix is in the code; the commit message has the context.",
    "- Anything already documented in CLAUDE.md files.",
    "- Ephemeral task details: in-progress work, temporary state, current conversation context.",
    "",
    "These exclusions apply even when the user explicitly asks you to save. If they ask you to save a PR list or activity summary, ask what was *surprising* or *non-obvious* about it — that is the part worth keeping.",
]

MEMORY_DRIFT_CAVEAT = "- Memory records can become stale over time. Use memory as context for what was true at a given point in time. Before answering the user or building assumptions based solely on information in memory records, verify that the memory is still correct and up-to-date by reading the current state of the files or resources. If a recalled memory conflicts with current information, trust what you observe now — and update or remove the stale memory rather than acting on it."

WHEN_TO_ACCESS_SECTION = [
    "## When to access memories",
    "- When memories seem relevant, or the user references prior-conversation work.",
    "- You MUST access memory when the user explicitly asks you to check, recall, or remember.",
    "- If the user says to *ignore* or *not use* memory: proceed as if MEMORY.md were empty. Do not apply remembered facts, cite, compare against, or mention memory content.",
    MEMORY_DRIFT_CAVEAT,
]

TRUSTING_RECALL_SECTION = [
    "## Before recommending from memory",
    "",
    "A memory that names a specific function, file, or flag is a claim that it existed *when the memory was written*. It may have been renamed, removed, or never merged. Before recommending it:",
    "",
    "- If the memory names a file path: check the file exists.",
    "- If the memory names a function or flag: grep for it.",
    "- If the user is about to act on your recommendation (not just asking about history), verify first.",
    "",
    "\"The memory says X exists\" is not the same as \"X exists now.\"",
    "",
    "A memory that summarizes repo state (activity logs, architecture snapshots) is frozen in time. If the user asks about *recent* or *current* state, prefer `git log` or reading the code over recalling the snapshot.",
]


def truncate_entrypoint_content(raw: str) -> str:
    """限制 memory 入口体积，并尽量在自然边界截断。"""
    trimmed = raw.strip()
    lines = trimmed.split("\n")
    was_line_truncated = len(lines) > MAX_ENTRYPOINT_LINES
    was_byte_truncated = len(trimmed.encode("utf-8")) > MAX_ENTRYPOINT_BYTES
    if not was_line_truncated and not was_byte_truncated:
        return trimmed
    truncated = "\n".join(lines[:MAX_ENTRYPOINT_LINES]) if was_line_truncated else trimmed
    encoded = truncated.encode("utf-8")
    if len(encoded) > MAX_ENTRYPOINT_BYTES:
        truncated = encoded[:MAX_ENTRYPOINT_BYTES].decode("utf-8", errors="ignore")
        if "\n" in truncated:
            truncated = truncated.rsplit("\n", 1)[0]
    reason = (
        f"{len(trimmed.encode('utf-8'))} bytes (limit: {MAX_ENTRYPOINT_BYTES} bytes) — index entries are too long"
        if was_byte_truncated and not was_line_truncated
        else f"{len(lines)} lines (limit: {MAX_ENTRYPOINT_LINES})"
        if was_line_truncated and not was_byte_truncated
        else f"{len(lines)} lines and {len(trimmed.encode('utf-8'))} bytes"
    )
    return (
        truncated
        + f"\n\n> WARNING: {ENTRYPOINT_NAME} is {reason}. Only part of it was loaded. Keep index entries to one line under ~200 chars; move detail into topic files."
    )


@dataclass
class MemoryLoader:
    """把 config_home、git root/cwd 映射为项目 memory 目录。"""
    config: KernelConfig
    auto_memory_path_override: Path | None = None

    def get_auto_mem_path(self) -> Path:
        """获取auto mem 路径，供项目 memory流程使用。"""
        if self.auto_memory_path_override is not None:
            # 测试和嵌入式调用可覆盖路径，但不改变默认项目键算法。
            return self.auto_memory_path_override
        # git root 让同一仓库不同子目录共享 memory；非 git 项目退回 cwd。
        base = find_git_root(self.config.cwd) or self.config.cwd
        return self.config.config_home / "projects" / sanitize_path(base) / "memory"

    def get_daily_log_path(self) -> Path:
        """获取daily log 路径，供项目 memory流程使用。"""
        from datetime import datetime

        today = datetime.now()
        yyyy = f"{today.year:04d}"
        mm = f"{today.month:02d}"
        dd = f"{today.day:02d}"
        return self.get_auto_mem_path() / "logs" / yyyy / mm / f"{yyyy}-{mm}-{dd}.md"

    def ensure_memory_dir_exists(self) -> None:
        """确保memory dir exists，供项目 memory流程使用。"""
        self.get_auto_mem_path().mkdir(parents=True, exist_ok=True)

    def build_memory_lines(self, display_name: str, memory_dir: Path, *, skip_index: bool = False) -> list[str]:
        """构造memory 行集合，供项目 memory流程使用。"""
        # daily-log 模式由后台过程维护索引，因此不会要求模型同步更新 MEMORY.md。
        if skip_index:
            how_to_save = [
                "## How to save memories",
                "",
                "Write each memory to its own file (e.g., `user_role.md`, `feedback_testing.md`) using this frontmatter format:",
                "",
                *MEMORY_FRONTMATTER_EXAMPLE,
                "",
                "- Keep the name, description, and type fields in memory files up-to-date with the content",
                "- Organize memory semantically by topic, not chronologically",
                "- Update or remove memories that turn out to be wrong or outdated",
                "- Do not write duplicate memories. First check if there is an existing memory you can update before writing a new one.",
            ]
        else:
            how_to_save = [
                "## How to save memories",
                "",
                "Saving a memory is a two-step process:",
                "",
                "**Step 1** — write the memory to its own file (e.g., `user_role.md`, `feedback_testing.md`) using this frontmatter format:",
                "",
                *MEMORY_FRONTMATTER_EXAMPLE,
                "",
                f"**Step 2** — add a pointer to that file in `{ENTRYPOINT_NAME}`. `{ENTRYPOINT_NAME}` is an index, not a memory — each entry should be one line, under ~150 characters: `- [Title](file.md) — one-line hook`. It has no frontmatter. Never write memory content directly into `{ENTRYPOINT_NAME}`.",
                "",
                f"- `{ENTRYPOINT_NAME}` is always loaded into your conversation context — lines after {MAX_ENTRYPOINT_LINES} will be truncated, so keep the index concise",
                "- Keep the name, description, and type fields in memory files up-to-date with the content",
                "- Organize memory semantically by topic, not chronologically",
                "- Update or remove memories that turn out to be wrong or outdated",
                "- Do not write duplicate memories. First check if there is an existing memory you can update before writing a new one.",
            ]
        lines = [
            f"# {display_name}",
            "",
            f"You have a persistent, file-based memory system at `{memory_dir}/`. {DIR_EXISTS_GUIDANCE}",
            "",
            "You should build up this memory system over time so that future conversations can have a complete picture of who the user is, how they'd like to collaborate with you, what behaviors to avoid or repeat, and the context behind the work the user gives you.",
            "",
            "If the user explicitly asks you to remember something, save it immediately as whichever type fits best. If they ask you to forget something, find and remove the relevant entry.",
            "",
            *TYPES_SECTION_INDIVIDUAL,
            *WHAT_NOT_TO_SAVE_SECTION,
            "",
            *how_to_save,
            "",
            *WHEN_TO_ACCESS_SECTION,
            "",
            *TRUSTING_RECALL_SECTION,
            "",
            "## Memory and other forms of persistence",
            "Memory is one of several persistence mechanisms available to you as you assist the user in a given conversation. The distinction is often that memory can be recalled in future conversations and should not be used for persisting information that is only useful within the scope of the current conversation.",
            "- When to use or update a plan instead of memory: If you are about to start a non-trivial implementation task and would like to reach alignment with the user on your approach you should use a Plan rather than saving this information to memory. Similarly, if you already have a plan within the conversation and you have changed your approach persist that change by updating the plan rather than saving a memory.",
            "- When to use or update tasks instead of memory: When you need to break your work in current conversation into discrete steps or keep track of your progress use tasks instead of saving to memory. Tasks are great for persisting information about the work that needs to be done in the current conversation, but memory should be reserved for information that will be useful in future conversations.",
            "",
        ]
        return lines

    def load_memory_prompt(self) -> str | None:
        """构造默认 memory 系统提示；v0.1 不自动内联 MEMORY.md 内容。"""
        if not self.config.auto_memory_enabled:
            return None
        memory_dir = self.get_auto_mem_path()
        self.ensure_memory_dir_exists()
        # Kairos 长会话写 append-only daily log；普通模式使用主题文件和索引。
        if self.config.features.kairos and self.config.kairos_active:
            return self.build_assistant_daily_log_prompt()
        return "\n".join(self.build_memory_lines(AUTO_MEM_DISPLAY_NAME, memory_dir))

    def build_memory_prompt_with_content(self) -> str:
        """构造memory 提示词 with 内容，供项目 memory流程使用。"""
        memory_dir = self.get_auto_mem_path()
        self.ensure_memory_dir_exists()
        lines = self.build_memory_lines(AUTO_MEM_DISPLAY_NAME, memory_dir)
        entrypoint = memory_dir / ENTRYPOINT_NAME
        try:
            content = entrypoint.read_text(encoding="utf-8")
        except FileNotFoundError:
            content = ""
        if content.strip():
            # 入口内容有独立行数/字节上限，避免长期增长吞噬上下文。
            lines.extend([f"## {ENTRYPOINT_NAME}", "", truncate_entrypoint_content(content)])
        else:
            lines.extend(
                [
                    f"## {ENTRYPOINT_NAME}",
                    "",
                    f"Your {ENTRYPOINT_NAME} is currently empty. When you save new memories, they will appear here.",
                ]
            )
        return "\n".join(lines)

    def build_assistant_daily_log_prompt(self, *, skip_index: bool = False) -> str:
        """构造assistant daily log 提示词，供项目 memory流程使用。"""
        memory_dir = self.get_auto_mem_path()
        log_path_pattern = memory_dir / "logs" / "YYYY" / "MM" / "YYYY-MM-DD.md"
        lines = [
            "# auto memory",
            "",
            f"You have a persistent, file-based memory system found at: `{memory_dir}/`",
            "",
            "This session is long-lived. As you work, record anything worth remembering by **appending** to today's daily log file:",
            "",
            f"`{log_path_pattern}`",
            "",
            "Substitute today's date (from `currentDate` in your context) for `YYYY-MM-DD`. When the date rolls over mid-session, start appending to the new day's file.",
            "",
            "Write each entry as a short timestamped bullet. Create the file (and parent directories) on first write if it does not exist. Do not rewrite or reorganize the log — it is append-only. A separate nightly process distills these logs into `MEMORY.md` and topic files.",
            "",
            "## What to log",
            '- User corrections and preferences ("use bun, not npm"; "stop summarizing diffs")',
            "- Facts about the user, their role, or their goals",
            "- Project context that is not derivable from the code (deadlines, incidents, decisions and their rationale)",
            "- Pointers to external systems (dashboards, Linear projects, Slack channels)",
            "- Anything the user explicitly asks you to remember",
            "",
            *WHAT_NOT_TO_SAVE_SECTION,
            "",
        ]
        if not skip_index:
            lines.extend(
                [
                    f"## {ENTRYPOINT_NAME}",
                    f"`{ENTRYPOINT_NAME}` is the distilled index (maintained nightly from your logs) and is loaded into your context automatically. Read it for orientation, but do not edit it directly — record new information in today's log instead.",
                    "",
                ]
            )
        return "\n".join(lines)
