"""提示词拼接精度、memory 路径和 session resume/parent 链测试。

重点断言 section 顺序和源码关键原文没有漂移，并通过临时 JSONL 展示 parentUuid、compact
boundary、preserved segment 与 microcompact 在 resume 时如何恢复。
"""

from __future__ import annotations

from pathlib import Path
import json

from agent_kernel.config import CachedMicrocompactConfig, FeatureFlags, KernelConfig, MCPClientConfig, OutputStyleConfig
from agent_kernel.context_compaction import create_compact_boundary_message, format_compact_summary, get_compact_prompt
from agent_kernel.memory import ENTRYPOINT_NAME, MemoryLoader
from agent_kernel.messages import create_assistant_message, create_tool_result_message, create_user_message
from agent_kernel.prompt_composer import SYSTEM_PROMPT_DYNAMIC_BOUNDARY, PromptComposer, build_effective_system_prompt
from agent_kernel.session import SessionStore
from agent_kernel.tools import BashTool, EditTool, FileReadTool, FileWriteTool, TodoWriteTool


def make_config(tmp_path: Path) -> KernelConfig:
    """为当前测试创建隔离 cwd 和 config_home 的最小 KernelConfig。"""
    cwd = tmp_path / "repo"
    cwd.mkdir()
    return KernelConfig(cwd=cwd, config_home=tmp_path / ".claude", session_start_date="2026-06-14")


def test_prompt_composer_keeps_core_section_order_and_append(tmp_path: Path) -> None:
    """验证 ``prompt composer keeps core section order and append`` 场景的行为、消息形状和关键不变量。"""
    config = make_config(tmp_path)
    composer = PromptComposer(config, MemoryLoader(config))
    tools = [BashTool(), FileReadTool(), FileWriteTool(), EditTool()]

    system_prompt, user_context, system_context = composer.fetch_system_prompt_parts(
        tools=tools,
        model="agent-kernel-frontier",
        append_system_prompt="APPENDED",
    )

    assert system_prompt[0].startswith("\nYou are an interactive agent")
    assert "# System" in system_prompt[1]
    assert "# Doing tasks" in system_prompt[2]
    assert "# Executing actions with care" in system_prompt[3]
    assert "# Using your tools" in system_prompt[4]
    assert "# Tone and style" in system_prompt[5]
    assert "# Output efficiency" in system_prompt[6]
    assert SYSTEM_PROMPT_DYNAMIC_BOUNDARY in system_prompt
    assert any("# auto memory" in section for section in system_prompt)
    assert any("Primary working directory" in section for section in system_prompt)
    assert system_prompt[-1] == "APPENDED"
    assert user_context["currentDate"] == "Today's date is 2026-06-14."
    assert system_context == {}


def test_prompt_composer_dynamic_sections_follow_source_order(tmp_path: Path) -> None:
    """验证 ``prompt composer dynamic sections follow source order`` 场景的行为、消息形状和关键不变量。"""
    config = KernelConfig(
        cwd=tmp_path / "repo",
        config_home=tmp_path / ".claude",
        session_start_date="2026-06-14",
        language="Spanish",
        features=FeatureFlags(cached_microcompact=True),
        scratchpad_enabled=True,
        scratchpad_dir=tmp_path / ".claude" / "scratchpad" / "session-1",
        cached_microcompact=CachedMicrocompactConfig(
            enabled=True,
            system_prompt_suggest_summaries=True,
            keep_recent=5,
            supported_models=("agent-kernel-frontier",),
        ),
        mcp_clients=(
            MCPClientConfig(name="github", instructions="Use pull request numbers exactly."),
            MCPClientConfig(name="disconnected", instructions="ignore me", type="disconnected"),
        ),
    )
    config.cwd.mkdir()
    memory_dir = MemoryLoader(config).get_auto_mem_path()
    memory_dir.mkdir(parents=True)
    (memory_dir / ENTRYPOINT_NAME).write_text("Remember the repo conventions.\n", encoding="utf-8")
    composer = PromptComposer(config, MemoryLoader(config))

    system_prompt = composer.get_system_prompt(tools=[BashTool(), FileReadTool()], model="agent-kernel-frontier")
    boundary = system_prompt.index(SYSTEM_PROMPT_DYNAMIC_BOUNDARY)
    dynamic_tail = system_prompt[boundary + 1 :]

    assert dynamic_tail[0].startswith("# Session-specific guidance")
    assert dynamic_tail[1].startswith("# auto memory")
    assert "Remember the repo conventions." not in dynamic_tail[1]
    assert dynamic_tail[2].startswith("# Environment")
    assert dynamic_tail[3].startswith("# Language")
    assert dynamic_tail[4].startswith("# MCP Server Instructions")
    assert dynamic_tail[5].startswith("# Scratchpad Directory")
    assert dynamic_tail[6].startswith("# Function Result Clearing")
    assert dynamic_tail[7] == "When working with tool results, write down any important information you might need later in your response, as the original tool result may be cleared later."
    assert "## github\nUse pull request numbers exactly." in dynamic_tail[4]
    assert "disconnected" not in dynamic_tail[4]
    assert f"`{config.scratchpad_dir}`" in dynamic_tail[5]
    assert "The 5 most recent results are always kept." in dynamic_tail[6]


def test_prompt_composer_source_precise_tool_guidance_and_env_bool(tmp_path: Path) -> None:
    """验证 ``prompt composer source precise tool guidance and env bool`` 场景的行为、消息形状和关键不变量。"""
    config = make_config(tmp_path)
    composer = PromptComposer(config, MemoryLoader(config))

    system_prompt = composer.get_system_prompt(
        tools=[BashTool(), FileReadTool(), FileWriteTool(), EditTool(), TodoWriteTool()],
        model="agent-kernel-frontier",
    )
    using_tools = next(section for section in system_prompt if section.startswith("# Using your tools"))
    env_info = next(section for section in system_prompt if section.startswith("# Environment"))
    session_guidance = composer.get_session_specific_guidance_section({"Agent", "Glob", "Grep"})

    assert "Break down and manage your work with the TodoWrite tool. These tools are helpful for planning your work and helping the user track your progress. Mark each task as completed as soon as you are done with the task. Do not batch up multiple tasks before marking them as completed." in using_tools
    assert "Is a git repository: false" in env_info
    assert "You are powered by the model named frontier model. The exact model ID is agent-kernel-frontier." in env_info
    assert session_guidance is not None
    assert "Use the Agent tool with specialized agents when the task at hand matches the agent's description." in session_guidance
    assert "For simple, directed codebase searches (e.g. for a specific file/class/function) use the Glob or Grep directly." in session_guidance
    assert "subagent_type=Explore" in session_guidance


def test_prompt_composer_preserves_ant_source_sections(tmp_path: Path) -> None:
    """验证 ``prompt composer preserves ant source sections`` 场景的行为、消息形状和关键不变量。"""
    config = KernelConfig(
        cwd=tmp_path / "repo",
        config_home=tmp_path / ".claude",
        session_start_date="2026-06-14",
        user_type="ant",
    )
    config.cwd.mkdir()
    composer = PromptComposer(config, MemoryLoader(config))

    system_prompt = composer.get_system_prompt(tools=[BashTool(), FileReadTool()], model="agent-kernel-frontier")

    assert any("Default to writing no comments. Only add one when the WHY is non-obvious" in section for section in system_prompt)
    assert any(section.startswith("# Communicating with the user") for section in system_prompt)
    assert not any("Your responses should be short and concise." in section for section in system_prompt)


def test_prompt_composer_mcp_delta_omits_mcp_instructions(tmp_path: Path) -> None:
    """验证 ``prompt composer mcp delta omits mcp instructions`` 场景的行为、消息形状和关键不变量。"""
    config = KernelConfig(
        cwd=tmp_path / "repo",
        config_home=tmp_path / ".claude",
        session_start_date="2026-06-14",
        features=FeatureFlags(mcp_instructions_delta=True),
        mcp_clients=(MCPClientConfig(name="github", instructions="Use PR context."),),
    )
    config.cwd.mkdir()
    composer = PromptComposer(config, MemoryLoader(config))

    system_prompt = composer.get_system_prompt(tools=[BashTool()], model="agent-kernel-frontier")

    assert not any(section.startswith("# MCP Server Instructions") for section in system_prompt)


def test_prompt_composer_output_style_intro_section_and_coding_gate(tmp_path: Path) -> None:
    """验证 ``prompt composer output style intro section and coding gate`` 场景的行为、消息形状和关键不变量。"""
    config = KernelConfig(
        cwd=tmp_path / "repo",
        config_home=tmp_path / ".claude",
        session_start_date="2026-06-14",
        output_style=OutputStyleConfig(
            name="Review",
            prompt="Prefer terse review findings.",
            keep_coding_instructions=False,
        ),
    )
    config.cwd.mkdir()
    composer = PromptComposer(config, MemoryLoader(config))

    system_prompt = composer.get_system_prompt(tools=[BashTool()], model="agent-kernel-frontier")
    boundary = system_prompt.index(SYSTEM_PROMPT_DYNAMIC_BOUNDARY)
    dynamic_tail = system_prompt[boundary + 1 :]

    assert 'according to your "Output Style" below' in system_prompt[0]
    assert not any(section.startswith("# Doing tasks") for section in system_prompt)
    assert any(section == "# Output Style: Review\nPrefer terse review findings." for section in dynamic_tail)


def test_prompt_composer_output_style_can_keep_coding_instructions(tmp_path: Path) -> None:
    """验证 ``prompt composer output style can keep coding instructions`` 场景的行为、消息形状和关键不变量。"""
    config = KernelConfig(
        cwd=tmp_path / "repo",
        config_home=tmp_path / ".claude",
        session_start_date="2026-06-14",
        output_style=OutputStyleConfig(
            name="Explanatory",
            prompt="Explain implementation choices.",
            keep_coding_instructions=True,
        ),
    )
    config.cwd.mkdir()
    composer = PromptComposer(config, MemoryLoader(config))

    system_prompt = composer.get_system_prompt(tools=[BashTool()], model="agent-kernel-frontier")

    assert any(section.startswith("# Doing tasks") for section in system_prompt)


def test_session_guidance_follows_interactive_gate(tmp_path: Path) -> None:
    """验证 ``session guidance follows interactive gate`` 场景的行为、消息形状和关键不变量。"""
    interactive = make_config(tmp_path)
    composer = PromptComposer(interactive, MemoryLoader(interactive))
    assert "suggest they type `! <command>`" in composer.get_session_specific_guidance_section(set())

    noninteractive = KernelConfig(
        cwd=tmp_path / "noninteractive",
        config_home=tmp_path / ".claude2",
        session_start_date="2026-06-14",
        is_non_interactive_session=True,
    )
    noninteractive.cwd.mkdir()
    composer = PromptComposer(noninteractive, MemoryLoader(noninteractive))
    assert composer.get_session_specific_guidance_section(set()) is None


def test_build_effective_system_prompt_priority() -> None:
    """验证 ``build effective system prompt priority`` 场景的行为、消息形状和关键不变量。"""
    assert build_effective_system_prompt(default_system_prompt=["default"], custom_system_prompt="custom") == ["custom"]
    assert build_effective_system_prompt(default_system_prompt=["default"], override_system_prompt="override") == ["override"]
    assert build_effective_system_prompt(default_system_prompt=["default"], append_system_prompt="tail") == ["default", "tail"]
    assert build_effective_system_prompt(
        default_system_prompt=["default"],
        agent_system_prompt="agent",
        custom_system_prompt="custom",
        append_system_prompt="tail",
    ) == ["agent", "tail"]
    assert build_effective_system_prompt(
        default_system_prompt=["default"],
        agent_system_prompt="agent",
        append_system_prompt="tail",
        proactive_active=True,
    ) == ["default", "\n# Custom Agent Instructions\nagent", "tail"]
    assert build_effective_system_prompt(
        default_system_prompt=["default"],
        override_system_prompt="override",
        append_system_prompt="tail",
    ) == ["override"]


def test_fetch_system_prompt_parts_custom_prompt_skips_default_prompt(tmp_path: Path) -> None:
    """验证 ``fetch system prompt parts custom prompt skips default prompt`` 场景的行为、消息形状和关键不变量。"""
    config = make_config(tmp_path)
    memory_dir = MemoryLoader(config).get_auto_mem_path()
    memory_dir.mkdir(parents=True)
    (memory_dir / ENTRYPOINT_NAME).write_text("This memory must not be in custom system prompt.\n", encoding="utf-8")
    composer = PromptComposer(config, MemoryLoader(config))

    system_prompt, user_context, system_context = composer.fetch_system_prompt_parts(
        tools=[BashTool()],
        model="agent-kernel-frontier",
        custom_system_prompt="CUSTOM",
        append_system_prompt="APPEND",
    )

    assert system_prompt == ["CUSTOM", "APPEND"]
    assert user_context["currentDate"] == "Today's date is 2026-06-14."
    assert system_context == {}

    empty_custom_prompt, _, _ = composer.fetch_system_prompt_parts(
        tools=[BashTool()],
        model="agent-kernel-frontier",
        custom_system_prompt="",
        append_system_prompt="APPEND",
    )
    assert empty_custom_prompt == ["APPEND"]


def test_context_compaction_prompt_and_summary_formatting() -> None:
    """验证 ``context compaction prompt and summary formatting`` 场景的行为、消息形状和关键不变量。"""
    prompt = get_compact_prompt("Keep exact file paths.")

    assert prompt.startswith("CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.")
    assert "Your task is to create a detailed summary of the conversation so far" in prompt
    assert "Additional Instructions:\nKeep exact file paths." in prompt
    assert prompt.endswith("Tool calls will be rejected and you will fail the task.")
    assert format_compact_summary("<analysis>draft</analysis>\n\n<summary>\n1. Done\n</summary>") == "Summary:\n1. Done"


def test_memory_loader_paths_and_entrypoint_content(tmp_path: Path) -> None:
    """验证 ``memory loader paths and entrypoint content`` 场景的行为、消息形状和关键不变量。"""
    config = make_config(tmp_path)
    loader = MemoryLoader(config)
    memory_dir = loader.get_auto_mem_path()
    memory_dir.mkdir(parents=True)
    (memory_dir / ENTRYPOINT_NAME).write_text("- [Preference](preference.md) — concise responses\n", encoding="utf-8")

    prompt = loader.build_memory_prompt_with_content()

    assert str(memory_dir).endswith("memory")
    assert "# auto memory" in prompt
    assert "## MEMORY.md" in prompt
    assert "- [Preference](preference.md) — concise responses" in prompt


def test_session_store_writes_jsonl_with_parent_chain(tmp_path: Path) -> None:
    """验证 ``session store writes jsonl with parent chain`` 场景的行为、消息形状和关键不变量。"""
    config = make_config(tmp_path)
    store = SessionStore(config, session_id="session-1")
    user = create_user_message("hello", uuid="u1")
    assistant = create_assistant_message("hi", uuid="a1", message_id="m1")

    store.record_transcript([user, assistant])
    rows = [json.loads(line) for line in store.transcript_path.read_text(encoding="utf-8").splitlines()]

    assert rows[0]["parentUuid"] is None
    assert rows[1]["parentUuid"] == "u1"
    assert rows[0]["sessionId"] == "session-1"
    assert rows[0]["cwd"] == str(config.cwd)
    assert store.load_messages()[1]["uuid"] == "a1"
    assert "parentUuid" not in store.load_messages()[1]


def test_session_store_resume_tracks_seen_prefix_without_duplicates(tmp_path: Path) -> None:
    """验证 ``session store resume tracks seen prefix without duplicates`` 场景的行为、消息形状和关键不变量。"""
    config = make_config(tmp_path)
    user = create_user_message("hello", uuid="u1")
    assistant = create_assistant_message("hi", uuid="a1", message_id="m1")
    first_store = SessionStore(config, session_id="session-1")
    first_store.record_transcript([user, assistant])

    resumed_store = SessionStore(config, session_id="session-1")
    next_user = create_user_message("next", uuid="u2")
    resumed_store.record_transcript([user, assistant, next_user])

    rows = [json.loads(line) for line in resumed_store.transcript_path.read_text(encoding="utf-8").splitlines()]
    assert [row["uuid"] for row in rows] == ["u1", "a1", "u2"]
    assert rows[2]["parentUuid"] == "a1"


def test_session_store_bridges_legacy_progress_parent_on_load(tmp_path: Path) -> None:
    """验证 ``session store bridges legacy progress parent on load`` 场景的行为、消息形状和关键不变量。"""
    config = make_config(tmp_path)
    store = SessionStore(config, session_id="session-1")
    store.project_dir.mkdir(parents=True)
    entries = [
        {"type": "user", "uuid": "u1", "message": {"role": "user", "content": []}, "parentUuid": None},
        {"type": "progress", "uuid": "p1", "parentUuid": "u1"},
        {"type": "assistant", "uuid": "a1", "message": {"id": "m1", "role": "assistant", "content": []}, "parentUuid": "p1"},
    ]
    store.transcript_path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8")

    reloaded = SessionStore(config, session_id="session-1")

    assert reloaded._load_transcript_messages()[1]["parentUuid"] == "u1"
    assert [message["uuid"] for message in reloaded.load_messages()] == ["u1", "a1"]


def test_tool_result_transcript_uses_source_assistant_uuid(tmp_path: Path) -> None:
    """验证 ``tool result transcript uses source assistant uuid`` 场景的行为、消息形状和关键不变量。"""
    config = make_config(tmp_path)
    store = SessionStore(config, session_id="session-1")
    user = create_user_message("read", uuid="u1")
    assistant = create_assistant_message(
        [{"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "x"}}],
        uuid="a1",
        message_id="m1",
    )
    unrelated_system = {"type": "system", "uuid": "s1", "content": "progress-like metadata"}
    tool_result = create_tool_result_message(
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"},
        uuid="u2",
        source_tool_assistant_uuid="a1",
    )

    store.record_transcript([user, assistant, unrelated_system, tool_result])
    rows = [json.loads(line) for line in store.transcript_path.read_text(encoding="utf-8").splitlines()]

    assert rows[3]["parentUuid"] == "a1"


def test_session_store_resume_loads_only_after_compact_boundary(tmp_path: Path) -> None:
    """验证 ``session store resume loads only after compact boundary`` 场景的行为、消息形状和关键不变量。"""
    config = make_config(tmp_path)
    store = SessionStore(config, session_id="session-1")
    old_user = create_user_message("old prompt", uuid="u1")
    old_assistant = create_assistant_message("old answer", uuid="a1", message_id="m1")
    boundary = create_compact_boundary_message("auto", 100, old_assistant["uuid"], messages_summarized=2)
    summary = create_user_message(
        "This session is being continued from a previous conversation.",
        uuid="summary-1",
        is_compact_summary=True,
        is_visible_in_transcript_only=True,
    )

    store.record_transcript([old_user, old_assistant, boundary, summary])
    rows = [json.loads(line) for line in store.transcript_path.read_text(encoding="utf-8").splitlines()]
    reloaded = SessionStore(config, session_id="session-1")

    assert [row["type"] for row in rows] == ["user", "assistant", "system", "user"]
    assert rows[2]["subtype"] == "compact_boundary"
    assert [message["type"] for message in reloaded.load_messages()] == ["system", "user"]
    assert reloaded.load_messages()[1]["isCompactSummary"] is True


def test_session_store_relinks_preserved_segment_after_partial_compact(tmp_path: Path) -> None:
    """验证 ``session store relinks preserved segment after partial compact`` 场景的行为、消息形状和关键不变量。"""
    config = make_config(tmp_path)
    store = SessionStore(config, session_id="session-1")
    old_user = create_user_message("old prompt", uuid="u1")
    old_assistant = create_assistant_message("old answer", uuid="a1", message_id="m1")
    kept_user = create_user_message("recent prompt", uuid="u2")
    kept_assistant = create_assistant_message("recent answer", uuid="a2", message_id="m2")
    summary = create_user_message(
        "This session is being continued from a previous conversation.",
        uuid="summary-1",
        is_compact_summary=True,
        is_visible_in_transcript_only=True,
    )
    boundary = create_compact_boundary_message(
        "auto",
        100,
        old_assistant["uuid"],
        messages_summarized=2,
        preserved_segment={"headUuid": "u2", "anchorUuid": "summary-1", "tailUuid": "a2"},
    )

    store.record_transcript([old_user, old_assistant, kept_user, kept_assistant, boundary, summary])
    reloaded = SessionStore(config, session_id="session-1")
    loaded_entries = reloaded._load_transcript_messages()

    assert [message["uuid"] for message in loaded_entries] == [boundary["uuid"], "summary-1", "u2", "a2"]
    assert loaded_entries[2]["parentUuid"] == "summary-1"
    assert [message["uuid"] for message in reloaded.load_messages()] == [boundary["uuid"], "summary-1", "u2", "a2"]


def test_session_store_applies_microcompact_boundary_on_resume(tmp_path: Path) -> None:
    """验证 ``session store applies microcompact boundary on resume`` 场景的行为、消息形状和关键不变量。"""
    config = make_config(tmp_path)
    store = SessionStore(config, session_id="session-1")
    tool_result = create_tool_result_message(
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "very large output"},
        uuid="tool-result-1",
    )
    boundary = {
        "type": "system",
        "subtype": "microcompact_boundary",
        "uuid": "micro-1",
        "content": "Context microcompacted",
        "microcompactMetadata": {
            "trigger": "auto",
            "preTokens": 100,
            "tokensSaved": 50,
            "compactedToolIds": ["toolu_1"],
            "clearedAttachmentUUIDs": [],
        },
    }

    store.record_transcript([tool_result, boundary])
    reloaded = SessionStore(config, session_id="session-1")
    loaded = reloaded.load_messages()

    assert loaded[0]["message"]["content"][0]["content"] == "[Old tool result content cleared]"
    assert loaded[1]["subtype"] == "microcompact_boundary"
