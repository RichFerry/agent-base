"""Skill frontmatter、prompt 展示、调用权限和 hook 注入测试。

用临时 SKILL.md 验证“system prompt 只放索引、调用后才展开正文”的按需加载语义，并
覆盖 disable-model-invocation、side-effect 权限和 skill 自带 hooks。
"""

from __future__ import annotations

from pathlib import Path
import asyncio

from agent_kernel.config import KernelConfig, SkillConfig
from agent_kernel.model_provider import FakeModelProvider
from agent_kernel.permissions import ToolPermissionContext
from agent_kernel.query_engine import QueryEngine
from agent_kernel.skills import SkillTool, load_skills


async def _collect(iterator):
    """消费异步生成器并把全部事件收集为列表，便于同步断言。"""
    return [event async for event in iterator]


def _make_config(tmp_path: Path) -> KernelConfig:
    """为当前测试创建隔离 cwd 和 config_home 的最小 KernelConfig。"""
    repo = tmp_path / "repo"
    repo.mkdir()
    return KernelConfig(cwd=repo, config_home=tmp_path / ".claude", session_start_date="2026-06-14")


def _write_skill(root: Path, name: str, body: str) -> Path:
    """为当前测试封装 ``_write_skill`` 辅助步骤。"""
    skill_dir = root / ".claude" / "skills" / name
    skill_dir.mkdir(parents=True)
    path = skill_dir / "SKILL.md"
    path.write_text(body, encoding="utf-8")
    return path


def _all_text(messages: list[dict]) -> str:
    """为当前测试封装 ``_all_text`` 辅助步骤。"""
    parts: list[str] = []
    for message in messages:
        payload = message.get("message")
        content = payload.get("content") if isinstance(payload, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if isinstance(block.get("text"), str):
                parts.append(block["text"])
            if isinstance(block.get("content"), str):
                parts.append(block["content"])
    return "\n".join(parts)


def test_skill_loader_reads_project_skill_frontmatter(tmp_path: Path) -> None:
    """验证 ``skill loader reads project skill frontmatter`` 场景的行为、消息形状和关键不变量。"""
    config = _make_config(tmp_path)
    skill_path = _write_skill(
        config.cwd,
        "commit",
        """---
description: Create a git commit
when_to_use: Use when the user asks to commit current changes
allowed-tools: Bash(git status:*), Bash(git diff:*)
argument-hint: message
arguments: message
version: 1.2.3
---
# Commit
Use the repository state to create a commit.
""",
    )

    skills = load_skills(config)

    assert len(skills) == 1
    assert skills[0].name == "commit"
    assert skills[0].description == "Create a git commit"
    assert skills[0].when_to_use == "Use when the user asks to commit current changes"
    assert skills[0].allowed_tools == ("Bash(git status:*)", "Bash(git diff:*)")
    assert skills[0].argument_names == ("message",)
    assert skills[0].version == "1.2.3"
    assert skills[0].base_dir == skill_path.parent


def test_query_engine_registers_skill_tool_prompt_and_sdk_init(tmp_path: Path) -> None:
    """验证 ``query engine registers skill tool prompt and sdk init`` 场景的行为、消息形状和关键不变量。"""
    config = _make_config(tmp_path)
    _write_skill(
        config.cwd,
        "review-pr",
        """---
description: Review a pull request
when_to_use: Use for pull request review
---
Review the PR carefully.
""",
    )
    engine = QueryEngine(model_provider=FakeModelProvider(["ok"]), config=config, session_id="session-1")
    prompt = "\n\n".join(engine.prompt_composer.get_system_prompt(tools=engine.tools, model=engine.model))
    init = engine.get_system_init_message()

    assert any(isinstance(tool, SkillTool) for tool in engine.tools)
    assert "Skill" in init["tools"]
    assert init["skills"] == ["review-pr"]
    assert "# User-invocable skills" in prompt
    assert "- review-pr: Review a pull request - Use for pull request review" in prompt
    assert "/<skill-name> (e.g., /commit) is shorthand" in prompt


def test_skill_tool_inline_invocation_expands_prompt_into_next_model_turn(tmp_path: Path) -> None:
    """验证 ``skill tool inline invocation expands prompt into next model turn`` 场景的行为、消息形状和关键不变量。"""
    config = _make_config(tmp_path)
    skill_path = _write_skill(
        config.cwd,
        "commit",
        """---
description: Create a git commit
---
Use git status.
Arguments: $ARGUMENTS
Skill dir: ${CLAUDE_SKILL_DIR}
Session: ${CLAUDE_SESSION_ID}
""",
    )
    provider = FakeModelProvider(
        [
            [{"type": "tool_use", "id": "toolu_skill", "name": "Skill", "input": {"skill": "/commit", "args": "-m ok"}}],
            "done",
        ]
    )
    engine = QueryEngine(model_provider=provider, config=config, session_id="session-1")

    events = asyncio.run(_collect(engine.submit_message("commit this", max_turns=3)))
    tool_messages = [event for event in events if event.get("type") == "user"]
    second_call_text = _all_text(provider.calls[1]["messages"])

    assert tool_messages[0]["message"]["content"][0]["content"] == "Launching skill: commit"
    assert "<command-name>commit</command-name>" in second_call_text
    assert "<command-args>-m ok</command-args>" in second_call_text
    assert f"Base directory for this skill: {skill_path.parent}" in second_call_text
    assert "Arguments: -m ok" in second_call_text
    assert "Skill dir: " + str(skill_path.parent) in second_call_text
    assert "Session: session-1" in second_call_text
    assert events[-1]["terminal"]["reason"] == "completed"


def test_skill_disable_model_invocation_blocks_tool_use(tmp_path: Path) -> None:
    """验证 ``skill disable model invocation blocks tool use`` 场景的行为、消息形状和关键不变量。"""
    config = KernelConfig(
        cwd=tmp_path / "repo",
        config_home=tmp_path / ".claude",
        skills=(
            SkillConfig(
                name="secret",
                description="Do secret work",
                content="Never model invoke.",
                disable_model_invocation=True,
            ),
        ),
    )
    config.cwd.mkdir()
    provider = FakeModelProvider(
        [
            [{"type": "tool_use", "id": "toolu_skill", "name": "Skill", "input": {"skill": "secret"}}],
            "done",
        ]
    )
    engine = QueryEngine(model_provider=provider, config=config)

    events = asyncio.run(_collect(engine.submit_message("secret", max_turns=2)))
    block = next(event for event in events if event.get("type") == "user")["message"]["content"][0]

    assert block["is_error"] is True
    assert "disable-model-invocation" in block["content"]


def test_skill_permission_ask_denies_and_bypass_allows_side_effect_skill(tmp_path: Path) -> None:
    """验证 ``skill permission ask denies and bypass allows side effect skill`` 场景的行为、消息形状和关键不变量。"""
    repo = tmp_path / "repo"
    repo.mkdir()
    config = KernelConfig(
        cwd=repo,
        config_home=tmp_path / ".claude",
        skills=(
            SkillConfig(
                name="danger",
                description="Runs a dangerous workflow",
                content="Use Bash carefully.",
                allowed_tools=("Bash(rm:*)",),
            ),
        ),
    )
    denied_provider = FakeModelProvider(
        [
            [{"type": "tool_use", "id": "toolu_skill", "name": "Skill", "input": {"skill": "danger"}}],
            "done",
        ]
    )
    denied_engine = QueryEngine(model_provider=denied_provider, config=config)

    denied_events = asyncio.run(_collect(denied_engine.submit_message("danger", max_turns=2)))
    denied_block = next(event for event in denied_events if event.get("type") == "user")["message"]["content"][0]

    allowed_provider = FakeModelProvider(
        [
            [{"type": "tool_use", "id": "toolu_skill", "name": "Skill", "input": {"skill": "danger"}}],
            "done",
        ]
    )
    allowed_engine = QueryEngine(model_provider=allowed_provider, config=config)
    allowed_engine.tool_use_context.app_state.tool_permission_context = ToolPermissionContext(mode="bypass")

    allowed_events = asyncio.run(_collect(allowed_engine.submit_message("danger", max_turns=2)))
    allowed_block = next(event for event in allowed_events if event.get("type") == "user")["message"]["content"][0]

    assert denied_block["is_error"] is True
    assert "Execute skill: danger" in denied_block["content"]
    assert allowed_block.get("is_error") is not True
    assert allowed_block["content"] == "Launching skill: danger"
