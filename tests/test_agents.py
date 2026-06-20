"""Subagent 定义加载、工具过滤、同步/后台执行和 fork 语义测试。

重点观察 FakeModelProvider 如何驱动嵌套 query，以及主 session、sidechain、fork child 的
消息与工具集如何隔离。这里也是理解 AgentTool 输出形状的最快入口。
"""

from __future__ import annotations

from pathlib import Path
import asyncio
import json

from agent_kernel.agents import AgentTool, build_forked_messages, load_agents, resolve_agent_tools, run_subagent
from agent_kernel.config import AgentConfig, FeatureFlags, KernelConfig
from agent_kernel.messages import create_assistant_message, create_user_message
from agent_kernel.model_provider import FakeModelProvider
from agent_kernel.query_engine import QueryEngine
from agent_kernel.tools import BashTool, EditTool, FileReadTool, FileWriteTool, ReadFileStateEntry, ToolUseContext


async def _collect(iterator):
    """消费异步生成器并把全部事件收集为列表，便于同步断言。"""
    return [event async for event in iterator]


def _make_config(tmp_path: Path, **kwargs) -> KernelConfig:
    """为当前测试创建隔离 cwd 和 config_home 的最小 KernelConfig。"""
    repo = tmp_path / "repo"
    repo.mkdir()
    return KernelConfig(cwd=repo, config_home=tmp_path / ".claude", session_start_date="2026-06-14", **kwargs)


def test_agent_loader_includes_builtins_and_project_markdown_override(tmp_path: Path) -> None:
    """验证 ``agent loader includes builtins and project markdown override`` 场景的行为、消息形状和关键不变量。"""
    config = _make_config(tmp_path)
    agents_dir = config.cwd / ".claude" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "reviewer.md").write_text(
        """---
name: reviewer
description: Review code changes
tools: Read, Bash
disallowedTools: Write, Edit
maxTurns: 2
---
You review code and report findings.
""",
        encoding="utf-8",
    )

    agents = load_agents(config)
    reviewer = next(agent for agent in agents if agent.agent_type == "reviewer")

    assert any(agent.agent_type == "general-purpose" for agent in agents)
    assert reviewer.when_to_use == "Review code changes"
    assert reviewer.tools == ("Read", "Bash")
    assert reviewer.disallowed_tools == ("Write", "Edit")
    assert reviewer.max_turns == 2
    assert "report findings" in reviewer.system_prompt


def test_query_engine_registers_agent_tool_and_sdk_init_agents(tmp_path: Path) -> None:
    """验证 ``query engine registers agent tool and sdk init agents`` 场景的行为、消息形状和关键不变量。"""
    engine = QueryEngine(model_provider=FakeModelProvider(["ok"]), config=_make_config(tmp_path), session_id="session-1")
    init = engine.get_system_init_message()
    agent_tool = next(tool for tool in engine.tools if isinstance(tool, AgentTool))
    prompt = asyncio.run(agent_tool.prompt())

    assert "Task" in init["tools"]
    assert "general-purpose" in init["agents"]
    assert "Launch a new agent to handle complex, multi-step tasks autonomously." in prompt
    assert "- general-purpose:" in prompt


def test_custom_tool_list_is_not_mutated_with_agent_tool(tmp_path: Path) -> None:
    """验证 ``custom tool list is not mutated with agent tool`` 场景的行为、消息形状和关键不变量。"""
    engine = QueryEngine(
        model_provider=FakeModelProvider(["ok"]),
        config=_make_config(tmp_path),
        tools=[BashTool(), FileReadTool()],
    )

    assert [tool.name for tool in engine.tools] == ["Bash", "Read"]


def test_resolve_agent_tools_filters_disallowed_and_allowlist(tmp_path: Path) -> None:
    """验证 ``resolve agent tools filters disallowed and allowlist`` 场景的行为、消息形状和关键不变量。"""
    agent = AgentConfig(
        name="reader",
        description="Read only",
        prompt="Read files.",
        tools=("Read", "Bash", "Write"),
        disallowed_tools=("Write",),
    )
    config = _make_config(tmp_path, agents=(agent,))
    loaded = next(item for item in load_agents(config) if item.agent_type == "reader")

    resolved = resolve_agent_tools(loaded, [BashTool(), FileReadTool(), FileWriteTool(), EditTool()])

    assert [tool.name for tool in resolved] == ["Bash", "Read"]


def test_agent_tool_sync_subagent_runs_nested_query_and_returns_result(tmp_path: Path) -> None:
    """验证 ``agent tool sync subagent runs nested query and returns result`` 场景的行为、消息形状和关键不变量。"""
    config = _make_config(
        tmp_path,
        disable_builtin_agents=True,
        agents=(
            AgentConfig(
                name="researcher",
                description="Research a question",
                prompt="You are a focused researcher.",
                tools=("Read",),
                max_turns=1,
            ),
        ),
    )
    provider = FakeModelProvider(
        [
            [
                {
                    "type": "tool_use",
                    "id": "toolu_agent",
                    "name": "Agent",
                    "input": {
                        "description": "Research answer",
                        "subagent_type": "researcher",
                        "prompt": "Find the answer.",
                    },
                }
            ],
            "Subagent found the answer.",
            "Parent saw the result.",
        ]
    )
    engine = QueryEngine(model_provider=provider, config=config, session_id="session-1")

    events = asyncio.run(_collect(engine.submit_message("delegate", max_turns=3)))
    tool_message = next(event for event in events if event.get("type") == "user")
    block = tool_message["message"]["content"][0]
    text_blocks = block["content"]
    sidechain_dir = config.config_home / "projects"
    sidechain_files = list(sidechain_dir.rglob("subagents/*.jsonl"))

    assert block["type"] == "tool_result"
    assert text_blocks[0]["text"] == "Subagent found the answer."
    assert "agentId:" in text_blocks[1]["text"]
    assert provider.calls[1]["options"]["querySource"] == "agent:flagSettings:researcher"
    assert provider.calls[1]["system_prompt"][0] == "You are a focused researcher."
    assert sidechain_files
    rows = [json.loads(line) for line in sidechain_files[0].read_text(encoding="utf-8").splitlines()]
    assert all(row["isSidechain"] is True for row in rows)
    assert any(row.get("agentId") for row in rows)


def test_agent_tool_background_returns_async_launched(tmp_path: Path) -> None:
    """验证 ``agent tool background returns async launched`` 场景的行为、消息形状和关键不变量。"""
    config = _make_config(
        tmp_path,
        disable_builtin_agents=True,
        agents=(
            AgentConfig(
                name="worker",
                description="Background work",
                prompt="You run in background.",
                tools=("Read",),
            ),
        ),
    )
    provider = FakeModelProvider(
        [
            [
                {
                    "type": "tool_use",
                    "id": "toolu_agent",
                    "name": "Agent",
                    "input": {
                        "description": "Background answer",
                        "subagent_type": "worker",
                        "prompt": "Do the work.",
                        "run_in_background": True,
                    },
                }
            ],
            "Background result.",
            "Parent continues.",
        ]
    )
    engine = QueryEngine(model_provider=provider, config=config, session_id="session-1")

    events = asyncio.run(_collect(engine.submit_message("delegate background", max_turns=2)))
    block = next(event for event in events if event.get("type") == "user")["message"]["content"][0]
    text = block["content"][0]["text"]

    assert "Async agent launched successfully." in text
    assert "output_file:" in text
    assert engine.tool_use_context.background_tasks


def test_build_forked_messages_preserves_parent_tool_uses_with_placeholders() -> None:
    """验证 ``build forked messages preserves parent tool uses with placeholders`` 场景的行为、消息形状和关键不变量。"""
    parent = create_assistant_message(
        [
            {"type": "text", "text": "Launching work."},
            {"type": "tool_use", "id": "toolu_agent", "name": "Agent", "input": {"description": "Audit", "prompt": "Audit"}},
            {"type": "tool_use", "id": "toolu_read", "name": "Read", "input": {"file_path": "/tmp/example.txt"}},
        ]
    )

    forked = build_forked_messages("Audit only the API surface.", parent)

    assert forked[0]["type"] == "assistant"
    assert forked[0]["uuid"] != parent["uuid"]
    assert forked[0]["message"]["content"] == parent["message"]["content"]
    user_blocks = forked[1]["message"]["content"]
    assert [block["tool_use_id"] for block in user_blocks[:2]] == ["toolu_agent", "toolu_read"]
    assert all(block["content"][0]["text"] == "Fork started — processing in background" for block in user_blocks[:2])
    assert user_blocks[2]["type"] == "text"
    assert "<fork-boilerplate>" in user_blocks[2]["text"]
    assert "Your directive: Audit only the API surface." in user_blocks[2]["text"]


def test_fork_subagent_recursive_guard(tmp_path: Path) -> None:
    """验证 ``fork subagent recursive guard`` 场景的行为、消息形状和关键不变量。"""
    config = _make_config(tmp_path, features=FeatureFlags(fork_subagent=True))
    tool = AgentTool(load_agents(config), config=config)
    context = ToolUseContext(
        config=config,
        tools=[tool],
        messages=[create_user_message("<fork-boilerplate>\nAlready forked.")],
        agent_type="fork",
    )

    result = asyncio.run(tool.validate_input({"description": "Nested fork", "prompt": "Fork again."}, context))

    assert result.result is False
    assert result.message == "Fork is not available inside a forked worker. Complete your task directly using your tools."


def test_fork_subagent_omitted_type_inherits_parent_context_and_runs_async(tmp_path: Path) -> None:
    """验证 ``fork subagent omitted type inherits parent context and runs async`` 场景的行为、消息形状和关键不变量。"""
    config = _make_config(tmp_path, features=FeatureFlags(fork_subagent=True))
    provider = FakeModelProvider(["Scope: API surface\nResult: ok"])
    tool = AgentTool(load_agents(config), config=config)
    parent_assistant = create_assistant_message(
        [
            {"type": "text", "text": "I'll fork this."},
            {
                "type": "tool_use",
                "id": "toolu_agent",
                "name": "Agent",
                "input": {"description": "API audit", "prompt": "Audit API surface.", "name": "api-audit"},
            },
        ]
    )
    context_tools = [FileReadTool(), FileWriteTool(), tool]
    context = ToolUseContext(
        config=config,
        tools=context_tools,
        model_provider=provider,
        web_fetch_model="parent-model",
        session_id="session-1",
        messages=[create_user_message("Parent context the fork should inherit."), parent_assistant],
        rendered_system_prompt=["PARENT SYSTEM"],
        user_context={"currentDate": "2026-06-16"},
        system_context={"gitStatus": "stale"},
        read_file_state={"/tmp/example.txt": ReadFileStateEntry(content="cached", timestamp=123)},
    )
    args = {
        "description": "API audit",
        "prompt": "Audit API surface.",
        "name": "api-audit",
        "model": "haiku",
    }

    async def _run():
        """运行当前测试所需的异步场景并返回事件。"""
        validation = await tool.validate_input(args, context)
        result = await tool.call(args, context, None, parent_assistant)
        await asyncio.wait_for(context.background_tasks[result.data["agentId"]]["task"], timeout=1)
        return validation, result

    validation, result = asyncio.run(_run())

    assert validation.result is True
    assert result.data["status"] == "async_launched"
    assert result.data["name"] == "api-audit"
    task_info = context.background_tasks[result.data["agentId"]]
    assert task_info["agentType"] == "fork"
    assert task_info["name"] == "api-audit"
    call = provider.calls[0]
    assert call["options"]["model"] == "parent-model"
    assert call["system_prompt"] == ["PARENT SYSTEM"]
    assert [tool.name for tool in call["tools"]] == [tool.name for tool in context_tools]
    serialized_messages = json.dumps(call["messages"], ensure_ascii=False)
    assert "Parent context the fork should inherit." in serialized_messages
    assert "Fork started — processing in background" in serialized_messages
    assert "Your directive: Audit API surface." in serialized_messages
    assert "toolu_agent" in serialized_messages


def test_subagent_omits_claude_md_and_stale_git_status_for_explore(tmp_path: Path) -> None:
    """验证 ``subagent omits claude md and stale git status for explore`` 场景的行为、消息形状和关键不变量。"""
    config = _make_config(tmp_path)
    agent = next(item for item in load_agents(config) if item.agent_type == "Explore")
    provider = FakeModelProvider(["Found it."])
    parent_context = ToolUseContext(
        config=config,
        tools=[BashTool(), FileReadTool(), FileWriteTool()],
        model_provider=provider,
        web_fetch_model="parent-model",
        session_id="session-1",
        user_context={"claudeMd": "project rules", "currentDate": "2026-06-16"},
        system_context={"gitStatus": "old status", "other": "kept"},
    )

    result = asyncio.run(
        run_subagent(
            agent=agent,
            prompt="Find a file.",
            description="Find file",
            parent_context=parent_context,
            model_provider=provider,
            model="parent-model",
        )
    )

    assert result["status"] == "completed"
    call = provider.calls[0]
    serialized_messages = json.dumps(call["messages"], ensure_ascii=False)
    system_prompt = "\n\n".join(call["system_prompt"])
    assert "project rules" not in serialized_messages
    assert "claudeMd" not in serialized_messages
    assert "currentDate" in serialized_messages
    assert "gitStatus" not in system_prompt
    assert "old status" not in system_prompt
    assert "other: kept" in system_prompt


def test_subagent_model_argument_overrides_agent_definition_model(tmp_path: Path) -> None:
    """验证 ``subagent model argument overrides agent definition model`` 场景的行为、消息形状和关键不变量。"""
    config = _make_config(
        tmp_path,
        disable_builtin_agents=True,
        agents=(
            AgentConfig(
                name="modelled",
                description="Model test",
                prompt="Use the configured model unless overridden.",
                tools=("Read",),
                model="haiku",
            ),
        ),
    )
    agent = next(item for item in load_agents(config) if item.agent_type == "modelled")
    provider = FakeModelProvider(["Done."])
    parent_context = ToolUseContext(
        config=config,
        tools=[FileReadTool()],
        model_provider=provider,
        web_fetch_model="parent-model",
        session_id="session-1",
    )

    result = asyncio.run(
        run_subagent(
            agent=agent,
            prompt="Use a different model.",
            description="Model override",
            parent_context=parent_context,
            model_provider=provider,
            model="parent-model",
            model_override="sonnet",
        )
    )

    assert result["status"] == "completed"
    assert provider.calls[0]["options"]["model"] == "sonnet"
