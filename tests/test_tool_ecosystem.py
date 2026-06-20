"""Glob/Grep/LS/Todo/MultiEdit/Notebook 等外围工具行为测试。

测试既验证单工具输出，也验证工具通过 QueryEngine 回灌下一模型轮；MultiEdit 和
Notebook 用例重点展示 read-before-write 与失败不落盘的不变量。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from agent_kernel.config import KernelConfig
from agent_kernel.messages import create_assistant_message
from agent_kernel.model_provider import FakeModelProvider
from agent_kernel.permissions import ToolPermissionContext
from agent_kernel.query_engine import QueryEngine
from agent_kernel.tools import (
    AppState,
    FileReadTool,
    GlobTool,
    GrepTool,
    LSTool,
    MultiEditTool,
    NotebookEditTool,
    TodoWriteTool,
    ToolUseContext,
)


async def _collect(generator):
    """消费异步生成器并把全部事件收集为列表，便于同步断言。"""
    return [event async for event in generator]


def _context(tmp_path: Path) -> ToolUseContext:
    """为当前测试封装 ``_context`` 辅助步骤。"""
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    tools = [
        GlobTool(),
        GrepTool(),
        LSTool(),
        FileReadTool(),
        MultiEditTool(),
        NotebookEditTool(),
        TodoWriteTool(),
    ]
    return ToolUseContext(
        config=KernelConfig(cwd=repo, config_home=tmp_path / ".claude"),
        tools=tools,
        app_state=AppState(ToolPermissionContext(mode="bypass")),
        session_id="session-1",
    )


def test_query_engine_default_tools_include_ecosystem_and_preserve_web_tail(tmp_path: Path) -> None:
    """验证 ``query engine default tools include ecosystem and preserve web tail`` 场景的行为、消息形状和关键不变量。"""
    repo = tmp_path / "repo"
    repo.mkdir()

    engine = QueryEngine(model_provider=FakeModelProvider(["done"]), config=KernelConfig(cwd=repo, config_home=tmp_path / ".claude"))

    names = [tool.name for tool in engine.tools]
    assert "Glob" in names
    assert "Grep" in names
    assert "LS" in names
    assert "MultiEdit" in names
    assert "NotebookEdit" in names
    assert "TodoWrite" in names
    assert names[-2:] == ["WebSearch", "WebFetch"]


def test_glob_grep_and_ls_tools(tmp_path: Path) -> None:
    """验证 ``glob grep and ls tools`` 场景的行为、消息形状和关键不变量。"""
    context = _context(tmp_path)
    repo = context.config.cwd
    src = repo / "src"
    src.mkdir()
    (src / "a.py").write_text("alpha\nneedle\n", encoding="utf-8")
    (src / "b.txt").write_text("needle elsewhere\n", encoding="utf-8")
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    parent = create_assistant_message("tool call")

    glob_result = asyncio.run(GlobTool().call({"pattern": "**/*.py"}, context, None, parent))
    glob_block = GlobTool().map_tool_result_to_tool_result_block_param(glob_result.data, "toolu_glob")
    grep_files = asyncio.run(GrepTool().call({"pattern": "needle", "path": str(repo), "glob": "**/*.py"}, context, None, parent))
    grep_files_block = GrepTool().map_tool_result_to_tool_result_block_param(grep_files.data, "toolu_grep_files")
    grep_content = asyncio.run(GrepTool().call({"pattern": "needle", "path": str(repo), "output_mode": "content", "-n": True}, context, None, parent))
    grep_content_block = GrepTool().map_tool_result_to_tool_result_block_param(grep_content.data, "toolu_grep_content")
    ls_result = asyncio.run(LSTool().call({"path": str(repo), "ignore": ["README.md"]}, context, None, parent))
    ls_block = LSTool().map_tool_result_to_tool_result_block_param(ls_result.data, "toolu_ls")

    assert "src/a.py" in glob_block["content"]
    assert grep_files_block["content"].startswith("Found 1 file")
    assert "src/a.py" in grep_files_block["content"]
    assert "src/a.py:2:needle" in grep_content_block["content"]
    assert "src/" in ls_block["content"]
    assert "README.md" not in ls_block["content"]


def test_todo_write_updates_app_state_and_clears_completed_list(tmp_path: Path) -> None:
    """验证 ``todo write updates app state and clears completed list`` 场景的行为、消息形状和关键不变量。"""
    context = _context(tmp_path)
    tool = TodoWriteTool()
    parent = create_assistant_message("tool call")
    todos = [
        {"content": "implement", "status": "in_progress", "activeForm": "Implementing"},
        {"content": "verify", "status": "pending", "activeForm": "Verifying"},
    ]

    invalid = asyncio.run(
        tool.validate_input(
            {
                "todos": [
                    {"content": "one", "status": "in_progress", "activeForm": "Doing one"},
                    {"content": "two", "status": "in_progress", "activeForm": "Doing two"},
                ]
            },
            context,
        )
    )
    result = asyncio.run(tool.call({"todos": todos}, context, None, parent))
    completed = asyncio.run(
        tool.call(
            {"todos": [{"content": "implement", "status": "completed", "activeForm": "Implementing"}]},
            context,
            None,
            parent,
        )
    )

    assert invalid.result is False
    assert context.get_app_state().todos["session-1"] == []
    assert result.data["newTodos"] == todos
    assert completed.data["newTodos"][0]["status"] == "completed"


def test_multi_edit_requires_read_and_applies_edits_atomically(tmp_path: Path) -> None:
    """验证 ``multi edit requires read and applies edits atomically`` 场景的行为、消息形状和关键不变量。"""
    context = _context(tmp_path)
    path = context.config.cwd / "main.py"
    path.write_text("first\nsecond\nsecond\n", encoding="utf-8")
    parent = create_assistant_message("tool call")
    tool = MultiEditTool()
    args = {
        "file_path": str(path),
        "edits": [
            {"old_string": "first", "new_string": "1st"},
            {"old_string": "second", "new_string": "2nd", "replace_all": True},
        ],
    }

    before_read = asyncio.run(tool.validate_input(args, context))
    asyncio.run(FileReadTool().call({"file_path": str(path)}, context, None, parent))
    after_read = asyncio.run(tool.validate_input(args, context))
    result = asyncio.run(tool.call(args, context, None, parent))
    block = tool.map_tool_result_to_tool_result_block_param(result.data, "toolu_multi")

    assert before_read.result is False
    assert after_read.result is True
    assert path.read_text(encoding="utf-8") == "1st\n2nd\n2nd\n"
    assert block["content"] == f"Applied 2 edits to {path} successfully."


def test_notebook_edit_requires_read_and_replaces_cell(tmp_path: Path) -> None:
    """验证 ``notebook edit requires read and replaces cell`` 场景的行为、消息形状和关键不变量。"""
    context = _context(tmp_path)
    path = context.config.cwd / "nb.ipynb"
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 7,
                "id": "abc",
                "metadata": {},
                "outputs": [{"output_type": "stream", "name": "stdout", "text": ["old\n"]}],
                "source": ["print('old')\n"],
            }
        ],
        "metadata": {"language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook), encoding="utf-8")
    parent = create_assistant_message("tool call")
    tool = NotebookEditTool()
    args = {"notebook_path": str(path), "cell_id": "abc", "new_source": "print('new')\n"}

    before_read = asyncio.run(tool.validate_input(args, context))
    asyncio.run(FileReadTool().call({"file_path": str(path)}, context, None, parent))
    after_read = asyncio.run(tool.validate_input(args, context))
    result = asyncio.run(tool.call(args, context, None, parent))
    updated = json.loads(path.read_text(encoding="utf-8"))
    block = tool.map_tool_result_to_tool_result_block_param(result.data, "toolu_nb")

    assert before_read.result is False
    assert after_read.result is True
    assert updated["cells"][0]["source"] == "print('new')\n"
    assert updated["cells"][0]["execution_count"] is None
    assert updated["cells"][0]["outputs"] == []
    assert block["content"] == "Updated cell abc with print('new')\n"


def test_query_engine_grep_tool_use_roundtrips_into_second_turn(tmp_path: Path) -> None:
    """验证 ``query engine grep tool use roundtrips into second turn`` 场景的行为、消息形状和关键不变量。"""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("needle\n", encoding="utf-8")
    provider = FakeModelProvider(
        [
            [
                {
                    "type": "tool_use",
                    "id": "toolu_grep",
                    "name": "Grep",
                    "input": {"pattern": "needle", "path": str(repo), "output_mode": "files_with_matches"},
                }
            ],
            "done",
        ]
    )
    engine = QueryEngine(model_provider=provider, config=KernelConfig(cwd=repo, config_home=tmp_path / ".claude"))
    engine.tool_use_context.app_state.tool_permission_context.mode = "bypass"

    events = asyncio.run(_collect(engine.submit_message("search", max_turns=3)))

    tool_message = next(event for event in events if event.get("type") == "user")
    assert tool_message["message"]["content"][0]["tool_use_id"] == "toolu_grep"
    assert "Found 1 file" in tool_message["message"]["content"][0]["content"]
    assert "app.py" in tool_message["message"]["content"][0]["content"]
    assert provider.calls[1]["messages"][-1]["message"]["content"][0]["type"] == "tool_result"
    assert events[-1]["terminal"]["reason"] == "completed"
