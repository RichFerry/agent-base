"""核心 query loop、权限、工具、compact、streaming 和取消的主行为测试集。

这是项目最大的行为规范文件：从 ask/bypass、Bash/File/Web 工具，到多轮 tool loop、
错误恢复、上下文压缩、真实 API payload 和 abort pairing 均有端到端断言。
"""

from __future__ import annotations

from pathlib import Path
import asyncio
import json

import pytest

from agent_kernel.config import ContextCompactionConfig, KernelConfig
from agent_kernel.messages import create_attachment_message, create_assistant_message, create_tool_result_message, create_user_message
from agent_kernel.model_provider import AnthropicAPIError, AnthropicModelProvider, FakeModelProvider
from agent_kernel.permissions import PermissionDecision, ToolPermissionContext, has_permissions_to_use_tool
from agent_kernel.query_engine import QueryEngine
from agent_kernel.tools import AppState, BashTool, EditTool, FileReadTool, FileWriteTool, ReadFileStateEntry, Tool, ToolResult, ToolUseContext, WebFetchTool, WebSearchTool
from agent_kernel.tools.prompts import bash_tool_prompt, edit_tool_prompt, read_tool_prompt


def make_context(tmp_path: Path, permission_context: ToolPermissionContext | None = None) -> ToolUseContext:
    """创建包含指定工具和权限模式的隔离 ToolUseContext。"""
    cwd = tmp_path / "repo"
    cwd.mkdir(exist_ok=True)
    config = KernelConfig(cwd=cwd, config_home=tmp_path / ".claude", session_start_date="2026-06-14")
    tools = [BashTool(), FileReadTool(), FileWriteTool(), EditTool(), WebSearchTool(), WebFetchTool()]
    return ToolUseContext(
        config=config,
        tools=tools,
        app_state=AppState(permission_context or ToolPermissionContext()),
    )


def test_permission_resolver_has_only_ask_and_bypass_modes(tmp_path: Path) -> None:
    """验证 ``permission resolver has only ask and bypass modes`` 场景的行为、消息形状和关键不变量。"""
    read = FileReadTool()
    write = FileWriteTool()
    ask_context = make_context(tmp_path)
    target = ask_context.config.cwd / "x.txt"
    target.write_text("x", encoding="utf-8")

    read_decision = asyncio.run(has_permissions_to_use_tool(read, {"file_path": str(target)}, ask_context))
    write_decision = asyncio.run(has_permissions_to_use_tool(write, {"file_path": str(ask_context.config.cwd / "new.txt"), "content": "x"}, ask_context))

    assert ask_context.get_app_state().tool_permission_context.mode == "ask"
    assert read_decision.behavior == "allow"
    assert write_decision.behavior == "deny"
    assert "File write requires approval" in (write_decision.message or "")

    approved_context = make_context(tmp_path)
    approved_context.permission_callback = lambda *_: PermissionDecision.allow()
    approved = asyncio.run(has_permissions_to_use_tool(write, {"file_path": str(approved_context.config.cwd / "approved.txt"), "content": "x"}, approved_context))
    assert approved.behavior == "allow"

    bypass_context = make_context(tmp_path, ToolPermissionContext(mode="bypass"))
    bypassed = asyncio.run(has_permissions_to_use_tool(write, {"file_path": str(bypass_context.config.cwd / "bypassed.txt"), "content": "x"}, bypass_context))
    assert bypassed.behavior == "allow"


def test_permission_resolver_supports_source_mode_aliases(tmp_path: Path) -> None:
    """验证 ``permission resolver supports source mode aliases`` 场景的行为、消息形状和关键不变量。"""
    read = FileReadTool()
    write = FileWriteTool()
    bash = BashTool()
    default_context = make_context(tmp_path, ToolPermissionContext(mode="default"))
    default_decision = asyncio.run(
        has_permissions_to_use_tool(
            write,
            {"file_path": str(default_context.config.cwd / "default.txt"), "content": "x"},
            default_context,
        )
    )
    accept_context = make_context(tmp_path, ToolPermissionContext(mode="acceptEdits"))
    accept_decision = asyncio.run(
        has_permissions_to_use_tool(
            write,
            {"file_path": str(accept_context.config.cwd / "accepted.txt"), "content": "x"},
            accept_context,
        )
    )
    bypass_context = make_context(tmp_path, ToolPermissionContext(mode="bypassPermissions"))
    bypass_decision = asyncio.run(has_permissions_to_use_tool(bash, {"command": "npm publish --dry-run"}, bypass_context))
    plan_context = make_context(tmp_path, ToolPermissionContext(mode="plan"))
    plan_file = plan_context.config.cwd / "plan.txt"
    plan_file.write_text("ok", encoding="utf-8")
    plan_read = asyncio.run(has_permissions_to_use_tool(read, {"file_path": str(plan_file)}, plan_context))
    plan_write = asyncio.run(
        has_permissions_to_use_tool(
            write,
            {"file_path": str(plan_context.config.cwd / "plan-write.txt"), "content": "x"},
            plan_context,
        )
    )
    dont_ask_context = make_context(tmp_path, ToolPermissionContext(mode="dontAsk"))
    dont_ask = asyncio.run(
        has_permissions_to_use_tool(
            write,
            {"file_path": str(dont_ask_context.config.cwd / "nope.txt"), "content": "x"},
            dont_ask_context,
        )
    )

    assert default_decision.behavior == "deny"
    assert accept_decision.behavior == "allow"
    assert bypass_decision.behavior == "allow"
    assert plan_read.behavior == "allow"
    assert plan_write.behavior == "deny"
    assert dont_ask.behavior == "deny"


def test_file_path_safety_blocks_sensitive_files_even_in_bypass(tmp_path: Path) -> None:
    """验证 ``file path safety blocks sensitive files even in bypass`` 场景的行为、消息形状和关键不变量。"""
    write = FileWriteTool()
    ask_context = make_context(tmp_path)
    sensitive = ask_context.config.cwd / ".git" / "config"

    asked = asyncio.run(has_permissions_to_use_tool(write, {"file_path": str(sensitive), "content": "x"}, ask_context))

    assert asked.behavior == "deny"
    assert "sensitive file" in (asked.message or "")

    bypass_context = make_context(tmp_path, ToolPermissionContext(mode="bypass"))
    bypass_sensitive = bypass_context.config.cwd / ".claude" / "settings.json"
    bypassed = asyncio.run(has_permissions_to_use_tool(write, {"file_path": str(bypass_sensitive), "content": "x"}, bypass_context))

    assert bypassed.behavior == "deny"
    assert "sensitive file" in (bypassed.message or "")


def test_edit_permission_asks_and_bypass_allows_safe_edits(tmp_path: Path) -> None:
    """验证 ``edit permission asks and bypass allows safe edits`` 场景的行为、消息形状和关键不变量。"""
    edit = EditTool()
    ask_context = make_context(tmp_path)
    target = ask_context.config.cwd / "sample.txt"
    target.write_text("old\n", encoding="utf-8")

    ask_decision = asyncio.run(
        has_permissions_to_use_tool(
            edit,
            {"file_path": str(target), "old_string": "old", "new_string": "new"},
            ask_context,
        )
    )
    approved_context = make_context(tmp_path)
    approved_context.permission_callback = lambda *_: PermissionDecision.allow()
    approved = asyncio.run(
        has_permissions_to_use_tool(
            edit,
            {"file_path": str(target), "old_string": "old", "new_string": "new"},
            approved_context,
        )
    )
    bypass_context = make_context(tmp_path, ToolPermissionContext(mode="bypass"))
    bypassed = asyncio.run(
        has_permissions_to_use_tool(
            edit,
            {"file_path": str(target), "old_string": "old", "new_string": "new"},
            bypass_context,
        )
    )
    sensitive = asyncio.run(
        has_permissions_to_use_tool(
            edit,
            {"file_path": str(ask_context.config.cwd / ".vscode" / "settings.json"), "old_string": "", "new_string": "{}"},
            bypass_context,
        )
    )

    assert ask_decision.behavior == "deny"
    assert approved.behavior == "allow"
    assert bypassed.behavior == "allow"
    assert sensitive.behavior == "deny"
    assert "sensitive file" in (sensitive.message or "")


def test_additional_working_directory_allows_read_and_writes_follow_mode(tmp_path: Path) -> None:
    """验证 ``additional working directory allows read and writes follow mode`` 场景的行为、消息形状和关键不变量。"""
    read = FileReadTool()
    write = FileWriteTool()
    extra = tmp_path / "extra"
    extra.mkdir()
    readable = extra / "note.txt"
    readable.write_text("ok", encoding="utf-8")
    context = make_context(
        tmp_path,
        ToolPermissionContext(
            additional_working_directories={str(extra): "extra"},
        ),
    )

    read_decision = asyncio.run(has_permissions_to_use_tool(read, {"file_path": str(readable)}, context))
    write_decision = asyncio.run(has_permissions_to_use_tool(write, {"file_path": str(extra / "new.txt"), "content": "x"}, context))
    bypass_context = make_context(
        tmp_path,
        ToolPermissionContext(
            mode="bypass",
            additional_working_directories={str(extra): "extra"},
        ),
    )
    bypass_write = asyncio.run(has_permissions_to_use_tool(write, {"file_path": str(extra / "new.txt"), "content": "x"}, bypass_context))

    assert read_decision.behavior == "allow"
    assert write_decision.behavior == "deny"
    assert bypass_write.behavior == "allow"


def test_file_path_validation_blocks_glob_and_shell_expansion_writes(tmp_path: Path) -> None:
    """验证 ``file path validation blocks glob and shell expansion writes`` 场景的行为、消息形状和关键不变量。"""
    write = FileWriteTool()
    context = make_context(tmp_path, ToolPermissionContext(mode="bypass"))

    glob_decision = asyncio.run(has_permissions_to_use_tool(write, {"file_path": str(context.config.cwd / "*.txt"), "content": "x"}, context))
    expansion_decision = asyncio.run(has_permissions_to_use_tool(write, {"file_path": "$HOME/secret.txt", "content": "x"}, context))

    assert glob_decision.behavior == "deny"
    assert "Glob patterns" in (glob_decision.message or "")
    assert expansion_decision.behavior == "deny"
    assert "Shell expansion syntax" in (expansion_decision.message or "")


def test_bash_ask_and_bypass_modes(tmp_path: Path) -> None:
    """验证 ``bash ask and bypass modes`` 场景的行为、消息形状和关键不变量。"""
    bash = BashTool()
    ask_context = make_context(tmp_path)
    bypass_context = make_context(tmp_path, ToolPermissionContext(mode="bypass"))

    allowed = asyncio.run(has_permissions_to_use_tool(bash, {"command": "ls -la"}, ask_context))
    asked = asyncio.run(has_permissions_to_use_tool(bash, {"command": "npm publish --dry-run"}, ask_context))
    bypassed = asyncio.run(has_permissions_to_use_tool(bash, {"command": "npm publish --dry-run"}, bypass_context))

    assert allowed.behavior == "allow"
    assert asked.behavior == "deny"
    assert "requires approval" in (asked.message or "")
    assert bypassed.behavior == "allow"


def test_bash_output_redirection_uses_path_safety(tmp_path: Path) -> None:
    """验证 ``bash output redirection uses path safety`` 场景的行为、消息形状和关键不变量。"""
    bash = BashTool()
    context = make_context(tmp_path, ToolPermissionContext(mode="bypass"))

    decision = asyncio.run(has_permissions_to_use_tool(bash, {"command": "echo hacked > .claude/settings.json"}, context))

    assert decision.behavior == "deny"
    assert "sensitive file" in (decision.message or "")


def test_bash_cd_with_redirection_and_process_substitution_are_bypass_immune(tmp_path: Path) -> None:
    """验证 ``bash cd with redirection and process substitution are bypass immune`` 场景的行为、消息形状和关键不变量。"""
    bash = BashTool()
    context = make_context(tmp_path, ToolPermissionContext(mode="bypass"))

    cd_redirect = asyncio.run(has_permissions_to_use_tool(bash, {"command": "cd .claude && echo hacked > settings.json"}, context))
    process_substitution = asyncio.run(has_permissions_to_use_tool(bash, {"command": "echo secret > >(tee .git/config)"}, context))

    assert cd_redirect.behavior == "deny"
    assert "change directories" in (cd_redirect.message or "")
    assert process_substitution.behavior == "deny"
    assert "Process substitution" in (process_substitution.message or "")


def test_bash_path_commands_use_filesystem_safety(tmp_path: Path) -> None:
    """验证 ``bash path commands use filesystem safety`` 场景的行为、消息形状和关键不变量。"""
    bash = BashTool()
    context = make_context(tmp_path, ToolPermissionContext(mode="bypass"))

    rm_sensitive = asyncio.run(has_permissions_to_use_tool(bash, {"command": "rm .git/config"}, context))
    wrapped_rm_sensitive = asyncio.run(has_permissions_to_use_tool(bash, {"command": "timeout 5 rm .git/config"}, context))
    chmod_sensitive = asyncio.run(has_permissions_to_use_tool(bash, {"command": "chmod 600 .claude/settings.json"}, context))
    dd_sensitive = asyncio.run(has_permissions_to_use_tool(bash, {"command": "dd if=/dev/zero of=.git/config bs=1 count=1"}, context))
    touch_glob = asyncio.run(has_permissions_to_use_tool(bash, {"command": "touch *.txt"}, context))
    cd_rm = asyncio.run(has_permissions_to_use_tool(bash, {"command": "cd subdir && rm file.txt"}, context))
    safe_mkdir = asyncio.run(has_permissions_to_use_tool(bash, {"command": "mkdir generated"}, context))

    assert rm_sensitive.behavior == "deny"
    assert "sensitive file" in (rm_sensitive.message or "")
    assert wrapped_rm_sensitive.behavior == "deny"
    assert chmod_sensitive.behavior == "deny"
    assert dd_sensitive.behavior == "deny"
    assert touch_glob.behavior == "deny"
    assert "Glob patterns" in (touch_glob.message or "")
    assert cd_rm.behavior == "deny"
    assert "change directories" in (cd_rm.message or "")
    assert safe_mkdir.behavior == "allow"


def test_bash_background_large_output_sleep_and_exit_code(tmp_path: Path) -> None:
    """验证 ``bash background large output sleep and exit code`` 场景的行为、消息形状和关键不变量。"""
    context = make_context(tmp_path, ToolPermissionContext(mode="bypass"))
    bash = BashTool()
    parent = create_assistant_message("running")

    sleep_validation = asyncio.run(bash.validate_input({"command": "sleep 3"}, context))
    background_sleep_validation = asyncio.run(bash.validate_input({"command": "sleep 3", "run_in_background": True}, context))
    exit_result = asyncio.run(bash.call({"command": "sh -c 'exit 7'"}, context, None, parent))
    exit_block = bash.map_tool_result_to_tool_result_block_param(exit_result.data, "toolu_exit")
    large_result = asyncio.run(bash.call({"command": "python3 -c \"print('x'*31050)\""}, context, None, parent))
    large_block = bash.map_tool_result_to_tool_result_block_param(large_result.data, "toolu_large")

    async def run_background():
        """为当前测试提供 ``run_background`` 辅助行为。"""
        result = await bash.call(
            {"command": "python3 -c \"import time; time.sleep(0.05); print('done')\"", "run_in_background": True},
            context,
            None,
            parent,
        )
        task_id = result.data["backgroundTaskId"]
        await asyncio.wait_for(context.background_tasks[task_id]["task"], timeout=1)
        return result

    background_result = asyncio.run(run_background())
    background_block = bash.map_tool_result_to_tool_result_block_param(background_result.data, "toolu_bg")
    output_path = Path(background_result.data["outputPath"])

    assert sleep_validation.result is False
    assert "Run blocking commands in the background" in (sleep_validation.message or "")
    assert background_sleep_validation.result is True
    assert "exit code 7" in exit_block["content"]
    assert "<persisted-output" in large_block["content"]
    assert Path(large_result.data["persistedOutputPath"]).exists()
    assert "Command running in background with ID" in background_block["content"]
    assert output_path.exists()
    assert "done" in output_path.read_text(encoding="utf-8")


def test_bash_internal_timeout_marks_interrupted_and_returns_output(tmp_path: Path) -> None:
    """验证 ``bash internal timeout marks interrupted and returns output`` 场景的行为、消息形状和关键不变量。"""
    context = make_context(tmp_path, ToolPermissionContext(mode="bypass"))
    bash = BashTool()
    parent = create_assistant_message("running")

    result = asyncio.run(
        bash.call(
            {"command": "python3 -c \"import time; print('before', flush=True); time.sleep(2)\"", "timeout": 50},
            context,
            None,
            parent,
        )
    )
    block = bash.map_tool_result_to_tool_result_block_param(result.data, "toolu_timeout")

    assert result.data["interrupted"] is True
    assert result.data["timeoutMs"] == 50
    assert "before" in block["content"]
    assert "Command was aborted before completion" in block["content"]
    assert block["is_error"] is True


def test_file_read_tool_result_shape_and_state(tmp_path: Path) -> None:
    """验证 ``file read tool result shape and state`` 场景的行为、消息形状和关键不变量。"""
    context = make_context(tmp_path)
    target = context.config.cwd / "sample.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")
    tool = FileReadTool()
    parent = create_assistant_message("reading")

    result = asyncio.run(tool.call({"file_path": str(target)}, context, None, parent))
    block = tool.map_tool_result_to_tool_result_block_param(result.data, "toolu_1")

    assert block["tool_use_id"] == "toolu_1"
    assert "1\talpha" in block["content"]
    assert str(target.resolve()) in context.read_file_state


def test_file_read_permissions_and_relative_paths_use_agent_cwd(tmp_path: Path) -> None:
    """验证 ``file read permissions and relative paths use agent cwd`` 场景的行为、消息形状和关键不变量。"""
    context = make_context(tmp_path)
    target = context.config.cwd / "relative.txt"
    target.write_text("inside\n", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    tool = FileReadTool()
    parent = create_assistant_message("reading")

    inside_decision = asyncio.run(has_permissions_to_use_tool(tool, {"file_path": "relative.txt"}, context))
    outside_decision = asyncio.run(has_permissions_to_use_tool(tool, {"file_path": str(outside)}, context))
    result = asyncio.run(tool.call({"file_path": "relative.txt"}, context, None, parent))

    assert inside_decision.behavior == "allow"
    assert outside_decision.behavior == "deny"
    assert result.data["file"]["content"] == "inside"


def test_file_read_handles_notebook_image_pdf_and_binary_shapes(tmp_path: Path) -> None:
    """验证 ``file read handles notebook image pdf and binary shapes`` 场景的行为、消息形状和关键不变量。"""
    context = make_context(tmp_path)
    tool = FileReadTool()
    parent = create_assistant_message("reading")
    notebook = context.config.cwd / "demo.ipynb"
    notebook.write_text(
        json.dumps({"cells": [{"cell_type": "code", "source": ["print('hi')\n"], "outputs": [{"text": ["hi\n"]}]}]}),
        encoding="utf-8",
    )
    image = context.config.cwd / "pixel.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8 + (1).to_bytes(4, "big") + (1).to_bytes(4, "big"))
    pdf = context.config.cwd / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n1 0 obj <<>> stream\n(Hello PDF)\nendstream\n%%EOF")
    binary = context.config.cwd / "blob.bin"
    binary.write_bytes(b"\x00\x01\x02")

    notebook_result = asyncio.run(tool.call({"file_path": str(notebook)}, context, None, parent))
    image_block = tool.map_tool_result_to_tool_result_block_param(
        asyncio.run(tool.call({"file_path": str(image)}, context, None, parent)).data,
        "toolu_image",
    )
    pdf_block = tool.map_tool_result_to_tool_result_block_param(
        asyncio.run(tool.call({"file_path": str(pdf)}, context, None, parent)).data,
        "toolu_pdf",
    )
    binary_block = tool.map_tool_result_to_tool_result_block_param(
        asyncio.run(tool.call({"file_path": str(binary)}, context, None, parent)).data,
        "toolu_binary",
    )

    assert "print('hi')" in notebook_result.data["file"]["content"]
    assert isinstance(image_block["content"], list)
    assert image_block["content"][1]["type"] == "image"
    assert "PDF file read" in pdf_block["content"]
    assert "Hello PDF" in pdf_block["content"]
    assert "Binary file" in binary_block["content"]


def test_file_read_dedup_partial_view_and_edit_guard(tmp_path: Path) -> None:
    """验证 ``file read dedup partial view and edit guard`` 场景的行为、消息形状和关键不变量。"""
    context = make_context(tmp_path, ToolPermissionContext(mode="bypass"))
    tool = FileReadTool()
    edit = EditTool()
    parent = create_assistant_message("reading")
    target = context.config.cwd / "long.txt"
    target.write_text("one\ntwo\nthree\n", encoding="utf-8")

    first = asyncio.run(tool.call({"file_path": str(target)}, context, None, parent))
    second = asyncio.run(tool.call({"file_path": str(target)}, context, None, parent))
    partial = asyncio.run(tool.call({"file_path": str(target), "offset": 2, "limit": 1}, context, None, parent))
    partial_block = tool.map_tool_result_to_tool_result_block_param(partial.data, "toolu_partial")
    edit_validation = asyncio.run(
        edit.validate_input(
            {"file_path": str(target), "old_string": "two", "new_string": "TWO"},
            context,
        )
    )

    assert first.data["type"] == "text"
    assert second.data["type"] == "file_unchanged"
    assert "File has more lines after line 2" in partial_block["content"]
    assert edit_validation.result is False
    assert "Read it first" in (edit_validation.message or "")


def test_write_and_edit_return_structured_patch(tmp_path: Path) -> None:
    """验证 ``write and edit return structured patch`` 场景的行为、消息形状和关键不变量。"""
    context = make_context(tmp_path)
    parent = create_assistant_message("editing")
    write = FileWriteTool()
    edit = EditTool()
    target = context.config.cwd / "patch.txt"

    write_result = asyncio.run(write.call({"file_path": str(target), "content": "one\n"}, context, None, parent))
    edit_result = asyncio.run(
        edit.call(
            {"file_path": str(target), "old_string": "one", "new_string": "two"},
            context,
            None,
            parent,
        )
    )

    assert write_result.data["structuredPatch"][0]["patch"].startswith("---")
    assert "-one" in edit_result.data["structuredPatch"][0]["patch"]
    assert "+two" in edit_result.data["structuredPatch"][0]["patch"]


def test_edit_quote_normalization_trailing_newline_delete_and_crlf_preserve(tmp_path: Path) -> None:
    """验证 ``edit quote normalization trailing newline delete and crlf preserve`` 场景的行为、消息形状和关键不变量。"""
    context = make_context(tmp_path, ToolPermissionContext(mode="bypass"))
    read = FileReadTool()
    edit = EditTool()
    parent = create_assistant_message("editing")
    quote_file = context.config.cwd / "quotes.txt"
    quote_file.write_text("title = \u201cOld\u201d\n", encoding="utf-8")
    crlf_file = context.config.cwd / "crlf.txt"
    crlf_file.write_bytes(b"alpha\r\nremove\r\nomega\r\n")

    asyncio.run(read.call({"file_path": str(quote_file)}, context, None, parent))
    quote_validation = asyncio.run(
        edit.validate_input(
            {"file_path": str(quote_file), "old_string": 'title = "Old"', "new_string": 'title = "New"'},
            context,
        )
    )
    quote_result = asyncio.run(
        edit.call(
            {"file_path": str(quote_file), "old_string": 'title = "Old"', "new_string": 'title = "New"'},
            context,
            None,
            parent,
        )
    )
    asyncio.run(read.call({"file_path": str(crlf_file)}, context, None, parent))
    delete_result = asyncio.run(
        edit.call(
            {"file_path": str(crlf_file), "old_string": "remove", "new_string": ""},
            context,
            None,
            parent,
        )
    )

    assert quote_validation.result is True
    assert quote_validation.meta["actualOldString"] == "title = \u201cOld\u201d"
    assert quote_result.data["oldString"] == "title = \u201cOld\u201d"
    assert quote_file.read_text(encoding="utf-8") == "title = \u201cNew\u201d\n"
    assert b"\r\n" in crlf_file.read_bytes()
    assert crlf_file.read_bytes() == b"alpha\r\nomega\r\n"
    assert "-remove" in delete_result.data["structuredPatch"][0]["patch"]


def test_tool_prompts_preserve_source_critical_sections() -> None:
    """验证 ``tool prompts preserve source critical sections`` 场景的行为、消息形状和关键不变量。"""
    edit_prompt = edit_tool_prompt()
    bash_prompt = bash_tool_prompt()
    read_prompt = read_tool_prompt()

    assert "line number + tab" in edit_prompt
    assert "spaces + line number + arrow" not in edit_prompt
    assert "# Committing changes with git" in bash_prompt
    assert "Git Safety Protocol:" in bash_prompt
    assert "# Creating pull requests" in bash_prompt
    assert "NEVER commit changes unless the user explicitly asks you to." in bash_prompt
    assert '- This tool can read PDF files (.pdf). For large PDFs (more than 10 pages), you MUST provide the pages parameter to read specific page ranges (e.g., pages: "1-5"). Reading a large PDF without the pages parameter will fail. Maximum 20 pages per request.' in read_prompt


def test_web_search_prompt_permissions_validation_and_result_format(tmp_path: Path, monkeypatch) -> None:
    """验证 ``web search prompt permissions validation and result format`` 场景的行为、消息形状和关键不变量。"""
    monkeypatch.setenv("CLAUDE_CODE_OVERRIDE_DATE", "2026-06-14")
    tool = WebSearchTool()
    ask_context = make_context(tmp_path)
    bypass_context = make_context(tmp_path, ToolPermissionContext(mode="bypass"))
    parent = create_assistant_message("searching")

    prompt = asyncio.run(tool.prompt())
    ask_decision = asyncio.run(has_permissions_to_use_tool(tool, {"query": "React docs"}, ask_context))
    bypass_decision = asyncio.run(has_permissions_to_use_tool(tool, {"query": "React docs"}, bypass_context))
    invalid = asyncio.run(tool.validate_input({"query": "x"}, bypass_context))
    conflicting_domains = asyncio.run(
        tool.validate_input(
            {"query": "React docs", "allowed_domains": ["react.dev"], "blocked_domains": ["example.com"]},
            bypass_context,
        )
    )
    bypass_context.web_search_handler = lambda args: [
        {"title": "React Docs", "url": "https://react.dev/learn"},
        {"title": "API Reference", "url": "https://react.dev/reference/react"},
    ]
    result = asyncio.run(tool.call({"query": "React docs"}, bypass_context, None, parent))
    block = tool.map_tool_result_to_tool_result_block_param(result.data, "toolu_search")

    assert "The current month is June 2026" in prompt
    assert "Sources:" in prompt
    assert ask_decision.behavior == "deny"
    assert "WebSearchTool requires permission" in (ask_decision.message or "")
    assert bypass_decision.behavior == "allow"
    assert invalid.result is False
    assert invalid.message == "Error: Missing query"
    assert conflicting_domains.result is False
    assert "Cannot specify both allowed_domains and blocked_domains" in (conflicting_domains.message or "")
    assert block["tool_use_id"] == "toolu_search"
    assert 'Web search results for query: "React docs"' in block["content"]
    assert 'Links: [{"title":"React Docs","url":"https://react.dev/learn"}' in block["content"]
    assert "REMINDER: You MUST include the sources above" in block["content"]


def test_web_fetch_permissions_validation_processing_and_redirect(tmp_path: Path) -> None:
    """验证 ``web fetch permissions validation processing and redirect`` 场景的行为、消息形状和关键不变量。"""
    tool = WebFetchTool()
    ask_context = make_context(tmp_path)
    bypass_context = make_context(tmp_path, ToolPermissionContext(mode="bypass"))
    parent = create_assistant_message("fetching")

    prompt = asyncio.run(tool.prompt())
    preapproved = asyncio.run(has_permissions_to_use_tool(tool, {"url": "https://docs.python.org/3/", "prompt": "summarize"}, ask_context))
    normal_ask = asyncio.run(has_permissions_to_use_tool(tool, {"url": "https://example.com/page", "prompt": "summarize"}, ask_context))
    normal_bypass = asyncio.run(has_permissions_to_use_tool(tool, {"url": "https://example.com/page", "prompt": "summarize"}, bypass_context))
    invalid = asyncio.run(tool.validate_input({"url": "ftp://example.com/file", "prompt": "summarize"}, bypass_context))

    bypass_context.web_fetch_handler = lambda url: {
        "bytes": 11,
        "code": 200,
        "codeText": "OK",
        "contentType": "text/html",
        "content": "Fetched content",
    }
    bypass_context.web_fetch_apply_handler = lambda prompt, content, is_preapproved: f"applied:{prompt}:{content}:{is_preapproved}"
    result = asyncio.run(tool.call({"url": "https://example.com/page", "prompt": "summarize"}, bypass_context, None, parent))
    block = tool.map_tool_result_to_tool_result_block_param(result.data, "toolu_fetch")

    model_context = make_context(tmp_path, ToolPermissionContext(mode="bypass"))
    model_context.model_provider = FakeModelProvider(["model summary"])
    model_context.web_fetch_model = "small-fast-model"
    model_context.web_fetch_handler = lambda url: {
        "bytes": 5,
        "code": 200,
        "codeText": "OK",
        "contentType": "text/plain",
        "content": "hello",
    }
    model_result = asyncio.run(tool.call({"url": "https://example.com/model", "prompt": "summarize"}, model_context, None, parent))

    bypass_context.web_fetch_handler = lambda url: {
        "type": "redirect",
        "originalUrl": "https://example.com/start",
        "redirectUrl": "https://other.example/final",
        "statusCode": 302,
    }
    redirect = asyncio.run(tool.call({"url": "https://example.com/start", "prompt": "summarize"}, bypass_context, None, parent))

    assert preapproved.behavior == "allow"
    assert "IMPORTANT: WebFetch WILL FAIL for authenticated or private URLs." in prompt
    assert "HTTP URLs will be automatically upgraded to HTTPS" in prompt
    assert normal_ask.behavior == "deny"
    assert "WebFetch" in (normal_ask.message or "")
    assert normal_bypass.behavior == "allow"
    assert invalid.result is False
    assert invalid.meta == {"reason": "invalid_url"}
    assert block == {"tool_use_id": "toolu_fetch", "type": "tool_result", "content": "applied:summarize:Fetched content:False"}
    assert model_result.data["result"] == "model summary"
    assert model_context.model_provider.calls[0]["options"]["querySource"] == "web_fetch_apply"
    assert model_context.model_provider.calls[0]["tools"] == []
    assert "REDIRECT DETECTED" in redirect.data["result"]
    assert 'Redirect URL: https://other.example/final' in redirect.data["result"]
    assert '- prompt: "summarize"' in redirect.data["result"]


def test_web_fetch_tool_result_in_query_engine_default_tools(tmp_path: Path) -> None:
    """验证 ``web fetch tool result in query engine default tools`` 场景的行为、消息形状和关键不变量。"""
    repo = tmp_path / "repo"
    repo.mkdir()
    provider = FakeModelProvider(
        [
            [
                {
                    "type": "tool_use",
                    "id": "toolu_fetch",
                    "name": "WebFetch",
                    "input": {"url": "https://example.com/page", "prompt": "summarize"},
                }
            ],
            "done",
        ]
    )
    engine = QueryEngine(model_provider=provider, config=KernelConfig(cwd=repo, config_home=tmp_path / ".claude"))
    engine.tool_use_context.app_state.tool_permission_context.mode = "bypass"
    engine.tool_use_context.web_fetch_handler = lambda url: {
        "bytes": 5,
        "code": 200,
        "codeText": "OK",
        "contentType": "text/plain",
        "content": "hello",
    }
    engine.tool_use_context.web_fetch_apply_handler = lambda prompt, content, is_preapproved: "summary from web"

    events = asyncio.run(_collect(engine.submit_message("fetch it", max_turns=3)))

    tool_message = next(event for event in events if event.get("type") == "user")
    assert tool_message["message"]["content"][0]["content"] == "summary from web"
    assert [tool.name for tool in engine.tools][-2:] == ["WebSearch", "WebFetch"]
    assert events[-1]["terminal"]["reason"] == "completed"


def test_write_permission_denied_tool_result_in_query_engine(tmp_path: Path) -> None:
    """验证 ``write permission denied tool result in query engine`` 场景的行为、消息形状和关键不变量。"""
    target = tmp_path / "repo" / "created.txt"
    provider = FakeModelProvider(
        [
            [
                {
                    "type": "tool_use",
                    "id": "toolu_write",
                    "name": "Write",
                    "input": {"file_path": str(target), "content": "hello"},
                }
            ],
            "done",
        ]
    )
    engine = QueryEngine(model_provider=provider, config=KernelConfig(cwd=tmp_path / "repo", config_home=tmp_path / ".claude"))
    events = asyncio.run(_collect(engine.submit_message("write it", max_turns=3)))

    tool_messages = [event for event in events if event.get("type") == "user"]
    assert tool_messages
    block = tool_messages[0]["message"]["content"][0]
    assert block["type"] == "tool_result"
    assert block["is_error"] is True
    assert "Permission denied" in block["content"]
    assert not target.exists()


def test_fake_provider_file_read_follow_up_loop(tmp_path: Path) -> None:
    """验证 ``fake provider file read follow up loop`` 场景的行为、消息形状和关键不变量。"""
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "sample.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")
    provider = FakeModelProvider(
        [
            [
                {
                    "type": "tool_use",
                    "id": "toolu_read",
                    "name": "Read",
                    "input": {"file_path": str(target)},
                }
            ],
            "I read it.",
        ]
    )
    engine = QueryEngine(model_provider=provider, config=KernelConfig(cwd=repo, config_home=tmp_path / ".claude"))
    events = asyncio.run(_collect(engine.submit_message("read sample", max_turns=3)))

    assert [event["type"] for event in events].count("stream_request_start") == 2
    tool_messages = [event for event in events if event.get("type") == "user"]
    assert "alpha" in tool_messages[0]["message"]["content"][0]["content"]
    assert events[-1]["type"] == "terminal"
    assert events[-1]["terminal"]["reason"] == "completed"


class ProgressTool(Tool):
    """提供 ``ProgressTool`` 测试替身，用于隔离外部依赖或触发指定分支。"""
    name = "Progress"
    input_schema = {}

    async def check_permissions(self, input: dict, context: ToolUseContext) -> PermissionDecision:
        """返回测试场景需要的固定权限决策。"""
        return PermissionDecision.allow()

    async def call(self, args: dict, context: ToolUseContext, can_use_tool, parent_message, on_progress=None) -> ToolResult:
        """执行测试工具并返回预设结果或异常。"""
        if on_progress:
            on_progress({"message": "halfway"})
        await asyncio.sleep(0)
        return ToolResult("ok")


class SlowTool(ProgressTool):
    """提供 ``SlowTool`` 测试替身，用于隔离外部依赖或触发指定分支。"""
    name = "Slow"

    async def call(self, args: dict, context: ToolUseContext, can_use_tool, parent_message, on_progress=None) -> ToolResult:
        """执行测试工具并返回预设结果或异常。"""
        await asyncio.sleep(0.05)
        return ToolResult("too late")


class CancellableTool(ProgressTool):
    """提供 ``CancellableTool`` 测试替身，用于隔离外部依赖或触发指定分支。"""
    name = "Cancellable"

    def __init__(self) -> None:
        """初始化测试替身的计数器和可观察状态。"""
        self.cancelled = False

    async def call(self, args: dict, context: ToolUseContext, can_use_tool, parent_message, on_progress=None) -> ToolResult:
        """执行测试工具并返回预设结果或异常。"""
        if on_progress:
            on_progress({"message": "started"})
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return ToolResult("finished")


class ConcurrentProbeTool(Tool):
    """提供 ``ConcurrentProbeTool`` 测试替身，用于隔离外部依赖或触发指定分支。"""
    name = "ConcurrentProbe"
    input_schema = {"label": str}
    required_fields = ("label",)

    def __init__(self) -> None:
        """初始化测试替身的计数器和可观察状态。"""
        self.active = 0
        self.max_active = 0

    def is_read_only(self, input: dict) -> bool:
        """声明测试工具是否只读。"""
        return True

    def is_concurrency_safe(self, input: dict) -> bool:
        """声明测试工具是否允许并发执行。"""
        return True

    async def check_permissions(self, input: dict, context: ToolUseContext) -> PermissionDecision:
        """返回测试场景需要的固定权限决策。"""
        return PermissionDecision.allow()

    async def call(self, args: dict, context: ToolUseContext, can_use_tool, parent_message, on_progress=None) -> ToolResult:
        """执行测试工具并返回预设结果或异常。"""
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.05)
        self.active -= 1
        return ToolResult(args["label"])


class UnstableValidateTool(Tool):
    """提供 ``UnstableValidateTool`` 测试替身，用于隔离外部依赖或触发指定分支。"""
    name = "UnstableValidate"
    input_schema = {"label": str}
    required_fields = ("label",)

    def is_read_only(self, input: dict) -> bool:
        """声明测试工具是否只读。"""
        return True

    def is_concurrency_safe(self, input: dict) -> bool:
        """声明测试工具是否允许并发执行。"""
        return True

    async def check_permissions(self, input: dict, context: ToolUseContext) -> PermissionDecision:
        """返回测试场景需要的固定权限决策。"""
        return PermissionDecision.allow()

    async def validate_input(self, input: dict, context: ToolUseContext):
        """执行当前测试工具的输入校验分支。"""
        if input["label"] == "bad":
            raise RuntimeError("validation exploded")
        return await super().validate_input(input, context)

    async def call(self, args: dict, context: ToolUseContext, can_use_tool, parent_message, on_progress=None) -> ToolResult:
        """执行测试工具并返回预设结果或异常。"""
        return ToolResult(args["label"])


class ParentProbeTool(Tool):
    """提供 ``ParentProbeTool`` 测试替身，用于隔离外部依赖或触发指定分支。"""
    name = "ParentProbe"
    input_schema = {}

    async def check_permissions(self, input: dict, context: ToolUseContext) -> PermissionDecision:
        """返回测试场景需要的固定权限决策。"""
        return PermissionDecision.allow()

    async def call(self, args: dict, context: ToolUseContext, can_use_tool, parent_message, on_progress=None) -> ToolResult:
        """执行测试工具并返回预设结果或异常。"""
        return ToolResult(parent_message["uuid"])


def test_tool_progress_and_timeout_events(tmp_path: Path) -> None:
    """验证 ``tool progress and timeout events`` 场景的行为、消息形状和关键不变量。"""
    repo = tmp_path / "repo"
    repo.mkdir()
    progress_provider = FakeModelProvider(
        [
            [{"type": "tool_use", "id": "toolu_progress", "name": "Progress", "input": {}}],
            "done",
        ]
    )
    progress_engine = QueryEngine(
        model_provider=progress_provider,
        config=KernelConfig(cwd=repo, config_home=tmp_path / ".claude"),
        tools=[ProgressTool()],
    )
    progress_events = asyncio.run(_collect(progress_engine.submit_message("run", max_turns=3)))
    assert any(event.get("type") == "tool_progress" and event["progress"]["message"] == "halfway" for event in progress_events)

    slow_provider = FakeModelProvider(
        [
            [{"type": "tool_use", "id": "toolu_slow", "name": "Slow", "input": {}}],
            "done",
        ]
    )
    slow_engine = QueryEngine(
        model_provider=slow_provider,
        config=KernelConfig(cwd=repo, config_home=tmp_path / ".claude2"),
        tools=[SlowTool()],
    )
    slow_engine.tool_use_context.tool_timeout_seconds = 0.001
    slow_events = asyncio.run(_collect(slow_engine.submit_message("run", max_turns=3)))
    tool_message = next(event for event in slow_events if event.get("type") == "user")
    block = tool_message["message"]["content"][0]
    assert block["is_error"] is True
    assert "timed out" in block["content"]


def test_query_engine_cancel_aborts_model_stream(tmp_path: Path) -> None:
    """验证 ``query engine cancel aborts model stream`` 场景的行为、消息形状和关键不变量。"""
    class AbortAwareProvider:
        """提供 ``AbortAwareProvider`` 测试替身，用于隔离外部依赖或触发指定分支。"""
        def __init__(self) -> None:
            """初始化测试替身的计数器和可观察状态。"""
            self.calls = []

        async def stream(self, *, messages, system_prompt, tools, options):
            """产生当前测试场景预设的模型消息或错误。"""
            self.calls.append(options)
            while not options["abortSignal"].aborted:
                await asyncio.sleep(0.01)
            raise asyncio.CancelledError("cancelled")
            if False:
                yield create_assistant_message("unreachable")

    repo = tmp_path / "repo"
    repo.mkdir()
    provider = AbortAwareProvider()
    engine = QueryEngine(
        model_provider=provider,
        config=KernelConfig(cwd=repo, config_home=tmp_path / ".claude"),
        tools=[],
    )

    async def run():
        """运行当前测试所需的异步场景并返回事件。"""
        events = []
        async for event in engine.submit_message("run", max_turns=1):
            events.append(event)
            if event.get("type") == "stream_request_start":
                engine.cancel("user_cancelled")
        return events

    events = asyncio.run(run())

    assert provider.calls
    assert provider.calls[0]["abortSignal"].aborted is True
    assert events[-1]["type"] == "terminal"
    assert events[-1]["terminal"]["reason"] == "aborted"
    assert events[-1]["terminal"]["message"] == "user_cancelled"
    assert any(
        event.get("type") == "user"
        and event["message"]["content"] == [{"type": "text", "text": "[Request interrupted by user]"}]
        for event in events
    )


def test_query_engine_cancel_aborts_inflight_tool(tmp_path: Path) -> None:
    """验证 ``query engine cancel aborts inflight tool`` 场景的行为、消息形状和关键不变量。"""
    repo = tmp_path / "repo"
    repo.mkdir()
    tool = CancellableTool()
    provider = FakeModelProvider(
        [
            [{"type": "tool_use", "id": "toolu_cancel", "name": "Cancellable", "input": {}}],
            "done",
        ]
    )
    engine = QueryEngine(
        model_provider=provider,
        config=KernelConfig(cwd=repo, config_home=tmp_path / ".claude"),
        tools=[tool],
    )

    async def run():
        """运行当前测试所需的异步场景并返回事件。"""
        events = []
        async for event in engine.submit_message("run", max_turns=3):
            events.append(event)
            if event.get("type") == "tool_progress":
                engine.cancel("stop_tools")
        return events

    events = asyncio.run(run())

    assert tool.cancelled is True
    assert events[-1]["type"] == "terminal"
    assert events[-1]["terminal"]["reason"] == "aborted"
    assert events[-1]["terminal"]["message"] == "stop_tools"
    tool_result = next(
        event["message"]["content"][0]
        for event in events
        if event.get("type") == "user" and event["message"]["content"][0].get("type") == "tool_result"
    )
    assert tool_result == {
        "type": "tool_result",
        "tool_use_id": "toolu_cancel",
        "content": "Interrupted by user",
        "is_error": True,
    }
    assert any(
        event.get("type") == "user"
        and event["message"]["content"] == [{"type": "text", "text": "[Request interrupted by user for tool use]"}]
        for event in events
    )


def test_tool_progress_is_not_replayed_to_model_context(tmp_path: Path) -> None:
    """验证 ``tool progress is not replayed to model context`` 场景的行为、消息形状和关键不变量。"""
    repo = tmp_path / "repo"
    repo.mkdir()
    provider = FakeModelProvider(
        [
            [{"type": "tool_use", "id": "toolu_progress", "name": "Progress", "input": {}}],
            "done",
        ]
    )
    engine = QueryEngine(
        model_provider=provider,
        config=KernelConfig(cwd=repo, config_home=tmp_path / ".claude"),
        tools=[ProgressTool()],
    )

    events = asyncio.run(_collect(engine.submit_message("run", max_turns=3)))

    assert any(event.get("type") == "tool_progress" for event in events)
    second_call_messages = json.dumps(provider.calls[1]["messages"], ensure_ascii=False)
    assert "tool_progress" not in second_call_messages


def test_concurrency_safe_tool_batch_runs_concurrently(tmp_path: Path) -> None:
    """验证 ``concurrency safe tool batch runs concurrently`` 场景的行为、消息形状和关键不变量。"""
    repo = tmp_path / "repo"
    repo.mkdir()
    tool = ConcurrentProbeTool()
    provider = FakeModelProvider(
        [
            [
                {"type": "tool_use", "id": "toolu_a", "name": "ConcurrentProbe", "input": {"label": "a"}},
                {"type": "tool_use", "id": "toolu_b", "name": "ConcurrentProbe", "input": {"label": "b"}},
            ],
            "done",
        ]
    )
    engine = QueryEngine(
        model_provider=provider,
        config=KernelConfig(cwd=repo, config_home=tmp_path / ".claude"),
        tools=[tool],
    )

    asyncio.run(_collect(engine.submit_message("run", max_turns=3)))

    assert tool.max_active == 2


def test_tool_validation_exception_in_concurrent_batch_returns_error_without_losing_siblings(tmp_path: Path) -> None:
    """验证 ``tool validation exception in concurrent batch returns error without losing siblings`` 场景的行为、消息形状和关键不变量。"""
    repo = tmp_path / "repo"
    repo.mkdir()
    provider = FakeModelProvider(
        [
            [
                {"type": "tool_use", "id": "toolu_good", "name": "UnstableValidate", "input": {"label": "good"}},
                {"type": "tool_use", "id": "toolu_bad", "name": "UnstableValidate", "input": {"label": "bad"}},
            ],
            "done",
        ]
    )
    engine = QueryEngine(
        model_provider=provider,
        config=KernelConfig(cwd=repo, config_home=tmp_path / ".claude"),
        tools=[UnstableValidateTool()],
    )

    events = asyncio.run(_collect(engine.submit_message("run", max_turns=3)))

    tool_blocks = [event["message"]["content"][0] for event in events if event.get("type") == "user"]
    by_id = {block["tool_use_id"]: block for block in tool_blocks}
    assert by_id["toolu_good"]["content"] == "good"
    assert by_id["toolu_bad"]["is_error"] is True
    assert "validation exploded" in by_id["toolu_bad"]["content"]
    assert events[-1]["terminal"]["reason"] == "completed"


def test_post_tool_hook_exception_does_not_discard_successful_tool_result(tmp_path: Path) -> None:
    """验证 ``post tool hook exception does not discard successful tool result`` 场景的行为、消息形状和关键不变量。"""
    repo = tmp_path / "repo"
    repo.mkdir()
    provider = FakeModelProvider(
        [
            [{"type": "tool_use", "id": "toolu_progress", "name": "Progress", "input": {}}],
            "done",
        ]
    )
    engine = QueryEngine(
        model_provider=provider,
        config=KernelConfig(cwd=repo, config_home=tmp_path / ".claude"),
        tools=[ProgressTool()],
    )

    def failing_post_hook(_input):
        """模拟 ``failing_post_hook`` hook，并返回当前用例需要的控制结果。"""
        raise RuntimeError("post hook exploded")

    engine.tool_use_context.hook_registry.register("PostToolUse", failing_post_hook, matcher="Progress")
    events = asyncio.run(_collect(engine.submit_message("run", max_turns=3)))

    tool_message = next(event for event in events if event.get("type") == "user")
    assert tool_message["message"]["content"][0]["content"] == "ok"
    hook_error = next(event for event in events if event.get("type") == "attachment" and event["message"]["attachment"]["type"] == "hook_execution_error")
    assert "post hook exploded" in hook_error["message"]["content"][0]["text"]


def test_tool_result_parent_matches_origin_assistant_message(tmp_path: Path) -> None:
    """验证 ``tool result parent matches origin assistant message`` 场景的行为、消息形状和关键不变量。"""
    repo = tmp_path / "repo"
    repo.mkdir()
    first = create_assistant_message(
        [{"type": "tool_use", "id": "toolu_one", "name": "ParentProbe", "input": {}}],
        uuid="assistant-one",
    )
    second = create_assistant_message(
        [{"type": "tool_use", "id": "toolu_two", "name": "ParentProbe", "input": {}}],
        uuid="assistant-two",
    )

    class MultiAssistantProvider:
        """提供 ``MultiAssistantProvider`` 测试替身，用于隔离外部依赖或触发指定分支。"""
        def __init__(self):
            """初始化测试替身的计数器和可观察状态。"""
            self.calls = []
            self.count = 0

        async def stream(self, *, messages, system_prompt, tools, options):
            """产生当前测试场景预设的模型消息或错误。"""
            self.calls.append({"messages": messages, "system_prompt": system_prompt, "tools": tools, "options": options})
            self.count += 1
            if self.count == 1:
                yield first
                yield second
            else:
                yield create_assistant_message("done")

    engine = QueryEngine(
        model_provider=MultiAssistantProvider(),
        config=KernelConfig(cwd=repo, config_home=tmp_path / ".claude"),
        tools=[ParentProbeTool()],
    )

    events = asyncio.run(_collect(engine.submit_message("run", max_turns=3)))

    tool_messages = [event for event in events if event.get("type") == "user"]
    assert [message["sourceToolAssistantUUID"] for message in tool_messages] == ["assistant-one", "assistant-two"]
    assert tool_messages[0]["message"]["content"][0]["content"] == "assistant-one"
    assert tool_messages[1]["message"]["content"][0]["content"] == "assistant-two"


def test_max_turns_executes_current_tool_batch_before_stopping(tmp_path: Path) -> None:
    """验证 ``max turns executes current tool batch before stopping`` 场景的行为、消息形状和关键不变量。"""
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "max-turns.txt"
    provider = FakeModelProvider(
        [
            [
                {
                    "type": "tool_use",
                    "id": "toolu_write",
                    "name": "Write",
                    "input": {"file_path": str(target), "content": "done"},
                }
            ],
            "should not be called",
        ]
    )
    engine = QueryEngine(model_provider=provider, config=KernelConfig(cwd=repo, config_home=tmp_path / ".claude"))
    engine.tool_use_context.app_state.tool_permission_context.mode = "bypass"

    events = asyncio.run(_collect(engine.submit_message("write", max_turns=1)))

    assert target.read_text(encoding="utf-8") == "done"
    assert len(provider.calls) == 1
    assert any(event.get("type") == "user" for event in events)
    attachment = next(event for event in events if event.get("type") == "attachment")
    assert attachment["message"]["attachment"]["type"] == "max_turns_reached"
    assert events[-1]["terminal"]["reason"] == "max_turns"
    assert events[-1]["terminal"]["turns"] == 2


def test_tool_use_errors_are_wrapped_like_source(tmp_path: Path) -> None:
    """验证 ``tool use errors are wrapped like source`` 场景的行为、消息形状和关键不变量。"""
    repo = tmp_path / "repo"
    repo.mkdir()
    provider = FakeModelProvider(
        [
            [{"type": "tool_use", "id": "toolu_missing", "name": "MissingTool", "input": {}}],
            "done",
        ]
    )
    engine = QueryEngine(model_provider=provider, config=KernelConfig(cwd=repo, config_home=tmp_path / ".claude"), tools=[])

    events = asyncio.run(_collect(engine.submit_message("run", max_turns=3)))

    tool_message = next(event for event in events if event.get("type") == "user")
    block = tool_message["message"]["content"][0]
    assert block["is_error"] is True
    assert block["content"].startswith("<tool_use_error>")
    assert block["content"].endswith("</tool_use_error>")


def test_query_engine_model_error_yields_system_and_terminal(tmp_path: Path) -> None:
    """验证 ``query engine model error yields system and terminal`` 场景的行为、消息形状和关键不变量。"""
    class ErrorProvider:
        """提供 ``ErrorProvider`` 测试替身，用于隔离外部依赖或触发指定分支。"""
        async def stream(self, *, messages, system_prompt, tools, options):
            """产生当前测试场景预设的模型消息或错误。"""
            raise RuntimeError("network down")
            yield

    repo = tmp_path / "repo"
    repo.mkdir()
    engine = QueryEngine(model_provider=ErrorProvider(), config=KernelConfig(cwd=repo, config_home=tmp_path / ".claude"))
    events = asyncio.run(_collect(engine.submit_message("hi", max_turns=1)))

    assert any(event.get("type") == "system" and event.get("subtype") == "api_error" for event in events)
    assert events[-1]["type"] == "terminal"
    assert events[-1]["terminal"]["reason"] == "error"


def test_query_engine_model_error_closes_partial_tool_use(tmp_path: Path) -> None:
    """验证 ``query engine model error closes partial tool use`` 场景的行为、消息形状和关键不变量。"""
    partial = create_assistant_message(
        [{"type": "tool_use", "id": "toolu_partial", "name": "Read", "input": {"file_path": "/tmp/x"}}],
        uuid="assistant-partial",
    )

    class PartialErrorProvider:
        """提供 ``PartialErrorProvider`` 测试替身，用于隔离外部依赖或触发指定分支。"""
        async def stream(self, *, messages, system_prompt, tools, options):
            """产生当前测试场景预设的模型消息或错误。"""
            yield partial
            raise RuntimeError("stream broke")

    repo = tmp_path / "repo"
    repo.mkdir()
    engine = QueryEngine(
        model_provider=PartialErrorProvider(),
        config=KernelConfig(cwd=repo, config_home=tmp_path / ".claude"),
        tools=[FileReadTool()],
    )

    events = asyncio.run(_collect(engine.submit_message("hi", max_turns=1)))

    tool_result_message = next(
        event
        for event in events
        if event.get("type") == "user" and event["message"]["content"][0].get("type") == "tool_result"
    )
    assert tool_result_message["sourceToolAssistantUUID"] == "assistant-partial"
    assert tool_result_message["message"]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "toolu_partial",
        "content": "stream broke",
        "is_error": True,
    }
    assert events[-1]["terminal"]["reason"] == "error"


def test_query_engine_auto_compacts_before_main_model_request(tmp_path: Path) -> None:
    """验证 ``query engine auto compacts before main model request`` 场景的行为、消息形状和关键不变量。"""
    repo = tmp_path / "repo"
    repo.mkdir()
    provider = FakeModelProvider(
        [
            "<analysis>draft</analysis>\n\n<summary>\n1. Primary Request and Intent:\n   Continue the kernel port.\n</summary>",
            "continued after compact",
        ]
    )
    config = KernelConfig(
        cwd=repo,
        config_home=tmp_path / ".claude",
        context_compaction=ContextCompactionConfig(enabled=True, threshold_tokens=1),
    )
    engine = QueryEngine(model_provider=provider, config=config)

    events = asyncio.run(_collect(engine.submit_message("large original prompt", max_turns=1)))

    compact_events = [event for event in events if event.get("type") == "context_compacted"]
    assert compact_events
    assert provider.calls[0]["options"]["querySource"] == "compact"
    assert provider.calls[0]["tools"] == []
    assert provider.calls[0]["system_prompt"] == ["You are a helpful AI assistant tasked with summarizing conversations."]
    assert provider.calls[1]["options"]["querySource"] == "python-port"
    main_messages_json = json.dumps(provider.calls[1]["messages"], ensure_ascii=False)
    assert "This session is being continued from a previous conversation" in main_messages_json
    assert "large original prompt" not in main_messages_json
    assert [message["type"] for message in engine.mutable_messages] == ["system", "user", "assistant"]


def test_compact_prompt_too_long_retry_partial_keep_and_file_restore(tmp_path: Path) -> None:
    """验证 ``compact prompt too long retry partial keep and file restore`` 场景的行为、消息形状和关键不变量。"""
    repo = tmp_path / "repo"
    repo.mkdir()
    restored_file = repo / "open.txt"
    restored_file.write_text("important file state", encoding="utf-8")
    provider = FakeModelProvider(
        [
            "prompt is too long: 999 tokens > 100 maximum",
            "<analysis>draft</analysis><summary>Recovered summary</summary>",
            "continued",
        ]
    )
    config = KernelConfig(
        cwd=repo,
        config_home=tmp_path / ".claude",
        context_compaction=ContextCompactionConfig(
            enabled=True,
            threshold_tokens=1,
            partial_keep_recent_messages=1,
            post_compact_max_files_to_restore=1,
        ),
    )
    old_messages = [
        create_tool_result_message({"type": "tool_result", "tool_use_id": "toolu_old", "content": "old tool"}, uuid="old-tool"),
        create_assistant_message("old answer", uuid="old-assistant"),
    ]
    engine = QueryEngine(model_provider=provider, config=config, mutable_messages=old_messages)
    engine.tool_use_context.read_file_state[str(restored_file)] = ReadFileStateEntry("important file state", 1)

    events = asyncio.run(_collect(engine.submit_message("latest user request", max_turns=1)))

    compact_event = next(event for event in events if event.get("type") == "context_compacted")
    assert len(provider.calls) == 3
    assert compact_event["boundary"]["compactMetadata"]["preservedSegment"]["headUuid"] == engine.mutable_messages[2]["uuid"]
    main_messages_json = json.dumps(provider.calls[-1]["messages"], ensure_ascii=False)
    assert "Recent messages are preserved verbatim" in main_messages_json
    assert "latest user request" in main_messages_json
    assert "Post-compact restored file context" in main_messages_json
    assert str(restored_file) in engine.tool_use_context.read_file_state


def test_partial_compact_preserves_tool_use_result_pair_and_read_state_without_duplicate_attachment(tmp_path: Path) -> None:
    """验证 ``partial compact preserves tool use result pair and read state without duplicate attachment`` 场景的行为、消息形状和关键不变量。"""
    repo = tmp_path / "repo"
    repo.mkdir()
    preserved_file = repo / "preserved.txt"
    restored_file = repo / "restored.txt"
    preserved_file.write_text("preserved file state", encoding="utf-8")
    restored_file.write_text("restored file state", encoding="utf-8")
    provider = FakeModelProvider(
        [
            "<analysis>draft</analysis><summary>Partial pair summary</summary>",
            "continued",
        ]
    )
    read_assistant = create_assistant_message(
        [
            {
                "type": "tool_use",
                "id": "toolu_read",
                "name": "Read",
                "input": {"file_path": str(preserved_file)},
            }
        ],
        uuid="read-assistant",
    )
    read_result = create_tool_result_message(
        {
            "type": "tool_result",
            "tool_use_id": "toolu_read",
            "content": "preserved file state",
        },
        uuid="read-result",
    )
    config = KernelConfig(
        cwd=repo,
        config_home=tmp_path / ".claude",
        context_compaction=ContextCompactionConfig(
            enabled=True,
            threshold_tokens=1,
            partial_keep_recent_messages=2,
            post_compact_max_files_to_restore=5,
        ),
    )
    engine = QueryEngine(
        model_provider=provider,
        config=config,
        mutable_messages=[
            create_user_message("old context that should be summarized"),
            read_assistant,
            read_result,
        ],
    )
    engine.tool_use_context.read_file_state[str(preserved_file)] = ReadFileStateEntry("preserved file state", 10)
    engine.tool_use_context.read_file_state[str(restored_file)] = ReadFileStateEntry("restored file state", 20)

    events = asyncio.run(_collect(engine.submit_message("latest user request", max_turns=1)))

    compact_event = next(event for event in events if event.get("type") == "context_compacted")
    kept_uuids = [message["uuid"] for message in compact_event["messagesToKeep"]]
    assert kept_uuids[:2] == ["read-assistant", "read-result"]
    model_messages_json = json.dumps(provider.calls[-1]["messages"], ensure_ascii=False)
    assert "toolu_read" in model_messages_json
    assert "Post-compact restored file context" in model_messages_json
    assert str(restored_file) in model_messages_json
    assert f"Post-compact restored file context for `{preserved_file}`" not in model_messages_json
    assert str(preserved_file) in engine.tool_use_context.read_file_state
    assert str(restored_file) in engine.tool_use_context.read_file_state


def test_compact_retries_when_summary_api_raises_prompt_too_long(tmp_path: Path) -> None:
    """验证 ``compact retries when summary api raises prompt too long`` 场景的行为、消息形状和关键不变量。"""
    class PromptTooLongDuringCompactProvider:
        """提供 ``PromptTooLongDuringCompactProvider`` 测试替身，用于隔离外部依赖或触发指定分支。"""
        def __init__(self):
            """初始化测试替身的计数器和可观察状态。"""
            self.calls = []
            self.compact_attempts = 0

        async def stream(self, *, messages, system_prompt, tools, options):
            """产生当前测试场景预设的模型消息或错误。"""
            self.calls.append({"messages": messages, "system_prompt": system_prompt, "tools": tools, "options": options})
            if options["querySource"] == "compact":
                self.compact_attempts += 1
                if self.compact_attempts <= 2:
                    raise RuntimeError("prompt is too long: compact input overflow")
                yield create_assistant_message("<analysis>draft</analysis><summary>Recovered after API PTL</summary>")
                return
            yield create_assistant_message("continued")

    repo = tmp_path / "repo"
    repo.mkdir()
    provider = PromptTooLongDuringCompactProvider()
    config = KernelConfig(
        cwd=repo,
        config_home=tmp_path / ".claude",
        context_compaction=ContextCompactionConfig(
            enabled=True,
            threshold_tokens=1,
            max_prompt_too_long_retries=3,
        ),
    )
    engine = QueryEngine(
        model_provider=provider,
        config=config,
        mutable_messages=[create_user_message(f"old message {index}") for index in range(8)],
    )

    events = asyncio.run(_collect(engine.submit_message("latest", max_turns=1)))

    compact_event = next(event for event in events if event.get("type") == "context_compacted")
    assert provider.compact_attempts == 3
    assert compact_event["promptTooLongRetries"] == 2
    compact_call_lengths = [len(call["messages"]) for call in provider.calls if call["options"]["querySource"] == "compact"]
    assert compact_call_lengths == sorted(compact_call_lengths, reverse=True)
    assert events[-1]["terminal"]["reason"] == "completed"


def test_main_prompt_too_long_error_compacts_and_retries_model_request(tmp_path: Path) -> None:
    """验证 ``main prompt too long error compacts and retries model request`` 场景的行为、消息形状和关键不变量。"""
    class PromptTooLongThenCompactProvider:
        """提供 ``PromptTooLongThenCompactProvider`` 测试替身，用于隔离外部依赖或触发指定分支。"""
        def __init__(self):
            """初始化测试替身的计数器和可观察状态。"""
            self.calls = []
            self.main_attempts = 0

        async def stream(self, *, messages, system_prompt, tools, options):
            """产生当前测试场景预设的模型消息或错误。"""
            self.calls.append({"messages": messages, "system_prompt": system_prompt, "tools": tools, "options": options})
            if options["querySource"] == "compact":
                yield create_assistant_message("<analysis>draft</analysis><summary>Reactive summary</summary>")
                return
            self.main_attempts += 1
            if self.main_attempts == 1:
                raise RuntimeError("prompt is too long: main request overflow")
            yield create_assistant_message("continued after retry")

    repo = tmp_path / "repo"
    repo.mkdir()
    provider = PromptTooLongThenCompactProvider()
    config = KernelConfig(
        cwd=repo,
        config_home=tmp_path / ".claude",
        context_compaction=ContextCompactionConfig(enabled=True, threshold_tokens=999_999),
    )
    engine = QueryEngine(
        model_provider=provider,
        config=config,
        mutable_messages=[create_user_message("large prior context")],
    )

    events = asyncio.run(_collect(engine.submit_message("continue", max_turns=1)))

    compact_event = next(event for event in events if event.get("type") == "context_compacted")
    assert compact_event["recoveringFrom"] == "prompt_too_long"
    assert [call["options"]["querySource"] for call in provider.calls] == ["python-port", "compact", "python-port"]
    retry_messages_json = json.dumps(provider.calls[-1]["messages"], ensure_ascii=False)
    assert "This session is being continued from a previous conversation" in retry_messages_json
    assert events[-1]["terminal"]["reason"] == "completed"


def test_compact_failure_falls_back_to_original_messages(tmp_path: Path) -> None:
    """验证 ``compact failure falls back to original messages`` 场景的行为、消息形状和关键不变量。"""
    class FailingThenAnswerProvider:
        """提供 ``FailingThenAnswerProvider`` 测试替身，用于隔离外部依赖或触发指定分支。"""
        def __init__(self):
            """初始化测试替身的计数器和可观察状态。"""
            self.calls = []

        async def stream(self, *, messages, system_prompt, tools, options):
            """产生当前测试场景预设的模型消息或错误。"""
            self.calls.append({"messages": messages, "system_prompt": system_prompt, "tools": tools, "options": options})
            if options["querySource"] == "compact":
                raise RuntimeError("compact failed")
            yield create_assistant_message("fallback answer")

    repo = tmp_path / "repo"
    repo.mkdir()
    provider = FailingThenAnswerProvider()
    config = KernelConfig(
        cwd=repo,
        config_home=tmp_path / ".claude",
        context_compaction=ContextCompactionConfig(enabled=True, threshold_tokens=1),
    )
    engine = QueryEngine(model_provider=provider, config=config)

    events = asyncio.run(_collect(engine.submit_message("keep going", max_turns=1)))

    assert any(event.get("type") == "context_compaction_failed" for event in events)
    assert events[-1]["terminal"]["reason"] == "completed"
    assert provider.calls[-1]["options"]["querySource"] == "python-port"
    assert "keep going" in json.dumps(provider.calls[-1]["messages"])


def test_microcompact_clears_old_tool_results_before_model_request(tmp_path: Path) -> None:
    """验证 ``microcompact clears old tool results before model request`` 场景的行为、消息形状和关键不变量。"""
    repo = tmp_path / "repo"
    repo.mkdir()
    old_messages = [
        create_tool_result_message({"type": "tool_result", "tool_use_id": f"toolu_{idx}", "content": f"large result {idx}"}, uuid=f"tool-result-{idx}")
        for idx in range(4)
    ]
    provider = FakeModelProvider(["done"])
    config = KernelConfig(
        cwd=repo,
        config_home=tmp_path / ".claude",
        context_compaction=ContextCompactionConfig(microcompact_enabled=True, microcompact_keep_recent_tool_results=1),
    )
    engine = QueryEngine(model_provider=provider, config=config, mutable_messages=old_messages)

    events = asyncio.run(_collect(engine.submit_message("next", max_turns=1)))

    assert any(event.get("type") == "context_microcompacted" for event in events)
    model_messages_json = json.dumps(provider.calls[0]["messages"], ensure_ascii=False)
    assert "[Old tool result content cleared]" in model_messages_json
    assert "large result 3" in model_messages_json
    assert "large result 0" not in model_messages_json


def test_query_engine_resume_loads_session_messages(tmp_path: Path) -> None:
    """验证 ``query engine resume loads session messages`` 场景的行为、消息形状和关键不变量。"""
    repo = tmp_path / "repo"
    repo.mkdir()
    session_id = "session-1"
    first_provider = FakeModelProvider(["first answer"])
    first_engine = QueryEngine(
        model_provider=first_provider,
        config=KernelConfig(cwd=repo, config_home=tmp_path / ".claude"),
        session_id=session_id,
    )
    asyncio.run(_collect(first_engine.submit_message("first", max_turns=1)))

    second_provider = FakeModelProvider(["second answer"])
    second_engine = QueryEngine(
        model_provider=second_provider,
        config=KernelConfig(cwd=repo, config_home=tmp_path / ".claude"),
        session_id=session_id,
        resume=True,
    )
    assert [message["type"] for message in second_engine.mutable_messages] == ["user", "assistant"]

    asyncio.run(_collect(second_engine.submit_message("second", max_turns=1)))
    assert [message["message"]["role"] for message in second_provider.calls[0]["messages"][-3:]] == [
        "user",
        "assistant",
        "user",
    ]


def test_query_engine_passes_agent_and_override_system_prompts(tmp_path: Path) -> None:
    """验证 ``query engine passes agent and override system prompts`` 场景的行为、消息形状和关键不变量。"""
    repo = tmp_path / "repo"
    repo.mkdir()
    agent_provider = FakeModelProvider(["ok"])
    agent_engine = QueryEngine(
        model_provider=agent_provider,
        config=KernelConfig(cwd=repo, config_home=tmp_path / ".claude"),
    )

    asyncio.run(_collect(agent_engine.submit_message("hi", agent_system_prompt="AGENT", append_system_prompt="APPEND")))

    assert agent_provider.calls[0]["system_prompt"] == ["AGENT", "APPEND"]

    override_provider = FakeModelProvider(["ok"])
    override_engine = QueryEngine(
        model_provider=override_provider,
        config=KernelConfig(cwd=repo, config_home=tmp_path / ".claude"),
    )

    asyncio.run(
        _collect(
            override_engine.submit_message(
                "hi",
                override_system_prompt="OVERRIDE",
                append_system_prompt="APPEND",
            )
        )
    )

    assert override_provider.calls[0]["system_prompt"] == ["OVERRIDE"]


def test_anthropic_provider_builds_request_and_parses_tool_use(tmp_path: Path) -> None:
    """验证 ``anthropic provider builds request and parses tool use`` 场景的行为、消息形状和关键不变量。"""
    captured = {}

    def transport(url, headers, body, timeout):
        """模拟模型 HTTP transport，并捕获请求体供断言。"""
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = body
        captured["timeout"] = timeout
        return {
            "id": "msg_real",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "Read",
                    "input": {"file_path": "/tmp/example.txt"},
                }
            ],
        }

    provider = AnthropicModelProvider(
        base_url="https://api.deepseek.com/anthropic",
        auth_token="secret-token",
        model="deepseek-v4-pro",
        transport=transport,
    )
    messages = [
        {
            "type": "user",
            "uuid": "u1",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "hello"}],
            },
        }
    ]

    response = asyncio.run(_collect(provider.stream(messages=messages, system_prompt=["sys"], tools=[FileReadTool(), WebSearchTool()], options={})))

    assert captured["url"] == "https://api.deepseek.com/anthropic/v1/messages"
    assert captured["headers"]["Authorization"] == "Bearer secret-token"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    assert captured["body"]["model"] == "deepseek-v4-pro"
    assert captured["body"]["system"] == "sys"
    assert captured["body"]["messages"][0]["role"] == "user"
    assert captured["body"]["tools"][0]["name"] == "Read"
    assert "file_path" in captured["body"]["tools"][0]["input_schema"]["required"]
    assert captured["body"]["tools"][1]["name"] == "WebSearch"
    assert captured["body"]["tools"][1]["input_schema"]["properties"]["allowed_domains"]["type"] == "array"
    assert provider.calls[0]["headers"]["Authorization"] == "Bearer [REDACTED]"
    assert response[0]["message"]["id"] == "msg_real"
    assert response[0]["message"]["content"][0]["type"] == "tool_use"


def test_anthropic_provider_merges_adjacent_tool_results_before_pairing() -> None:
    """验证 ``anthropic provider merges adjacent tool results before pairing`` 场景的行为、消息形状和关键不变量。"""
    captured = {}

    def transport(url, headers, body, timeout):
        """模拟模型 HTTP transport，并捕获请求体供断言。"""
        captured["body"] = body
        return {"id": "msg_text", "content": [{"type": "text", "text": "ok"}]}

    assistant = create_assistant_message(
        [
            {"type": "tool_use", "id": "toolu_a", "name": "Read", "input": {"file_path": "/tmp/a"}},
            {"type": "tool_use", "id": "toolu_b", "name": "Read", "input": {"file_path": "/tmp/b"}},
        ]
    )
    messages = [
        create_user_message("read both"),
        assistant,
        create_tool_result_message({"type": "tool_result", "tool_use_id": "toolu_a", "content": "a"}),
        create_tool_result_message({"type": "tool_result", "tool_use_id": "toolu_b", "content": "b"}),
    ]
    provider = AnthropicModelProvider(auth_token="secret-token", transport=transport)

    asyncio.run(_collect(provider.stream(messages=messages, system_prompt=[], tools=[], options={})))

    api_messages = captured["body"]["messages"]
    assert [message["role"] for message in api_messages] == ["user", "assistant", "user"]
    result_blocks = api_messages[-1]["content"]
    assert [block["tool_use_id"] for block in result_blocks] == ["toolu_a", "toolu_b"]
    assert all(block["content"] != "[Tool result missing due to internal error]" for block in result_blocks)


def test_anthropic_provider_repairs_orphaned_and_missing_tool_results() -> None:
    """验证 ``anthropic provider repairs orphaned and missing tool results`` 场景的行为、消息形状和关键不变量。"""
    captured = {}

    def transport(url, headers, body, timeout):
        """模拟模型 HTTP transport，并捕获请求体供断言。"""
        captured["body"] = body
        return {"id": "msg_text", "content": [{"type": "text", "text": "ok"}]}

    messages = [
        create_user_message(
            [
                {"type": "text", "text": "resume"},
                {"type": "tool_result", "tool_use_id": "toolu_orphan", "content": "stale"},
            ]
        ),
        create_attachment_message("restored file context", attachment_type="file"),
        create_assistant_message(
            [
                {
                    "type": "tool_use",
                    "id": "toolu_missing",
                    "name": "Read",
                    "input": {"file_path": "/tmp/a"},
                    "caller": {"type": "direct"},
                }
            ]
        ),
    ]
    provider = AnthropicModelProvider(auth_token="secret-token", transport=transport)

    asyncio.run(_collect(provider.stream(messages=messages, system_prompt=[], tools=[], options={})))

    api_messages = captured["body"]["messages"]
    assert api_messages[0]["content"] == [
        {"type": "text", "text": "resume"},
        {"type": "text", "text": "restored file context"},
    ]
    assert set(api_messages[1]["content"][0]) == {"type", "id", "name", "input"}
    assert api_messages[-1]["role"] == "user"
    assert api_messages[-1]["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_missing",
            "content": "[Tool result missing due to internal error]",
            "is_error": True,
        }
    ]


def test_anthropic_provider_normalizes_streaming_text_and_tool_use() -> None:
    """验证 ``anthropic provider normalizes streaming text and tool use`` 场景的行为、消息形状和关键不变量。"""
    captured = {}

    def stream_transport(url, headers, body, timeout, abort_signal):
        """模拟 Anthropic SSE transport，返回预设事件序列。"""
        captured["url"] = url
        captured["body"] = body
        captured["timeout"] = timeout
        captured["abort_signal"] = abort_signal
        return [
            {"type": "message_start", "message": {"id": "msg_stream", "content": []}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hello "}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "world"}},
            {"type": "content_block_stop", "index": 0},
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {}},
            },
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"file_path\""}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": ":\"/tmp/a.txt\"}"}},
            {"type": "content_block_stop", "index": 1},
            {"type": "message_stop"},
        ]

    provider = AnthropicModelProvider(
        base_url="https://api.deepseek.com/anthropic",
        auth_token="secret-token",
        model="deepseek-v4-pro",
        stream_transport=stream_transport,
    )

    response = asyncio.run(_collect(provider.stream(messages=[], system_prompt=["sys"], tools=[], options={})))
    content = response[0]["message"]["content"]

    assert captured["body"]["stream"] is True
    assert provider.calls[0]["body"]["stream"] is True
    assert response[0]["message"]["id"] == "msg_stream"
    assert content[0] == {"type": "text", "text": "hello world"}
    assert content[1]["type"] == "tool_use"
    assert content[1]["input"] == {"file_path": "/tmp/a.txt"}


def test_anthropic_stream_error_event_raises_api_error() -> None:
    """验证 ``anthropic stream error event raises api error`` 场景的行为、消息形状和关键不变量。"""
    def stream_transport(url, headers, body, timeout, abort_signal):
        """模拟 Anthropic SSE transport，返回预设事件序列。"""
        return [{"type": "error", "error": {"message": "bad stream"}}]

    provider = AnthropicModelProvider(
        base_url="https://api.deepseek.com/anthropic",
        auth_token="secret-token",
        stream_transport=stream_transport,
    )

    with pytest.raises(AnthropicAPIError, match="bad stream"):
        asyncio.run(_collect(provider.stream(messages=[], system_prompt=[], tools=[], options={})))


def test_anthropic_provider_uses_x_api_key_when_no_auth_token() -> None:
    """验证 ``anthropic provider uses x api key when no auth token`` 场景的行为、消息形状和关键不变量。"""
    captured = {}

    def transport(url, headers, body, timeout):
        """模拟模型 HTTP transport，并捕获请求体供断言。"""
        captured["headers"] = headers
        return {"id": "msg_text", "content": [{"type": "text", "text": "ok"}]}

    provider = AnthropicModelProvider(
        base_url="https://api.anthropic.com/v1",
        auth_token="",
        api_key="api-key",
        transport=transport,
    )

    response = asyncio.run(_collect(provider.stream(messages=[], system_prompt=[], tools=[], options={"model": "claude-opus-4-6"})))

    assert captured["headers"]["x-api-key"] == "api-key"
    assert "Authorization" not in captured["headers"]
    assert response[0]["message"]["content"][0]["text"] == "ok"


async def _collect(aiter):
    """消费异步生成器并把全部事件收集为列表，便于同步断言。"""
    return [event async for event in aiter]
