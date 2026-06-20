"""Pre/Post/Permission/Stop/Subagent 等 hook 生命周期行为测试。

每个测试展示一种 HookResult 字段如何影响 input、权限、额外消息或 continuation；可与
hooks.py 的返回值归一化和 tool_execution.py 的消费位置对照阅读。
"""

from __future__ import annotations

from pathlib import Path
import asyncio
import json

from agent_kernel.config import KernelConfig
from agent_kernel.messages import create_assistant_message
from agent_kernel.model_provider import FakeModelProvider
from agent_kernel.permissions import PermissionDecision
from agent_kernel.query_engine import QueryEngine
from agent_kernel.tools import Tool, ToolResult, ToolUseContext


async def _collect(iterator):
    """消费异步生成器并把全部事件收集为列表，便于同步断言。"""
    return [event async for event in iterator]


class EchoTool(Tool):
    """提供 ``EchoTool`` 测试替身，用于隔离外部依赖或触发指定分支。"""
    name = "Echo"
    input_schema = {"label": str}
    required_fields = ("label",)

    def is_read_only(self, input: dict) -> bool:
        """声明测试工具是否只读。"""
        return True

    async def check_permissions(self, input: dict, context: ToolUseContext) -> PermissionDecision:
        """返回测试场景需要的固定权限决策。"""
        return PermissionDecision.allow()

    async def call(self, args: dict, context: ToolUseContext, can_use_tool, parent_message, on_progress=None) -> ToolResult:
        """执行测试工具并返回预设结果或异常。"""
        return ToolResult(args["label"])


class FailingTool(EchoTool):
    """提供 ``FailingTool`` 测试替身，用于隔离外部依赖或触发指定分支。"""
    name = "Failing"

    async def call(self, args: dict, context: ToolUseContext, can_use_tool, parent_message, on_progress=None) -> ToolResult:
        """执行测试工具并返回预设结果或异常。"""
        raise RuntimeError("boom")


def make_config(tmp_path: Path) -> KernelConfig:
    """为当前测试创建隔离 cwd 和 config_home 的最小 KernelConfig。"""
    repo = tmp_path / "repo"
    repo.mkdir()
    return KernelConfig(cwd=repo, config_home=tmp_path / ".claude", session_start_date="2026-06-14")


def attachment_types(events: list[dict]) -> list[str]:
    """提取事件流中 attachment 消息的类型集合。"""
    types = []
    for event in events:
        payload = event.get("message")
        attachment = payload.get("attachment") if isinstance(payload, dict) else None
        if isinstance(attachment, dict):
            types.append(attachment.get("type"))
    return types


def test_pre_tool_use_hook_updates_input_and_adds_context(tmp_path: Path) -> None:
    """验证 ``pre tool use hook updates input and adds context`` 场景的行为、消息形状和关键不变量。"""
    provider = FakeModelProvider(
        [
            [{"type": "tool_use", "id": "toolu_echo", "name": "Echo", "input": {"label": "original"}}],
            "done",
        ]
    )
    engine = QueryEngine(model_provider=provider, config=make_config(tmp_path), tools=[EchoTool()])

    def pre_tool_hook(hook_input: dict):
        """模拟 ``pre_tool_hook`` hook，并返回当前用例需要的控制结果。"""
        assert hook_input["hook_event_name"] == "PreToolUse"
        assert hook_input["tool_input"] == {"label": "original"}
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "updatedInput": {"label": "from hook"},
                "additionalContext": "context from pre hook",
            }
        }

    engine.tool_use_context.hook_registry.register("PreToolUse", pre_tool_hook, matcher="Echo")

    events = asyncio.run(_collect(engine.submit_message("run echo", max_turns=3)))

    tool_message = next(event for event in events if event.get("type") == "user")
    assert tool_message["message"]["content"][0]["content"] == "from hook"
    assert "hook_additional_context" in attachment_types(events)


def test_permission_request_hook_can_allow_ask_permission(tmp_path: Path) -> None:
    """验证 ``permission request hook can allow ask permission`` 场景的行为、消息形状和关键不变量。"""
    config = make_config(tmp_path)
    target = config.cwd / "allowed.txt"
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
    engine = QueryEngine(model_provider=provider, config=config)
    engine.tool_use_context.hook_registry.register(
        "PermissionRequest",
        lambda hook_input: {"permissionRequestResult": {"behavior": "allow"}},
        matcher="Write",
    )

    events = asyncio.run(_collect(engine.submit_message("write", max_turns=3)))

    assert target.read_text(encoding="utf-8") == "hello"
    assert "hook_permission_decision" in attachment_types(events)
    assert events[-1]["terminal"]["reason"] == "completed"


def test_permission_denied_hook_retry_message_is_returned(tmp_path: Path) -> None:
    """验证 ``permission denied hook retry message is returned`` 场景的行为、消息形状和关键不变量。"""
    config = make_config(tmp_path)
    target = config.cwd / "denied.txt"
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
        ]
    )
    engine = QueryEngine(model_provider=provider, config=config)
    engine.tool_use_context.hook_registry.register(
        "PermissionDenied",
        lambda hook_input: {"hookSpecificOutput": {"hookEventName": "PermissionDenied", "retry": True}},
        matcher="Write",
    )

    events = asyncio.run(_collect(engine.submit_message("write", max_turns=1)))
    event_json = json.dumps(events, ensure_ascii=False)

    assert not target.exists()
    assert "Permission denied" in event_json
    assert "PermissionDenied hook indicated this command is now approved" in event_json


def test_post_tool_use_hook_can_stop_continuation(tmp_path: Path) -> None:
    """验证 ``post tool use hook can stop continuation`` 场景的行为、消息形状和关键不变量。"""
    provider = FakeModelProvider(
        [
            [{"type": "tool_use", "id": "toolu_echo", "name": "Echo", "input": {"label": "ok"}}],
            "should not run",
        ]
    )
    engine = QueryEngine(model_provider=provider, config=make_config(tmp_path), tools=[EchoTool()])
    engine.tool_use_context.hook_registry.register(
        "PostToolUse",
        lambda hook_input: {"continue": False, "stopReason": "post hook stopped"},
        matcher="Echo",
    )

    events = asyncio.run(_collect(engine.submit_message("run echo", max_turns=3)))

    assert len(provider.calls) == 1
    assert "hook_stopped_continuation" in attachment_types(events)
    assert events[-1]["terminal"]["reason"] == "hook_stopped"


def test_post_tool_use_failure_hook_receives_errors(tmp_path: Path) -> None:
    """验证 ``post tool use failure hook receives errors`` 场景的行为、消息形状和关键不变量。"""
    provider = FakeModelProvider(
        [
            [{"type": "tool_use", "id": "toolu_fail", "name": "Failing", "input": {"label": "x"}}],
        ]
    )
    engine = QueryEngine(model_provider=provider, config=make_config(tmp_path), tools=[FailingTool()])
    seen: list[dict] = []

    def failure_hook(hook_input: dict):
        """模拟 ``failure_hook`` hook，并返回当前用例需要的控制结果。"""
        seen.append(hook_input)
        return {"hookSpecificOutput": {"hookEventName": "PostToolUseFailure", "additionalContext": "failure context"}}

    engine.tool_use_context.hook_registry.register("PostToolUseFailure", failure_hook, matcher="Failing")

    events = asyncio.run(_collect(engine.submit_message("run failure", max_turns=1)))

    assert seen[0]["hook_event_name"] == "PostToolUseFailure"
    assert "boom" in seen[0]["error"]
    assert "hook_additional_context" in attachment_types(events)


def test_stop_hook_blocking_error_reenters_model_loop(tmp_path: Path) -> None:
    """验证 ``stop hook blocking error reenters model loop`` 场景的行为、消息形状和关键不变量。"""
    provider = FakeModelProvider(["first answer", "second answer"])
    engine = QueryEngine(model_provider=provider, config=make_config(tmp_path), tools=[EchoTool()])
    calls = 0

    def stop_hook(hook_input: dict):
        """模拟 ``stop_hook`` hook，并返回当前用例需要的控制结果。"""
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"blockingError": "Please address the stop hook feedback."}
        return None

    engine.tool_use_context.hook_registry.register("Stop", stop_hook)

    events = asyncio.run(_collect(engine.submit_message("answer", max_turns=3)))

    assert [event["type"] for event in events].count("stream_request_start") == 2
    assert calls == 2
    assert "Please address the stop hook feedback." in json.dumps(provider.calls[1]["messages"], ensure_ascii=False)
    assert events[-1]["terminal"]["reason"] == "completed"


def test_stop_failure_hook_runs_on_model_error(tmp_path: Path) -> None:
    """验证 ``stop failure hook runs on model error`` 场景的行为、消息形状和关键不变量。"""
    class ErrorProvider:
        """提供 ``ErrorProvider`` 测试替身，用于隔离外部依赖或触发指定分支。"""
        async def stream(self, *, messages, system_prompt, tools, options):
            """产生当前测试场景预设的模型消息或错误。"""
            raise RuntimeError("network down")
            yield

    engine = QueryEngine(model_provider=ErrorProvider(), config=make_config(tmp_path), tools=[EchoTool()])
    seen: list[dict] = []
    engine.tool_use_context.hook_registry.register("StopFailure", lambda hook_input: seen.append(hook_input) or None)

    events = asyncio.run(_collect(engine.submit_message("answer", max_turns=1)))

    assert seen[0]["hook_event_name"] == "StopFailure"
    assert seen[0]["error"] == "network down"
    assert events[-1]["terminal"]["reason"] == "error"
