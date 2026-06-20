"""SDK 消息映射、init/result/error 事件和 transcript 兼容性测试。

这里同时比较 ``sdk_events=False`` 的原始内核流与 opt-in SDK 流，确保外围包装不会改变
核心 query 事件；旧版 compact metadata 字段也通过 replay 用例验证。
"""

from __future__ import annotations

from pathlib import Path
import asyncio
import json

from agent_kernel.config import KernelConfig, MCPClientConfig, OutputStyleConfig
from agent_kernel.context_compaction import create_compact_boundary_message
from agent_kernel.messages import create_assistant_message, create_user_message
from agent_kernel.model_provider import FakeModelProvider
from agent_kernel.query_engine import QueryEngine
from agent_kernel.sdk import (
    from_sdk_compact_metadata,
    to_internal_messages,
    to_sdk_compact_metadata,
    to_sdk_messages,
)
from agent_kernel.session import SessionStore
from agent_kernel.tools import BashTool, FileReadTool


async def _collect(iterator):
    """消费异步生成器并把全部事件收集为列表，便于同步断言。"""
    return [event async for event in iterator]


def make_config(tmp_path: Path) -> KernelConfig:
    """为当前测试创建隔离 cwd 和 config_home 的最小 KernelConfig。"""
    repo = tmp_path / "repo"
    repo.mkdir()
    return KernelConfig(cwd=repo, config_home=tmp_path / ".claude", session_start_date="2026-06-14")


def test_sdk_mapper_compact_metadata_and_synthetic_user(tmp_path: Path) -> None:
    """验证 ``sdk mapper compact metadata and synthetic user`` 场景的行为、消息形状和关键不变量。"""
    config = make_config(tmp_path)
    user = create_user_message("summary", uuid="u1", is_visible_in_transcript_only=True)
    assistant = create_assistant_message("answer", uuid="a1", message_id="m1")
    boundary = create_compact_boundary_message(
        "auto",
        123,
        "a1",
        messages_summarized=2,
        preserved_segment={"headUuid": "u1", "anchorUuid": "summary-1", "tailUuid": "a1"},
    )

    sdk_messages = to_sdk_messages([user, assistant, boundary], session_id="session-1")
    compact_meta = sdk_messages[2]["compact_metadata"]

    assert sdk_messages[0]["type"] == "user"
    assert sdk_messages[0]["isSynthetic"] is True
    assert sdk_messages[0]["session_id"] == "session-1"
    assert sdk_messages[1]["type"] == "assistant"
    assert sdk_messages[1]["parent_tool_use_id"] is None
    assert compact_meta == {
        "trigger": "auto",
        "pre_tokens": 123,
        "preserved_segment": {
            "head_uuid": "u1",
            "anchor_uuid": "summary-1",
            "tail_uuid": "a1",
        },
    }
    assert from_sdk_compact_metadata(compact_meta)["preservedSegment"]["anchorUuid"] == "summary-1"
    assert to_sdk_compact_metadata(from_sdk_compact_metadata(compact_meta)) == compact_meta

    internal = to_internal_messages([sdk_messages[0], sdk_messages[1], sdk_messages[2]])
    assert [message["type"] for message in internal] == ["user", "assistant", "system"]
    assert internal[2]["compactMetadata"]["preTokens"] == 123


def test_system_init_message_uses_source_shaped_fields(tmp_path: Path) -> None:
    """验证 ``system init message uses source shaped fields`` 场景的行为、消息形状和关键不变量。"""
    repo = tmp_path / "repo"
    repo.mkdir()
    config = KernelConfig(
        cwd=repo,
        config_home=tmp_path / ".claude",
        mcp_clients=(MCPClientConfig(name="github", type="connected"),),
        output_style=OutputStyleConfig(name="Review", prompt="Be terse."),
    )
    engine = QueryEngine(
        model_provider=FakeModelProvider(["ok"]),
        config=config,
        session_id="session-1",
        tools=[BashTool(), FileReadTool()],
        model="claude-opus-4-6",
    )
    engine.tool_use_context.app_state.tool_permission_context.mode = "bypass"

    init = engine.get_system_init_message()
    status = engine.get_sdk_status_message()

    assert init["type"] == "system"
    assert init["subtype"] == "init"
    assert init["cwd"] == str(repo)
    assert init["session_id"] == "session-1"
    assert init["tools"] == ["Bash", "Read"]
    assert init["mcp_servers"] == [{"name": "github", "status": "connected"}]
    assert init["model"] == "claude-opus-4-6"
    assert init["permissionMode"] == "bypass"
    assert init["slash_commands"] == []
    assert init["apiKeySource"] == "none"
    assert init["claude_code_version"] == "0.1.0-python-port"
    assert init["output_style"] == "Review"
    assert status["subtype"] == "status"
    assert status["permissionMode"] == "bypass"


def test_transcript_preserves_sdk_fields_and_loads_sdk_messages(tmp_path: Path) -> None:
    """验证 ``transcript preserves sdk fields and loads sdk messages`` 场景的行为、消息形状和关键不变量。"""
    config = make_config(tmp_path)
    store = SessionStore(config, session_id="session-1")
    user = create_user_message("hello", uuid="u1")
    user["requestId"] = "req-1"
    user["timestamp"] = "2026-06-14T00:00:00+00:00"
    user["toolUseResult"] = {"kind": "structured"}
    assistant = create_assistant_message("hi", uuid="a1", message_id="m1")

    store.record_transcript([user, assistant])
    rows = [json.loads(line) for line in store.transcript_path.read_text(encoding="utf-8").splitlines()]
    sdk_messages = store.load_sdk_messages()

    assert rows[0]["requestId"] == "req-1"
    assert rows[0]["timestamp"] == "2026-06-14T00:00:00+00:00"
    assert rows[0]["toolUseResult"] == {"kind": "structured"}
    assert sdk_messages[0]["tool_use_result"] == {"kind": "structured"}
    assert sdk_messages[1]["message"]["id"] == "m1"


def test_transcript_resume_accepts_sdk_compact_metadata_shape(tmp_path: Path) -> None:
    """验证 ``transcript resume accepts sdk compact metadata shape`` 场景的行为、消息形状和关键不变量。"""
    config = make_config(tmp_path)
    store = SessionStore(config, session_id="session-1")
    store.project_dir.mkdir(parents=True)
    entries = [
        {"type": "user", "uuid": "u1", "message": {"role": "user", "content": []}, "parentUuid": None},
        {
            "type": "system",
            "subtype": "compact_boundary",
            "uuid": "c1",
            "content": "Conversation compacted",
            "compact_metadata": {
                "trigger": "auto",
                "pre_tokens": 100,
                "preserved_segment": {
                    "head_uuid": "u1",
                    "anchor_uuid": "summary-1",
                    "tail_uuid": "u1",
                },
            },
            "parentUuid": "u1",
        },
        {
            "type": "user",
            "uuid": "summary-1",
            "message": {"role": "user", "content": []},
            "parentUuid": "c1",
            "isCompactSummary": True,
        },
    ]
    store.transcript_path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8")

    reloaded = SessionStore(config, session_id="session-1")
    loaded = reloaded._load_transcript_messages()
    sdk_messages = reloaded.load_sdk_messages()

    assert loaded[0]["compactMetadata"]["preservedSegment"]["anchorUuid"] == "summary-1"
    assert sdk_messages[0]["compact_metadata"]["preserved_segment"]["anchor_uuid"] == "summary-1"


def test_query_engine_sdk_events_are_opt_in_and_source_shaped(tmp_path: Path) -> None:
    """验证 ``query engine sdk events are opt in and source shaped`` 场景的行为、消息形状和关键不变量。"""
    config = make_config(tmp_path)
    default_engine = QueryEngine(
        model_provider=FakeModelProvider(["plain answer"]),
        config=config,
        session_id="default-session",
    )
    default_events = asyncio.run(_collect(default_engine.submit_message("hi", max_turns=1)))

    sdk_engine = QueryEngine(
        model_provider=FakeModelProvider(["sdk answer"]),
        config=KernelConfig(cwd=config.cwd, config_home=tmp_path / ".claude-sdk"),
        session_id="sdk-session",
    )
    sdk_events = asyncio.run(_collect(sdk_engine.submit_message("hi", max_turns=1, sdk_events=True)))

    assert default_events[0]["type"] == "stream_request_start"
    assert default_events[-1]["type"] == "terminal"
    assert sdk_events[0]["type"] == "system"
    assert sdk_events[0]["subtype"] == "init"
    assert sdk_events[-1]["type"] == "result"
    assert sdk_events[-1]["subtype"] == "success"
    assert sdk_events[-1]["is_error"] is False
    assert sdk_events[-1]["result"] == "sdk answer"


def test_query_engine_sdk_events_report_model_errors_without_swallowing_terminal(tmp_path: Path) -> None:
    """验证 ``query engine sdk events report model errors without swallowing terminal`` 场景的行为、消息形状和关键不变量。"""
    class ErrorProvider:
        """提供 ``ErrorProvider`` 测试替身，用于隔离外部依赖或触发指定分支。"""
        async def stream(self, *, messages, system_prompt, tools, options):
            """产生当前测试场景预设的模型消息或错误。"""
            raise RuntimeError("network down")
            yield

    config = make_config(tmp_path)
    engine = QueryEngine(model_provider=ErrorProvider(), config=config, session_id="session-err")

    events = asyncio.run(_collect(engine.submit_message("hi", max_turns=1, sdk_events=True)))

    assert any(event.get("type") == "system" and event.get("subtype") == "api_error" for event in events)
    assert any(event.get("type") == "terminal" and event["terminal"]["reason"] == "error" for event in events)
    assert any(event.get("type") == "system" and event.get("subtype") == "error" for event in events)
    assert events[-1]["type"] == "result"
    assert events[-1]["subtype"] == "error_during_execution"
    assert events[-1]["is_error"] is True
    assert "network down" in events[-1]["errors"][0]
