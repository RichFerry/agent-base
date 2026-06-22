"""SDK event and transcript contract tests for v0.2.7.

These tests freeze the observable event and JSONL transcript shapes around
tool_use/tool_result pairing, resume, compact boundaries, and combined local
runner capabilities. They use fake providers and local fixtures only.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from agent_kernel.config import ContextCompactionConfig, KernelConfig
from agent_kernel.model_provider import FakeModelProvider
from agent_kernel.query_engine import QueryEngine
from examples.local_agent import build_local_engine, run_local_agent_once


async def _collect(iterator):
    return [event async for event in iterator]


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    return repo


def _examples_dir() -> Path:
    return Path(__file__).parents[1] / "examples"


def _transcript_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _tool_result_blocks_from_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for event in events:
        if event.get("type") != "user":
            continue
        content = event.get("message", {}).get("content", [])
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                blocks.append(block)
    return blocks


def _tool_use_ids_from_rows(rows: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for row in rows:
        if row.get("type") != "assistant":
            continue
        for block in row.get("message", {}).get("content", []):
            if isinstance(block, dict) and block.get("type") == "tool_use":
                ids.append(str(block.get("id")))
    return ids


def _tool_result_ids_from_rows(rows: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for row in rows:
        if row.get("type") != "user":
            continue
        for block in row.get("message", {}).get("content", []):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                ids.append(str(block.get("tool_use_id")))
    return ids


def _contract_kind(row: dict[str, Any]) -> str:
    content = row.get("message", {}).get("content", [])
    first = content[0] if content else {}
    if row.get("type") == "system" and row.get("subtype") == "compact_boundary":
        return "system:compact_boundary"
    if row.get("type") == "system":
        return f"system:{row.get('subtype')}"
    if row.get("type") == "assistant" and any(block.get("type") == "tool_use" for block in content if isinstance(block, dict)):
        return "assistant:tool_use"
    if row.get("type") == "assistant":
        return "assistant:text"
    if row.get("type") == "user" and isinstance(first, dict) and first.get("type") == "tool_result":
        return f"user:tool_result:{first.get('tool_use_id')}"
    if row.get("type") == "user" and row.get("isCompactSummary"):
        return "user:compact_summary"
    if row.get("type") == "user" and row.get("isMeta"):
        return "user:skill_prompt"
    if row.get("type") == "user":
        return "user:prompt"
    return str(row.get("type"))


def _assert_no_orphan_tool_results(rows: list[dict[str, Any]]) -> None:
    seen_tool_uses: set[str] = set()
    for row in rows:
        for block in row.get("message", {}).get("content", []):
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                seen_tool_uses.add(str(block.get("id")))
            elif block.get("type") == "tool_result":
                assert str(block.get("tool_use_id")) in seen_tool_uses


def test_sdk_event_and_transcript_shape_for_tool_loop(tmp_path: Path) -> None:
    """SDK init/assistant/user/result events and JSONL rows keep stable shape."""
    repo = _repo(tmp_path)
    target = repo / "sample.txt"
    target.write_text("alpha\n", encoding="utf-8")
    provider = FakeModelProvider(
        [
            [{"type": "tool_use", "id": "toolu_read", "name": "Read", "input": {"file_path": str(target)}}],
            "Read complete.",
        ]
    )
    engine = build_local_engine(
        cwd=repo,
        config_home=tmp_path / ".claude",
        model_provider=provider,
        session_id="transcript-tool-loop",
        require_api_key=False,
    )

    result = asyncio.run(run_local_agent_once("read sample", engine=engine, max_turns=3))
    events = result.events
    rows = _transcript_rows(result.transcript_path)
    assistant_tool = next(event for event in events if event.get("type") == "assistant" and event["message"]["content"][0]["type"] == "tool_use")
    user_tool = next(event for event in events if event.get("type") == "user" and event["message"]["content"][0]["type"] == "tool_result")

    assert events[0]["type"] == "system"
    assert events[0]["subtype"] == "init"
    assert events[0]["session_id"] == "transcript-tool-loop"
    assert "Read" in events[0]["tools"]
    assert events[0]["permissionMode"] == "ask"
    assert assistant_tool["message"]["role"] == "assistant"
    assert assistant_tool["message"]["content"][0] == {
        "type": "tool_use",
        "id": "toolu_read",
        "name": "Read",
        "input": {"file_path": str(target)},
    }
    assert user_tool["message"]["role"] == "user"
    assert user_tool["message"]["content"][0]["type"] == "tool_result"
    assert user_tool["message"]["content"][0]["tool_use_id"] == "toolu_read"
    assert "alpha" in user_tool["message"]["content"][0]["content"]
    assert events[-1]["type"] == "result"
    assert events[-1]["subtype"] == "success"
    assert events[-1]["is_error"] is False
    assert events[-1]["result"] == "Read complete."

    assert [_contract_kind(row) for row in rows] == [
        "user:prompt",
        "assistant:tool_use",
        "user:tool_result:toolu_read",
        "assistant:text",
    ]
    assert all(row["sessionId"] == "transcript-tool-loop" for row in rows)
    assert all(row["cwd"] == str(repo) for row in rows)
    assert all(row["version"] == "0.1.0-python-port" for row in rows)
    assert rows[0]["parentUuid"] is None
    assert rows[2]["parentUuid"] == rows[1]["uuid"]
    assert rows[2]["sourceToolAssistantUUID"] == rows[1]["uuid"]
    assert rows[2]["message"]["content"][0]["tool_use_id"] == "toolu_read"
    _assert_no_orphan_tool_results(rows)


def test_resume_preserves_transcript_message_ordering_and_pairing(tmp_path: Path) -> None:
    """Resume loads the same ordered message chain and does not orphan tool_results."""
    repo = _repo(tmp_path)
    target = repo / "resume.txt"
    target.write_text("resume alpha\n", encoding="utf-8")
    provider = FakeModelProvider(
        [
            [{"type": "tool_use", "id": "toolu_resume", "name": "Read", "input": {"file_path": str(target)}}],
            "Initial done.",
        ]
    )
    config = KernelConfig(cwd=repo, config_home=tmp_path / ".claude")
    engine = QueryEngine(model_provider=provider, config=config, session_id="resume-contract")

    events = asyncio.run(_collect(engine.submit_message("read resume", max_turns=3, sdk_events=True)))
    rows = _transcript_rows(engine.session_store.transcript_path)
    resumed = QueryEngine(
        model_provider=FakeModelProvider(["After resume."]),
        config=KernelConfig(cwd=repo, config_home=tmp_path / ".claude"),
        session_id="resume-contract",
        resume=True,
    )

    assert events[-1]["type"] == "result"
    assert [_contract_kind(row) for row in rows] == [
        "user:prompt",
        "assistant:tool_use",
        "user:tool_result:toolu_resume",
        "assistant:text",
    ]
    assert [_contract_kind(row) for row in resumed.mutable_messages] == [_contract_kind(row) for row in rows]
    assert _tool_use_ids_from_rows(resumed.mutable_messages) == ["toolu_resume"]
    assert _tool_result_ids_from_rows(resumed.mutable_messages) == ["toolu_resume"]
    _assert_no_orphan_tool_results(resumed.mutable_messages)


def test_compact_boundary_event_and_transcript_order_contract(tmp_path: Path) -> None:
    """Auto compact records a stable boundary and transcript order."""
    repo = _repo(tmp_path)
    provider = FakeModelProvider(
        [
            "<analysis>draft</analysis>\n\n<summary>\n1. Primary Request and Intent:\n   Freeze transcript contracts.\n</summary>",
            "continued after compact",
        ]
    )
    config = KernelConfig(
        cwd=repo,
        config_home=tmp_path / ".claude",
        context_compaction=ContextCompactionConfig(enabled=True, threshold_tokens=1),
    )
    engine = QueryEngine(model_provider=provider, config=config, session_id="compact-contract")

    events = asyncio.run(_collect(engine.submit_message("large original prompt", max_turns=1, sdk_events=True)))
    compact_event = next(event for event in events if event.get("type") == "context_compacted")
    rows = _transcript_rows(engine.session_store.transcript_path)

    assert compact_event["boundary"]["type"] == "system"
    assert compact_event["boundary"]["subtype"] == "compact_boundary"
    assert compact_event["boundary"]["content"] == "Conversation compacted"
    assert compact_event["boundary"]["compactMetadata"]["trigger"] == "auto"
    assert compact_event["boundary"]["compactMetadata"]["messagesSummarized"] == 1
    assert compact_event["summary_messages"][0]["isCompactSummary"] is True
    assert compact_event["summary_messages"][0]["isVisibleInTranscriptOnly"] is True
    assert events[-1]["type"] == "result"
    assert events[-1]["subtype"] == "success"

    assert [_contract_kind(row) for row in rows] == [
        "user:prompt",
        "system:compact_boundary",
        "user:compact_summary",
        "assistant:text",
    ]
    boundary = rows[1]
    summary = rows[2]
    assert boundary["parentUuid"] == rows[0]["uuid"]
    assert boundary["compactMetadata"]["trigger"] == "auto"
    assert boundary["compactMetadata"]["messagesSummarized"] == 1
    assert summary["parentUuid"] == boundary["uuid"]
    assert summary["isCompactSummary"] is True
    assert summary["isVisibleInTranscriptOnly"] is True

    resumed = QueryEngine(
        model_provider=FakeModelProvider(["resumed compact"]),
        config=KernelConfig(cwd=repo, config_home=tmp_path / ".claude"),
        session_id="compact-contract",
        resume=True,
    )
    assert [_contract_kind(row) for row in resumed.mutable_messages] == [
        "system:compact_boundary",
        "user:compact_summary",
        "assistant:text",
    ]


def test_model_error_and_tool_error_contracts(tmp_path: Path) -> None:
    """Model errors and tool errors enter stable SDK/transcript paths."""
    class ErrorProvider:
        async def stream(self, *, messages, system_prompt, tools, options):
            raise RuntimeError("model down")
            yield

    error_engine = QueryEngine(
        model_provider=ErrorProvider(),
        config=KernelConfig(cwd=_repo(tmp_path), config_home=tmp_path / ".claude-model-error"),
        session_id="model-error-contract",
    )
    model_events = asyncio.run(_collect(error_engine.submit_message("hi", max_turns=1, sdk_events=True)))
    model_rows = _transcript_rows(error_engine.session_store.transcript_path)

    assert any(event.get("type") == "system" and event.get("subtype") == "api_error" for event in model_events)
    assert model_events[-1]["type"] == "result"
    assert model_events[-1]["subtype"] == "error_during_execution"
    assert model_events[-1]["is_error"] is True
    assert "model down" in model_events[-1]["errors"][0]
    assert [_contract_kind(row) for row in model_rows] == ["user:prompt", "system:api_error"]
    assert model_rows[1]["level"] == "error"
    assert model_rows[1]["error"] == "model down"

    provider = FakeModelProvider(
        [
            [{"type": "tool_use", "id": "toolu_missing", "name": "NoSuchTool", "input": {}}],
            "Recovered from tool error.",
        ]
    )
    tool_engine = QueryEngine(
        model_provider=provider,
        config=KernelConfig(cwd=_repo(tmp_path), config_home=tmp_path / ".claude-tool-error"),
        session_id="tool-error-contract",
    )
    tool_events = asyncio.run(_collect(tool_engine.submit_message("use missing tool", max_turns=3, sdk_events=True)))
    tool_rows = _transcript_rows(tool_engine.session_store.transcript_path)
    block = _tool_result_blocks_from_events(tool_events)[0]

    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "toolu_missing"
    assert block["is_error"] is True
    assert "Tool NoSuchTool does not exist" in block["content"]
    assert tool_events[-1]["subtype"] == "success"
    assert [_contract_kind(row) for row in tool_rows] == [
        "user:prompt",
        "assistant:tool_use",
        "user:tool_result:toolu_missing",
        "assistant:text",
    ]
    _assert_no_orphan_tool_results(tool_rows)


def test_combined_flow_satisfies_sdk_transcript_and_pairing_contracts(tmp_path: Path) -> None:
    """WebSearch, Skill, and MCP combined flow keeps the same transcript contract."""
    search_calls: list[dict[str, Any]] = []

    def search_handler(args: dict[str, Any]) -> list[dict[str, str]]:
        search_calls.append(dict(args))
        return [{"title": "Combined Contract", "url": "https://example.invalid/combined-contract"}]

    provider = FakeModelProvider(
        [
            [
                {"type": "tool_use", "id": "toolu_search", "name": "WebSearch", "input": {"query": "combined contract"}},
                {"type": "tool_use", "id": "toolu_skill", "name": "Skill", "input": {"skill": "echo", "args": "skill contract"}},
                {"type": "tool_use", "id": "toolu_mcp", "name": "mcp__local-echo__echo", "input": {"text": "mcp contract"}},
            ],
            "Combined transcript contract complete.",
        ]
    )
    engine = build_local_engine(
        cwd=_repo(tmp_path),
        config_home=tmp_path / ".claude",
        model_provider=provider,
        session_id="combined-transcript-contract",
        web_search_handler=search_handler,
        skills_dir=_examples_dir() / "skills",
        mcp_fixture=_examples_dir() / "mcp" / "echo-mcp.json",
        permission_mode="bypass",
        require_api_key=False,
    )

    result = asyncio.run(run_local_agent_once("combined transcript", engine=engine, max_turns=4))
    rows = _transcript_rows(result.transcript_path)
    blocks = _tool_result_blocks_from_events(result.events)

    assert result.events[0]["type"] == "system"
    assert result.events[0]["subtype"] == "init"
    assert result.events[0]["skills"] == ["echo"]
    assert result.events[0]["mcp_servers"] == [{"name": "local-echo", "status": "connected"}]
    assert result.events[-1]["type"] == "result"
    assert result.events[-1]["subtype"] == "success"
    assert [block["tool_use_id"] for block in blocks] == ["toolu_search", "toolu_skill", "toolu_mcp"]
    assert search_calls == [{"query": "combined contract"}]
    assert [_contract_kind(row) for row in rows] == [
        "user:prompt",
        "assistant:tool_use",
        "user:tool_result:toolu_search",
        "user:tool_result:toolu_skill",
        "user:skill_prompt",
        "user:tool_result:toolu_mcp",
        "assistant:text",
    ]
    assert _tool_use_ids_from_rows(rows) == ["toolu_search", "toolu_skill", "toolu_mcp"]
    assert _tool_result_ids_from_rows(rows) == ["toolu_search", "toolu_skill", "toolu_mcp"]
    assert rows[4]["isMeta"] is True
    assert "<command-name>echo</command-name>" in rows[4]["message"]["content"][0]["text"]
    _assert_no_orphan_tool_results(rows)
