"""v0.6 MCP -> session transcript -> memory extraction full-chain contracts."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from agent_kernel.config import KernelConfig
from agent_kernel.memory import ENTRYPOINT_NAME, MemoryLoader
from agent_kernel.memory_chain import apply_memory_candidates, extract_memory_candidates, load_candidate_json, memory_provenance, rebuild_memory_index, validate_memory_store
from agent_kernel.messages import create_assistant_message, create_tool_result_message, normalize_messages_for_api
from agent_kernel.model_provider import FakeModelProvider
from agent_kernel.session_diagnostics import redacted_session_entries, session_timeline, validate_session, validate_session_entries
from examples import local_agent
from examples.local_agent import build_local_engine, main, run_local_agent_once


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    return repo


def _example_mcp_fixture() -> Path:
    return Path(__file__).parents[1] / "examples" / "mcp" / "echo-mcp.json"


def _stdio_mcp_config() -> Path:
    return Path(__file__).parents[1] / "examples" / "mcp" / "stdio-config.json"


def _tool_result_blocks(events: list[dict]) -> list[dict]:
    blocks: list[dict] = []
    for event in events:
        if event.get("type") != "user":
            continue
        content = event.get("message", {}).get("content", [])
        blocks.extend(block for block in content if isinstance(block, dict) and block.get("type") == "tool_result")
    return blocks


def test_mcp_session_memory_full_chain_contract(tmp_path: Path) -> None:
    """MCP tool_result metadata can be audited, extracted into memory, and loaded on resume."""
    repo = _repo(tmp_path)
    config_home = tmp_path / ".claude"
    provider = FakeModelProvider(
        [
            [{"type": "tool_use", "id": "toolu_mcp", "name": "mcp__local-echo__echo", "input": {"text": "rememberable"}}],
            "done",
        ]
    )
    engine = build_local_engine(
        cwd=repo,
        config_home=config_home,
        model_provider=provider,
        mcp_fixture=_example_mcp_fixture(),
        permission_mode="bypass",
        require_api_key=False,
    )

    result = asyncio.run(run_local_agent_once("call mcp", engine=engine, max_turns=3))
    tool_result = _tool_result_blocks(result.events)[0]

    assert tool_result["content"] == '{"echo":"rememberable","source":"local-mcp-fixture"}'
    assert tool_result["mcpMetadata"]["serverName"] == "local-echo"
    assert tool_result["mcpMetadata"]["operation"] == "tools/call"
    assert tool_result["mcpMetadata"]["status"] == "ok"

    validation = validate_session(result.session_id, cwd=repo, config_home=config_home)
    assert validation["status"] == "ok"
    timeline = session_timeline(result.session_id, cwd=repo, config_home=config_home)
    assert [item["kind"] for item in timeline["events"][:3]] == ["user:message", "assistant:tool_use", "user:tool_result"]
    assert timeline["events"][2]["toolResults"][0]["mcpMetadata"]["serverName"] == "local-echo"

    candidates = extract_memory_candidates(result.session_id, cwd=repo, config_home=config_home)
    assert [candidate.sourceKind for candidate in candidates] == ["mcp_reference"]
    memory_dir = MemoryLoader(KernelConfig(cwd=repo, config_home=config_home)).get_auto_mem_path()
    assert not (memory_dir / ENTRYPOINT_NAME).exists()

    applied = apply_memory_candidates(candidates, session_id=result.session_id, cwd=repo, config_home=config_home)
    assert applied["candidateCount"] == 1
    written_path = applied["written"][0]["path"]
    assert (memory_dir / written_path).exists()
    assert memory_provenance(written_path, cwd=repo, config_home=config_home)["provenance"]["sourceKind"] == "mcp_reference"
    assert validate_memory_store(cwd=repo, config_home=config_home)["status"] == "ok"

    after_apply = session_timeline(result.session_id, cwd=repo, config_home=config_home)
    assert after_apply["events"][-1]["kind"] == "system:memory_extraction"

    resumed_provider = FakeModelProvider(["resumed"])
    resumed = build_local_engine(
        cwd=repo,
        config_home=config_home,
        model_provider=resumed_provider,
        session_id=result.session_id,
        resume=True,
        permission_mode="bypass",
        require_api_key=False,
    )
    asyncio.run(run_local_agent_once("continue with memory", engine=resumed, max_turns=1))
    system_prompt = "\n\n".join(resumed_provider.calls[0]["system_prompt"])
    assert "# auto memory" in system_prompt
    assert str(memory_dir) in system_prompt


def test_session_validation_reports_specific_chain_issues() -> None:
    """Session validation catches pairing, UUID, parent, and MCP metadata problems."""
    entries = [
        {"type": "user", "uuid": "u1", "sessionId": "s1", "parentUuid": None, "message": {"role": "user", "content": [{"type": "text", "text": "hi"}]}},
        {
            "type": "assistant",
            "uuid": "a1",
            "sessionId": "s1",
            "parentUuid": "u1",
            "message": {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "mcp__srv__echo", "input": {}}]},
        },
        {
            "type": "user",
            "uuid": "u2",
            "sessionId": "s1",
            "parentUuid": "a1",
            "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "orphan", "content": "oops", "mcpMetadata": {"serverName": "srv"}}]},
        },
        {"type": "assistant", "uuid": "u2", "sessionId": "s1", "parentUuid": "missing", "message": {"role": "assistant", "content": []}},
    ]

    result = validate_session_entries(entries, session_id="s1")
    codes = {issue["code"] for issue in result["issues"]}

    assert result["status"] == "error"
    assert {"orphan_tool_result", "missing_tool_result", "duplicate_uuid", "bad_parent_uuid", "bad_mcp_metadata"} <= codes


def test_mcp_metadata_is_transcript_only_for_model_api() -> None:
    """Provider API normalization strips MCP metadata from tool_result blocks."""
    assistant = create_assistant_message(
        [{"type": "tool_use", "id": "toolu_mcp", "name": "mcp__local__echo", "input": {}}],
        uuid="a1",
        message_id="m1",
    )
    tool_result = create_tool_result_message(
        {
            "type": "tool_result",
            "tool_use_id": "toolu_mcp",
            "content": "ok",
            "mcpMetadata": {"serverName": "local", "operation": "tools/call", "status": "ok"},
        },
        uuid="u1",
        source_tool_assistant_uuid="a1",
    )

    normalized = normalize_messages_for_api([assistant, tool_result])
    block = normalized[1]["message"]["content"][0]

    assert tool_result["message"]["content"][0]["mcpMetadata"]["serverName"] == "local"
    assert block == {"type": "tool_result", "tool_use_id": "toolu_mcp", "content": "ok"}


def test_missing_session_validate_and_extract_fail_clearly(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Missing transcripts do not look like valid empty sessions or empty extraction sets."""
    repo = _repo(tmp_path)
    config_home = tmp_path / ".claude"

    validate_result = validate_session("missing-session", cwd=repo, config_home=config_home)
    validate_code = main(["sessions", "validate", "missing-session", "--cwd", str(repo), "--config-home", str(config_home)])
    validate_output = capsys.readouterr()
    inspect_code = main(["sessions", "inspect", "missing-session", "--cwd", str(repo), "--config-home", str(config_home)])
    inspect_output = capsys.readouterr()
    timeline_code = main(["sessions", "timeline", "missing-session", "--cwd", str(repo), "--config-home", str(config_home)])
    timeline_output = capsys.readouterr()
    extract_code = main(["memory", "extract", "missing-session", "--cwd", str(repo), "--config-home", str(config_home), "--dry-run"])
    extract_output = capsys.readouterr()

    assert validate_result["status"] == "error"
    assert validate_result["issues"][0]["code"] == "missing_transcript"
    assert validate_code == 1
    assert "missing_transcript" in validate_output.out
    assert inspect_code == 2
    assert "Session transcript does not exist: missing-session" in inspect_output.err
    assert timeline_code == 2
    assert "Session transcript does not exist: missing-session" in timeline_output.err
    assert extract_code == 2
    assert "Session transcript does not exist: missing-session" in extract_output.err


def test_candidate_json_validation_is_clear(tmp_path: Path) -> None:
    """Candidate JSON review/apply path reports schema problems as ValueError."""
    candidate_path = tmp_path / "candidates.json"
    candidate_path.write_text(json.dumps({"candidates": [{"type": "project"}]}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing fields"):
        load_candidate_json(candidate_path)


def test_redacted_export_truncates_payloads_and_removes_fake_secrets(tmp_path: Path) -> None:
    """Redacted export preserves ordering while removing secret-like values and huge payloads."""
    repo = _repo(tmp_path)
    config = KernelConfig(cwd=repo, config_home=tmp_path / ".claude")
    store = local_agent.SessionStore(config, session_id="redact-session")
    store.record_transcript(
        [
            {
                "type": "user",
                "uuid": "u1",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "token sk-testsecret1234567890 " + ("x" * 800)}],
                },
            }
        ]
    )

    redacted = redacted_session_entries("redact-session", cwd=repo, config_home=tmp_path / ".claude")
    text = json.dumps(redacted, ensure_ascii=False)

    assert "sk-testsecret" not in text
    assert "[REDACTED]" in text
    assert "[REDACTED_TRUNCATED" in text


def test_sessions_gc_requires_age_for_confirmed_delete(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Confirmed session GC requires an age threshold; default remains dry-run."""
    repo = _repo(tmp_path)
    config_home = tmp_path / ".claude"

    code = main(["sessions", "gc", "--cwd", str(repo), "--config-home", str(config_home), "--yes"])
    captured = capsys.readouterr()

    assert code == 2
    assert "sessions gc --yes requires --older-than DAYS" in captured.err


def test_memory_validate_and_rebuild_index_contract(tmp_path: Path) -> None:
    """Memory validate catches index/frontmatter issues and rebuild-index writes a backup."""
    repo = _repo(tmp_path)
    config_home = tmp_path / ".claude"
    memory_dir = MemoryLoader(KernelConfig(cwd=repo, config_home=config_home)).get_auto_mem_path()
    memory_dir.mkdir(parents=True)
    (memory_dir / ENTRYPOINT_NAME).write_text("- [Gone](missing.md) - stale\n", encoding="utf-8")
    (memory_dir / "project").mkdir()
    (memory_dir / "project" / "one.md").write_text("---\nname: Duplicate\ndescription: one\ntype: project\n---\n\none\n", encoding="utf-8")
    (memory_dir / "project" / "two.md").write_text("---\nname: Duplicate\ndescription: two\ntype: invalid\n---\n\ntwo\n", encoding="utf-8")

    result = validate_memory_store(cwd=repo, config_home=config_home)
    codes = {issue["code"] for issue in result["issues"]}

    assert {"stale_index_link", "duplicate_frontmatter_name", "invalid_type", "missing_index_entry"} <= codes

    dry_run = rebuild_memory_index(cwd=repo, config_home=config_home)
    assert dry_run["status"] == "dry-run"
    applied = rebuild_memory_index(cwd=repo, config_home=config_home, apply=True)
    assert applied["status"] == "applied"
    assert applied["backupPath"] is not None
    assert Path(applied["backupPath"]).exists()
    assert "project/one.md" in (memory_dir / ENTRYPOINT_NAME).read_text(encoding="utf-8")


def test_mcp_doctor_static_start_and_inspect_cli(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """MCP doctor is static by default; --start and inspect are explicit local-only checks."""
    config_path = tmp_path / "static-mcp.json"
    config_path.write_text(json.dumps({"mcpServers": {"static": {"command": "python3", "args": ["missing.py"]}}}), encoding="utf-8")

    static_code = main(["mcp", "doctor", "--mcp-config", str(config_path), "--json"])
    static_output = capsys.readouterr()
    start_code = main(["mcp", "doctor", "--mcp-config", str(_stdio_mcp_config()), "--mcp-start", "--json"])
    start_output = capsys.readouterr()
    inspect_code = main(["mcp", "inspect", "local-echo", "--mcp-fixture", str(_example_mcp_fixture()), "--json"])
    inspect_output = capsys.readouterr()

    static_payload = json.loads(static_output.out)
    start_payload = json.loads(start_output.out)
    inspect_payload = json.loads(inspect_output.out)

    assert static_code == 0
    assert static_payload["started"] is False
    assert static_payload["servers"][0]["tools"] == []
    assert start_code == 0
    assert start_payload["started"] is True
    assert start_payload["servers"][0]["tools"] == ["mcp__stdio-echo__echo"]
    assert "command" not in start_payload["servers"][0]["diagnostics"]
    assert start_payload["servers"][0]["diagnostics"]["commandConfigured"] is True
    assert inspect_code == 0
    assert inspect_payload["toolDetails"][0]["fullName"] == "mcp__local-echo__echo"
    assert inspect_payload["resourceDetails"] == [
        {"uri": "fixture://local-echo/readme", "name": "Local Echo MCP Fixture", "mimeType": "text/plain"}
    ]
    assert "This fixture is local-only" not in inspect_output.out


def test_cli_json_events_schema_version_and_print_session_id(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runner JSON events add the v0.6 schema version without changing underlying events."""
    repo = _repo(tmp_path)
    original = local_agent.build_local_engine

    def fake_build_local_engine(**kwargs):
        return original(
            cwd=kwargs.get("cwd"),
            config_home=kwargs.get("config_home"),
            model_provider=FakeModelProvider(["json event final"]),
            session_id=kwargs.get("session_id"),
            permission_mode=kwargs.get("permission_mode") or "ask",
            require_api_key=False,
            memory_enabled=kwargs.get("memory_enabled"),
        )

    monkeypatch.setattr(local_agent, "build_local_engine", fake_build_local_engine)

    code = main(["--cwd", str(repo), "--config-home", str(tmp_path / ".claude"), "--json-events", "--print-session-id", "hello"])
    captured = capsys.readouterr()
    event_lines = [json.loads(line) for line in captured.err.splitlines() if line.startswith("{")]

    assert code == 0
    assert "json event final" in captured.out
    assert any(event.get("schemaVersion") == "0.6" and event.get("type") == "system" for event in event_lines)
    assert any(event.get("schemaVersion") == "0.6" and event.get("type") == "result" for event in event_lines)
    assert "session=" in captured.err
