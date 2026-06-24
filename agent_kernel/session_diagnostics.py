"""Session transcript validation, inspection, timeline, and redacted export.

This module is intentionally read-only: it loads JSONL transcript rows through
``SessionStore`` and derives diagnostics without changing session history.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
import json
import re

from .config import KernelConfig
from .session import SessionStore, is_compact_boundary_message, is_microcompact_boundary_message, is_transcript_message


SECRET_VALUE_RE = re.compile(r"\b(?:sk|ak|pk|rk)-[A-Za-z0-9_-]{8,}\b", re.IGNORECASE)
MAX_REDACTED_CONTENT_CHARS = 500


@dataclass(frozen=True)
class SessionIssue:
    code: str
    message: str
    severity: str = "error"
    uuid: str | None = None
    index: int | None = None

    def as_json(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "uuid": self.uuid,
            "index": self.index,
        }


def _store(session_id: str, *, cwd: str | Path | None = None, config_home: str | Path | None = None) -> SessionStore:
    config_kwargs: dict[str, Any] = {"cwd": Path(cwd).expanduser() if cwd is not None else Path.cwd()}
    if config_home is not None:
        config_kwargs["config_home"] = Path(config_home).expanduser()
    return SessionStore(KernelConfig(**config_kwargs), session_id=session_id)


def _content_blocks(entry: dict[str, Any]) -> list[dict[str, Any]]:
    payload = entry.get("message")
    content = payload.get("content") if isinstance(payload, dict) else None
    return [block for block in content if isinstance(block, dict)] if isinstance(content, list) else []


def _tool_uses(entry: dict[str, Any]) -> list[dict[str, Any]]:
    return [block for block in _content_blocks(entry) if block.get("type") == "tool_use"]


def _tool_results(entry: dict[str, Any]) -> list[dict[str, Any]]:
    return [block for block in _content_blocks(entry) if block.get("type") == "tool_result"]


def _is_mcp_tool_name(name: str) -> bool:
    return name.startswith("mcp__") or name in {"ListMcpResourcesTool", "ReadMcpResourceTool"}


def _mcp_metadata_issue(block: dict[str, Any], *, index: int, uuid: str | None) -> SessionIssue | None:
    metadata = block.get("mcpMetadata")
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        return SessionIssue("bad_mcp_metadata", "MCP metadata must be an object.", uuid=uuid, index=index)
    required = ("serverName", "operation", "status")
    missing = [field for field in required if not metadata.get(field)]
    if missing:
        return SessionIssue("bad_mcp_metadata", f"MCP metadata missing fields: {', '.join(missing)}", uuid=uuid, index=index)
    return None


def validate_session_entries(entries: list[dict[str, Any]], *, session_id: str | None = None) -> dict[str, Any]:
    """Return stable transcript diagnostics for already-loaded JSONL entries."""
    issues: list[SessionIssue] = []
    seen_uuids: set[str] = set()
    known_uuids: set[str] = set()
    tool_uses: dict[str, dict[str, Any]] = {}
    tool_results: dict[str, dict[str, Any]] = {}

    for index, entry in enumerate(entries):
        uuid = entry.get("uuid")
        if not isinstance(uuid, str) or not uuid:
            issues.append(SessionIssue("bad_uuid", "Transcript row must include a non-empty uuid.", index=index))
        else:
            if uuid in seen_uuids:
                issues.append(SessionIssue("duplicate_uuid", f"Duplicate uuid: {uuid}", uuid=uuid, index=index))
            seen_uuids.add(uuid)

        if session_id is not None and entry.get("sessionId") not in {None, session_id}:
            issues.append(SessionIssue("session_id_mismatch", f"Row sessionId does not match {session_id}.", uuid=uuid, index=index))

        if not is_transcript_message(entry):
            issues.append(SessionIssue("bad_row_shape", "Row is not a transcript message.", uuid=uuid, index=index))

        parent = entry.get("parentUuid")
        if parent is not None and parent not in known_uuids:
            issues.append(SessionIssue("bad_parent_uuid", f"parentUuid does not reference an earlier row: {parent}", uuid=uuid, index=index))
        if isinstance(uuid, str):
            known_uuids.add(uuid)

        if is_compact_boundary_message(entry):
            metadata = entry.get("compactMetadata")
            if not isinstance(metadata, dict):
                issues.append(SessionIssue("bad_compact_boundary", "Compact boundary must include compactMetadata.", uuid=uuid, index=index))
        if is_microcompact_boundary_message(entry):
            metadata = entry.get("microcompactMetadata")
            if not isinstance(metadata, dict):
                issues.append(SessionIssue("bad_microcompact_boundary", "Microcompact boundary must include microcompactMetadata.", uuid=uuid, index=index))

        for block in _tool_uses(entry):
            tool_id = block.get("id")
            if not isinstance(tool_id, str) or not tool_id:
                issues.append(SessionIssue("bad_tool_use", "tool_use block must include id.", uuid=uuid, index=index))
                continue
            if tool_id in tool_uses:
                issues.append(SessionIssue("duplicate_tool_use", f"Duplicate tool_use id: {tool_id}", uuid=uuid, index=index))
            tool_uses[tool_id] = {"uuid": uuid, "index": index, "name": block.get("name")}

        for block in _tool_results(entry):
            tool_id = block.get("tool_use_id")
            if not isinstance(tool_id, str) or not tool_id:
                issues.append(SessionIssue("bad_tool_result", "tool_result block must include tool_use_id.", uuid=uuid, index=index))
                continue
            if tool_id not in tool_uses:
                issues.append(SessionIssue("orphan_tool_result", f"tool_result has no prior tool_use: {tool_id}", uuid=uuid, index=index))
            if tool_id in tool_results:
                issues.append(SessionIssue("duplicate_tool_result", f"Duplicate tool_result for: {tool_id}", uuid=uuid, index=index))
            tool_results[tool_id] = {"uuid": uuid, "index": index}
            mcp_issue = _mcp_metadata_issue(block, index=index, uuid=uuid)
            if mcp_issue:
                issues.append(mcp_issue)

    for tool_id, tool_use in sorted(tool_uses.items(), key=lambda item: item[1]["index"]):
        if tool_id not in tool_results:
            issues.append(
                SessionIssue(
                    "missing_tool_result",
                    f"tool_use has no matching tool_result: {tool_id}",
                    uuid=tool_use.get("uuid"),
                    index=tool_use.get("index"),
                )
            )

    return {
        "status": "ok" if not issues else "error",
        "issueCount": len(issues),
        "issues": [issue.as_json() for issue in issues],
        "summary": {
            "messageCount": len(entries),
            "toolUseCount": len(tool_uses),
            "toolResultCount": len(tool_results),
            "mcpToolUseCount": sum(1 for item in tool_uses.values() if _is_mcp_tool_name(str(item.get("name") or ""))),
        },
    }


def validate_session(session_id: str, *, cwd: str | Path | None = None, config_home: str | Path | None = None) -> dict[str, Any]:
    store = _store(session_id, cwd=cwd, config_home=config_home)
    if not store.transcript_path.exists():
        return {
            "status": "error",
            "issueCount": 1,
            "issues": [
                SessionIssue(
                    "missing_transcript",
                    f"Session transcript does not exist: {session_id}",
                ).as_json()
            ],
            "summary": {
                "messageCount": 0,
                "toolUseCount": 0,
                "toolResultCount": 0,
                "mcpToolUseCount": 0,
            },
            "sessionId": session_id,
            "transcriptPath": str(store.transcript_path),
            "exists": False,
        }
    entries = store.load_entries()
    result = validate_session_entries(entries, session_id=session_id)
    result["sessionId"] = session_id
    result["transcriptPath"] = str(store.transcript_path)
    result["exists"] = store.transcript_path.exists()
    return result


def inspect_session(session_id: str, *, cwd: str | Path | None = None, config_home: str | Path | None = None) -> dict[str, Any]:
    store = _store(session_id, cwd=cwd, config_home=config_home)
    entries = store.load_entries()
    validation = validate_session(session_id, cwd=cwd, config_home=config_home)
    tool_uses = [block for entry in entries for block in _tool_uses(entry)]
    tool_results = [block for entry in entries for block in _tool_results(entry)]
    path = store.transcript_path
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat() if path.exists() else None
    return {
        "sessionId": session_id,
        "transcriptPath": str(path),
        "exists": path.exists(),
        "lastModified": modified,
        "messageCount": len(entries),
        "toolUseCount": len(tool_uses),
        "toolResultCount": len(tool_results),
        "mcpCallCount": sum(1 for block in tool_uses if _is_mcp_tool_name(str(block.get("name") or ""))),
        "permissionDenialCount": sum(1 for block in tool_results if block.get("is_error") and "Permission denied" in str(block.get("content", ""))),
        "compactionCount": sum(1 for entry in entries if is_compact_boundary_message(entry) or is_microcompact_boundary_message(entry)),
        "modelErrorCount": sum(1 for entry in entries if entry.get("type") == "system" and entry.get("subtype") in {"api_error", "error"}),
        "finalAssistantPresent": any(entry.get("type") == "assistant" for entry in reversed(entries)),
        "validationStatus": validation["status"],
        "validationIssueCount": validation["issueCount"],
    }


def session_timeline(session_id: str, *, cwd: str | Path | None = None, config_home: str | Path | None = None) -> dict[str, Any]:
    store = _store(session_id, cwd=cwd, config_home=config_home)
    if not store.transcript_path.exists():
        return {"sessionId": session_id, "transcriptPath": str(store.transcript_path), "exists": False, "events": []}
    rows: list[dict[str, Any]] = []
    for index, entry in enumerate(store.load_entries()):
        kind = str(entry.get("type") or "unknown")
        item: dict[str, Any] = {
            "index": index,
            "uuid": entry.get("uuid"),
            "parentUuid": entry.get("parentUuid"),
            "type": entry.get("type"),
            "timestamp": entry.get("timestamp"),
        }
        if entry.get("type") == "system":
            item["kind"] = f"system:{entry.get('subtype', 'informational')}"
            if entry.get("subtype") == "memory_extraction":
                item["memoryExtraction"] = entry.get("memoryExtraction")
        elif entry.get("type") == "assistant":
            tool_uses = _tool_uses(entry)
            item["kind"] = "assistant:tool_use" if tool_uses else "assistant:message"
            item["toolUses"] = [{"id": block.get("id"), "name": block.get("name")} for block in tool_uses]
        elif entry.get("type") == "user":
            tool_results = _tool_results(entry)
            item["kind"] = "user:tool_result" if tool_results else "user:message"
            item["toolResults"] = [
                {
                    "toolUseId": block.get("tool_use_id"),
                    "isError": bool(block.get("is_error")),
                    "mcpMetadata": block.get("mcpMetadata"),
                }
                for block in tool_results
            ]
        else:
            item["kind"] = kind
        rows.append(item)
    return {"sessionId": session_id, "transcriptPath": str(store.transcript_path), "exists": True, "events": rows}


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        redacted = SECRET_VALUE_RE.sub("[REDACTED]", value)
        if len(redacted) > MAX_REDACTED_CONTENT_CHARS:
            return f"{redacted[:MAX_REDACTED_CONTENT_CHARS]}...[REDACTED_TRUNCATED chars={len(redacted)}]"
        return redacted
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact_value(item) for key, item in value.items()}
    return value


def redacted_session_entries(session_id: str, *, cwd: str | Path | None = None, config_home: str | Path | None = None) -> list[dict[str, Any]]:
    store = _store(session_id, cwd=cwd, config_home=config_home)
    return [_redact_value(entry) for entry in store.load_entries()]


def collect_gc_targets(
    *,
    cwd: str | Path | None = None,
    config_home: str | Path | None = None,
    older_than_days: int | None = None,
) -> list[dict[str, Any]]:
    probe = _store("__probe__", cwd=cwd, config_home=config_home)
    project_dir = probe.project_dir
    if not project_dir.exists():
        return []
    cutoff = None
    if older_than_days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
    targets: list[dict[str, Any]] = []
    for path in sorted(project_dir.glob("*.jsonl")):
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if cutoff is not None and modified >= cutoff:
            continue
        targets.append({"sessionId": path.stem, "transcriptPath": str(path), "lastModified": modified.isoformat()})
    return targets


def delete_gc_targets(targets: list[dict[str, Any]]) -> list[str]:
    deleted: list[str] = []
    for target in targets:
        path = Path(str(target["transcriptPath"]))
        if path.exists() and path.is_file():
            path.unlink()
            deleted.append(str(path))
    return deleted
