"""Explicit transcript-to-memory workflows for the local Agent Base.

Memory extraction is deterministic and manual-only. It reads existing transcript
rows, proposes conservative candidates, and writes memory files only when the
caller explicitly applies candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import re
import shutil

from .config import KernelConfig
from .memory import ENTRYPOINT_NAME, MemoryLoader
from .messages import create_system_message
from .session import SessionStore


MEMORY_TYPES = {"user", "feedback", "project", "reference"}
SECRET_VALUE_RE = re.compile(r"\b(?:sk|ak|pk|rk)-[A-Za-z0-9_-]{8,}\b", re.IGNORECASE)
MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
MAX_CANDIDATE_BODY_CHARS = 1_200
MAX_INDEX_LINE_CHARS = 200


@dataclass(frozen=True)
class MemoryCandidate:
    type: str
    name: str
    path: str
    indexEntry: str
    body: str
    sourceSessionId: str
    sourceEventUuids: tuple[str, ...]
    sourceKind: str
    confidence: str = "medium"
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def as_json(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "name": self.name,
            "path": self.path,
            "indexEntry": self.indexEntry,
            "body": self.body,
            "sourceSessionId": self.sourceSessionId,
            "sourceEventUuids": list(self.sourceEventUuids),
            "sourceKind": self.sourceKind,
            "confidence": self.confidence,
            "warnings": list(self.warnings),
        }


def _config(cwd: str | Path | None = None, config_home: str | Path | None = None, *, memory_enabled: bool | None = None) -> KernelConfig:
    kwargs: dict[str, Any] = {"cwd": Path(cwd).expanduser() if cwd is not None else Path.cwd()}
    if config_home is not None:
        kwargs["config_home"] = Path(config_home).expanduser()
    if memory_enabled is not None:
        kwargs["auto_memory_enabled"] = memory_enabled
    return KernelConfig(**kwargs)


def _loader(cwd: str | Path | None = None, config_home: str | Path | None = None, *, memory_enabled: bool | None = None) -> MemoryLoader:
    return MemoryLoader(_config(cwd, config_home, memory_enabled=memory_enabled))


def _safe_relative_path(relative_path: str | Path) -> Path:
    path = Path(relative_path)
    if path.is_absolute() or any(part == ".." for part in path.parts) or not path.parts:
        raise ValueError("Memory path must be relative and stay inside the project memory directory.")
    return path


def _resolve_memory_path(loader: MemoryLoader, relative_path: str | Path) -> Path:
    relative = _safe_relative_path(relative_path)
    root = loader.get_auto_mem_path().resolve()
    target = (root / relative).resolve()
    if target != root and root not in target.parents:
        raise ValueError("Memory path must stay inside the project memory directory.")
    return target


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip().lower()).strip("-")
    return slug or "memory"


def _first_line(text: str, *, limit: int = 140) -> str:
    line = next((part.strip() for part in text.splitlines() if part.strip()), "").strip()
    return line[:limit] or "Memory extracted from session"


def _user_text(entry: dict[str, Any]) -> str:
    payload = entry.get("message")
    content = payload.get("content") if isinstance(payload, dict) else None
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    texts = [block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"]
    return "\n".join(text for text in texts if isinstance(text, str))


def _tool_uses(entry: dict[str, Any]) -> list[dict[str, Any]]:
    payload = entry.get("message")
    content = payload.get("content") if isinstance(payload, dict) else None
    return [block for block in content if isinstance(block, dict) and block.get("type") == "tool_use"] if isinstance(content, list) else []


def _tool_results(entry: dict[str, Any]) -> list[dict[str, Any]]:
    payload = entry.get("message")
    content = payload.get("content") if isinstance(payload, dict) else None
    return [block for block in content if isinstance(block, dict) and block.get("type") == "tool_result"] if isinstance(content, list) else []


def _looks_sensitive_or_noisy(text: str) -> bool:
    lowered = text.lower()
    return bool(
        SECRET_VALUE_RE.search(text)
        or len(text) > 6_000
        or "traceback (most recent call last)" in lowered
        or "authorization:" in lowered
        or "api_key" in lowered
    )


def _candidate_from_user_text(session_id: str, entry: dict[str, Any], text: str) -> MemoryCandidate | None:
    lowered = text.lower()
    if not any(marker in lowered for marker in ("remember", "prefer", "preference", "always ", "don't ", "do not ")):
        return None
    memory_type = "feedback" if any(marker in lowered for marker in ("prefer", "preference", "don't ", "do not ", "always ")) else "project"
    name = _first_line(text).removeprefix("remember").strip(" :") or _first_line(text)
    if len(name) > 80:
        name = name[:80].rstrip()
    relative = Path(memory_type) / f"{_slug(name)}.md"
    body = text.strip()[:MAX_CANDIDATE_BODY_CHARS]
    if _looks_sensitive_or_noisy(body):
        return None
    index = f"- [{name}]({relative.as_posix()}) - extracted from session {session_id}"
    return MemoryCandidate(
        type=memory_type,
        name=name,
        path=relative.as_posix(),
        indexEntry=index,
        body=body,
        sourceSessionId=session_id,
        sourceEventUuids=(str(entry.get("uuid")),),
        sourceKind="user_request",
        confidence="medium",
    )


def _candidate_from_mcp_result(session_id: str, tool_use: dict[str, Any], result_entry: dict[str, Any], result_block: dict[str, Any]) -> MemoryCandidate | None:
    name = str(tool_use.get("name") or "")
    if not name.startswith("mcp__") and name not in {"ListMcpResourcesTool", "ReadMcpResourceTool"}:
        return None
    metadata = result_block.get("mcpMetadata") if isinstance(result_block.get("mcpMetadata"), dict) else {}
    if metadata.get("serverName"):
        server = str(metadata["serverName"])
    elif "__" in name:
        server = name.split("__")[1]
    else:
        server = "mcp"
    operation = str(metadata.get("operation") or "tool_call")
    resource_uri = metadata.get("resourceUri")
    title = f"{server} {operation}".strip()
    relative = Path("reference") / f"{_slug(title)}.md"
    pointer = f"MCP {operation} on server `{server}`"
    if resource_uri:
        pointer += f" for resource `{resource_uri}`"
    body = "\n".join(
        [
            pointer + ".",
            "",
            f"Source session: `{session_id}`",
            f"Source tool: `{name}`",
            "This memory stores a pointer to the MCP source, not a raw dump of the MCP result.",
        ]
    )
    index = f"- [{title}]({relative.as_posix()}) - MCP reference from session {session_id}"
    return MemoryCandidate(
        type="reference",
        name=title,
        path=relative.as_posix(),
        indexEntry=index,
        body=body,
        sourceSessionId=session_id,
        sourceEventUuids=tuple(str(value) for value in (tool_use.get("_entryUuid"), result_entry.get("uuid")) if value),
        sourceKind="mcp_reference",
        confidence="medium",
    )


def extract_memory_candidates_from_entries(session_id: str, entries: list[dict[str, Any]]) -> list[MemoryCandidate]:
    candidates: list[MemoryCandidate] = []
    tool_uses_by_id: dict[str, dict[str, Any]] = {}
    seen_paths: set[str] = set()
    for entry in entries:
        if entry.get("type") == "assistant":
            for block in _tool_uses(entry):
                if isinstance(block.get("id"), str):
                    tool_uses_by_id[str(block["id"])] = {**block, "_entryUuid": entry.get("uuid")}
        if entry.get("type") == "user":
            text = _user_text(entry)
            if text:
                candidate = _candidate_from_user_text(session_id, entry, text)
                if candidate and candidate.path not in seen_paths:
                    candidates.append(candidate)
                    seen_paths.add(candidate.path)
            for result_block in _tool_results(entry):
                tool_id = str(result_block.get("tool_use_id") or "")
                tool_use = tool_uses_by_id.get(tool_id)
                if tool_use is None:
                    continue
                candidate = _candidate_from_mcp_result(session_id, tool_use, entry, result_block)
                if candidate and candidate.path not in seen_paths:
                    candidates.append(candidate)
                    seen_paths.add(candidate.path)
    return candidates


def extract_memory_candidates(
    session_id: str,
    *,
    cwd: str | Path | None = None,
    config_home: str | Path | None = None,
) -> list[MemoryCandidate]:
    store = SessionStore(_config(cwd, config_home), session_id=session_id)
    if not store.transcript_path.exists():
        raise FileNotFoundError(f"Session transcript does not exist: {session_id}")
    return extract_memory_candidates_from_entries(session_id, store.load_entries())


def load_candidate_json(path: str | Path) -> list[MemoryCandidate]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    items = payload.get("candidates") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ValueError("Candidate JSON must contain a candidates array.")
    candidates: list[MemoryCandidate] = []
    required_fields = {"type", "name", "path", "indexEntry", "body", "sourceSessionId", "sourceKind"}
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError("Candidate JSON entries must be objects.")
        missing = sorted(required_fields - set(item))
        if missing:
            raise ValueError(f"Candidate JSON entry {index} missing fields: {', '.join(missing)}.")
        candidates.append(
            MemoryCandidate(
                type=str(item["type"]),
                name=str(item["name"]),
                path=str(item["path"]),
                indexEntry=str(item["indexEntry"]),
                body=str(item["body"]),
                sourceSessionId=str(item["sourceSessionId"]),
                sourceEventUuids=tuple(str(value) for value in item.get("sourceEventUuids", [])),
                sourceKind=str(item["sourceKind"]),
                confidence=str(item.get("confidence") or "medium"),
                warnings=tuple(str(value) for value in item.get("warnings", [])),
            )
        )
    return candidates


def _frontmatter_content(candidate: MemoryCandidate) -> str:
    description = _first_line(candidate.body)
    return "\n".join(
        [
            "---",
            f"name: {candidate.name}",
            f"description: {description}",
            f"type: {candidate.type}",
            f"sourceSession: {candidate.sourceSessionId}",
            f"sourceKind: {candidate.sourceKind}",
            f"confidence: {candidate.confidence}",
            "---",
            "",
            candidate.body.rstrip(),
            "",
        ]
    )


def apply_memory_candidates(
    candidates: list[MemoryCandidate],
    *,
    session_id: str | None = None,
    cwd: str | Path | None = None,
    config_home: str | Path | None = None,
    memory_enabled: bool | None = None,
    memory_default_path: str | Path | None = None,
) -> dict[str, Any]:
    loader = _loader(cwd, config_home, memory_enabled=memory_enabled)
    memory_dir = loader.get_auto_mem_path()
    memory_root = memory_dir.resolve()
    validated: list[tuple[MemoryCandidate, Path]] = []
    seen_paths: set[str] = set()
    for candidate in candidates:
        if candidate.type not in MEMORY_TYPES:
            raise ValueError(f"Unsupported memory type: {candidate.type}")
        target = _resolve_memory_path(loader, candidate.path)
        relative_key = target.relative_to(memory_root).as_posix()
        if relative_key in seen_paths:
            raise ValueError(f"Duplicate memory candidate path: {relative_key}")
        seen_paths.add(relative_key)
        validated.append((candidate, target))
    memory_dir.mkdir(parents=True, exist_ok=True)
    written: list[dict[str, Any]] = []
    index_path = _resolve_memory_path(loader, memory_default_path or ENTRYPOINT_NAME)
    existing_index = index_path.read_text(encoding="utf-8") if index_path.exists() else ""
    index_lines = existing_index.splitlines()
    for candidate, target in validated:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_frontmatter_content(candidate), encoding="utf-8")
        target.chmod(0o600)
        provenance_path = target.with_suffix(target.suffix + ".provenance.json")
        provenance = candidate.as_json() | {"writtenAt": datetime.now(timezone.utc).isoformat()}
        provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        provenance_path.chmod(0o600)
        if candidate.path not in existing_index:
            index_lines.append(candidate.indexEntry)
            existing_index += "\n" + candidate.indexEntry
        written.append({"path": str(target.relative_to(memory_root)), "provenancePath": str(provenance_path.relative_to(memory_root))})
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text("\n".join(line for line in index_lines if line.strip()) + ("\n" if index_lines else ""), encoding="utf-8")
    index_path.chmod(0o600)
    if session_id:
        store = SessionStore(_config(cwd, config_home, memory_enabled=memory_enabled), session_id=session_id)
        store.record_transcript(
            [
                create_system_message(
                    f"Memory extraction wrote {len(written)} files.",
                    subtype="memory_extraction",
                    memoryExtraction={"written": written, "candidateCount": len(candidates)},
                )
            ]
        )
    return {"written": written, "candidateCount": len(candidates), "memoryDir": str(memory_dir)}


def _parse_frontmatter(text: str) -> dict[str, str]:
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---", 4)
    if end == -1:
        return {}
    result: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = value.strip()
    return result


def validate_memory_store(
    *,
    cwd: str | Path | None = None,
    config_home: str | Path | None = None,
    memory_enabled: bool | None = None,
    memory_default_path: str | Path | None = None,
) -> dict[str, Any]:
    loader = _loader(cwd, config_home, memory_enabled=memory_enabled)
    memory_dir = loader.get_auto_mem_path()
    if not memory_dir.exists():
        return {"status": "ok", "issues": [], "memoryDir": str(memory_dir)}
    root = memory_dir.resolve()
    issues: list[dict[str, Any]] = []
    index_path = _resolve_memory_path(loader, memory_default_path or ENTRYPOINT_NAME)
    index_links: set[str] = set()
    if index_path.exists():
        for line_number, line in enumerate(index_path.read_text(encoding="utf-8").splitlines(), start=1):
            if len(line) > MAX_INDEX_LINE_CHARS:
                issues.append({"code": "oversized_index_line", "path": str(index_path.relative_to(memory_dir)), "line": line_number})
            for match in MARKDOWN_LINK_RE.finditer(line):
                link = match.group(1)
                if not link.startswith(("http://", "https://")):
                    index_links.add(link)
                    try:
                        target = _resolve_memory_path(loader, link)
                    except ValueError:
                        issues.append({"code": "unsafe_index_link", "path": link})
                        continue
                    if not target.exists():
                        issues.append({"code": "stale_index_link", "path": link})
    seen_names: dict[str, str] = {}
    memory_files: set[str] = set()
    for path in sorted(memory_dir.rglob("*")):
        try:
            resolved = path.resolve()
        except OSError as exc:
            issues.append({"code": "unresolvable_path", "path": str(path), "message": str(exc)})
            continue
        if resolved != root and root not in resolved.parents:
            issues.append({"code": "symlink_escape", "path": str(path)})
            continue
        if not path.is_file() or path.name == ENTRYPOINT_NAME or path.name.endswith(".provenance.json"):
            continue
        if path.suffix != ".md":
            continue
        relative = path.relative_to(memory_dir).as_posix()
        memory_files.add(relative)
        meta = _parse_frontmatter(path.read_text(encoding="utf-8"))
        memory_type = meta.get("type")
        if memory_type not in MEMORY_TYPES:
            issues.append({"code": "invalid_type", "path": relative, "type": memory_type})
        name = meta.get("name")
        if name:
            if name in seen_names:
                issues.append({"code": "duplicate_frontmatter_name", "path": relative, "otherPath": seen_names[name], "name": name})
            seen_names[name] = relative
        if relative not in index_links:
            issues.append({"code": "missing_index_entry", "path": relative})
    for link in sorted(index_links):
        if link not in memory_files and not link.startswith(("http://", "https://")):
            target = memory_dir / link
            if target.exists() and target.name != ENTRYPOINT_NAME:
                continue
    return {"status": "ok" if not issues else "error", "issues": issues, "memoryDir": str(memory_dir)}


def rebuild_memory_index(
    *,
    cwd: str | Path | None = None,
    config_home: str | Path | None = None,
    memory_enabled: bool | None = None,
    memory_default_path: str | Path | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    loader = _loader(cwd, config_home, memory_enabled=memory_enabled)
    memory_dir = loader.get_auto_mem_path()
    entries: list[tuple[str, str, str]] = []
    if memory_dir.exists():
        for path in sorted(memory_dir.rglob("*.md")):
            if path.name == ENTRYPOINT_NAME:
                continue
            relative = path.relative_to(memory_dir).as_posix()
            meta = _parse_frontmatter(path.read_text(encoding="utf-8"))
            name = meta.get("name") or path.stem.replace("-", " ").title()
            memory_type = meta.get("type") if meta.get("type") in MEMORY_TYPES else "reference"
            entries.append((memory_type, name, relative))
    lines = [f"- [{name}]({relative}) - {memory_type}" for memory_type, name, relative in sorted(entries)]
    index_path = _resolve_memory_path(loader, memory_default_path or ENTRYPOINT_NAME)
    backup_path = None
    if apply:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        if index_path.exists():
            backup_path = index_path.with_suffix(index_path.suffix + ".bak")
            shutil.copy2(index_path, backup_path)
        index_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        index_path.chmod(0o600)
    return {
        "status": "applied" if apply else "dry-run",
        "indexPath": str(index_path),
        "backupPath": str(backup_path) if backup_path else None,
        "lines": lines,
    }


def memory_provenance(
    relative_path: str | Path,
    *,
    cwd: str | Path | None = None,
    config_home: str | Path | None = None,
    memory_enabled: bool | None = None,
) -> dict[str, Any]:
    loader = _loader(cwd, config_home, memory_enabled=memory_enabled)
    target = _resolve_memory_path(loader, relative_path)
    provenance_path = target.with_suffix(target.suffix + ".provenance.json")
    if not provenance_path.exists():
        return {"path": str(_safe_relative_path(relative_path)), "provenance": None}
    return {"path": str(_safe_relative_path(relative_path)), "provenance": json.loads(provenance_path.read_text(encoding="utf-8"))}
