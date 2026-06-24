"""JSONL session transcript 的路径、追加写入、resume 与 compact 重放。

存储形态为 ``<config_home>/projects/<sanitized-cwd>/<sessionId>.jsonl``。每行是独立
JSON entry，包含 cwd/sessionId/version/timestamp/uuid/parentUuid/isSidechain 等外围字段
和原始消息内容。追加写而非重写，使异常退出前已产生的消息仍可恢复。

``_seen`` 防止 QueryEngine 在 resume 或 compact 更新后重复落盘；``_last_uuid`` 维护
当前 parent 链。tool_result 使用 sourceToolAssistantUUID 保留其逻辑来源。加载时会：
规范旧版字段、定位最新 compact boundary、应用 microcompact 删除列表、恢复 partial
compact preserved segment，并桥接旧 progress parent。

``load_messages`` 返回内部 loop 视图，``load_sdk_messages`` 返回映射后的外部视图；
SessionStore 不调用模型，也不决定哪些消息应 compact。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import json

from .config import KernelConfig
from .messages import Message
from .path_utils import sanitize_path
from .sdk import from_sdk_compact_metadata, to_sdk_messages


TRANSCRIPT_MESSAGE_TYPES = {"user", "assistant", "attachment", "system"}


def is_transcript_message(entry: dict) -> bool:
    """判断transcript 消息，供transcript 持久化流程使用。"""
    return entry.get("type") in TRANSCRIPT_MESSAGE_TYPES and isinstance(entry.get("uuid"), str)


def is_chain_participant(message: dict) -> bool:
    """判断chain participant，供transcript 持久化流程使用。"""
    return message.get("type") != "progress"


def _strip_transcript_fields(entry: dict) -> dict:
    """移除transcript fields，供transcript 持久化流程使用。"""
    stripped = dict(entry)
    stripped.pop("parentUuid", None)
    stripped.pop("isSidechain", None)
    return stripped


def _normalize_transcript_entry(entry: dict) -> dict:
    """规范化transcript 条目，供transcript 持久化流程使用。"""
    normalized = dict(entry)
    if "tool_use_result" in normalized and "toolUseResult" not in normalized:
        normalized["toolUseResult"] = normalized["tool_use_result"]
    if normalized.get("type") == "system" and normalized.get("subtype") == "compact_boundary":
        metadata = normalized.get("compactMetadata") or normalized.get("compact_metadata")
        if isinstance(metadata, dict):
            normalized["compactMetadata"] = from_sdk_compact_metadata(metadata)
    return normalized


def is_compact_boundary_message(entry: dict) -> bool:
    """判断压缩 边界 消息，供transcript 持久化流程使用。"""
    return entry.get("type") == "system" and entry.get("subtype") == "compact_boundary"


def is_microcompact_boundary_message(entry: dict) -> bool:
    """判断microcompact 边界 消息，供transcript 持久化流程使用。"""
    return entry.get("type") == "system" and entry.get("subtype") == "microcompact_boundary"


@dataclass
class SessionStore:
    """一个 sessionId 对应一个 JSONL 文件的持久化适配器。"""
    config: KernelConfig
    session_id: str = field(default_factory=lambda: str(uuid4()))
    version: str = "0.1.0-python-port"
    # _last_uuid 是下一条 chain participant 的 parentUuid。
    _last_uuid: str | None = None
    # _seen 防止 resume 后把已经存在的消息再次追加到 JSONL。
    _seen: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        """完成 dataclass 创建后的派生字段初始化与规范化。"""
        # 初始化 seen/parent 状态，使 resume 后第一次追加不会重复旧前缀。
        messages = self._load_transcript_messages()
        self._seen = {entry["uuid"] for entry in messages}
        last = next((entry for entry in reversed(messages) if is_chain_participant(entry)), None)
        self._last_uuid = last["uuid"] if last else None

    @property
    def project_dir(self) -> Path:
        """返回当前 workspace 对应的项目 transcript 存储目录。"""
        return self.config.workspace_runtime.sessions_dir

    @property
    def legacy_project_dir(self) -> Path:
        """Return the pre-v0.7 cwd-keyed transcript directory for resume compatibility."""
        return self.config.config_home / "projects" / sanitize_path(self.config.cwd)

    @property
    def transcript_path(self) -> Path:
        """返回当前 session 的 JSONL transcript 文件路径。"""
        path = self.project_dir / f"{self.session_id}.jsonl"
        legacy_path = self.legacy_project_dir / f"{self.session_id}.jsonl"
        if legacy_path != path and legacy_path.exists() and not path.exists():
            return legacy_path
        return path

    def record_transcript(self, messages: list[Message]) -> None:
        """追加尚未写入的消息，并更新 parentUuid 链。"""
        entries = []
        parent_uuid = self._last_uuid
        # 一旦本批出现新消息，后续遇到旧消息就不能再回退 parent 指针。
        seen_new_message = False
        for message in messages:
            uuid = message.get("uuid")
            if not uuid or not is_transcript_message(message) or message["type"] == "tombstone":
                continue
            if uuid in self._seen:
                if not seen_new_message and is_chain_participant(message):
                    # compact 后的新历史前缀可能以已存在消息结束，需要从它续链。
                    parent_uuid = uuid
                continue
            entry = self._serialize_message(message, parent_uuid)
            entries.append(entry)
            seen_new_message = True
            self._seen.add(uuid)
            if is_chain_participant(message):
                parent_uuid = uuid
        if not entries:
            return
        self._last_uuid = parent_uuid
        # 采用 append-only JSONL，异常退出时已经 flush 的历史仍然可恢复。
        self.project_dir.mkdir(parents=True, exist_ok=True)
        with self.transcript_path.open("a", encoding="utf-8") as fh:
            for entry in entries:
                fh.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.transcript_path.chmod(0o600)

    def _serialize_message(self, message: Message, parent_uuid: str | None) -> dict:
        """完成 ``_serialize_message`` 对应的transcript 持久化内部步骤。"""
        effective_parent_uuid = parent_uuid
        if message.get("type") == "user" and message.get("sourceToolAssistantUUID"):
            # tool_result 的逻辑父节点是发出 tool_use 的 assistant，而非前一条进度消息。
            effective_parent_uuid = message.get("sourceToolAssistantUUID")
        timestamp = message.get("timestamp") or datetime.now(timezone.utc).isoformat()
        tool_use_result = message.get("toolUseResult") or message.get("tool_use_result")
        return {
            **message,
            "requestId": message.get("requestId"),
            "parentUuid": effective_parent_uuid,
            "isSidechain": False,
            "userType": self.config.user_type,
            "cwd": str(self.config.cwd),
            "sessionId": self.session_id,
            "timestamp": timestamp,
            "version": self.version,
            "gitBranch": None,
            **({"toolUseResult": tool_use_result} if tool_use_result is not None else {}),
        }

    def load_messages(self) -> list[dict]:
        """加载并恢复可供 agent loop 继续使用的内部消息。"""
        return [_strip_transcript_fields(entry) for entry in self._load_transcript_messages()]

    def load_sdk_messages(self) -> list[dict]:
        """加载 transcript 并映射为 SDK 消息视图。"""
        return to_sdk_messages(self._load_transcript_messages(), session_id=self.session_id)

    def load_entries(self) -> list[dict]:
        """读取 transcript 中的原始规范化 JSON entries。"""
        if not self.transcript_path.exists():
            return []
        entries: list[dict] = []
        with self.transcript_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    entries.append(json.loads(line))
        return entries

    def _load_transcript_messages(self) -> list[dict]:
        """加载transcript 消息集合，供transcript 持久化流程使用。"""
        if not self.transcript_path.exists():
            return []
        # uuid 去重兼容旧 transcript 中重复写入的消息。
        messages_by_uuid: dict[str, dict] = {}
        progress_bridge: dict[str, str | None] = {}
        microcompacted_tool_ids: set[str] = set()
        for entry in self.load_entries():
            entry = _normalize_transcript_entry(entry)
            if entry.get("type") == "progress" and isinstance(entry.get("uuid"), str):
                # progress 不进入模型历史，但后继 parent 需要桥接到 progress 的父节点。
                parent = entry.get("parentUuid")
                progress_bridge[entry["uuid"]] = progress_bridge.get(parent, parent) if parent else None
                continue
            if is_microcompact_boundary_message(entry):
                metadata = entry.get("microcompactMetadata") or {}
                microcompacted_tool_ids.update(str(tool_id) for tool_id in metadata.get("compactedToolIds", []))
            if not is_transcript_message(entry):
                continue
            parent_uuid = entry.get("parentUuid")
            if parent_uuid in progress_bridge:
                entry = dict(entry)
                entry["parentUuid"] = progress_bridge[parent_uuid]
            messages_by_uuid[entry["uuid"]] = entry
        messages = list(messages_by_uuid.values())
        if microcompacted_tool_ids:
            messages = self._apply_microcompact_boundaries(messages, microcompacted_tool_ids)
        # 只恢复最新 full compact 之后的有效窗口。
        for index in range(len(messages) - 1, -1, -1):
            if is_compact_boundary_message(messages[index]):
                return self._messages_after_compact_boundary(messages, index)
        return messages

    def _messages_after_compact_boundary(self, messages: list[dict], boundary_index: int) -> list[dict]:
        """完成 ``_messages_after_compact_boundary`` 对应的transcript 持久化内部步骤。"""
        sliced = list(messages[boundary_index:])
        boundary = sliced[0]
        metadata = boundary.get("compactMetadata") or {}
        preserved = metadata.get("preservedSegment")
        if not isinstance(preserved, dict):
            return sliced
        head_uuid = preserved.get("headUuid")
        tail_uuid = preserved.get("tailUuid")
        anchor_uuid = preserved.get("anchorUuid")
        if not head_uuid or not tail_uuid or not anchor_uuid:
            return sliced
        uuid_to_index = {message.get("uuid"): index for index, message in enumerate(messages)}
        head_index = uuid_to_index.get(head_uuid)
        tail_index = uuid_to_index.get(tail_uuid)
        if head_index is None or tail_index is None or head_index > tail_index:
            return sliced
        preserved_messages = [dict(message) for message in messages[head_index : tail_index + 1]]
        if not preserved_messages:
            return sliced
        # preserved segment 在新链中挂到 boundary 指定 anchor，而不是旧历史父节点。
        preserved_messages[0]["parentUuid"] = anchor_uuid
        existing = {message.get("uuid") for message in sliced}
        preserved_messages = [message for message in preserved_messages if message.get("uuid") not in existing]
        if not preserved_messages:
            return sliced
        anchor_index = next((index for index, message in enumerate(sliced) if message.get("uuid") == anchor_uuid), 0)
        return [*sliced[: anchor_index + 1], *preserved_messages, *sliced[anchor_index + 1 :]]

    def _apply_microcompact_boundaries(self, messages: list[dict], tool_ids: set[str]) -> list[dict]:
        """应用microcompact boundaries，供transcript 持久化流程使用。"""
        patched: list[dict] = []
        for message in messages:
            if message.get("type") != "user":
                patched.append(message)
                continue
            payload = message.get("message")
            content = payload.get("content") if isinstance(payload, dict) else None
            if not isinstance(content, list):
                patched.append(message)
                continue
            changed = False
            new_content = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result" and str(block.get("tool_use_id")) in tool_ids:
                    new_content.append({**block, "content": "[Old tool result content cleared]"})
                    changed = True
                else:
                    new_content.append(block)
            if changed:
                patched.append({**message, "message": {**payload, "content": new_content}})
            else:
                patched.append(message)
        return patched
