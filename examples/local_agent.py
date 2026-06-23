"""Minimal local runner for the Python Agent Kernel.

This is intentionally an example-layer entry point, not a product CLI or TUI.
It wires user input into QueryEngine.submit_message(), prints concise event
logs, and leaves the core agent loop untouched.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_kernel import KernelConfig, MCPClientConfig, ModelProvider, QueryEngine, SessionStore, build_model_provider_from_env
from agent_kernel.memory import ENTRYPOINT_NAME, MemoryLoader
from agent_kernel.mcp import MCP_CONFIG_ENV, MCPConfigurationError, close_mcp_clients, load_mcp_config
from agent_kernel.skills import SkillDefinition, skill_from_markdown
import agent_kernel.web_adapters as _web_adapters
from agent_kernel.web_adapters import (
    WEB_FETCH_MAX_BYTES_ENV,
    WEB_FETCH_MAX_CHARS_ENV,
    WEB_FETCH_PROVIDER_ENV,
    WEB_FETCH_TIMEOUT_ENV,
    WEB_SEARCH_API_KEY_ENV,
    WEB_SEARCH_MODEL_ENV,
    WEB_SEARCH_PROVIDER_ENV,
    WEB_SEARCH_STUB_RESULTS_ENV,
    WEB_SEARCH_TIMEOUT_ENV,
    WEB_SEARCH_URL_ENV,
    WebFetchConfigurationError,
    WebFetchHandler,
    WebSearchConfigurationError,
    WebSearchHandler,
    format_web_fetch_unavailable_message,
    format_web_search_unavailable_message,
)


EventLogger = Callable[[str], None]
# Compatibility shim for existing tests that monkeypatch examples.local_agent.urlopen.
urlopen = _web_adapters.urlopen


class MissingCredentialsError(RuntimeError):
    """Raised when the real local runner has no model API credentials."""


class SkillsConfigurationError(RuntimeError):
    """Raised when the example runner cannot configure local skills."""


class MCPFixtureConfigurationError(RuntimeError):
    """Raised when the example runner cannot configure a local MCP fixture."""


class MemoryConfigurationError(RuntimeError):
    """Raised when the example runner cannot safely access local memory files."""


@dataclass
class LocalAgentRun:
    """Result returned by the example runner helper."""

    events: list[dict[str, Any]]
    final_response: str
    logs: list[str]
    session_id: str
    transcript_path: Path


def has_api_credentials(env: Mapping[str, str] | None = None) -> bool:
    """Return whether credentials are present for the selected model provider."""
    values = env or os.environ
    provider = (values.get("AGENT_KERNEL_PROVIDER") or "anthropic").strip().lower()
    if provider in {"openai-chat", "openai-responses", "openai-response"}:
        return bool(values.get("AGENT_KERNEL_API_KEY") or values.get("OPENAI_API_KEY"))
    return bool(values.get("AGENT_KERNEL_API_KEY") or values.get("ANTHROPIC_AUTH_TOKEN") or values.get("ANTHROPIC_API_KEY"))


def _sync_web_adapter_urlopen() -> None:
    """Keep legacy local_agent.urlopen monkeypatches wired to web adapters."""
    _web_adapters.urlopen = urlopen


def build_web_search_handler_from_env(env: Mapping[str, str] | None = None) -> WebSearchHandler | None:
    """Compatibility wrapper around the internal WebSearch adapter builder."""
    _sync_web_adapter_urlopen()
    handler = _web_adapters.build_web_search_handler_from_env(env)
    if handler is None:
        return None

    def wrapped(args: dict[str, Any]) -> Any:
        _sync_web_adapter_urlopen()
        return handler(args)

    return wrapped


def build_web_fetch_handler_from_env(env: Mapping[str, str] | None = None) -> WebFetchHandler | None:
    """Compatibility wrapper around the internal WebFetch adapter builder."""
    _sync_web_adapter_urlopen()
    handler = _web_adapters.build_web_fetch_handler_from_env(env)
    if handler is None:
        return None

    def wrapped(url: str) -> Any:
        _sync_web_adapter_urlopen()
        return handler(url)

    return wrapped


def make_http_web_fetch_handler(
    *,
    timeout_seconds: float = 10.0,
    max_bytes: int = 1_000_000,
    max_chars: int = 100_000,
) -> WebFetchHandler:
    """Compatibility wrapper around the internal HTTP WebFetch handler."""
    _sync_web_adapter_urlopen()
    handler = _web_adapters.make_http_web_fetch_handler(
        timeout_seconds=timeout_seconds,
        max_bytes=max_bytes,
        max_chars=max_chars,
    )

    def wrapped(url: str) -> Any:
        _sync_web_adapter_urlopen()
        return handler(url)

    return wrapped


make_stub_web_search_handler = _web_adapters.make_stub_web_search_handler
make_unavailable_web_fetch_handler = _web_adapters.make_unavailable_web_fetch_handler


def _apply_permission_mode(engine: QueryEngine, permission_mode: str) -> None:
    if permission_mode not in {"ask", "bypass"}:
        raise ValueError("permission_mode must be 'ask' or 'bypass'.")
    engine.tool_use_context.app_state.tool_permission_context.mode = permission_mode


def discover_local_skills(skills_dir: str | Path) -> list[SkillDefinition]:
    """Return valid skills under a local skills directory or raise a clear error."""
    root = Path(skills_dir).expanduser()
    if not root.exists():
        raise SkillsConfigurationError(f"Skills directory does not exist: {root}")
    if not root.is_dir():
        raise SkillsConfigurationError(f"Skills path is not a directory: {root}")
    skills: list[SkillDefinition] = []
    seen: dict[str, Path] = {}
    for child in sorted(root.iterdir(), key=lambda path: path.name):
        if not child.is_dir():
            continue
        skill = skill_from_markdown(child / "SKILL.md", name=child.name, loaded_from=str(root), source="local-runner")
        if skill is not None:
            if skill.name in seen:
                raise SkillsConfigurationError(f"Duplicate skill name '{skill.name}' in {root}: {seen[skill.name]} and {child / 'SKILL.md'}")
            seen[skill.name] = child / "SKILL.md"
            skills.append(skill)
    if not skills:
        raise SkillsConfigurationError(f"No valid skills found in {root}. Expected child directories containing SKILL.md.")
    return skills


def _runner_config(cwd: str | Path | None = None, config_home: str | Path | None = None) -> KernelConfig:
    config_kwargs: dict[str, Any] = {"cwd": Path(cwd).expanduser() if cwd is not None else Path.cwd()}
    if config_home is not None:
        config_kwargs["config_home"] = Path(config_home).expanduser()
    return KernelConfig(**config_kwargs)


def _session_project_dir(cwd: str | Path | None = None, config_home: str | Path | None = None) -> Path:
    config = _runner_config(cwd, config_home)
    return SessionStore(config, session_id="__probe__").project_dir


def list_local_sessions(cwd: str | Path | None = None, config_home: str | Path | None = None) -> list[str]:
    """Return deterministic local transcript session ids for the runner project."""
    project_dir = _session_project_dir(cwd, config_home)
    if not project_dir.exists():
        return []
    return sorted(path.stem for path in project_dir.glob("*.jsonl") if path.is_file())


def latest_local_session_id(cwd: str | Path | None = None, config_home: str | Path | None = None) -> str | None:
    """Return the most recently modified local transcript session id."""
    project_dir = _session_project_dir(cwd, config_home)
    if not project_dir.exists():
        return None
    sessions = [path for path in project_dir.glob("*.jsonl") if path.is_file()]
    if not sessions:
        return None
    latest = max(sessions, key=lambda path: (path.stat().st_mtime, path.name))
    return latest.stem


def _memory_loader(cwd: str | Path | None = None, config_home: str | Path | None = None) -> MemoryLoader:
    return MemoryLoader(_runner_config(cwd, config_home))


def _resolve_memory_path(
    loader: MemoryLoader,
    relative_path: str | Path | None,
) -> Path:
    path = Path(relative_path or ENTRYPOINT_NAME)
    if path.is_absolute():
        raise MemoryConfigurationError("Memory path must be relative.")
    if not path.parts or any(part == ".." for part in path.parts):
        raise MemoryConfigurationError("Memory path must stay inside the project memory directory.")
    memory_dir = loader.get_auto_mem_path()
    target = (memory_dir / path).resolve()
    root = memory_dir.resolve()
    if target != root and root not in target.parents:
        raise MemoryConfigurationError("Memory path must stay inside the project memory directory.")
    return target


def memory_status_lines(cwd: str | Path | None = None, config_home: str | Path | None = None) -> list[str]:
    """Return local memory status without creating or modifying memory files."""
    loader = _memory_loader(cwd, config_home)
    memory_dir = loader.get_auto_mem_path()
    entrypoint = memory_dir / ENTRYPOINT_NAME
    return [
        f"memory_dir={memory_dir}",
        f"memory_dir_exists={str(memory_dir.exists()).lower()}",
        f"entrypoint={entrypoint}",
        f"entrypoint_exists={str(entrypoint.exists()).lower()}",
    ]


def read_memory_file(
    relative_path: str | Path | None = None,
    *,
    cwd: str | Path | None = None,
    config_home: str | Path | None = None,
) -> str:
    """Read a safe relative memory file path."""
    target = _resolve_memory_path(_memory_loader(cwd, config_home), relative_path)
    try:
        return target.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise MemoryConfigurationError(f"Memory file does not exist: {relative_path or ENTRYPOINT_NAME}") from exc
    except OSError as exc:
        raise MemoryConfigurationError(f"Unable to read memory file: {exc}") from exc


def write_memory_file(
    relative_path: str | Path,
    text: str,
    *,
    cwd: str | Path | None = None,
    config_home: str | Path | None = None,
) -> Path:
    """Write a safe relative memory file path, creating parent directories."""
    target = _resolve_memory_path(_memory_loader(cwd, config_home), relative_path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        target.chmod(0o600)
    except OSError as exc:
        raise MemoryConfigurationError(f"Unable to write memory file: {exc}") from exc
    return target


def _load_json_file(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise MCPFixtureConfigurationError(f"Unable to read {label}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise MCPFixtureConfigurationError(f"{label} is not valid JSON: {exc}") from exc


class _TemplateDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _render_fixture_value(value: Any, args: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return value.format_map(_TemplateDict({key: str(item) for key, item in args.items()}))
    if isinstance(value, list):
        return [_render_fixture_value(item, args) for item in value]
    if isinstance(value, dict):
        return {key: _render_fixture_value(item, args) for key, item in value.items()}
    return value


def load_mcp_fixture(fixture_path: str | Path) -> MCPClientConfig:
    """Load a local-only MCP smoke fixture into an MCPClientConfig."""
    path = Path(fixture_path).expanduser()
    if not path.exists():
        raise MCPFixtureConfigurationError(f"MCP fixture does not exist: {path}")
    if not path.is_file():
        raise MCPFixtureConfigurationError(f"MCP fixture path is not a file: {path}")
    fixture = _load_json_file(path, "MCP fixture")
    if not isinstance(fixture, dict):
        raise MCPFixtureConfigurationError("MCP fixture must be a JSON object.")

    server_name = str(fixture.get("name") or fixture.get("server") or "").strip()
    if not server_name:
        raise MCPFixtureConfigurationError("MCP fixture must include a non-empty 'name'.")

    raw_tools = fixture.get("tools")
    if not isinstance(raw_tools, list) or not raw_tools:
        raise MCPFixtureConfigurationError("MCP fixture must include a non-empty 'tools' array.")

    tools: list[dict[str, Any]] = []
    results_by_tool: dict[str, Any] = {}
    for index, item in enumerate(raw_tools):
        if not isinstance(item, dict):
            raise MCPFixtureConfigurationError(f"MCP fixture tool at index {index} must be an object.")
        tool_name = str(item.get("name") or "").strip()
        if not tool_name:
            raise MCPFixtureConfigurationError(f"MCP fixture tool at index {index} must include a non-empty 'name'.")
        tool_def = {
            key: value
            for key, value in item.items()
            if key not in {"result", "response", "responseTemplate"}
        }
        tool_def.setdefault("description", f"Local MCP fixture tool: {tool_name}")
        tool_def.setdefault(
            "inputSchema",
            {"type": "object", "properties": {}, "additionalProperties": True},
        )
        tools.append(tool_def)
        results_by_tool[tool_name] = item.get("result", item.get("response", item.get("responseTemplate", f"{tool_name} completed.")))

    raw_resources = fixture.get("resources") or []
    if not isinstance(raw_resources, list):
        raise MCPFixtureConfigurationError("MCP fixture 'resources' must be an array when provided.")
    resources = tuple(resource for resource in raw_resources if isinstance(resource, dict))
    calls: list[dict[str, Any]] = []

    def call_tool(tool_name: str, args: dict[str, Any]) -> Any:
        calls.append({"tool_name": tool_name, "args": dict(args)})
        if tool_name not in results_by_tool:
            raise RuntimeError(f'MCP fixture tool "{tool_name}" not found.')
        return _render_fixture_value(results_by_tool[tool_name], args)

    def read_resource(uri: str) -> dict[str, Any]:
        resource = next((item for item in resources if item.get("uri") == uri), None)
        if resource is None:
            raise RuntimeError(f'MCP fixture resource "{uri}" not found.')
        return {"contents": [resource]}

    setattr(call_tool, "calls", calls)
    return MCPClientConfig(
        name=server_name,
        instructions=str(fixture.get("instructions") or ""),
        type=str(fixture.get("type") or "connected"),
        tools=tuple(tools),
        resources=resources,
        call_tool_handler=call_tool,
        read_resource_handler=read_resource if resources else None,
    )


def build_local_engine(
    *,
    cwd: str | Path | None = None,
    config_home: str | Path | None = None,
    model_provider: ModelProvider | None = None,
    session_id: str | None = None,
    model: str | None = None,
    web_search_handler: WebSearchHandler | None = None,
    web_fetch_handler: WebFetchHandler | None = None,
    skills_dir: str | Path | None = None,
    mcp_fixture: str | Path | None = None,
    mcp_config: str | Path | None = None,
    permission_mode: str = "ask",
    require_api_key: bool = True,
    resume: bool = False,
) -> QueryEngine:
    """Build a QueryEngine for local example use.

    Tests may pass a fake provider and set ``require_api_key=False``. Real CLI
    use requires API credentials so a missing key fails before any network path.
    """
    skills_path: Path | None = None
    if skills_dir is not None:
        skills_path = Path(skills_dir).expanduser()
        discover_local_skills(skills_path)
    mcp_fixture_path = Path(mcp_fixture).expanduser() if mcp_fixture is not None else None
    if mcp_fixture_path is not None:
        if not mcp_fixture_path.exists():
            raise MCPFixtureConfigurationError(f"MCP fixture does not exist: {mcp_fixture_path}")
        if not mcp_fixture_path.is_file():
            raise MCPFixtureConfigurationError(f"MCP fixture path is not a file: {mcp_fixture_path}")
    mcp_config_path = Path(mcp_config).expanduser() if mcp_config is not None else None
    if mcp_config_path is not None:
        if not mcp_config_path.exists():
            raise MCPConfigurationError(f"MCP config does not exist: {mcp_config_path}")
        if not mcp_config_path.is_file():
            raise MCPConfigurationError(f"MCP config path is not a file: {mcp_config_path}")
    if model_provider is None:
        if require_api_key and not has_api_credentials():
            raise MissingCredentialsError(
                "Missing Anthropic-compatible API credentials. Set AGENT_KERNEL_API_KEY, "
                "ANTHROPIC_AUTH_TOKEN, or ANTHROPIC_API_KEY. For OpenAI modes, set "
                "AGENT_KERNEL_PROVIDER=openai-chat or openai-responses and provide "
                "AGENT_KERNEL_API_KEY or OPENAI_API_KEY. Optional: AGENT_KERNEL_BASE_URL "
                "and AGENT_KERNEL_MODEL."
            )
        model_provider = build_model_provider_from_env(require_credentials=require_api_key)

    mcp_clients: tuple[MCPClientConfig, ...] = ()
    if mcp_fixture_path is not None:
        mcp_clients = (*mcp_clients, load_mcp_fixture(mcp_fixture_path))
    if mcp_config_path is not None:
        mcp_clients = (*mcp_clients, *load_mcp_config(mcp_config_path, cwd=cwd))

    config_kwargs: dict[str, Any] = {"cwd": Path(cwd).expanduser() if cwd is not None else Path.cwd()}
    if config_home is not None:
        config_kwargs["config_home"] = Path(config_home).expanduser()
    if skills_path is not None:
        config_kwargs["skill_paths"] = (skills_path,)
    config_kwargs["skill_discovery_mode"] = "explicit"
    if mcp_clients:
        config_kwargs["mcp_clients"] = mcp_clients
    config = KernelConfig(**config_kwargs)

    engine_kwargs: dict[str, Any] = {
        "model_provider": model_provider,
        "config": config,
    }
    if session_id is not None:
        engine_kwargs["session_id"] = session_id
    if model is not None:
        engine_kwargs["model"] = model
    if resume:
        engine_kwargs["resume"] = True
    engine = QueryEngine(**engine_kwargs)
    setattr(engine, "_agent_kernel_owned_mcp_clients", mcp_clients)
    _apply_permission_mode(engine, permission_mode)
    if web_search_handler is not None:
        engine.tool_use_context.web_search_handler = web_search_handler
    engine.tool_use_context.web_fetch_handler = web_fetch_handler or make_unavailable_web_fetch_handler()
    return engine


def _text_blocks(event: dict[str, Any]) -> list[str]:
    payload = event.get("message")
    content = payload.get("content") if isinstance(payload, dict) else None
    if not isinstance(content, list):
        return []
    return [
        block["text"]
        for block in content
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
    ]


def _tool_use_blocks(event: dict[str, Any]) -> list[dict[str, Any]]:
    payload = event.get("message")
    content = payload.get("content") if isinstance(payload, dict) else None
    if not isinstance(content, list):
        return []
    return [block for block in content if isinstance(block, dict) and block.get("type") == "tool_use"]


def _tool_result_blocks(event: dict[str, Any]) -> list[dict[str, Any]]:
    payload = event.get("message")
    content = payload.get("content") if isinstance(payload, dict) else None
    if not isinstance(content, list):
        return []
    return [block for block in content if isinstance(block, dict) and block.get("type") == "tool_result"]


def _extract_assistant_text(events: list[dict[str, Any]]) -> str:
    for event in reversed(events):
        if event.get("type") != "assistant":
            continue
        texts = _text_blocks(event)
        if texts:
            return texts[-1]
    return ""


def _shorten(value: Any, *, limit: int = 160) -> str:
    text = str(value).replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1]}..."


def format_event_log(event: dict[str, Any]) -> list[str]:
    """Convert kernel/SDK events into concise local-runner log lines."""
    event_type = event.get("type")
    if event_type == "system" and event.get("subtype") == "init":
        return [
            "[sdk:init] "
            f"session={event.get('session_id')} "
            f"model={event.get('model')} "
            f"permission={event.get('permissionMode')}"
        ]
    if event_type == "system" and event.get("subtype") in {"api_error", "error"}:
        return [f"[error] {_shorten(event.get('error') or event.get('content') or 'unknown error')}"]
    if event_type == "stream_request_start":
        return ["[model] request"]
    if event_type == "assistant":
        lines = []
        for block in _tool_use_blocks(event):
            lines.append(f"[tool_use] {block.get('name')} input={_shorten(block.get('input', {}))}")
        for text in _text_blocks(event):
            lines.append(f"[assistant] {_shorten(text)}")
        return lines
    if event_type == "user":
        lines = []
        for block in _tool_result_blocks(event):
            content = _shorten(block.get("content", ""))
            status = "error" if block.get("is_error") else "ok"
            if block.get("is_error") and "Permission denied" in content:
                lines.append(f"[permission] denied {content}")
            lines.append(f"[tool_result:{status}] {block.get('tool_use_id')} {content}")
        return lines
    if event_type == "context_compacted":
        return [f"[compact] pre={event.get('preCompactTokenCount')} post={event.get('postCompactTokenCount')}"]
    if event_type == "context_microcompacted":
        return [f"[compact:micro] saved={event.get('tokensSaved')} tool_ids={event.get('compactedToolIds')}"]
    if event_type == "context_compaction_failed":
        return [f"[compact:error] {_shorten(event.get('error', 'unknown error'))}"]
    if event_type == "tool_progress":
        progress = event.get("progress") or {}
        tool_name = event.get("tool_name") or "tool"
        if isinstance(progress, dict) and progress.get("type") == "query_update":
            return [f"[tool_progress] {tool_name} query={_shorten(progress.get('query', ''))}"]
        if isinstance(progress, dict) and progress.get("type") == "search_results_received":
            return [f"[tool_progress] {tool_name} results={progress.get('resultCount')} query={_shorten(progress.get('query', ''))}"]
        if isinstance(progress, dict) and progress.get("type") == "mcp_progress":
            return [
                "[tool_progress] "
                f"{tool_name} mcp={progress.get('serverName')}/{progress.get('toolName')} "
                f"status={progress.get('status')}"
            ]
        return [f"[tool_progress] {tool_name} {_shorten(progress)}"]
    if event_type == "terminal":
        terminal = event.get("terminal") or {}
        return [f"[terminal] reason={terminal.get('reason')} turns={terminal.get('turns')}"]
    if event_type == "result":
        if event.get("is_error"):
            return [f"[sdk:result] error stop_reason={event.get('stop_reason')}"]
        return [f"[sdk:result] success turns={event.get('num_turns')}"]
    return []


async def run_local_agent_once(
    prompt: str,
    *,
    engine: QueryEngine | None = None,
    cwd: str | Path | None = None,
    config_home: str | Path | None = None,
    model_provider: ModelProvider | None = None,
    session_id: str | None = None,
    model: str | None = None,
    web_search_handler: WebSearchHandler | None = None,
    web_fetch_handler: WebFetchHandler | None = None,
    skills_dir: str | Path | None = None,
    mcp_fixture: str | Path | None = None,
    mcp_config: str | Path | None = None,
    permission_mode: str | None = None,
    max_turns: int = 10,
    sdk_events: bool = True,
    event_logger: EventLogger | None = None,
    require_api_key: bool = True,
) -> LocalAgentRun:
    """Run one prompt through QueryEngine and collect logs plus final text."""
    created_engine = engine is None
    if engine is None:
        engine = build_local_engine(
            cwd=cwd,
            config_home=config_home,
            model_provider=model_provider,
            session_id=session_id,
            model=model,
            web_search_handler=web_search_handler,
            web_fetch_handler=web_fetch_handler,
            skills_dir=skills_dir,
            mcp_fixture=mcp_fixture,
            mcp_config=mcp_config,
            permission_mode=permission_mode or "ask",
            require_api_key=require_api_key,
        )
    else:
        if permission_mode is not None:
            _apply_permission_mode(engine, permission_mode)
        if web_search_handler is not None:
            engine.tool_use_context.web_search_handler = web_search_handler
        if web_fetch_handler is not None:
            engine.tool_use_context.web_fetch_handler = web_fetch_handler
    events: list[dict[str, Any]] = []
    logs = [f"[session] session={engine.session_id} transcript={engine.session_store.transcript_path}"]
    for line in logs:
        if event_logger is not None:
            event_logger(line)

    try:
        async for event in engine.submit_message(prompt, max_turns=max_turns, sdk_events=sdk_events):
            events.append(event)
            for line in format_event_log(event):
                logs.append(line)
                if event_logger is not None:
                    event_logger(line)
    finally:
        if created_engine:
            close_mcp_clients(getattr(engine, "_agent_kernel_owned_mcp_clients", ()))

    final_response = ""
    for event in reversed(events):
        if event.get("type") == "result" and not event.get("is_error"):
            final_response = str(event.get("result") or "")
            break
    if not final_response:
        final_response = _extract_assistant_text(events)

    return LocalAgentRun(
        events=events,
        final_response=final_response,
        logs=logs,
        session_id=engine.session_id,
        transcript_path=engine.session_store.transcript_path,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one prompt through the local Python Agent Kernel.")
    parser.add_argument("prompt", nargs="*", help="Prompt text. If omitted, the runner asks for one line on stdin.")
    parser.add_argument("--repl", action="store_true", help="Keep the same QueryEngine session open for repeated prompts.")
    parser.add_argument("--cwd", type=Path, default=Path.cwd(), help="Working directory for tool permissions and prompt context.")
    parser.add_argument("--config-home", type=Path, help="Override local config/transcript home.")
    parser.add_argument("--session-id", help="Use a stable transcript session id.")
    parser.add_argument("--list-sessions", action="store_true", help="List local transcript session ids for this cwd.")
    parser.add_argument("--resume", metavar="SESSION_ID", help="Resume an existing transcript session id.")
    parser.add_argument("--continue", dest="continue_session", action="store_true", help="Resume the most recently modified local session.")
    parser.add_argument("--memory-status", action="store_true", help="Show local project memory status without mutating files.")
    parser.add_argument("--memory-read", nargs="?", const=ENTRYPOINT_NAME, metavar="RELATIVE_PATH", help="Read a relative project memory file.")
    parser.add_argument("--memory-write", metavar="RELATIVE_PATH", help="Write a relative project memory file.")
    parser.add_argument("--memory-text", help="Text to write with --memory-write.")
    parser.add_argument("--model", help="Override the configured model for this runner.")
    parser.add_argument("--max-turns", type=int, default=10, help="Maximum model turns per submitted prompt.")
    parser.add_argument("--permission-mode", choices=("ask", "bypass"), default="ask", help="Permission mode to pass through to the kernel.")
    parser.add_argument("--enable-web-search", action="store_true", help=f"Enable example WebSearch provider from {WEB_SEARCH_PROVIDER_ENV}.")
    parser.add_argument("--web-search-provider", choices=("stub", "http-json", "anthropic-compatible"), help="Example WebSearch provider override.")
    parser.add_argument("--web-search-stub-results", type=Path, help=f"JSON file used by the 'stub' WebSearch provider.")
    parser.add_argument("--enable-web-fetch", action="store_true", help=f"Enable example WebFetch provider from {WEB_FETCH_PROVIDER_ENV}.")
    parser.add_argument("--web-fetch-provider", choices=("http",), help="Example WebFetch provider override.")
    parser.add_argument("--skills-dir", type=Path, help="Load local skills from child directories containing SKILL.md.")
    parser.add_argument("--list-skills", action="store_true", help="List skills from --skills-dir without calling a model.")
    parser.add_argument("--mcp-fixture", type=Path, help="Load a local-only MCP smoke fixture JSON file.")
    parser.add_argument("--mcp-config", type=Path, help=f"Load local stdio MCP servers from config JSON. Env: {MCP_CONFIG_ENV}.")
    parser.add_argument("--quiet", action="store_true", help="Only print assistant final responses to stdout.")
    return parser


async def _run_cli(args: argparse.Namespace) -> int:
    if args.list_sessions:
        sessions = list_local_sessions(args.cwd, args.config_home)
        if not sessions:
            print("No sessions found.")
            return 0
        for session_id in sessions:
            print(session_id)
        return 0

    if args.memory_status:
        for line in memory_status_lines(args.cwd, args.config_home):
            print(line)
        return 0

    if args.memory_read is not None:
        try:
            print(read_memory_file(args.memory_read, cwd=args.cwd, config_home=args.config_home), end="")
        except MemoryConfigurationError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        return 0

    if args.memory_write is not None:
        if args.memory_text is None:
            print("error: --memory-write requires --memory-text.", file=sys.stderr)
            return 2
        try:
            path = write_memory_file(args.memory_write, args.memory_text, cwd=args.cwd, config_home=args.config_home)
        except MemoryConfigurationError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"wrote {path}")
        return 0

    if args.list_skills:
        if args.skills_dir is None:
            print("No skills loaded. Pass --skills-dir PATH to inspect local skills.")
            return 0
        try:
            skills = discover_local_skills(args.skills_dir)
        except SkillsConfigurationError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        for skill in skills:
            print(f"{skill.name}\t{skill.display_description()}")
        return 0

    if args.skills_dir is not None:
        try:
            discover_local_skills(args.skills_dir)
        except SkillsConfigurationError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    search_env = dict(os.environ)
    if args.web_search_provider:
        search_env[WEB_SEARCH_PROVIDER_ENV] = args.web_search_provider
    elif args.web_search_stub_results:
        search_env[WEB_SEARCH_PROVIDER_ENV] = "stub"
    if args.web_search_stub_results:
        search_env[WEB_SEARCH_STUB_RESULTS_ENV] = str(args.web_search_stub_results)
    web_search_handler = None
    if args.enable_web_search or args.web_search_provider or args.web_search_stub_results:
        try:
            web_search_handler = build_web_search_handler_from_env(search_env)
        except WebSearchConfigurationError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if web_search_handler is None:
            print(f"error: {format_web_search_unavailable_message()}", file=sys.stderr)
            print(
                f"hint: set {WEB_SEARCH_PROVIDER_ENV}=stub or set {WEB_SEARCH_PROVIDER_ENV}=http-json "
                f"with {WEB_SEARCH_URL_ENV}; anthropic-compatible also requires {WEB_SEARCH_API_KEY_ENV} "
                f"and {WEB_SEARCH_MODEL_ENV}",
                file=sys.stderr,
            )
            return 2

    fetch_env = dict(os.environ)
    if args.web_fetch_provider:
        fetch_env[WEB_FETCH_PROVIDER_ENV] = args.web_fetch_provider
    web_fetch_handler = None
    if args.enable_web_fetch or args.web_fetch_provider:
        try:
            web_fetch_handler = build_web_fetch_handler_from_env(fetch_env)
        except WebFetchConfigurationError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if web_fetch_handler is None:
            print(f"error: {format_web_fetch_unavailable_message()}", file=sys.stderr)
            print(f"hint: set {WEB_FETCH_PROVIDER_ENV}=http or pass --web-fetch-provider http", file=sys.stderr)
            return 2

    mcp_config = args.mcp_config or (Path(os.environ[MCP_CONFIG_ENV]).expanduser() if os.environ.get(MCP_CONFIG_ENV) else None)
    if args.resume and args.continue_session:
        print("error: use either --resume SESSION_ID or --continue, not both.", file=sys.stderr)
        return 2
    if args.session_id and args.resume:
        print("error: use either --session-id or --resume SESSION_ID, not both.", file=sys.stderr)
        return 2
    session_id = args.session_id
    resume = False
    if args.resume:
        session_id = args.resume
        resume = True
    elif args.continue_session:
        latest_session = latest_local_session_id(args.cwd, args.config_home)
        if latest_session is None:
            print("error: no local sessions found to continue.", file=sys.stderr)
            return 2
        session_id = latest_session
        resume = True

    try:
        engine = build_local_engine(
            cwd=args.cwd,
            config_home=args.config_home,
            session_id=session_id,
            model=args.model,
            web_search_handler=web_search_handler,
            web_fetch_handler=web_fetch_handler,
            skills_dir=args.skills_dir,
            mcp_fixture=args.mcp_fixture,
            mcp_config=mcp_config,
            permission_mode=args.permission_mode,
            resume=resume,
        )
    except MissingCredentialsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except MCPFixtureConfigurationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except MCPConfigurationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    def log(line: str) -> None:
        if not args.quiet:
            print(line, file=sys.stderr)

    prompt = " ".join(args.prompt).strip()
    try:
        if args.repl:
            if prompt:
                prompts = [prompt]
            else:
                prompts = []
            while True:
                if prompts:
                    next_prompt = prompts.pop(0)
                else:
                    try:
                        next_prompt = input("user> ").strip()
                    except EOFError:
                        break
                if not next_prompt or next_prompt.lower() in {"exit", "quit"}:
                    break
                result = await run_local_agent_once(next_prompt, engine=engine, max_turns=args.max_turns, event_logger=log)
                print(result.final_response)
            return 0

        if not prompt:
            try:
                prompt = input("user> ").strip()
            except EOFError:
                prompt = ""
        if not prompt:
            print("error: prompt is empty", file=sys.stderr)
            return 2
        result = await run_local_agent_once(prompt, engine=engine, max_turns=args.max_turns, event_logger=log)
        print(result.final_response)
        return 0
    finally:
        close_mcp_clients(getattr(engine, "_agent_kernel_owned_mcp_clients", ()))


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for ``python3 examples/local_agent.py``."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(_run_cli(args))


if __name__ == "__main__":
    raise SystemExit(main())
