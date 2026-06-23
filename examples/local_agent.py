"""Minimal local runner for the Python Agent Kernel.

This is intentionally an example-layer entry point, not a product CLI or TUI.
It wires user input into QueryEngine.submit_message(), prints concise event
logs, and leaves the core agent loop untouched.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_kernel import KernelConfig, MCPClientConfig, ModelProvider, QueryEngine, SessionStore, build_model_provider_from_env
from agent_kernel.memory import ENTRYPOINT_NAME, MemoryLoader
from agent_kernel.mcp import MCP_CONFIG_ENV, MCPConfigurationError, build_mcp_tool_name, close_mcp_clients, load_mcp_config, normalize_name_for_mcp
from agent_kernel.path_utils import find_git_root
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


class LocalConfigError(RuntimeError):
    """Raised when the local runner config file is invalid or unsafe."""


DEFAULT_LOCAL_CONFIG_NAME = "settings.json"


DEFAULT_LOCAL_CONFIG_TEMPLATE: dict[str, Any] = {
    "provider": {
        "type": "anthropic",
        "model": "",
        "baseUrl": "",
        "timeout": 60,
        "maxTokens": None,
    },
    "runner": {
        "permissionMode": "ask",
        "maxTurns": 10,
        "quiet": False,
        "jsonEvents": False,
        "printTranscriptPath": False,
    },
    "webSearch": {
        "enabled": False,
        "provider": "stub",
        "stubResults": "",
        "timeout": 10,
    },
    "webFetch": {
        "enabled": False,
        "provider": "http",
        "timeout": 10,
        "maxBytes": 1000000,
        "maxChars": 100000,
    },
    "skills": {
        "dirs": [],
        "discoveryMode": "explicit",
        "strictValidation": False,
    },
    "mcp": {
        "fixtures": [],
        "configs": [],
        "startupTimeout": 5,
        "toolTimeout": 5,
    },
    "session": {
        "defaultMode": "new",
    },
    "memory": {
        "enabled": True,
        "defaultPath": "MEMORY.md",
    },
    "debug": {
        "config": False,
        "tools": False,
        "provider": False,
        "redact": True,
    },
}


@dataclass(frozen=True)
class LocalRunnerConfig:
    """Example-layer persisted runner settings loaded from settings.json."""

    path: Path | None = None
    paths: tuple[Path, ...] = ()
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    provider_timeout: float | None = None
    provider_max_tokens: int | None = None
    permission_mode: str | None = None
    max_turns: int | None = None
    quiet: bool | None = None
    json_events: bool | None = None
    print_transcript_path: bool | None = None
    web_search_enabled: bool | None = None
    web_search_provider: str | None = None
    web_search_stub_results: Path | None = None
    web_search_timeout: float | None = None
    web_fetch_enabled: bool | None = None
    web_fetch_provider: str | None = None
    web_fetch_timeout: float | None = None
    web_fetch_max_bytes: int | None = None
    web_fetch_max_chars: int | None = None
    skill_dirs: tuple[Path, ...] = ()
    skill_discovery_mode: str | None = None
    skill_strict_validation: bool | None = None
    mcp_fixtures: tuple[Path, ...] = ()
    mcp_configs: tuple[Path, ...] = ()
    mcp_startup_timeout: float | None = None
    mcp_tool_timeout: float | None = None
    session_default_mode: str | None = None
    memory_enabled: bool | None = None
    memory_default_path: str | None = None
    debug_config: bool | None = None
    debug_tools: bool | None = None
    debug_provider: bool | None = None

    @property
    def skills_dir(self) -> Path | None:
        return self.skill_dirs[0] if self.skill_dirs else None

    @property
    def mcp_fixture(self) -> Path | None:
        return self.mcp_fixtures[0] if self.mcp_fixtures else None

    @property
    def mcp_config(self) -> Path | None:
        return self.mcp_configs[0] if self.mcp_configs else None


@dataclass
class LocalAgentRun:
    """Result returned by the example runner helper."""

    events: list[dict[str, Any]]
    final_response: str
    logs: list[str]
    session_id: str
    transcript_path: Path


class _NoopModelProvider:
    async def stream(self, *, messages: list[dict], system_prompt: list[str], tools: list[object], options: dict):
        if False:
            yield {}


def _as_non_empty_str(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise LocalConfigError(f"{label} must be a string.")
    stripped = value.strip()
    return stripped or None


def _as_optional_bool(value: Any, label: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise LocalConfigError(f"{label} must be a boolean.")
    return value


def _as_optional_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise LocalConfigError(f"{label} must be a positive integer.")
    return value


def _as_optional_float(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise LocalConfigError(f"{label} must be a positive number.")
    return float(value)


def _memory_default_path(value: Any) -> str | None:
    text = _as_non_empty_str(value, "memory.defaultPath")
    if text is None:
        return None
    path = Path(text)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise LocalConfigError("memory.defaultPath must be a relative path inside the project memory directory.")
    return text


def _config_section(payload: dict[str, Any], name: str) -> dict[str, Any]:
    value = payload.get(name) or {}
    if not isinstance(value, dict):
        raise LocalConfigError(f"{name} must be an object.")
    return value


def _config_path(value: Any, label: str, base_dir: Path) -> Path | None:
    text = _as_non_empty_str(value, label)
    if text is None:
        return None
    path = Path(text).expanduser()
    return path if path.is_absolute() else base_dir / path


def _config_paths(value: Any, label: str, base_dir: Path) -> tuple[Path, ...]:
    if value is None:
        return ()
    raw_values = value if isinstance(value, list) else [value]
    if not isinstance(raw_values, list):
        raise LocalConfigError(f"{label} must be a string or array of strings.")
    paths: list[Path] = []
    for index, item in enumerate(raw_values):
        path = _config_path(item, f"{label}[{index}]", base_dir)
        if path is not None:
            paths.append(path)
    return tuple(paths)


SECRET_KEY_PATTERNS = ("api_key", "apikey", "token", "secret", "password", "auth")


def _looks_secret_value(value: str) -> bool:
    stripped = value.strip()
    return bool(re.match(r"^(sk|ak|pk|rk)-[A-Za-z0-9_-]{16,}$", stripped))


def _reject_secret_config(value: Any, *, path: str = "settings") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            lowered = key_text.lower().replace("-", "_")
            if any(pattern in lowered for pattern in SECRET_KEY_PATTERNS):
                raise LocalConfigError(f"{path}.{key_text} looks like a secret. Put API keys and tokens in environment variables.")
            _reject_secret_config(item, path=f"{path}.{key_text}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_secret_config(item, path=f"{path}[{index}]")
    elif isinstance(value, str) and _looks_secret_value(value):
        raise LocalConfigError(f"{path} looks like a secret. Put API keys and tokens in environment variables.")


def _load_json_config_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise LocalConfigError(f"Unable to read local config: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise LocalConfigError(f"Local config is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise LocalConfigError("Local config must be a JSON object.")
    return payload


LOCAL_CONFIG_SECTIONS = {"provider", "runner", "webSearch", "webFetch", "skills", "mcp", "session", "memory", "debug"}


def _supported_local_config_payload(payload: dict[str, Any]) -> dict[str, Any]:
    supported = {key: value for key, value in payload.items() if key in LOCAL_CONFIG_SECTIONS}
    _reject_secret_config(supported)
    return supported


def _nearest_project_settings(cwd: Path) -> Path | None:
    root = find_git_root(cwd) or cwd
    current = cwd.resolve()
    root = root.resolve()
    for candidate in [current, *current.parents]:
        settings = candidate / DEFAULT_LOCAL_CONFIG_NAME
        if settings.exists():
            return settings
        if candidate == root:
            break
    return None


def _parse_local_config_payload(payload: dict[str, Any], path: Path) -> LocalRunnerConfig:
    payload = _supported_local_config_payload(payload)
    base_dir = path.parent
    provider = _config_section(payload, "provider")
    runner = _config_section(payload, "runner")
    web_search = _config_section(payload, "webSearch")
    web_fetch = _config_section(payload, "webFetch")
    skills = _config_section(payload, "skills")
    mcp = _config_section(payload, "mcp")
    session = _config_section(payload, "session")
    memory = _config_section(payload, "memory")
    debug = _config_section(payload, "debug")

    permission_mode = _as_non_empty_str(runner.get("permissionMode"), "runner.permissionMode")
    if permission_mode is not None and permission_mode not in {"ask", "bypass"}:
        raise LocalConfigError("runner.permissionMode must be 'ask' or 'bypass'.")
    discovery_mode = _as_non_empty_str(skills.get("discoveryMode"), "skills.discoveryMode")
    if discovery_mode is not None and discovery_mode not in {"ambient", "explicit"}:
        raise LocalConfigError("skills.discoveryMode must be 'ambient' or 'explicit'.")
    session_default_mode = _as_non_empty_str(session.get("defaultMode"), "session.defaultMode")
    if session_default_mode is not None and session_default_mode not in {"new", "continue"}:
        raise LocalConfigError("session.defaultMode must be 'new' or 'continue'.")

    skill_dirs = _config_paths(skills.get("dirs"), "skills.dirs", base_dir)
    legacy_skill_dir = _config_path(skills.get("dir"), "skills.dir", base_dir)
    if legacy_skill_dir is not None:
        skill_dirs = (*skill_dirs, legacy_skill_dir)
    mcp_fixtures = _config_paths(mcp.get("fixtures"), "mcp.fixtures", base_dir)
    legacy_mcp_fixture = _config_path(mcp.get("fixture"), "mcp.fixture", base_dir)
    if legacy_mcp_fixture is not None:
        mcp_fixtures = (*mcp_fixtures, legacy_mcp_fixture)
    mcp_configs = _config_paths(mcp.get("configs"), "mcp.configs", base_dir)
    legacy_mcp_config = _config_path(mcp.get("config"), "mcp.config", base_dir)
    if legacy_mcp_config is not None:
        mcp_configs = (*mcp_configs, legacy_mcp_config)

    return LocalRunnerConfig(
        path=path,
        paths=(path,),
        provider=_as_non_empty_str(provider.get("type"), "provider.type"),
        model=_as_non_empty_str(provider.get("model"), "provider.model"),
        base_url=_as_non_empty_str(provider.get("baseUrl"), "provider.baseUrl"),
        provider_timeout=_as_optional_float(provider.get("timeout"), "provider.timeout"),
        provider_max_tokens=_as_optional_int(provider.get("maxTokens"), "provider.maxTokens"),
        permission_mode=permission_mode,
        max_turns=_as_optional_int(runner.get("maxTurns"), "runner.maxTurns"),
        quiet=_as_optional_bool(runner.get("quiet"), "runner.quiet"),
        json_events=_as_optional_bool(runner.get("jsonEvents"), "runner.jsonEvents"),
        print_transcript_path=_as_optional_bool(runner.get("printTranscriptPath"), "runner.printTranscriptPath"),
        web_search_enabled=_as_optional_bool(web_search.get("enabled"), "webSearch.enabled"),
        web_search_provider=_as_non_empty_str(web_search.get("provider"), "webSearch.provider"),
        web_search_stub_results=_config_path(web_search.get("stubResults"), "webSearch.stubResults", base_dir),
        web_search_timeout=_as_optional_float(web_search.get("timeout"), "webSearch.timeout"),
        web_fetch_enabled=_as_optional_bool(web_fetch.get("enabled"), "webFetch.enabled"),
        web_fetch_provider=_as_non_empty_str(web_fetch.get("provider"), "webFetch.provider"),
        web_fetch_timeout=_as_optional_float(web_fetch.get("timeout"), "webFetch.timeout"),
        web_fetch_max_bytes=_as_optional_int(web_fetch.get("maxBytes"), "webFetch.maxBytes"),
        web_fetch_max_chars=_as_optional_int(web_fetch.get("maxChars"), "webFetch.maxChars"),
        skill_dirs=skill_dirs,
        skill_discovery_mode=discovery_mode,
        skill_strict_validation=_as_optional_bool(skills.get("strictValidation"), "skills.strictValidation"),
        mcp_fixtures=mcp_fixtures,
        mcp_configs=mcp_configs,
        mcp_startup_timeout=_as_optional_float(mcp.get("startupTimeout"), "mcp.startupTimeout"),
        mcp_tool_timeout=_as_optional_float(mcp.get("toolTimeout"), "mcp.toolTimeout"),
        session_default_mode=session_default_mode,
        memory_enabled=_as_optional_bool(memory.get("enabled"), "memory.enabled"),
        memory_default_path=_memory_default_path(memory.get("defaultPath")),
        debug_config=_as_optional_bool(debug.get("config"), "debug.config"),
        debug_tools=_as_optional_bool(debug.get("tools"), "debug.tools"),
        debug_provider=_as_optional_bool(debug.get("provider"), "debug.provider"),
    )


def _first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _merge_paths(*groups: tuple[Path, ...]) -> tuple[Path, ...]:
    merged: list[Path] = []
    seen: set[str] = set()
    for group in groups:
        for path in group:
            key = str(path.expanduser())
            if key not in seen:
                merged.append(path)
                seen.add(key)
    return tuple(merged)


def _merge_local_configs(configs: Sequence[LocalRunnerConfig]) -> LocalRunnerConfig:
    if not configs:
        return LocalRunnerConfig()
    # Later entries have higher precedence: user < project < explicit.
    ordered = list(configs)
    rev = list(reversed(ordered))
    return LocalRunnerConfig(
        path=rev[0].path,
        paths=tuple(config.path for config in ordered if config.path is not None),
        provider=_first_non_none(*(config.provider for config in rev)),
        model=_first_non_none(*(config.model for config in rev)),
        base_url=_first_non_none(*(config.base_url for config in rev)),
        provider_timeout=_first_non_none(*(config.provider_timeout for config in rev)),
        provider_max_tokens=_first_non_none(*(config.provider_max_tokens for config in rev)),
        permission_mode=_first_non_none(*(config.permission_mode for config in rev)),
        max_turns=_first_non_none(*(config.max_turns for config in rev)),
        quiet=_first_non_none(*(config.quiet for config in rev)),
        json_events=_first_non_none(*(config.json_events for config in rev)),
        print_transcript_path=_first_non_none(*(config.print_transcript_path for config in rev)),
        web_search_enabled=_first_non_none(*(config.web_search_enabled for config in rev)),
        web_search_provider=_first_non_none(*(config.web_search_provider for config in rev)),
        web_search_stub_results=_first_non_none(*(config.web_search_stub_results for config in rev)),
        web_search_timeout=_first_non_none(*(config.web_search_timeout for config in rev)),
        web_fetch_enabled=_first_non_none(*(config.web_fetch_enabled for config in rev)),
        web_fetch_provider=_first_non_none(*(config.web_fetch_provider for config in rev)),
        web_fetch_timeout=_first_non_none(*(config.web_fetch_timeout for config in rev)),
        web_fetch_max_bytes=_first_non_none(*(config.web_fetch_max_bytes for config in rev)),
        web_fetch_max_chars=_first_non_none(*(config.web_fetch_max_chars for config in rev)),
        skill_dirs=_merge_paths(*(config.skill_dirs for config in ordered)),
        skill_discovery_mode=_first_non_none(*(config.skill_discovery_mode for config in rev)),
        skill_strict_validation=_first_non_none(*(config.skill_strict_validation for config in rev)),
        mcp_fixtures=_merge_paths(*(config.mcp_fixtures for config in ordered)),
        mcp_configs=_merge_paths(*(config.mcp_configs for config in ordered)),
        mcp_startup_timeout=_first_non_none(*(config.mcp_startup_timeout for config in rev)),
        mcp_tool_timeout=_first_non_none(*(config.mcp_tool_timeout for config in rev)),
        session_default_mode=_first_non_none(*(config.session_default_mode for config in rev)),
        memory_enabled=_first_non_none(*(config.memory_enabled for config in rev)),
        memory_default_path=_first_non_none(*(config.memory_default_path for config in rev)),
        debug_config=_first_non_none(*(config.debug_config for config in rev)),
        debug_tools=_first_non_none(*(config.debug_tools for config in rev)),
        debug_provider=_first_non_none(*(config.debug_provider for config in rev)),
    )


def load_local_runner_config(
    config_path: str | Path | None = None,
    *,
    cwd: str | Path | None = None,
    config_home: str | Path | None = None,
) -> LocalRunnerConfig:
    """Load optional settings.json defaults for the local runner."""
    base_cwd = Path(cwd).expanduser() if cwd is not None else Path.cwd()
    configs: list[LocalRunnerConfig] = []

    config_home_path = Path(config_home).expanduser() if config_home is not None else _runner_config(base_cwd).config_home
    user_path = config_home_path / DEFAULT_LOCAL_CONFIG_NAME
    if user_path.exists():
        if not user_path.is_file():
            raise LocalConfigError(f"Local config path is not a file: {user_path}")
        configs.append(_parse_local_config_payload(_load_json_config_file(user_path), user_path))

    project_path = _nearest_project_settings(base_cwd)
    if project_path is not None and project_path != user_path:
        if not project_path.is_file():
            raise LocalConfigError(f"Local config path is not a file: {project_path}")
        configs.append(_parse_local_config_payload(_load_json_config_file(project_path), project_path))

    if config_path is not None:
        explicit_path = Path(config_path).expanduser()
        if not explicit_path.exists():
            raise LocalConfigError(f"Local config does not exist: {explicit_path}")
        if not explicit_path.is_file():
            raise LocalConfigError(f"Local config path is not a file: {explicit_path}")
        if explicit_path not in {config.path for config in configs}:
            configs.append(_parse_local_config_payload(_load_json_config_file(explicit_path), explicit_path))

    return _merge_local_configs(configs)


def write_default_local_config(path: str | Path) -> Path:
    """Create a starter local runner config without secrets."""
    target = Path(path).expanduser()
    if target.exists():
        raise LocalConfigError(f"Local config already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(DEFAULT_LOCAL_CONFIG_TEMPLATE, indent=2) + "\n", encoding="utf-8")
    return target


def local_config_env(config: LocalRunnerConfig, env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Overlay non-secret settings defaults onto an environment mapping."""
    values = dict(env or os.environ)
    if config.provider and "AGENT_KERNEL_PROVIDER" not in values:
        values["AGENT_KERNEL_PROVIDER"] = config.provider
    if config.model and not any(key in values for key in ("AGENT_KERNEL_MODEL", "ANTHROPIC_MODEL", "OPENAI_MODEL")):
        values["AGENT_KERNEL_MODEL"] = config.model
    if config.base_url and not any(key in values for key in ("AGENT_KERNEL_BASE_URL", "ANTHROPIC_BASE_URL", "OPENAI_BASE_URL")):
        values["AGENT_KERNEL_BASE_URL"] = config.base_url
    if config.provider_timeout and "AGENT_KERNEL_TIMEOUT" not in values:
        values["AGENT_KERNEL_TIMEOUT"] = str(config.provider_timeout)
    if config.provider_max_tokens and "AGENT_KERNEL_MAX_TOKENS" not in values:
        values["AGENT_KERNEL_MAX_TOKENS"] = str(config.provider_max_tokens)
    if config.web_search_provider and WEB_SEARCH_PROVIDER_ENV not in values:
        values[WEB_SEARCH_PROVIDER_ENV] = config.web_search_provider
    if config.web_search_stub_results and WEB_SEARCH_STUB_RESULTS_ENV not in values:
        values[WEB_SEARCH_STUB_RESULTS_ENV] = str(config.web_search_stub_results)
    if config.web_search_timeout and WEB_SEARCH_TIMEOUT_ENV not in values:
        values[WEB_SEARCH_TIMEOUT_ENV] = str(config.web_search_timeout)
    if config.web_fetch_provider and WEB_FETCH_PROVIDER_ENV not in values:
        values[WEB_FETCH_PROVIDER_ENV] = config.web_fetch_provider
    if config.web_fetch_timeout and WEB_FETCH_TIMEOUT_ENV not in values:
        values[WEB_FETCH_TIMEOUT_ENV] = str(config.web_fetch_timeout)
    if config.web_fetch_max_bytes and WEB_FETCH_MAX_BYTES_ENV not in values:
        values[WEB_FETCH_MAX_BYTES_ENV] = str(config.web_fetch_max_bytes)
    if config.web_fetch_max_chars and WEB_FETCH_MAX_CHARS_ENV not in values:
        values[WEB_FETCH_MAX_CHARS_ENV] = str(config.web_fetch_max_chars)
    if config.mcp_configs and MCP_CONFIG_ENV not in values:
        values[MCP_CONFIG_ENV] = os.pathsep.join(str(path) for path in config.mcp_configs)
    return values


@contextmanager
def _temporary_environ(values: Mapping[str, str]):
    original = os.environ.copy()
    os.environ.clear()
    os.environ.update(values)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(original)


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


@dataclass(frozen=True)
class SkillValidationReport:
    skills: tuple[SkillDefinition, ...]
    warnings: tuple[str, ...] = ()

    def as_json(self) -> dict[str, Any]:
        return {
            "skills": [skill.as_sdk_dict() | {"baseDir": str(skill.base_dir) if skill.base_dir else None} for skill in self.skills],
            "warnings": list(self.warnings),
        }


def _skill_identity(path: Path) -> str:
    try:
        return str(path.resolve(strict=True))
    except OSError:
        return str(path.absolute())


def discover_local_skills(skills_dir: str | Path) -> list[SkillDefinition]:
    """Return valid skills under one local skills directory or raise a clear error."""
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


def validate_local_skills(
    skills_dirs: Sequence[str | Path],
    *,
    strict: bool = False,
) -> SkillValidationReport:
    """Load and validate local skills from one or more explicit directories."""
    all_skills: list[SkillDefinition] = []
    warnings: list[str] = []
    seen_names: dict[str, Path] = {}
    seen_paths: dict[str, str] = {}
    for skills_dir in skills_dirs:
        skills = discover_local_skills(skills_dir)
        for skill in skills:
            skill_path = (skill.base_dir / "SKILL.md") if skill.base_dir is not None else Path(str(skills_dir)) / skill.name / "SKILL.md"
            identity = _skill_identity(skill_path)
            if skill.name in seen_names:
                raise SkillsConfigurationError(f"Duplicate skill name '{skill.name}': {seen_names[skill.name]} and {skill_path}")
            if identity in seen_paths:
                raise SkillsConfigurationError(f"Duplicate skill file through symlink or repeated path: {seen_paths[identity]} and {skill_path}")
            if skill.context == "fork":
                warnings.append(f"Skill '{skill.name}' declares context=fork; forked skills are not implemented.")
            if skill.extra_frontmatter:
                unknown = ", ".join(sorted(skill.extra_frontmatter))
                warnings.append(f"Skill '{skill.name}' has unsupported frontmatter keys: {unknown}.")
            seen_names[skill.name] = skill_path
            seen_paths[identity] = str(skill_path)
            all_skills.append(skill)
    if strict and warnings:
        raise SkillsConfigurationError("; ".join(warnings))
    return SkillValidationReport(tuple(all_skills), tuple(warnings))


def _format_skill_line(skill: SkillDefinition) -> str:
    return f"{skill.name}\t{skill.display_description()}"


def _find_skill_by_name(skills: Sequence[SkillDefinition], name: str) -> SkillDefinition | None:
    normalized = name[1:] if name.startswith("/") else name
    return next((skill for skill in skills if skill.name == normalized), None)


def _runner_config(
    cwd: str | Path | None = None,
    config_home: str | Path | None = None,
    *,
    memory_enabled: bool | None = None,
) -> KernelConfig:
    config_kwargs: dict[str, Any] = {"cwd": Path(cwd).expanduser() if cwd is not None else Path.cwd()}
    if config_home is not None:
        config_kwargs["config_home"] = Path(config_home).expanduser()
    if memory_enabled is not None:
        config_kwargs["auto_memory_enabled"] = memory_enabled
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


def _session_store_for(session_id: str, cwd: str | Path | None = None, config_home: str | Path | None = None) -> SessionStore:
    return SessionStore(_runner_config(cwd, config_home), session_id=session_id)


def _tool_result_pairing_issues(entries: Sequence[dict[str, Any]]) -> list[str]:
    tool_use_ids: set[str] = set()
    issues: list[str] = []
    for entry in entries:
        payload = entry.get("message")
        content = payload.get("content") if isinstance(payload, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and block.get("id"):
                tool_use_ids.add(str(block["id"]))
            if block.get("type") == "tool_result":
                tool_id = str(block.get("tool_use_id") or "")
                if tool_id and tool_id not in tool_use_ids:
                    issues.append(f"orphan tool_result: {tool_id}")
    return issues


def session_info(
    session_id: str,
    *,
    cwd: str | Path | None = None,
    config_home: str | Path | None = None,
) -> dict[str, Any]:
    store = _session_store_for(session_id, cwd, config_home)
    path = store.transcript_path
    entries = store.load_entries()
    pairing_issues = _tool_result_pairing_issues(entries)
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat() if path.exists() else None
    return {
        "sessionId": session_id,
        "transcriptPath": str(path),
        "exists": path.exists(),
        "lastModified": modified,
        "messageCount": len(entries),
        "hasToolResultPairingIssues": bool(pairing_issues),
        "toolResultPairingIssues": pairing_issues,
    }


def list_local_session_infos(cwd: str | Path | None = None, config_home: str | Path | None = None) -> list[dict[str, Any]]:
    return [session_info(session_id, cwd=cwd, config_home=config_home) for session_id in list_local_sessions(cwd, config_home)]


def delete_local_session(
    session_id: str,
    *,
    cwd: str | Path | None = None,
    config_home: str | Path | None = None,
) -> Path:
    store = _session_store_for(session_id, cwd, config_home)
    path = store.transcript_path
    if not path.exists():
        raise FileNotFoundError(f"Session transcript does not exist: {session_id}")
    path.unlink()
    return path


def _memory_loader(
    cwd: str | Path | None = None,
    config_home: str | Path | None = None,
    *,
    memory_enabled: bool | None = None,
) -> MemoryLoader:
    return MemoryLoader(_runner_config(cwd, config_home, memory_enabled=memory_enabled))


def _default_memory_path(memory_default_path: str | Path | None = None) -> str | Path:
    return memory_default_path or ENTRYPOINT_NAME


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


def memory_status_lines(
    cwd: str | Path | None = None,
    config_home: str | Path | None = None,
    *,
    memory_enabled: bool | None = None,
    memory_default_path: str | Path | None = None,
) -> list[str]:
    """Return local memory status without creating or modifying memory files."""
    loader = _memory_loader(cwd, config_home, memory_enabled=memory_enabled)
    memory_dir = loader.get_auto_mem_path()
    entrypoint = _resolve_memory_path(loader, _default_memory_path(memory_default_path))
    return [
        f"enabled={str(loader.config.auto_memory_enabled).lower()}",
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
    memory_enabled: bool | None = None,
    memory_default_path: str | Path | None = None,
) -> str:
    """Read a safe relative memory file path."""
    target = _resolve_memory_path(
        _memory_loader(cwd, config_home, memory_enabled=memory_enabled),
        relative_path or _default_memory_path(memory_default_path),
    )
    try:
        return target.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise MemoryConfigurationError(f"Memory file does not exist: {relative_path or _default_memory_path(memory_default_path)}") from exc
    except OSError as exc:
        raise MemoryConfigurationError(f"Unable to read memory file: {exc}") from exc


def list_memory_files(
    *,
    cwd: str | Path | None = None,
    config_home: str | Path | None = None,
    memory_enabled: bool | None = None,
) -> list[dict[str, Any]]:
    loader = _memory_loader(cwd, config_home, memory_enabled=memory_enabled)
    memory_dir = loader.get_auto_mem_path()
    if not memory_dir.exists():
        return []
    root = memory_dir.resolve()
    files: list[dict[str, Any]] = []
    for path in sorted(memory_dir.rglob("*")):
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved != root and root not in resolved.parents:
            continue
        if not path.is_file():
            continue
        try:
            relative = path.relative_to(memory_dir)
        except ValueError:
            continue
        files.append(
            {
                "path": str(relative),
                "bytes": path.stat().st_size,
                "modified": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
            }
        )
    return files


def write_memory_file(
    relative_path: str | Path,
    text: str,
    *,
    cwd: str | Path | None = None,
    config_home: str | Path | None = None,
    memory_enabled: bool | None = None,
) -> Path:
    """Write a safe relative memory file path, creating parent directories."""
    target = _resolve_memory_path(_memory_loader(cwd, config_home, memory_enabled=memory_enabled), relative_path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        target.chmod(0o600)
    except OSError as exc:
        raise MemoryConfigurationError(f"Unable to write memory file: {exc}") from exc
    return target


def append_memory_file(
    relative_path: str | Path,
    text: str,
    *,
    cwd: str | Path | None = None,
    config_home: str | Path | None = None,
    memory_enabled: bool | None = None,
) -> Path:
    target = _resolve_memory_path(_memory_loader(cwd, config_home, memory_enabled=memory_enabled), relative_path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(text)
        target.chmod(0o600)
    except OSError as exc:
        raise MemoryConfigurationError(f"Unable to append memory file: {exc}") from exc
    return target


def delete_memory_file(
    relative_path: str | Path,
    *,
    cwd: str | Path | None = None,
    config_home: str | Path | None = None,
    memory_enabled: bool | None = None,
) -> Path:
    target = _resolve_memory_path(_memory_loader(cwd, config_home, memory_enabled=memory_enabled), relative_path)
    if not target.exists():
        raise MemoryConfigurationError(f"Memory file does not exist: {relative_path}")
    try:
        target.unlink()
    except OSError as exc:
        raise MemoryConfigurationError(f"Unable to delete memory file: {exc}") from exc
    return target


def _memory_slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip().lower()).strip("-")
    return slug or "memory"


def remember_memory(
    *,
    memory_type: str,
    name: str,
    text: str,
    cwd: str | Path | None = None,
    config_home: str | Path | None = None,
    memory_enabled: bool | None = None,
    memory_default_path: str | Path | None = None,
) -> Path:
    if memory_type not in {"user", "feedback", "project", "reference"}:
        raise MemoryConfigurationError("Memory type must be one of: user, feedback, project, reference.")
    slug = _memory_slug(name)
    relative_path = Path(memory_type) / f"{slug}.md"
    content = "\n".join(
        [
            "---",
            f"name: {name}",
            f"description: {text.strip().splitlines()[0][:160] if text.strip() else name}",
            f"type: {memory_type}",
            "---",
            "",
            text.rstrip(),
            "",
        ]
    )
    target = write_memory_file(relative_path, content, cwd=cwd, config_home=config_home, memory_enabled=memory_enabled)
    index_line = f"- [{name}]({relative_path.as_posix()}) - {memory_type}\n"
    loader = _memory_loader(cwd, config_home, memory_enabled=memory_enabled)
    entrypoint = _resolve_memory_path(loader, _default_memory_path(memory_default_path))
    existing = entrypoint.read_text(encoding="utf-8") if entrypoint.exists() else ""
    if relative_path.as_posix() not in existing:
        append_memory_file(_default_memory_path(memory_default_path), index_line, cwd=cwd, config_home=config_home, memory_enabled=memory_enabled)
    return target


def validate_memory(
    *,
    cwd: str | Path | None = None,
    config_home: str | Path | None = None,
    memory_enabled: bool | None = None,
) -> list[str]:
    loader = _memory_loader(cwd, config_home, memory_enabled=memory_enabled)
    memory_dir = loader.get_auto_mem_path()
    if not memory_dir.exists():
        return []
    issues: list[str] = []
    root = memory_dir.resolve()
    for path in memory_dir.rglob("*"):
        try:
            resolved = path.resolve()
        except OSError as exc:
            issues.append(f"{path}: unable to resolve path: {exc}")
            continue
        if resolved != root and root not in resolved.parents:
            issues.append(f"{path}: escapes memory directory")
    return issues


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


def _ensure_unique_mcp_clients(clients: Sequence[MCPClientConfig]) -> None:
    seen_servers: dict[str, str] = {}
    seen_tools: dict[str, str] = {}
    for client in clients:
        normalized_server = normalize_name_for_mcp(client.name)
        if normalized_server in seen_servers:
            raise MCPConfigurationError(f'MCP server "{client.name}" collides with "{seen_servers[normalized_server]}" after name normalization.')
        seen_servers[normalized_server] = client.name
        for tool in client.tools:
            tool_name = str(tool.get("name") or "").strip()
            if not tool_name:
                raise MCPConfigurationError(f'MCP server "{client.name}" has a tool without a name.')
            full_name = build_mcp_tool_name(client.name, tool_name)
            if full_name in seen_tools:
                raise MCPConfigurationError(f'MCP tool "{full_name}" is registered more than once.')
            seen_tools[full_name] = client.name


def load_mcp_clients_for_runner(
    *,
    fixtures: Sequence[str | Path] = (),
    configs: Sequence[str | Path] = (),
    cwd: str | Path | None = None,
    startup_timeout_seconds: float | None = None,
    tool_timeout_seconds: float | None = None,
) -> tuple[MCPClientConfig, ...]:
    clients: tuple[MCPClientConfig, ...] = ()
    try:
        for fixture in fixtures:
            clients = (*clients, load_mcp_fixture(fixture))
        for config in configs:
            clients = (
                *clients,
                *load_mcp_config(
                    config,
                    cwd=cwd,
                    startup_timeout_seconds=startup_timeout_seconds,
                    tool_timeout_seconds=tool_timeout_seconds,
                ),
            )
        _ensure_unique_mcp_clients(clients)
        return clients
    except Exception:
        close_mcp_clients(clients)
        raise


def mcp_clients_summary(clients: Sequence[MCPClientConfig]) -> list[dict[str, Any]]:
    return [
        {
            "name": client.name,
            "type": client.type,
            "tools": [build_mcp_tool_name(client.name, str(tool.get("name") or "")) for tool in client.tools],
            "resources": [resource.get("uri") for resource in client.resources if isinstance(resource, dict)],
        }
        for client in clients
    ]


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
    skills_dirs: Sequence[str | Path] | None = None,
    skill_discovery_mode: str = "explicit",
    mcp_fixture: str | Path | None = None,
    mcp_fixtures: Sequence[str | Path] | None = None,
    mcp_config: str | Path | None = None,
    mcp_configs: Sequence[str | Path] | None = None,
    mcp_startup_timeout: float | None = None,
    mcp_tool_timeout: float | None = None,
    permission_mode: str = "ask",
    require_api_key: bool = True,
    resume: bool = False,
    provider_env: Mapping[str, str] | None = None,
    memory_enabled: bool | None = None,
) -> QueryEngine:
    """Build a QueryEngine for local example use.

    Tests may pass a fake provider and set ``require_api_key=False``. Real CLI
    use requires API credentials so a missing key fails before any network path.
    """
    skills_paths: tuple[Path, ...] = ()
    if skills_dir is not None:
        skills_paths = (*skills_paths, Path(skills_dir).expanduser())
    if skills_dirs:
        skills_paths = (*skills_paths, *(Path(path).expanduser() for path in skills_dirs))
    if skills_paths:
        validate_local_skills(skills_paths)
    if skill_discovery_mode not in {"ambient", "explicit"}:
        raise ValueError("skill_discovery_mode must be 'ambient' or 'explicit'.")

    mcp_fixture_paths: tuple[Path, ...] = ()
    if mcp_fixture is not None:
        mcp_fixture_paths = (*mcp_fixture_paths, Path(mcp_fixture).expanduser())
    if mcp_fixtures:
        mcp_fixture_paths = (*mcp_fixture_paths, *(Path(path).expanduser() for path in mcp_fixtures))
    for mcp_fixture_path in mcp_fixture_paths:
        if not mcp_fixture_path.exists():
            raise MCPFixtureConfigurationError(f"MCP fixture does not exist: {mcp_fixture_path}")
        if not mcp_fixture_path.is_file():
            raise MCPFixtureConfigurationError(f"MCP fixture path is not a file: {mcp_fixture_path}")

    mcp_config_paths: tuple[Path, ...] = ()
    if mcp_config is not None:
        mcp_config_paths = (*mcp_config_paths, Path(mcp_config).expanduser())
    if mcp_configs:
        mcp_config_paths = (*mcp_config_paths, *(Path(path).expanduser() for path in mcp_configs))
    for mcp_config_path in mcp_config_paths:
        if not mcp_config_path.exists():
            raise MCPConfigurationError(f"MCP config does not exist: {mcp_config_path}")
        if not mcp_config_path.is_file():
            raise MCPConfigurationError(f"MCP config path is not a file: {mcp_config_path}")
    if model_provider is None:
        credential_env = provider_env or os.environ
        if require_api_key and not has_api_credentials(credential_env):
            raise MissingCredentialsError(
                "Missing Anthropic-compatible API credentials. Set AGENT_KERNEL_API_KEY, "
                "ANTHROPIC_AUTH_TOKEN, or ANTHROPIC_API_KEY. For OpenAI modes, set "
                "AGENT_KERNEL_PROVIDER=openai-chat or openai-responses and provide "
                "AGENT_KERNEL_API_KEY or OPENAI_API_KEY. Optional: AGENT_KERNEL_BASE_URL "
                "and AGENT_KERNEL_MODEL."
            )
        if provider_env is None:
            model_provider = build_model_provider_from_env(require_credentials=require_api_key)
        else:
            with _temporary_environ(provider_env):
                model_provider = build_model_provider_from_env(require_credentials=require_api_key)

    mcp_clients = load_mcp_clients_for_runner(
        fixtures=mcp_fixture_paths,
        configs=mcp_config_paths,
        cwd=cwd,
        startup_timeout_seconds=mcp_startup_timeout,
        tool_timeout_seconds=mcp_tool_timeout,
    )

    config_kwargs: dict[str, Any] = {"cwd": Path(cwd).expanduser() if cwd is not None else Path.cwd()}
    if config_home is not None:
        config_kwargs["config_home"] = Path(config_home).expanduser()
    if memory_enabled is not None:
        config_kwargs["auto_memory_enabled"] = memory_enabled
    if skills_paths:
        config_kwargs["skill_paths"] = skills_paths
    config_kwargs["skill_discovery_mode"] = skill_discovery_mode
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
    event_observer: Callable[[dict[str, Any]], None] | None = None,
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
            if event_observer is not None:
                event_observer(event)
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
    parser.add_argument("--agent-config", type=Path, help=f"Load local runner defaults from JSON. Default: ./{DEFAULT_LOCAL_CONFIG_NAME} when present.")
    parser.add_argument("--init-config", nargs="?", const="", metavar="PATH", help=f"Write a starter local runner config and exit. Default: ./{DEFAULT_LOCAL_CONFIG_NAME}.")
    parser.add_argument("--doctor", action="store_true", help="Inspect local runner config without calling a model or network.")
    parser.add_argument("--doctor-json", action="store_true", help="Print local runner diagnostics as JSON.")
    parser.add_argument("--print-effective-config", action="store_true", help="Print redacted effective local runner config and exit.")
    parser.add_argument("--validate-config", action="store_true", help="Validate local runner settings and exit.")
    parser.add_argument("--dry-run-config", action="store_true", help="Validate and print effective config without model, MCP, or network calls.")
    parser.add_argument("--repl", action="store_true", help="Keep the same QueryEngine session open for repeated prompts.")
    parser.add_argument("--cwd", type=Path, default=Path.cwd(), help="Working directory for tool permissions and prompt context.")
    parser.add_argument("--config-home", type=Path, help="Override local config/transcript home.")
    parser.add_argument("--json", dest="json_output", action="store_true", help="Print supported management command output as JSON.")
    parser.add_argument("--json-events", action="store_true", help="Print SDK events as JSON lines while running.")
    parser.add_argument("--print-transcript-path", action="store_true", help="Print the transcript path after a run.")
    parser.add_argument("--debug-tools", action="store_true", help="Print resolved tool names without calling a model.")
    parser.add_argument("--debug-provider", action="store_true", help="Print redacted provider diagnostics without calling a model.")
    parser.add_argument("--session-id", help="Use a stable transcript session id.")
    parser.add_argument("--list-sessions", action="store_true", help="List local transcript session ids for this cwd.")
    parser.add_argument("--session-info", metavar="SESSION_ID", help="Show transcript metadata for a local session.")
    parser.add_argument("--session-transcript-path", metavar="SESSION_ID", help="Print transcript path for a local session.")
    parser.add_argument("--session-export", metavar="SESSION_ID", help="Print transcript JSONL for a local session.")
    parser.add_argument("--session-delete", metavar="SESSION_ID", help="Delete a local session transcript; requires --yes.")
    parser.add_argument("--resume", metavar="SESSION_ID", help="Resume an existing transcript session id.")
    parser.add_argument("--continue", dest="continue_session", action="store_true", help="Resume the most recently modified local session.")
    parser.add_argument("--yes", action="store_true", help="Confirm destructive local management commands.")
    parser.add_argument("--memory-status", action="store_true", help="Show local project memory status without mutating files.")
    parser.add_argument("--memory-list", action="store_true", help="List local memory files.")
    parser.add_argument("--memory-read", nargs="?", const="", metavar="RELATIVE_PATH", help="Read a relative project memory file.")
    parser.add_argument("--memory-write", metavar="RELATIVE_PATH", help="Write a relative project memory file.")
    parser.add_argument("--memory-append", metavar="RELATIVE_PATH", help="Append to a relative project memory file.")
    parser.add_argument("--memory-delete", metavar="RELATIVE_PATH", help="Delete a relative project memory file; requires --yes.")
    parser.add_argument("--memory-remember", action="store_true", help="Write a structured memory file and update MEMORY.md.")
    parser.add_argument("--memory-forget", metavar="RELATIVE_PATH", help="Delete a memory file; requires --yes.")
    parser.add_argument("--memory-validate", action="store_true", help="Validate memory paths stay inside the memory directory.")
    parser.add_argument("--memory-type", choices=("user", "feedback", "project", "reference"), help="Memory type for --memory-remember.")
    parser.add_argument("--memory-name", help="Memory name for --memory-remember.")
    parser.add_argument("--memory-text", help="Text to write with --memory-write.")
    parser.add_argument("--model", help="Override the configured model for this runner.")
    parser.add_argument("--max-turns", type=int, help="Maximum model turns per submitted prompt.")
    parser.add_argument("--permission-mode", choices=("ask", "bypass"), help="Permission mode to pass through to the kernel.")
    parser.add_argument("--enable-web-search", action="store_true", default=None, help=f"Enable example WebSearch provider from {WEB_SEARCH_PROVIDER_ENV}.")
    parser.add_argument("--web-search-provider", choices=("stub", "http-json", "anthropic-compatible"), help="Example WebSearch provider override.")
    parser.add_argument("--web-search-stub-results", type=Path, help=f"JSON file used by the 'stub' WebSearch provider.")
    parser.add_argument("--enable-web-fetch", action="store_true", default=None, help=f"Enable example WebFetch provider from {WEB_FETCH_PROVIDER_ENV}.")
    parser.add_argument("--web-fetch-provider", choices=("http",), help="Example WebFetch provider override.")
    parser.add_argument("--skills-dir", type=Path, action="append", help="Load local skills from child directories containing SKILL.md. Repeatable.")
    parser.add_argument("--list-skills", action="store_true", help="List skills from --skills-dir without calling a model.")
    parser.add_argument("--validate-skills", action="store_true", help="Validate configured local skills without calling a model.")
    parser.add_argument("--strict-skills", action="store_true", help="Treat skill validation warnings as errors.")
    parser.add_argument("--skill-info", metavar="NAME", help="Print one local skill definition summary.")
    parser.add_argument("--mcp-fixture", type=Path, action="append", help="Load a local-only MCP smoke fixture JSON file. Repeatable.")
    parser.add_argument("--mcp-config", type=Path, action="append", help=f"Load local stdio MCP servers from config JSON. Env: {MCP_CONFIG_ENV}. Repeatable.")
    parser.add_argument("--mcp-list", action="store_true", help="List configured MCP servers/tools without calling a model.")
    parser.add_argument("--mcp-doctor", action="store_true", help="Validate configured MCP files without calling a model.")
    parser.add_argument("--mcp-validate-config", type=Path, help="Validate one local stdio MCP config path.")
    parser.add_argument("--quiet", action="store_true", help="Only print assistant final responses to stdout.")
    return parser


def _configured_path_status(label: str, path: Path | None) -> str:
    if path is None:
        return f"{label}=not-configured"
    if path.exists():
        kind = "dir" if path.is_dir() else "file" if path.is_file() else "other"
        return f"{label}={path} status=ok type={kind}"
    return f"{label}={path} status=missing"


def local_doctor_lines(
    config: LocalRunnerConfig,
    *,
    env: Mapping[str, str],
    permission_mode: str,
    max_turns: int,
    web_search_enabled: bool,
    web_fetch_enabled: bool,
    skills_dirs: Sequence[Path],
    mcp_fixtures: Sequence[Path],
    mcp_configs: Sequence[Path],
) -> list[str]:
    """Return non-secret local runner diagnostics."""
    provider = (env.get("AGENT_KERNEL_PROVIDER") or "anthropic").strip().lower()
    model = env.get("AGENT_KERNEL_MODEL") or env.get("ANTHROPIC_MODEL") or env.get("OPENAI_MODEL") or ""
    return [
        f"config_path={config.path if config.path is not None else 'not-found'}",
        f"config_paths={','.join(str(path) for path in config.paths) if config.paths else 'none'}",
        f"provider={provider}",
        f"model={model or 'not-configured'}",
        f"base_url={'configured' if env.get('AGENT_KERNEL_BASE_URL') or env.get('ANTHROPIC_BASE_URL') or env.get('OPENAI_BASE_URL') else 'default'}",
        f"credentials={'present' if has_api_credentials(env) else 'missing'}",
        f"permission_mode={permission_mode}",
        f"max_turns={max_turns}",
        f"web_search={'enabled' if web_search_enabled else 'disabled'} provider={env.get(WEB_SEARCH_PROVIDER_ENV, 'not-configured')}",
        f"web_fetch={'enabled' if web_fetch_enabled else 'disabled'} provider={env.get(WEB_FETCH_PROVIDER_ENV, 'not-configured')}",
        *([_configured_path_status("skills_dir", path) for path in skills_dirs] or [_configured_path_status("skills_dir", None)]),
        *([_configured_path_status("mcp_fixture", path) for path in mcp_fixtures] or [_configured_path_status("mcp_fixture", None)]),
        *([_configured_path_status("mcp_config", path) for path in mcp_configs] or [_configured_path_status("mcp_config", None)]),
        "default_tests=offline",
    ]


def local_doctor_json(
    config: LocalRunnerConfig,
    *,
    env: Mapping[str, str],
    permission_mode: str,
    max_turns: int,
    web_search_enabled: bool,
    web_fetch_enabled: bool,
    skills_dirs: Sequence[Path],
    mcp_fixtures: Sequence[Path],
    mcp_configs: Sequence[Path],
) -> dict[str, Any]:
    provider = (env.get("AGENT_KERNEL_PROVIDER") or "anthropic").strip().lower()
    model = env.get("AGENT_KERNEL_MODEL") or env.get("ANTHROPIC_MODEL") or env.get("OPENAI_MODEL") or ""
    return {
        "configPath": str(config.path) if config.path is not None else None,
        "configPaths": [str(path) for path in config.paths],
        "provider": provider,
        "model": model or None,
        "baseUrlConfigured": bool(env.get("AGENT_KERNEL_BASE_URL") or env.get("ANTHROPIC_BASE_URL") or env.get("OPENAI_BASE_URL")),
        "credentialsPresent": has_api_credentials(env),
        "permissionMode": permission_mode,
        "maxTurns": max_turns,
        "webSearch": {"enabled": web_search_enabled, "provider": env.get(WEB_SEARCH_PROVIDER_ENV)},
        "webFetch": {"enabled": web_fetch_enabled, "provider": env.get(WEB_FETCH_PROVIDER_ENV)},
        "skillsDirs": [str(path) for path in skills_dirs],
        "mcpFixtures": [str(path) for path in mcp_fixtures],
        "mcpConfigs": [str(path) for path in mcp_configs],
        "defaultTests": "offline",
    }


def effective_config_json(
    *,
    config: LocalRunnerConfig,
    env: Mapping[str, str],
    permission_mode: str,
    max_turns: int,
    quiet: bool,
    json_events: bool,
    print_transcript_path: bool,
    web_search_enabled: bool,
    web_fetch_enabled: bool,
    skills_dirs: Sequence[Path],
    mcp_fixtures: Sequence[Path],
    mcp_configs: Sequence[Path],
) -> dict[str, Any]:
    return {
        "settingsFile": DEFAULT_LOCAL_CONFIG_NAME,
        "configPaths": [str(path) for path in config.paths],
        "provider": {
            "type": env.get("AGENT_KERNEL_PROVIDER") or "anthropic",
            "model": env.get("AGENT_KERNEL_MODEL") or env.get("ANTHROPIC_MODEL") or env.get("OPENAI_MODEL") or None,
            "baseUrlConfigured": bool(env.get("AGENT_KERNEL_BASE_URL") or env.get("ANTHROPIC_BASE_URL") or env.get("OPENAI_BASE_URL")),
            "credentials": "present" if has_api_credentials(env) else "missing",
        },
        "runner": {
            "permissionMode": permission_mode,
            "maxTurns": max_turns,
            "quiet": quiet,
            "jsonEvents": json_events,
            "printTranscriptPath": print_transcript_path,
        },
        "webSearch": {"enabled": web_search_enabled, "provider": env.get(WEB_SEARCH_PROVIDER_ENV)},
        "webFetch": {"enabled": web_fetch_enabled, "provider": env.get(WEB_FETCH_PROVIDER_ENV)},
        "skills": {"dirs": [str(path) for path in skills_dirs], "discoveryMode": config.skill_discovery_mode or "explicit"},
        "mcp": {
            "fixtures": [str(path) for path in mcp_fixtures],
            "configs": [str(path) for path in mcp_configs],
            "startupTimeout": config.mcp_startup_timeout,
            "toolTimeout": config.mcp_tool_timeout,
        },
        "memory": {
            "enabled": True if config.memory_enabled is None else config.memory_enabled,
            "defaultPath": config.memory_default_path or ENTRYPOINT_NAME,
        },
        "debug": {
            "config": bool(config.debug_config),
            "tools": bool(config.debug_tools),
            "provider": bool(config.debug_provider),
        },
    }


async def _run_cli(args: argparse.Namespace) -> int:
    init_config = getattr(args, "init_config", None)
    if init_config is not None:
        target = Path(init_config) if init_config else (args.agent_config or args.cwd / DEFAULT_LOCAL_CONFIG_NAME)
        try:
            path = write_default_local_config(target)
        except LocalConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"wrote {path}")
        return 0

    try:
        local_config = load_local_runner_config(args.agent_config, cwd=args.cwd, config_home=args.config_home)
    except LocalConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    configured_env = local_config_env(local_config, os.environ)
    if args.model:
        configured_env["AGENT_KERNEL_MODEL"] = args.model

    permission_mode = args.permission_mode or local_config.permission_mode or "ask"
    max_turns = args.max_turns if args.max_turns is not None else local_config.max_turns or 10
    quiet = bool(args.quiet or local_config.quiet)
    json_events = bool(args.json_events or local_config.json_events)
    print_transcript_path = bool(args.print_transcript_path or local_config.print_transcript_path)
    debug_config = bool(args.dry_run_config or local_config.debug_config)
    debug_tools = bool(args.debug_tools or local_config.debug_tools)
    debug_provider = bool(args.debug_provider or local_config.debug_provider)
    skills_dirs = tuple(args.skills_dir) if args.skills_dir else local_config.skill_dirs
    skill_discovery_mode = local_config.skill_discovery_mode or "explicit"
    web_search_provider = args.web_search_provider or local_config.web_search_provider
    web_search_stub_results = args.web_search_stub_results or local_config.web_search_stub_results
    enable_web_search = args.enable_web_search if args.enable_web_search is not None else bool(local_config.web_search_enabled)
    web_fetch_provider = args.web_fetch_provider or local_config.web_fetch_provider
    enable_web_fetch = args.enable_web_fetch if args.enable_web_fetch is not None else bool(local_config.web_fetch_enabled)
    mcp_fixtures = tuple(args.mcp_fixture) if args.mcp_fixture else local_config.mcp_fixtures
    if args.mcp_config:
        mcp_configs = tuple(args.mcp_config)
    elif os.environ.get(MCP_CONFIG_ENV):
        mcp_configs = tuple(Path(path).expanduser() for path in os.environ[MCP_CONFIG_ENV].split(os.pathsep) if path)
    else:
        mcp_configs = local_config.mcp_configs
    model = args.model or local_config.model
    effective_search_enabled = enable_web_search or bool(web_search_provider or web_search_stub_results)
    effective_fetch_enabled = enable_web_fetch or bool(web_fetch_provider)

    if args.validate_config:
        print(f"{DEFAULT_LOCAL_CONFIG_NAME} ok")
        return 0

    if args.doctor_json:
        print(
            json.dumps(
                local_doctor_json(
                    local_config,
                    env=configured_env,
                    permission_mode=permission_mode,
                    max_turns=max_turns,
                    web_search_enabled=effective_search_enabled,
                    web_fetch_enabled=effective_fetch_enabled,
                    skills_dirs=skills_dirs,
                    mcp_fixtures=mcp_fixtures,
                    mcp_configs=mcp_configs,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.doctor:
        for line in local_doctor_lines(
            local_config,
            env=configured_env,
            permission_mode=permission_mode,
            max_turns=max_turns,
            web_search_enabled=effective_search_enabled,
            web_fetch_enabled=effective_fetch_enabled,
            skills_dirs=skills_dirs,
            mcp_fixtures=mcp_fixtures,
            mcp_configs=mcp_configs,
        ):
            print(line)
        return 0

    effective = effective_config_json(
        config=local_config,
        env=configured_env,
        permission_mode=permission_mode,
        max_turns=max_turns,
        quiet=quiet,
        json_events=json_events,
        print_transcript_path=print_transcript_path,
        web_search_enabled=effective_search_enabled,
        web_fetch_enabled=effective_fetch_enabled,
        skills_dirs=skills_dirs,
        mcp_fixtures=mcp_fixtures,
        mcp_configs=mcp_configs,
    )
    if args.print_effective_config or args.dry_run_config or debug_config:
        print(json.dumps(effective, ensure_ascii=False, indent=2))
        return 0

    if debug_provider:
        print(json.dumps(effective["provider"], ensure_ascii=False, indent=2))
        return 0

    if args.list_sessions:
        if args.json_output:
            print(json.dumps(list_local_session_infos(args.cwd, args.config_home), ensure_ascii=False, indent=2))
            return 0
        session_ids = list_local_sessions(args.cwd, args.config_home)
        if not session_ids:
            print("No sessions found.")
            return 0
        for session_id in session_ids:
            print(session_id)
        return 0

    if args.session_info:
        info = session_info(args.session_info, cwd=args.cwd, config_home=args.config_home)
        if args.json_output:
            print(json.dumps(info, ensure_ascii=False, indent=2))
        else:
            for key, value in info.items():
                print(f"{key}={value}")
        return 0

    if args.session_transcript_path:
        print(session_info(args.session_transcript_path, cwd=args.cwd, config_home=args.config_home)["transcriptPath"])
        return 0

    if args.session_export:
        store = _session_store_for(args.session_export, args.cwd, args.config_home)
        if not store.transcript_path.exists():
            print(f"error: Session transcript does not exist: {args.session_export}", file=sys.stderr)
            return 2
        print(store.transcript_path.read_text(encoding="utf-8"), end="")
        return 0

    if args.session_delete:
        if not args.yes:
            print("error: --session-delete requires --yes.", file=sys.stderr)
            return 2
        try:
            path = delete_local_session(args.session_delete, cwd=args.cwd, config_home=args.config_home)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"deleted {path}")
        return 0

    if args.memory_status:
        for line in memory_status_lines(
            args.cwd,
            args.config_home,
            memory_enabled=local_config.memory_enabled,
            memory_default_path=local_config.memory_default_path,
        ):
            print(line)
        return 0

    if args.memory_list:
        files = list_memory_files(cwd=args.cwd, config_home=args.config_home, memory_enabled=local_config.memory_enabled)
        if args.json_output:
            print(json.dumps(files, ensure_ascii=False, indent=2))
        elif not files:
            print("No memory files found.")
        else:
            for item in files:
                print(f"{item['path']}\t{item['bytes']} bytes")
        return 0

    if args.memory_read is not None:
        try:
            print(
                read_memory_file(
                    args.memory_read,
                    cwd=args.cwd,
                    config_home=args.config_home,
                    memory_enabled=local_config.memory_enabled,
                    memory_default_path=local_config.memory_default_path,
                ),
                end="",
            )
        except MemoryConfigurationError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        return 0

    if args.memory_write is not None:
        if args.memory_text is None:
            print("error: --memory-write requires --memory-text.", file=sys.stderr)
            return 2
        try:
            path = write_memory_file(
                args.memory_write,
                args.memory_text,
                cwd=args.cwd,
                config_home=args.config_home,
                memory_enabled=local_config.memory_enabled,
            )
        except MemoryConfigurationError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"wrote {path}")
        return 0

    if args.memory_append is not None:
        if args.memory_text is None:
            print("error: --memory-append requires --memory-text.", file=sys.stderr)
            return 2
        try:
            path = append_memory_file(
                args.memory_append,
                args.memory_text,
                cwd=args.cwd,
                config_home=args.config_home,
                memory_enabled=local_config.memory_enabled,
            )
        except MemoryConfigurationError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"appended {path}")
        return 0

    if args.memory_delete is not None or args.memory_forget is not None:
        if not args.yes:
            print("error: memory delete/forget requires --yes.", file=sys.stderr)
            return 2
        target = args.memory_delete or args.memory_forget
        try:
            path = delete_memory_file(target, cwd=args.cwd, config_home=args.config_home, memory_enabled=local_config.memory_enabled)
        except MemoryConfigurationError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"deleted {path}")
        return 0

    if args.memory_remember:
        if not args.memory_type or not args.memory_name or args.memory_text is None:
            print("error: --memory-remember requires --memory-type, --memory-name, and --memory-text.", file=sys.stderr)
            return 2
        try:
            path = remember_memory(
                memory_type=args.memory_type,
                name=args.memory_name,
                text=args.memory_text,
                cwd=args.cwd,
                config_home=args.config_home,
                memory_enabled=local_config.memory_enabled,
                memory_default_path=local_config.memory_default_path,
            )
        except MemoryConfigurationError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"remembered {path}")
        return 0

    if args.memory_validate:
        issues = validate_memory(cwd=args.cwd, config_home=args.config_home, memory_enabled=local_config.memory_enabled)
        if args.json_output:
            print(json.dumps({"issues": issues}, ensure_ascii=False, indent=2))
        elif issues:
            for issue in issues:
                print(issue)
        else:
            print("memory ok")
        return 1 if issues else 0

    if args.list_skills:
        if not skills_dirs:
            print("No skills loaded. Pass --skills-dir PATH to inspect local skills.")
            return 0
        try:
            report = validate_local_skills(skills_dirs, strict=bool(args.strict_skills or local_config.skill_strict_validation))
        except SkillsConfigurationError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if args.json_output:
            print(json.dumps(report.as_json(), ensure_ascii=False, indent=2))
        else:
            for warning in report.warnings:
                print(f"warning: {warning}", file=sys.stderr)
            for skill in report.skills:
                print(_format_skill_line(skill))
        return 0

    if args.validate_skills:
        if not skills_dirs:
            print("No skills loaded. Pass --skills-dir PATH to validate local skills.")
            return 0
        try:
            report = validate_local_skills(skills_dirs, strict=bool(args.strict_skills or local_config.skill_strict_validation))
        except SkillsConfigurationError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if args.json_output:
            print(json.dumps(report.as_json(), ensure_ascii=False, indent=2))
        else:
            for warning in report.warnings:
                print(f"warning: {warning}")
            print(f"skills ok ({len(report.skills)})")
        return 0

    if args.skill_info:
        if not skills_dirs:
            print("error: --skill-info requires --skills-dir or configured skills.dirs.", file=sys.stderr)
            return 2
        try:
            report = validate_local_skills(skills_dirs)
        except SkillsConfigurationError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        skill = _find_skill_by_name(report.skills, args.skill_info)
        if skill is None:
            print(f"error: Skill not found: {args.skill_info}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                skill.as_sdk_dict()
                | {
                    "whenToUse": skill.when_to_use,
                    "allowedTools": list(skill.allowed_tools),
                    "argumentHint": skill.argument_hint,
                    "argumentNames": list(skill.argument_names),
                    "context": skill.context,
                    "baseDir": str(skill.base_dir) if skill.base_dir else None,
                    "paths": list(skill.paths),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if skills_dirs:
        try:
            validate_local_skills(skills_dirs, strict=bool(local_config.skill_strict_validation))
        except SkillsConfigurationError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    if args.mcp_validate_config:
        clients: tuple[MCPClientConfig, ...] = ()
        try:
            clients = load_mcp_clients_for_runner(
                configs=(args.mcp_validate_config,),
                cwd=args.cwd,
                startup_timeout_seconds=local_config.mcp_startup_timeout,
                tool_timeout_seconds=local_config.mcp_tool_timeout,
            )
            print("mcp config ok")
            return 0
        except (MCPConfigurationError, MCPFixtureConfigurationError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        finally:
            close_mcp_clients(clients)

    if args.mcp_list or args.mcp_doctor:
        clients: tuple[MCPClientConfig, ...] = ()
        try:
            clients = load_mcp_clients_for_runner(
                fixtures=mcp_fixtures,
                configs=mcp_configs,
                cwd=args.cwd,
                startup_timeout_seconds=local_config.mcp_startup_timeout,
                tool_timeout_seconds=local_config.mcp_tool_timeout,
            )
            summary = mcp_clients_summary(clients)
            if args.json_output:
                print(json.dumps({"servers": summary}, ensure_ascii=False, indent=2))
            elif args.mcp_doctor:
                print(f"mcp ok ({len(summary)} servers)")
                for server in summary:
                    print(f"{server['name']}\ttools={len(server['tools'])}\tresources={len(server['resources'])}")
            else:
                if not summary:
                    print("No MCP servers configured.")
                for server in summary:
                    for tool in server["tools"]:
                        print(f"{server['name']}\t{tool}")
            return 0
        except (MCPConfigurationError, MCPFixtureConfigurationError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        finally:
            close_mcp_clients(clients)

    search_env = dict(configured_env)
    if web_search_provider:
        search_env[WEB_SEARCH_PROVIDER_ENV] = web_search_provider
    elif web_search_stub_results:
        search_env[WEB_SEARCH_PROVIDER_ENV] = "stub"
    if web_search_stub_results:
        search_env[WEB_SEARCH_STUB_RESULTS_ENV] = str(web_search_stub_results)
    web_search_handler = None
    if effective_search_enabled:
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

    fetch_env = dict(configured_env)
    if web_fetch_provider:
        fetch_env[WEB_FETCH_PROVIDER_ENV] = web_fetch_provider
    web_fetch_handler = None
    if effective_fetch_enabled:
        try:
            web_fetch_handler = build_web_fetch_handler_from_env(fetch_env)
        except WebFetchConfigurationError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if web_fetch_handler is None:
            print(f"error: {format_web_fetch_unavailable_message()}", file=sys.stderr)
            print(f"hint: set {WEB_FETCH_PROVIDER_ENV}=http or pass --web-fetch-provider http", file=sys.stderr)
            return 2

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
    elif args.continue_session or local_config.session_default_mode == "continue":
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
            model_provider=_NoopModelProvider() if debug_tools else None,
            session_id=session_id,
            model=model,
            web_search_handler=web_search_handler,
            web_fetch_handler=web_fetch_handler,
            skills_dirs=skills_dirs,
            skill_discovery_mode=skill_discovery_mode,
            mcp_fixtures=mcp_fixtures,
            mcp_configs=mcp_configs,
            mcp_startup_timeout=local_config.mcp_startup_timeout,
            mcp_tool_timeout=local_config.mcp_tool_timeout,
            permission_mode=permission_mode,
            resume=resume,
            provider_env=configured_env,
            require_api_key=not debug_tools,
            memory_enabled=local_config.memory_enabled,
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

    if debug_tools:
        try:
            for tool in engine.tools:
                print(tool.name)
        finally:
            close_mcp_clients(getattr(engine, "_agent_kernel_owned_mcp_clients", ()))
        return 0

    def log(line: str) -> None:
        if not quiet and not json_events:
            print(line, file=sys.stderr)

    def observe_event(event: dict[str, Any]) -> None:
        if json_events:
            print(json.dumps(event, ensure_ascii=False, separators=(",", ":")), file=sys.stderr)

    prompt = " ".join(args.prompt).strip()
    try:
        if args.repl:
            prompts = [prompt] if prompt else []
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
                result = await run_local_agent_once(
                    next_prompt,
                    engine=engine,
                    max_turns=max_turns,
                    event_logger=log,
                    event_observer=observe_event,
                )
                print(result.final_response)
                if print_transcript_path:
                    print(f"transcript={result.transcript_path}", file=sys.stderr)
            return 0

        if not prompt:
            try:
                prompt = input("user> ").strip()
            except EOFError:
                prompt = ""
        if not prompt:
            print("error: prompt is empty", file=sys.stderr)
            return 2
        result = await run_local_agent_once(prompt, engine=engine, max_turns=max_turns, event_logger=log, event_observer=observe_event)
        print(result.final_response)
        if print_transcript_path:
            print(f"transcript={result.transcript_path}", file=sys.stderr)
        return 0
    finally:
        close_mcp_clients(getattr(engine, "_agent_kernel_owned_mcp_clients", ()))


def _expand_management_command(argv: Sequence[str] | None) -> list[str] | None:
    if argv is None:
        raw = sys.argv[1:]
    else:
        raw = list(argv)
    if not raw or raw[0].startswith("-"):
        return list(raw) if argv is not None else None
    head, *tail = raw
    if head == "config":
        if not tail or tail[0] in {"doctor", "status"}:
            return ["--doctor", *tail[1:]]
        if tail[0] == "doctor-json":
            return ["--doctor-json", *tail[1:]]
        if tail[0] == "init":
            return ["--init-config", *tail[1:]]
        if tail[0] == "validate":
            return ["--validate-config", *tail[1:]]
        if tail[0] in {"print", "effective", "show"}:
            return ["--print-effective-config", *tail[1:]]
    if head == "skills":
        if not tail or tail[0] == "list":
            return ["--list-skills", *tail[1:]]
        if tail[0] == "validate":
            return ["--validate-skills", *tail[1:]]
        if tail[0] == "info" and len(tail) >= 2:
            return ["--skill-info", tail[1], *tail[2:]]
    if head == "sessions":
        if not tail or tail[0] == "list":
            return ["--list-sessions", *tail[1:]]
        if tail[0] == "info" and len(tail) >= 2:
            return ["--session-info", tail[1], *tail[2:]]
        if tail[0] in {"path", "transcript-path"} and len(tail) >= 2:
            return ["--session-transcript-path", tail[1], *tail[2:]]
        if tail[0] == "export" and len(tail) >= 2:
            return ["--session-export", tail[1], *tail[2:]]
        if tail[0] == "delete" and len(tail) >= 2:
            return ["--session-delete", tail[1], *tail[2:]]
    if head == "memory":
        if not tail or tail[0] == "status":
            return ["--memory-status", *tail[1:]]
        if tail[0] == "list":
            return ["--memory-list", *tail[1:]]
        if tail[0] == "read":
            return ["--memory-read", *tail[1:]]
        if tail[0] == "write" and len(tail) >= 2:
            return ["--memory-write", tail[1], *tail[2:]]
        if tail[0] == "append" and len(tail) >= 2:
            return ["--memory-append", tail[1], *tail[2:]]
        if tail[0] == "delete" and len(tail) >= 2:
            return ["--memory-delete", tail[1], *tail[2:]]
        if tail[0] == "remember":
            return ["--memory-remember", *tail[1:]]
        if tail[0] == "forget" and len(tail) >= 2:
            return ["--memory-forget", tail[1], *tail[2:]]
        if tail[0] == "validate":
            return ["--memory-validate", *tail[1:]]
    if head == "mcp":
        if not tail or tail[0] == "list":
            return ["--mcp-list", *tail[1:]]
        if tail[0] == "doctor":
            return ["--mcp-doctor", *tail[1:]]
        if tail[0] == "validate-config" and len(tail) >= 2:
            return ["--mcp-validate-config", tail[1], *tail[2:]]
    return list(raw) if argv is not None else None


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for ``python3 examples/local_agent.py``."""
    parser = _build_parser()
    args = parser.parse_args(_expand_management_command(argv))
    return asyncio.run(_run_cli(args))


if __name__ == "__main__":
    raise SystemExit(main())
