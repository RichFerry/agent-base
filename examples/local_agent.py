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
import socket
import sys
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_kernel import AnthropicModelProvider, KernelConfig, MCPClientConfig, ModelProvider, QueryEngine
from agent_kernel.skills import SkillDefinition, skill_from_markdown


EventLogger = Callable[[str], None]
WebSearchHandler = Callable[[dict[str, Any]], Any]
WebFetchHandler = Callable[[str], Any]
WEB_SEARCH_PROVIDER_ENV = "AGENT_KERNEL_WEB_SEARCH_PROVIDER"
WEB_SEARCH_STUB_RESULTS_ENV = "AGENT_KERNEL_WEB_SEARCH_STUB_RESULTS"
WEB_SEARCH_URL_ENV = "AGENT_KERNEL_WEB_SEARCH_URL"
WEB_SEARCH_API_KEY_ENV = "AGENT_KERNEL_WEB_SEARCH_API_KEY"
WEB_SEARCH_MODEL_ENV = "AGENT_KERNEL_WEB_SEARCH_MODEL"
WEB_SEARCH_TIMEOUT_ENV = "AGENT_KERNEL_WEB_SEARCH_TIMEOUT"
WEB_FETCH_PROVIDER_ENV = "AGENT_KERNEL_WEB_FETCH_PROVIDER"
WEB_FETCH_TIMEOUT_ENV = "AGENT_KERNEL_WEB_FETCH_TIMEOUT"
WEB_FETCH_MAX_BYTES_ENV = "AGENT_KERNEL_WEB_FETCH_MAX_BYTES"
WEB_FETCH_MAX_CHARS_ENV = "AGENT_KERNEL_WEB_FETCH_MAX_CHARS"


class MissingCredentialsError(RuntimeError):
    """Raised when the real local runner has no model API credentials."""


class WebSearchConfigurationError(RuntimeError):
    """Raised when the example runner cannot configure WebSearch."""


class WebFetchConfigurationError(RuntimeError):
    """Raised when the example runner cannot configure WebFetch."""


class SkillsConfigurationError(RuntimeError):
    """Raised when the example runner cannot configure local skills."""


class MCPFixtureConfigurationError(RuntimeError):
    """Raised when the example runner cannot configure a local MCP fixture."""


@dataclass
class LocalAgentRun:
    """Result returned by the example runner helper."""

    events: list[dict[str, Any]]
    final_response: str
    logs: list[str]
    session_id: str
    transcript_path: Path


def has_api_credentials(env: Mapping[str, str] | None = None) -> bool:
    """Return whether Anthropic-compatible credentials are present."""
    values = env or os.environ
    return bool(values.get("ANTHROPIC_AUTH_TOKEN") or values.get("ANTHROPIC_API_KEY"))


def format_web_search_unavailable_message() -> str:
    """Return the shared example-layer message for missing WebSearch setup."""
    return "WebSearch is not configured. Provide a web_search_handler or configure the local runner search provider."


def format_web_fetch_unavailable_message() -> str:
    """Return the shared example-layer message for missing WebFetch setup."""
    return "WebFetch is not configured. Set AGENT_KERNEL_WEB_FETCH_PROVIDER=http or provide a web_fetch_handler."


def _load_stub_results(path: str | Path) -> Any:
    try:
        return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except OSError as exc:
        raise WebSearchConfigurationError(f"Unable to read WebSearch stub results file: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise WebSearchConfigurationError(f"WebSearch stub results file is not valid JSON: {exc}") from exc


def make_stub_web_search_handler(results: Any | None = None) -> WebSearchHandler:
    """Build a deterministic, no-network WebSearch handler for local examples/tests."""
    configured_results = results

    def handler(args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query") or "")
        if configured_results is None:
            content = [
                {
                    "title": f"Stub search result for {query}",
                    "url": "https://example.invalid/search",
                    "snippet": "Configure a real example adapter to return live search results.",
                }
            ]
            return {"query": query, "results": [{"content": content}]}
        if isinstance(configured_results, dict):
            output = dict(configured_results)
            output.setdefault("query", query)
            return output
        return configured_results

    return handler


def _normalise_http_json_result_item(item: Any) -> dict[str, str]:
    if not isinstance(item, dict):
        text = str(item)
        return {"title": text, "url": "", "snippet": text}
    title = item.get("title") or item.get("name") or item.get("url") or item.get("link") or item.get("snippet") or "Result"
    url = item.get("url") or item.get("link") or item.get("href") or ""
    snippet = item.get("snippet") or item.get("description") or item.get("content") or item.get("summary") or item.get("text") or ""
    output = {"title": str(title), "url": str(url)}
    if snippet:
        output["snippet"] = str(snippet)
    return output


def _normalise_http_json_result_list(items: Any, query: str) -> dict[str, Any]:
    if not isinstance(items, list):
        raise RuntimeError("http-json WebSearch provider returned unsupported JSON shape. Expected a result list.")
    content: list[dict[str, str]] = []
    text_results: list[str] = []
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("content"), list):
            content.extend(_normalise_http_json_result_item(content_item) for content_item in item["content"])
        elif isinstance(item, dict):
            content.append(_normalise_http_json_result_item(item))
        elif item is not None:
            text_results.append(str(item))
    results: list[Any] = []
    if content:
        results.append({"content": content})
    results.extend(text_results)
    return {"query": query, "results": results}


def _normalise_http_json_search_payload(payload: Any, query: str) -> dict[str, Any]:
    if isinstance(payload, list):
        return _normalise_http_json_result_list(payload, query)
    if not isinstance(payload, dict):
        raise RuntimeError("http-json WebSearch provider returned unsupported JSON shape. Expected a JSON object or list.")
    if "results" in payload:
        return _normalise_http_json_result_list(payload["results"], query)
    if "items" in payload:
        return _normalise_http_json_result_list(payload["items"], query)
    if "data" in payload and isinstance(payload["data"], (dict, list)):
        return _normalise_http_json_search_payload(payload["data"], query)
    if any(key in payload for key in ("title", "url", "link", "href", "snippet", "description", "content", "text")):
        return {"query": query, "results": [{"content": [_normalise_http_json_result_item(payload)]}]}
    raise RuntimeError("http-json WebSearch provider returned unsupported JSON shape. Expected results, items, or data.")


def _anthropic_compatible_messages_endpoint(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1/messages"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/messages"
    return f"{normalized}/v1/messages"


def _short_snippet(text: str, *, limit: int = 500) -> str:
    snippet = " ".join(text.split())
    if len(snippet) <= limit:
        return snippet
    return f"{snippet[: limit - 1]}..."


def _extract_text_from_anthropic_content(content: Any) -> str:
    parts: list[str] = []
    if not isinstance(content, list):
        return ""
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
        elif isinstance(block, str):
            parts.append(block)
    return _short_snippet("\n".join(parts))


def _extract_citation_hits_from_text_block(block: dict[str, Any]) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    citations = block.get("citations")
    if not isinstance(citations, list):
        return hits
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        url = citation.get("url") or citation.get("source") or citation.get("uri") or ""
        title = citation.get("title") or citation.get("document_title") or url or "Citation"
        snippet = citation.get("cited_text") or citation.get("text") or ""
        hit = {"title": str(title), "url": str(url)}
        if snippet:
            hit["snippet"] = _short_snippet(str(snippet), limit=240)
        hits.append(hit)
    return hits


def _normalise_anthropic_compatible_search_response(
    payload: Any,
    *,
    query: str,
    duration_seconds: float,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError("anthropic-compatible WebSearch provider returned unsupported JSON shape. Expected a JSON object.")
    content = payload.get("content")
    if not isinstance(content, list):
        raise RuntimeError("anthropic-compatible WebSearch provider response is missing message content.")

    hits: list[dict[str, str]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "web_search_tool_result":
            result_content = block.get("content")
            if isinstance(result_content, list):
                hits.extend(_normalise_http_json_result_item(item) for item in result_content)
        elif block.get("type") == "text":
            hits.extend(_extract_citation_hits_from_text_block(block))

    if hits:
        return {"query": query, "results": [{"content": hits}], "durationSeconds": duration_seconds}

    snippet = _extract_text_from_anthropic_content(content)
    if not snippet:
        snippet = "Anthropic-compatible provider returned no structured search results."
    return {
        "query": query,
        "results": [
            {
                "content": [
                    {
                        "title": "Anthropic-compatible search result",
                        "url": "",
                        "snippet": snippet,
                    }
                ]
            }
        ],
        "durationSeconds": duration_seconds,
    }


def make_anthropic_compatible_web_search_handler(
    base_url: str,
    *,
    api_key: str,
    model: str,
    timeout_seconds: float = 10.0,
) -> WebSearchHandler:
    """Build an opt-in Anthropic-compatible search adapter for WebSearch."""
    url = base_url.strip()
    key = api_key.strip()
    model_name = model.strip()
    if not url:
        raise WebSearchConfigurationError(f"anthropic-compatible WebSearch provider requires {WEB_SEARCH_URL_ENV}.")
    if not key:
        raise WebSearchConfigurationError(f"anthropic-compatible WebSearch provider requires {WEB_SEARCH_API_KEY_ENV}.")
    if not model_name:
        raise WebSearchConfigurationError(f"anthropic-compatible WebSearch provider requires {WEB_SEARCH_MODEL_ENV}.")

    endpoint = _anthropic_compatible_messages_endpoint(url)

    def handler(args: dict[str, Any]) -> dict[str, Any]:
        start = time.perf_counter()
        query = str(args.get("query") or "")
        search_tool: dict[str, Any] = {"type": "web_search_20250305", "name": "web_search", "max_uses": 8}
        for field_name in ("allowed_domains", "blocked_domains"):
            value = args.get(field_name)
            if isinstance(value, list):
                search_tool[field_name] = value
        prompt_lines = [f"Perform a web search for the query: {query}"]
        if isinstance(args.get("allowed_domains"), list):
            prompt_lines.append(f"Only include results from these domains: {', '.join(args['allowed_domains'])}")
        if isinstance(args.get("blocked_domains"), list):
            prompt_lines.append(f"Do not include results from these domains: {', '.join(args['blocked_domains'])}")
        prompt_lines.append("Return concise search results with titles, URLs, and short snippets when available.")
        request_body = {
            "model": model_name,
            "max_tokens": 1024,
            "system": "You are an assistant for performing a web search tool use.",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "\n".join(prompt_lines)}],
                }
            ],
            "tools": [search_tool],
            "stream": False,
        }
        payload = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "Authorization": f"Bearer {key}",
        }
        request = Request(endpoint, data=payload, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            raise RuntimeError(f"anthropic-compatible WebSearch provider request failed with HTTP {exc.code}.") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise RuntimeError("anthropic-compatible WebSearch provider request timed out.") from exc
        except URLError as exc:
            raise RuntimeError(f"anthropic-compatible WebSearch provider request failed: {exc.reason}") from exc
        except OSError as exc:
            raise RuntimeError(f"anthropic-compatible WebSearch provider request failed: {exc.__class__.__name__}.") from exc
        try:
            response_payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("anthropic-compatible WebSearch provider returned invalid JSON.") from exc
        duration_seconds = time.perf_counter() - start
        return _normalise_anthropic_compatible_search_response(
            response_payload,
            query=query,
            duration_seconds=duration_seconds,
        )

    return handler


def make_http_json_web_search_handler(
    search_url: str,
    *,
    api_key: str | None = None,
    timeout_seconds: float = 10.0,
) -> WebSearchHandler:
    """Build an opt-in, standard-library HTTP JSON WebSearch adapter."""
    url = search_url.strip()
    if not url:
        raise WebSearchConfigurationError(f"http-json WebSearch provider requires {WEB_SEARCH_URL_ENV}.")

    def handler(args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query") or "")
        request_body: dict[str, Any] = {"query": query}
        for field_name in ("allowed_domains", "blocked_domains"):
            value = args.get(field_name)
            if isinstance(value, list):
                request_body[field_name] = value
        payload = json.dumps(request_body).encode("utf-8")
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = Request(url, data=payload, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            raise RuntimeError(f"http-json WebSearch provider request failed with HTTP {exc.code}.") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise RuntimeError("http-json WebSearch provider request timed out.") from exc
        except URLError as exc:
            raise RuntimeError(f"http-json WebSearch provider request failed: {exc.reason}") from exc
        except OSError as exc:
            raise RuntimeError(f"http-json WebSearch provider request failed: {exc.__class__.__name__}.") from exc
        try:
            response_payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("http-json WebSearch provider returned invalid JSON.") from exc
        return _normalise_http_json_search_payload(response_payload, query)

    return handler


def _parse_web_search_timeout(value: str | None) -> float:
    if value is None or not value.strip():
        return 10.0
    try:
        timeout = float(value)
    except ValueError as exc:
        raise WebSearchConfigurationError(f"{WEB_SEARCH_TIMEOUT_ENV} must be a positive number.") from exc
    if timeout <= 0:
        raise WebSearchConfigurationError(f"{WEB_SEARCH_TIMEOUT_ENV} must be a positive number.")
    return timeout


def _parse_web_fetch_timeout(value: str | None) -> float:
    if value is None or not value.strip():
        return 10.0
    try:
        timeout = float(value)
    except ValueError as exc:
        raise WebFetchConfigurationError(f"{WEB_FETCH_TIMEOUT_ENV} must be a positive number.") from exc
    if timeout <= 0:
        raise WebFetchConfigurationError(f"{WEB_FETCH_TIMEOUT_ENV} must be a positive number.")
    return timeout


def _parse_positive_int_env(value: str | None, env_name: str, default: int) -> int:
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise WebFetchConfigurationError(f"{env_name} must be a positive integer.") from exc
    if parsed <= 0:
        raise WebFetchConfigurationError(f"{env_name} must be a positive integer.")
    return parsed


def _validate_fetch_url(url: str) -> None:
    try:
        parsed = urlparse(url)
    except ValueError as exc:
        raise RuntimeError(f"WebFetch invalid URL: {url}") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError("WebFetch invalid URL. Only http and https URLs are supported.")


def make_unavailable_web_fetch_handler() -> WebFetchHandler:
    """Build the local runner's default no-network WebFetch handler."""

    def handler(url: str) -> dict[str, Any]:
        raise RuntimeError(format_web_fetch_unavailable_message())

    return handler


def make_http_web_fetch_handler(
    *,
    timeout_seconds: float = 10.0,
    max_bytes: int = 1_000_000,
    max_chars: int = 100_000,
) -> WebFetchHandler:
    """Build an opt-in, standard-library HTTP(S) WebFetch handler."""

    def handler(url: str) -> dict[str, Any]:
        _validate_fetch_url(url)
        request = Request(
            url,
            headers={
                "Accept": "text/markdown, text/plain, text/html, */*",
                "User-Agent": "agent-kernel-local-runner/0.3",
            },
            method="GET",
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read(max_bytes + 1)
                if len(raw) > max_bytes:
                    raise RuntimeError(f"WebFetch response exceeds {max_bytes} byte limit.")
                content_type = response.headers.get("content-type", "") if getattr(response, "headers", None) is not None else ""
                status_code = int(getattr(response, "status", response.getcode() if hasattr(response, "getcode") else 200))
                reason = str(getattr(response, "reason", "") or "")
        except HTTPError as exc:
            raise RuntimeError(f"WebFetch request failed with HTTP {exc.code}.") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise RuntimeError("WebFetch request timed out.") from exc
        except URLError as exc:
            raise RuntimeError(f"WebFetch request failed: {exc.reason}") from exc
        except OSError as exc:
            raise RuntimeError(f"WebFetch request failed: {exc.__class__.__name__}.") from exc
        content = raw.decode("utf-8", errors="replace")
        if len(content) > max_chars:
            raise RuntimeError(f"WebFetch response exceeds {max_chars} character limit.")
        return {
            "bytes": len(raw),
            "code": status_code,
            "codeText": reason or ("OK" if 200 <= status_code < 300 else "HTTP Error"),
            "content": content,
            "contentType": content_type,
            "url": url,
        }

    return handler


def build_web_fetch_handler_from_env(env: Mapping[str, str] | None = None) -> WebFetchHandler | None:
    """Build an example-level WebFetch handler from environment settings."""
    values = env or os.environ
    provider = (values.get(WEB_FETCH_PROVIDER_ENV) or "").strip().lower()
    if not provider:
        return None
    if provider != "http":
        raise WebFetchConfigurationError(f"Unsupported WebFetch provider '{provider}'. Local runner supports opt-in 'http'.")
    timeout_seconds = _parse_web_fetch_timeout(values.get(WEB_FETCH_TIMEOUT_ENV))
    max_bytes = _parse_positive_int_env(values.get(WEB_FETCH_MAX_BYTES_ENV), WEB_FETCH_MAX_BYTES_ENV, 1_000_000)
    max_chars = _parse_positive_int_env(values.get(WEB_FETCH_MAX_CHARS_ENV), WEB_FETCH_MAX_CHARS_ENV, 100_000)
    return make_http_web_fetch_handler(timeout_seconds=timeout_seconds, max_bytes=max_bytes, max_chars=max_chars)


def build_web_search_handler_from_env(env: Mapping[str, str] | None = None) -> WebSearchHandler | None:
    """Build an example-level WebSearch handler from environment settings.

    ``stub`` is deterministic and local. ``http-json`` and
    ``anthropic-compatible`` are opt-in only and use Python's standard library
    to call caller-provided endpoints.
    """
    values = env or os.environ
    provider = (values.get(WEB_SEARCH_PROVIDER_ENV) or "").strip().lower()
    if not provider:
        return None
    if provider == "stub":
        results_path = values.get(WEB_SEARCH_STUB_RESULTS_ENV)
        results = _load_stub_results(results_path) if results_path else None
        return make_stub_web_search_handler(results)
    if provider == "http-json":
        search_url = values.get(WEB_SEARCH_URL_ENV) or ""
        timeout_seconds = _parse_web_search_timeout(values.get(WEB_SEARCH_TIMEOUT_ENV))
        return make_http_json_web_search_handler(
            search_url,
            api_key=values.get(WEB_SEARCH_API_KEY_ENV),
            timeout_seconds=timeout_seconds,
        )
    if provider == "anthropic-compatible":
        timeout_seconds = _parse_web_search_timeout(values.get(WEB_SEARCH_TIMEOUT_ENV))
        return make_anthropic_compatible_web_search_handler(
            values.get(WEB_SEARCH_URL_ENV) or "",
            api_key=values.get(WEB_SEARCH_API_KEY_ENV) or "",
            model=values.get(WEB_SEARCH_MODEL_ENV) or "",
            timeout_seconds=timeout_seconds,
        )
    else:
        raise WebSearchConfigurationError(
            f"Unsupported WebSearch provider '{provider}'. Local runner supports 'stub', opt-in 'http-json', and opt-in 'anthropic-compatible'."
        )


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
    for child in sorted(root.iterdir(), key=lambda path: path.name):
        if not child.is_dir():
            continue
        skill = skill_from_markdown(child / "SKILL.md", name=child.name, loaded_from=str(root), source="local-runner")
        if skill is not None:
            skills.append(skill)
    if not skills:
        raise SkillsConfigurationError(f"No valid skills found in {root}. Expected child directories containing SKILL.md.")
    return skills


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
    permission_mode: str = "ask",
    require_api_key: bool = True,
) -> QueryEngine:
    """Build a QueryEngine for local example use.

    Tests may pass a fake provider and set ``require_api_key=False``. Real CLI
    use requires API credentials so a missing key fails before any network path.
    """
    skills_path: Path | None = None
    if skills_dir is not None:
        skills_path = Path(skills_dir).expanduser()
        discover_local_skills(skills_path)
    mcp_clients: tuple[MCPClientConfig, ...] = ()
    if mcp_fixture is not None:
        mcp_clients = (load_mcp_fixture(mcp_fixture),)

    if model_provider is None:
        if require_api_key and not has_api_credentials():
            raise MissingCredentialsError(
                "Missing Anthropic-compatible API credentials. Set ANTHROPIC_AUTH_TOKEN "
                "or ANTHROPIC_API_KEY. Optional: ANTHROPIC_BASE_URL and ANTHROPIC_MODEL."
            )
        model_provider = AnthropicModelProvider.from_env()

    config_kwargs: dict[str, Any] = {"cwd": Path(cwd).expanduser() if cwd is not None else Path.cwd()}
    if config_home is not None:
        config_kwargs["config_home"] = Path(config_home).expanduser()
    if skills_path is not None:
        config_kwargs["skill_paths"] = (skills_path,)
    if mcp_clients:
        config_kwargs["mcp_clients"] = mcp_clients
    config = KernelConfig(**config_kwargs)
    # The local runner loads only the explicitly supplied --skills-dir. This
    # keeps examples deterministic without changing QueryEngine defaults.
    setattr(config, "_agent_kernel_skill_paths_only", True)

    engine_kwargs: dict[str, Any] = {
        "model_provider": model_provider,
        "config": config,
    }
    if session_id is not None:
        engine_kwargs["session_id"] = session_id
    if model is not None:
        engine_kwargs["model"] = model
    engine = QueryEngine(**engine_kwargs)
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
    permission_mode: str | None = None,
    max_turns: int = 10,
    sdk_events: bool = True,
    event_logger: EventLogger | None = None,
    require_api_key: bool = True,
) -> LocalAgentRun:
    """Run one prompt through QueryEngine and collect logs plus final text."""
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

    async for event in engine.submit_message(prompt, max_turns=max_turns, sdk_events=sdk_events):
        events.append(event)
        for line in format_event_log(event):
            logs.append(line)
            if event_logger is not None:
                event_logger(line)

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
    parser.add_argument("--config-home", type=Path, help="Override CLAUDE-style config/transcript home.")
    parser.add_argument("--session-id", help="Use a stable transcript session id.")
    parser.add_argument("--model", help="Override ANTHROPIC_MODEL for this runner.")
    parser.add_argument("--max-turns", type=int, default=10, help="Maximum model turns per submitted prompt.")
    parser.add_argument("--permission-mode", choices=("ask", "bypass"), default="ask", help="Permission mode to pass through to the kernel.")
    parser.add_argument("--enable-web-search", action="store_true", help=f"Enable example WebSearch provider from {WEB_SEARCH_PROVIDER_ENV}.")
    parser.add_argument("--web-search-provider", choices=("stub", "http-json", "anthropic-compatible"), help="Example WebSearch provider override.")
    parser.add_argument("--web-search-stub-results", type=Path, help=f"JSON file used by the 'stub' WebSearch provider.")
    parser.add_argument("--enable-web-fetch", action="store_true", help=f"Enable example WebFetch provider from {WEB_FETCH_PROVIDER_ENV}.")
    parser.add_argument("--web-fetch-provider", choices=("http",), help="Example WebFetch provider override.")
    parser.add_argument("--skills-dir", type=Path, help="Load local skills from child directories containing SKILL.md.")
    parser.add_argument("--mcp-fixture", type=Path, help="Load a local-only MCP smoke fixture JSON file.")
    parser.add_argument("--quiet", action="store_true", help="Only print assistant final responses to stdout.")
    return parser


async def _run_cli(args: argparse.Namespace) -> int:
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

    try:
        engine = build_local_engine(
            cwd=args.cwd,
            config_home=args.config_home,
            session_id=args.session_id,
            model=args.model,
            web_search_handler=web_search_handler,
            web_fetch_handler=web_fetch_handler,
            skills_dir=args.skills_dir,
            mcp_fixture=args.mcp_fixture,
            permission_mode=args.permission_mode,
        )
    except MissingCredentialsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except MCPFixtureConfigurationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    def log(line: str) -> None:
        if not args.quiet:
            print(line, file=sys.stderr)

    prompt = " ".join(args.prompt).strip()
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


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for ``python3 examples/local_agent.py``."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(_run_cli(args))


if __name__ == "__main__":
    raise SystemExit(main())
