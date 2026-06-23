"""Example-friendly WebSearch and WebFetch provider adapters.

These adapters are intentionally opt-in and stdlib-only. They convert local
runner environment/configuration into the handler contracts consumed by
WebSearchTool and WebFetchTool without changing the tools' core semantics.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import time
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


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


class WebSearchConfigurationError(RuntimeError):
    """Raised when WebSearch cannot be configured from explicit settings."""


class WebFetchConfigurationError(RuntimeError):
    """Raised when WebFetch cannot be configured from explicit settings."""


def format_web_search_unavailable_message() -> str:
    """Return the shared message for missing WebSearch setup."""
    return "WebSearch is not configured. Provide a web_search_handler or configure the local runner search provider."


def format_web_fetch_unavailable_message() -> str:
    """Return the shared message for missing WebFetch setup."""
    return "WebFetch is not configured. Set AGENT_KERNEL_WEB_FETCH_PROVIDER=http or provide a web_fetch_handler."


def _load_stub_results(path: str | Path) -> Any:
    try:
        return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except OSError as exc:
        raise WebSearchConfigurationError(f"Unable to read WebSearch stub results file: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise WebSearchConfigurationError(f"WebSearch stub results file is not valid JSON: {exc}") from exc


def make_stub_web_search_handler(results: Any | None = None) -> WebSearchHandler:
    """Build a deterministic, no-network WebSearch handler."""
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
            "messages": [{"role": "user", "content": [{"type": "text", "text": "\n".join(prompt_lines)}]}],
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
        return _normalise_anthropic_compatible_search_response(
            response_payload,
            query=query,
            duration_seconds=time.perf_counter() - start,
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
    """Build an example-level WebSearch handler from environment settings."""
    values = env or os.environ
    provider = (values.get(WEB_SEARCH_PROVIDER_ENV) or "").strip().lower()
    if not provider:
        return None
    if provider == "stub":
        results_path = values.get(WEB_SEARCH_STUB_RESULTS_ENV)
        results = _load_stub_results(results_path) if results_path else None
        return make_stub_web_search_handler(results)
    if provider == "http-json":
        timeout_seconds = _parse_web_search_timeout(values.get(WEB_SEARCH_TIMEOUT_ENV))
        return make_http_json_web_search_handler(
            values.get(WEB_SEARCH_URL_ENV) or "",
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
    raise WebSearchConfigurationError(
        f"Unsupported WebSearch provider '{provider}'. Local runner supports 'stub', opt-in 'http-json', and opt-in 'anthropic-compatible'."
    )
