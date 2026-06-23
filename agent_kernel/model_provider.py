"""模型调用边界：协议抽象、Anthropic-compatible HTTP/SSE 和测试 provider。

输入：内部 messages、分段 system prompt、当前工具对象和每次请求 options。
输出：异步产生标准 ``AssistantMessage``，query loop 不接触 HTTP wire format。

真实调用顺序：
1. 归一化消息并修复 tool_use/tool_result pairing。
2. 把 Tool 转为 Anthropic JSON Schema，拼出 ``/v1/messages`` 请求体。
3. 从 Bearer token 或 x-api-key 生成认证头，日志副本始终脱敏。
4. 默认以 ``stream=true`` 读取 SSE；显式 transport 保留可测试的非流式路径。
5. ``AnthropicStreamNormalizer`` 按 block index 聚合 text、thinking、citation 和
   tool input JSON delta，最终生成内核消息。

AbortSignal 会在请求前后检查，并在取消时关闭正在读取的 response。FakeModelProvider
消费预设 turns，用于无需网络的确定性 agent-loop 测试。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
import asyncio
import json
import os
from typing import Any, AsyncIterator, Callable, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from uuid import uuid4

from .messages import (
    AssistantMessage,
    ContentBlock,
    Message,
    create_assistant_message,
    ensure_tool_result_pairing,
    normalize_messages_for_api,
)


class ModelProvider(Protocol):
    """可注入模型后端必须实现的最小异步协议。"""
    async def stream(
        self,
        *,
        messages: list[dict],
        system_prompt: list[str],
        tools: list[object],
        options: dict,
    ) -> AsyncIterator[AssistantMessage]:
        """异步产生当前模型后端的 AssistantMessage 消息流。"""

        ...


Transport = Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]]
StreamTransport = Callable[[str, dict[str, str], dict[str, Any], float, Any | None], Iterable[dict[str, Any]]]


class AnthropicAPIError(RuntimeError):
    """表示模型调用阶段的专用错误。"""
    pass


class OpenAIAPIError(RuntimeError):
    """Raised when an OpenAI-compatible provider request cannot be normalized."""
    pass


def _endpoint_from_base_url(base_url: str) -> str:
    """完成 ``_endpoint_from_base_url`` 对应的模型调用内部步骤。"""
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1/messages"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/messages"
    return f"{normalized}/v1/messages"


def _openai_chat_endpoint_from_base_url(base_url: str) -> str:
    """Return the Chat Completions endpoint for an OpenAI-compatible base URL."""
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


def _openai_responses_endpoint_from_base_url(base_url: str) -> str:
    """Return the Responses endpoint for an OpenAI-compatible base URL."""
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1/responses"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/responses"
    return f"{normalized}/v1/responses"


def _json_schema_for_type(py_type: type | tuple[type, ...]) -> dict[str, Any]:
    """完成 ``_json_schema_for_type`` 对应的模型调用内部步骤。"""
    if py_type is str:
        return {"type": "string"}
    if py_type is int:
        return {"type": "integer"}
    if py_type is bool:
        return {"type": "boolean"}
    if py_type is list:
        return {"type": "array", "items": {"type": "string"}}
    if py_type is dict:
        return {"type": "object"}
    if isinstance(py_type, tuple):
        variants = [_json_schema_for_type(item) for item in py_type]
        return {"anyOf": variants}
    return {"type": "string"}


def _message_to_api(message: Message | dict[str, Any]) -> dict[str, Any] | None:
    """完成 ``_message_to_api`` 对应的模型调用内部步骤。"""
    if message.get("type") == "tombstone":
        return None
    payload = message.get("message")
    if not isinstance(payload, dict):
        return None
    role = payload.get("role")
    content = payload.get("content")
    if role not in {"user", "assistant"} or not isinstance(content, list):
        return None
    return {"role": role, "content": content}


def _default_transport(url: str, headers: dict[str, str], body: dict[str, Any], timeout: float) -> dict[str, Any]:
    """完成 ``_default_transport`` 对应的模型调用内部步骤。"""
    # 非流式 transport 主要用于兼容后端和单元测试；默认真实路径使用 SSE。
    request = Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
    except HTTPError as exc:
        error_payload = exc.read().decode("utf-8", errors="replace")
        raise AnthropicAPIError(f"Anthropic API request failed with HTTP {exc.code}: {error_payload}") from exc
    except URLError as exc:
        raise AnthropicAPIError(f"Anthropic API request failed: {exc.reason}") from exc
    return json.loads(payload)


def _default_openai_transport(url: str, headers: dict[str, str], body: dict[str, Any], timeout: float) -> dict[str, Any]:
    """POST JSON to an OpenAI-compatible endpoint using only stdlib."""
    request = Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
    except HTTPError as exc:
        error_payload = exc.read().decode("utf-8", errors="replace")
        raise OpenAIAPIError(f"OpenAI-compatible request failed with HTTP {exc.code}: {error_payload}") from exc
    except URLError as exc:
        raise OpenAIAPIError(f"OpenAI-compatible request failed: {exc.reason}") from exc
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise OpenAIAPIError("OpenAI-compatible provider returned invalid JSON.") from exc


def _raise_if_aborted(abort_signal: Any | None) -> None:
    """完成 ``_raise_if_aborted`` 对应的模型调用内部步骤。"""
    if abort_signal is not None and hasattr(abort_signal, "throw_if_aborted"):
        abort_signal.throw_if_aborted()


def _default_stream_transport(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout: float,
    abort_signal: Any | None = None,
) -> Iterable[dict[str, Any]]:
    """完成 ``_default_stream_transport`` 对应的模型调用内部步骤。"""
    # 复制 header，避免把 Accept 修改泄漏到 provider 的诊断快照。
    request_headers = dict(headers)
    request_headers["Accept"] = "text/event-stream"
    request = Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            remove_abort_callback = None
            if abort_signal is not None and hasattr(abort_signal, "add_callback"):
                # 在线程中阻塞读取时，关闭 response 是让取消尽快生效的可靠方式。
                remove_abort_callback = abort_signal.add_callback(lambda _reason: response.close())
            try:
                data_lines: list[str] = []
                for raw_line in response:
                    _raise_if_aborted(abort_signal)
                    line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                    if not line:
                        # SSE 以空行结束一个 event，data 可以由多行拼成。
                        if data_lines:
                            payload = "\n".join(data_lines)
                            data_lines.clear()
                            if payload == "[DONE]":
                                continue
                            yield json.loads(payload)
                        continue
                    if line.startswith(":"):
                        continue
                    if line.startswith("data:"):
                        data_lines.append(line[5:].lstrip())
                if data_lines:
                    payload = "\n".join(data_lines)
                    if payload != "[DONE]":
                        yield json.loads(payload)
            finally:
                if remove_abort_callback is not None:
                    remove_abort_callback()
    except HTTPError as exc:
        error_payload = exc.read().decode("utf-8", errors="replace")
        raise AnthropicAPIError(f"Anthropic API stream failed with HTTP {exc.code}: {error_payload}") from exc
    except URLError as exc:
        raise AnthropicAPIError(f"Anthropic API stream failed: {exc.reason}") from exc


async def _maybe_await(value):
    """完成 ``_maybe_await`` 对应的模型调用内部步骤。"""
    if hasattr(value, "__await__"):
        return await value
    return value


def _content_to_text(content: Any) -> str:
    """Convert text or structured tool_result content into stable text."""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, separators=(",", ":"))


def _parse_tool_arguments(raw_arguments: Any, *, provider_name: str) -> dict[str, Any]:
    """Parse provider-emitted function arguments into internal tool input."""
    if raw_arguments is None:
        return {}
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if raw_arguments == "":
        return {}
    if not isinstance(raw_arguments, str):
        raise OpenAIAPIError(f"{provider_name} emitted non-string tool arguments.")
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise OpenAIAPIError(f"{provider_name} emitted invalid tool arguments JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise OpenAIAPIError(f"{provider_name} emitted non-object tool arguments JSON.")
    return parsed


def _assistant_text_from_blocks(content: list[dict[str, Any]]) -> str:
    """Collect text blocks from an internal assistant content list."""
    return "\n".join(
        str(block.get("text", ""))
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _tool_result_blocks_from_content(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return tool_result blocks from an internal user content list."""
    return [
        block
        for block in content
        if isinstance(block, dict) and block.get("type") == "tool_result" and isinstance(block.get("tool_use_id"), str)
    ]


def _text_blocks_from_content(content: list[dict[str, Any]]) -> list[str]:
    """Return text block values from an internal user/assistant content list."""
    return [
        str(block.get("text", ""))
        for block in content
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
    ]


class AnthropicStreamNormalizer:
    """把 Anthropic SSE event 序列聚合成一条完整 assistant message。

    tool input 以 JSON 字符串碎片到达，必须等 content_block_stop 后再解析；text、
    thinking、signature 和 citation 则按 block index 累积。
    """
    def __init__(self) -> None:
        """初始化实例内部状态和后续处理所需的缓存。"""
        # message_id 来自 message_start；缺失时消息构造器会生成本地 ID。
        self.message_id: str | None = None
        # blocks 以 SSE index 为键，允许 text/tool/thinking 分片交错到达。
        self.blocks: dict[int, dict[str, Any]] = {}
        # tool input 必须先缓存 partial_json，content_block_stop 时一次解析。
        self.json_buffers: dict[int, list[str]] = {}

    def push(self, event: dict[str, Any]) -> None:
        """接收一个流式事件并更新当前聚合状态。"""
        event_type = event.get("type")
        if event_type in {None, "ping"}:
            # ping 只用于保活，不属于 assistant content。
            return
        if event_type == "error":
            error = event.get("error")
            if isinstance(error, dict):
                message = error.get("message") or json.dumps(error, ensure_ascii=False)
            else:
                message = str(error or "Unknown streaming error.")
            raise AnthropicAPIError(message)
        if event_type == "message_start":
            message = event.get("message")
            if isinstance(message, dict):
                if isinstance(message.get("id"), str):
                    self.message_id = message["id"]
                for index, block in enumerate(message.get("content") or []):
                    if isinstance(block, dict):
                        self._start_block(index, block)
            return
        if event_type == "content_block_start":
            index = int(event.get("index", len(self.blocks)))
            block = event.get("content_block") or {}
            self._start_block(index, block if isinstance(block, dict) else {})
            return
        if event_type == "content_block_delta":
            # delta 必须应用到对应 index，不能假设 blocks 严格串行到达。
            index = int(event.get("index", len(self.blocks)))
            delta = event.get("delta") or {}
            if isinstance(delta, dict):
                self._apply_delta(index, delta)
            return
        if event_type == "content_block_stop":
            self._finish_block(int(event.get("index", len(self.blocks))))
            return
        if event_type in {"message_delta", "message_stop"}:
            return

    def message(self) -> AssistantMessage:
        """根据已聚合的流式状态构造最终 assistant 消息。"""
        for index in list(self.blocks):
            self._finish_block(index)
        content = [self.blocks[index] for index in sorted(self.blocks)]
        if not content:
            raise AnthropicAPIError("Anthropic stream did not produce a message.")
        return create_assistant_message(content, message_id=self.message_id)

    def _start_block(self, index: int, block: dict[str, Any]) -> None:
        """完成 ``_start_block`` 对应的模型调用内部步骤。"""
        block_type = block.get("type")
        if block_type == "text":
            self.blocks[index] = {"type": "text", "text": str(block.get("text", ""))}
        elif block_type == "tool_use":
            self.blocks[index] = {
                "type": "tool_use",
                "id": str(block.get("id", "")),
                "name": str(block.get("name", "")),
                "input": block.get("input") if isinstance(block.get("input"), dict) else {},
            }
            self.json_buffers[index] = []
        else:
            self.blocks[index] = dict(block)

    def _apply_delta(self, index: int, delta: dict[str, Any]) -> None:
        """应用delta，供模型调用流程使用。"""
        delta_type = delta.get("type")
        block = self.blocks.setdefault(index, {"type": "text", "text": ""})
        if delta_type == "text_delta":
            block["text"] = str(block.get("text", "")) + str(delta.get("text", ""))
        elif delta_type == "input_json_delta":
            self.json_buffers.setdefault(index, []).append(str(delta.get("partial_json", "")))
        elif delta_type == "thinking_delta":
            block["thinking"] = str(block.get("thinking", "")) + str(delta.get("thinking", ""))
        elif delta_type == "signature_delta":
            block["signature"] = delta.get("signature")
        elif delta_type == "citations_delta":
            citations = block.setdefault("citations", [])
            if isinstance(citations, list):
                citations.append(delta.get("citation"))

    def _finish_block(self, index: int) -> None:
        """完成 ``_finish_block`` 对应的模型调用内部步骤。"""
        if index not in self.json_buffers:
            return
        # tool input 的 partial_json 只有在 block stop 后才保证是完整 JSON。
        partial = "".join(self.json_buffers.pop(index))
        if not partial:
            return
        block = self.blocks.setdefault(index, {"type": "tool_use", "id": "", "name": "", "input": {}})
        try:
            parsed = json.loads(partial)
        except json.JSONDecodeError as exc:
            raise AnthropicAPIError(f"Anthropic stream emitted invalid tool input JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise AnthropicAPIError("Anthropic stream emitted non-object tool input JSON.")
        block["input"] = parsed


def normalize_anthropic_stream_events(events: Iterable[dict[str, Any]]) -> AssistantMessage:
    """规范化anthropic 流 事件集合，供模型调用流程使用。"""
    normalizer = AnthropicStreamNormalizer()
    for event in events:
        normalizer.push(event)
    return normalizer.message()


@dataclass
class AnthropicModelProvider:
    """支持 Bearer token 或 x-api-key 的 Anthropic-compatible provider。"""
    # 连接与认证参数；auth_token 优先于 api_key。
    base_url: str | None = None
    auth_token: str | None = None
    api_key: str | None = None
    model: str | None = None
    max_tokens: int = 4096
    timeout: float = 120.0
    # transport 用于非流式测试；stream_transport 可替换真实 SSE reader。
    transport: Transport | None = None
    stream_transport: StreamTransport | None = None
    streaming: bool = True
    # calls 仅保存脱敏请求摘要，供测试和诊断检查，不保存真实凭据。
    calls: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "AnthropicModelProvider":
        """从当前进程环境变量创建配置完整的实例。"""
        return cls(
            base_url=os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
            auth_token=os.environ.get("ANTHROPIC_AUTH_TOKEN"),
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            model=os.environ.get("ANTHROPIC_MODEL"),
        )

    def __post_init__(self) -> None:
        """完成 dataclass 创建后的派生字段初始化与规范化。"""
        self.base_url = self.base_url or os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
        self.auth_token = self.auth_token if self.auth_token is not None else os.environ.get("ANTHROPIC_AUTH_TOKEN")
        self.api_key = self.api_key if self.api_key is not None else os.environ.get("ANTHROPIC_API_KEY")
        self.model = self.model or os.environ.get("ANTHROPIC_MODEL") or "claude-opus-4-6"

    async def stream(
        self,
        *,
        messages: list[dict],
        system_prompt: list[str],
        tools: list[object],
        options: dict,
    ) -> AsyncIterator[AssistantMessage]:
        # 自定义 transport 保留非流式测试路径；默认真实 HTTP 使用 SSE。
        """异步产生当前模型后端的 AssistantMessage 消息流。"""
        body = await self._build_request_body(messages, system_prompt, tools, options)
        headers = self._headers()
        url = _endpoint_from_base_url(self.base_url or "https://api.anthropic.com")
        abort_signal = options.get("abortSignal")
        # 显式普通 transport 优先保留非流式行为；默认网络请求走 SSE。
        should_stream = self.stream_transport is not None or (self.streaming and self.transport is None)
        body["stream"] = should_stream
        self.calls.append({"url": url, "body": body, "headers": self._redacted_headers(headers)})
        _raise_if_aborted(abort_signal)
        if should_stream:
            stream_transport = self.stream_transport or _default_stream_transport
            events = await asyncio.to_thread(lambda: list(stream_transport(url, headers, body, self.timeout, abort_signal)))
            _raise_if_aborted(abort_signal)
            yield normalize_anthropic_stream_events(events)
            return
        transport = self.transport or _default_transport
        response = await asyncio.to_thread(transport, url, headers, body, self.timeout)
        _raise_if_aborted(abort_signal)
        yield self._response_to_message(response)

    async def _build_request_body(
        self,
        messages: list[dict],
        system_prompt: list[str],
        tools: list[object],
        options: dict,
    ) -> dict[str, Any]:
        """构造请求 body，供模型调用流程使用。"""
        # pairing repair 必须在归一化之后，因为连续 user tool_result 会先被合并。
        model_messages = normalize_messages_for_api(messages)
        repaired_messages = ensure_tool_result_pairing(model_messages)
        api_messages = [_message_to_api(message) for message in repaired_messages]
        body: dict[str, Any] = {
            "model": options.get("model") or self.model,
            "max_tokens": options.get("max_tokens", self.max_tokens),
            "messages": [message for message in api_messages if message is not None],
            "system": "\n\n".join(system_prompt),
            "stream": False,
        }
        tool_specs = [await self._tool_to_api(tool) for tool in tools]
        if tool_specs:
            body["tools"] = tool_specs
        return body

    async def _tool_to_api(self, tool: object) -> dict[str, Any]:
        """完成 ``_tool_to_api`` 对应的模型调用内部步骤。"""
        if hasattr(tool, "to_api_spec"):
            return await _maybe_await(tool.to_api_spec())
        properties = {
            field_name: _json_schema_for_type(field_type)
            for field_name, field_type in getattr(tool, "input_schema", {}).items()
        }
        description_parts = []
        if hasattr(tool, "description"):
            description_parts.append(await tool.description(None))
        if hasattr(tool, "prompt"):
            description_parts.append(await tool.prompt())
        description = "\n\n".join(part for part in description_parts if part)
        return {
            "name": getattr(tool, "name"),
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": list(getattr(tool, "required_fields", ())),
                "additionalProperties": False,
            },
        }

    def _headers(self) -> dict[str, str]:
        """完成 ``_headers`` 对应的模型调用内部步骤。"""
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        elif self.api_key:
            headers["x-api-key"] = self.api_key
        else:
            host = urlparse(self.base_url or "").netloc or "Anthropic-compatible"
            raise AnthropicAPIError(f"{host} API credentials are missing. Set ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY.")
        return headers

    def _redacted_headers(self, headers: dict[str, str]) -> dict[str, str]:
        """完成 ``_redacted_headers`` 对应的模型调用内部步骤。"""
        redacted = dict(headers)
        if "Authorization" in redacted:
            redacted["Authorization"] = "Bearer [REDACTED]"
        if "x-api-key" in redacted:
            redacted["x-api-key"] = "[REDACTED]"
        return redacted

    def _response_to_message(self, response: dict[str, Any]) -> AssistantMessage:
        """完成 ``_response_to_message`` 对应的模型调用内部步骤。"""
        content = response.get("content")
        if not isinstance(content, list):
            raise AnthropicAPIError("Anthropic API response did not include a content block list.")
        return create_assistant_message(
            content,
            message_id=response.get("id") if isinstance(response.get("id"), str) else None,
        )


@dataclass
class OpenAIChatModelProvider:
    """OpenAI Chat Completions adapter that preserves the internal message protocol."""
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    max_tokens: int = 4096
    timeout: float = 120.0
    transport: Transport | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "OpenAIChatModelProvider":
        """Create an OpenAI Chat provider from env-first v0.4 settings."""
        return cls(
            base_url=os.environ.get("AGENT_KERNEL_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com",
            api_key=os.environ.get("AGENT_KERNEL_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            model=os.environ.get("AGENT_KERNEL_MODEL") or os.environ.get("OPENAI_MODEL"),
        )

    def __post_init__(self) -> None:
        """Fill OpenAI Chat defaults without reading secrets into logs."""
        self.base_url = self.base_url or os.environ.get("AGENT_KERNEL_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com"
        self.api_key = self.api_key if self.api_key is not None else os.environ.get("AGENT_KERNEL_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.model = self.model or os.environ.get("AGENT_KERNEL_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4.1"

    async def stream(
        self,
        *,
        messages: list[dict],
        system_prompt: list[str],
        tools: list[object],
        options: dict,
    ) -> AsyncIterator[AssistantMessage]:
        """Produce one normalized assistant message from Chat Completions."""
        body = await self._build_request_body(messages, system_prompt, tools, options)
        headers = self._headers()
        url = _openai_chat_endpoint_from_base_url(self.base_url or "https://api.openai.com")
        abort_signal = options.get("abortSignal")
        self.calls.append({"url": url, "body": body, "headers": self._redacted_headers(headers)})
        _raise_if_aborted(abort_signal)
        transport = self.transport or _default_openai_transport
        response = await asyncio.to_thread(transport, url, headers, body, self.timeout)
        _raise_if_aborted(abort_signal)
        yield self._response_to_message(response)

    async def _build_request_body(
        self,
        messages: list[dict],
        system_prompt: list[str],
        tools: list[object],
        options: dict,
    ) -> dict[str, Any]:
        """Map internal messages and tools to Chat Completions wire shape."""
        model_messages = normalize_messages_for_api(messages)
        repaired_messages = ensure_tool_result_pairing(model_messages)
        chat_messages = self._messages_to_chat(repaired_messages, system_prompt)
        body: dict[str, Any] = {
            "model": options.get("model") or self.model,
            "messages": chat_messages,
            "max_tokens": options.get("max_tokens", self.max_tokens),
        }
        tool_specs = [await self._tool_to_chat_spec(tool) for tool in tools]
        if tool_specs:
            body["tools"] = tool_specs
            body["tool_choice"] = "auto"
        return body

    def _messages_to_chat(self, messages: list[Message], system_prompt: list[str]) -> list[dict[str, Any]]:
        """Map Anthropic-shaped internal history to OpenAI Chat messages."""
        output: list[dict[str, Any]] = []
        if system_prompt:
            output.append({"role": "system", "content": "\n\n".join(system_prompt)})
        for message in messages:
            if message.get("type") == "tombstone":
                continue
            payload = message.get("message")
            if not isinstance(payload, dict):
                continue
            role = payload.get("role")
            content = payload.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, list):
                continue
            if role == "user":
                texts = _text_blocks_from_content(content)
                if texts:
                    output.append({"role": "user", "content": "\n".join(texts)})
                for block in _tool_result_blocks_from_content(content):
                    output.append(
                        {
                            "role": "tool",
                            "tool_call_id": block["tool_use_id"],
                            "content": _content_to_text(block.get("content", "")),
                        }
                    )
                continue
            tool_calls = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_calls.append(
                        {
                            "id": str(block.get("id", "")),
                            "type": "function",
                            "function": {
                                "name": str(block.get("name", "")),
                                "arguments": json.dumps(block.get("input") if isinstance(block.get("input"), dict) else {}, ensure_ascii=False),
                            },
                        }
                    )
            text = _assistant_text_from_blocks(content)
            chat_message: dict[str, Any] = {"role": "assistant", "content": text or None}
            if tool_calls:
                chat_message["tool_calls"] = tool_calls
            output.append(chat_message)
        return output

    async def _tool_to_chat_spec(self, tool: object) -> dict[str, Any]:
        """Map a kernel Tool to an OpenAI Chat function tool."""
        spec = await _kernel_tool_to_anthropic_spec(tool)
        return {
            "type": "function",
            "function": {
                "name": spec["name"],
                "description": spec.get("description", ""),
                "parameters": spec.get("input_schema") or {"type": "object", "properties": {}},
            },
        }

    def _headers(self) -> dict[str, str]:
        """Build auth headers from env/config without logging secrets."""
        if not self.api_key:
            host = urlparse(self.base_url or "").netloc or "OpenAI-compatible"
            raise OpenAIAPIError(f"{host} API credentials are missing. Set AGENT_KERNEL_API_KEY or OPENAI_API_KEY.")
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _redacted_headers(self, headers: dict[str, str]) -> dict[str, str]:
        """Redact credentials in diagnostic call snapshots."""
        redacted = dict(headers)
        if "Authorization" in redacted:
            redacted["Authorization"] = "Bearer [REDACTED]"
        return redacted

    def _response_to_message(self, response: dict[str, Any]) -> AssistantMessage:
        """Normalize Chat Completions output to an AssistantMessage."""
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise OpenAIAPIError("OpenAI Chat response did not include choices.")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            raise OpenAIAPIError("OpenAI Chat response did not include a message.")
        content_blocks: list[ContentBlock] = []
        content = message.get("content")
        if isinstance(content, str) and content:
            content_blocks.append({"type": "text", "text": content})
        for tool_call in message.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": str(tool_call.get("id") or f"toolu_{uuid4().hex}"),
                    "name": str(function.get("name") or ""),
                    "input": _parse_tool_arguments(function.get("arguments"), provider_name="OpenAI Chat"),
                }
            )
        if not content_blocks:
            content_blocks.append({"type": "text", "text": ""})
        return create_assistant_message(
            content_blocks,
            message_id=response.get("id") if isinstance(response.get("id"), str) else None,
        )


@dataclass
class OpenAIResponsesModelProvider:
    """OpenAI Responses adapter that preserves tool_use/tool_result semantics."""
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    max_tokens: int = 4096
    timeout: float = 120.0
    transport: Transport | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "OpenAIResponsesModelProvider":
        """Create an OpenAI Responses provider from env-first v0.4 settings."""
        return cls(
            base_url=os.environ.get("AGENT_KERNEL_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com",
            api_key=os.environ.get("AGENT_KERNEL_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            model=os.environ.get("AGENT_KERNEL_MODEL") or os.environ.get("OPENAI_MODEL"),
        )

    def __post_init__(self) -> None:
        """Fill Responses defaults without creating global state."""
        self.base_url = self.base_url or os.environ.get("AGENT_KERNEL_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com"
        self.api_key = self.api_key if self.api_key is not None else os.environ.get("AGENT_KERNEL_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.model = self.model or os.environ.get("AGENT_KERNEL_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4.1"

    async def stream(
        self,
        *,
        messages: list[dict],
        system_prompt: list[str],
        tools: list[object],
        options: dict,
    ) -> AsyncIterator[AssistantMessage]:
        """Produce one normalized assistant message from Responses."""
        body = await self._build_request_body(messages, system_prompt, tools, options)
        headers = self._headers()
        url = _openai_responses_endpoint_from_base_url(self.base_url or "https://api.openai.com")
        abort_signal = options.get("abortSignal")
        self.calls.append({"url": url, "body": body, "headers": self._redacted_headers(headers)})
        _raise_if_aborted(abort_signal)
        transport = self.transport or _default_openai_transport
        response = await asyncio.to_thread(transport, url, headers, body, self.timeout)
        _raise_if_aborted(abort_signal)
        yield self._response_to_message(response)

    async def _build_request_body(
        self,
        messages: list[dict],
        system_prompt: list[str],
        tools: list[object],
        options: dict,
    ) -> dict[str, Any]:
        """Map internal messages and tools to Responses wire shape."""
        model_messages = normalize_messages_for_api(messages)
        repaired_messages = ensure_tool_result_pairing(model_messages)
        body: dict[str, Any] = {
            "model": options.get("model") or self.model,
            "input": self._messages_to_responses_input(repaired_messages),
            "max_output_tokens": options.get("max_tokens", self.max_tokens),
        }
        if system_prompt:
            body["instructions"] = "\n\n".join(system_prompt)
        tool_specs = [await self._tool_to_responses_spec(tool) for tool in tools]
        if tool_specs:
            body["tools"] = tool_specs
            body["tool_choice"] = "auto"
        return body

    def _messages_to_responses_input(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Map Anthropic-shaped internal history to Responses input items."""
        output: list[dict[str, Any]] = []
        for message in messages:
            if message.get("type") == "tombstone":
                continue
            payload = message.get("message")
            if not isinstance(payload, dict):
                continue
            role = payload.get("role")
            content = payload.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, list):
                continue
            if role == "user":
                texts = _text_blocks_from_content(content)
                if texts:
                    output.append({"role": "user", "content": "\n".join(texts)})
                for block in _tool_result_blocks_from_content(content):
                    output.append(
                        {
                            "type": "function_call_output",
                            "call_id": block["tool_use_id"],
                            "output": _content_to_text(block.get("content", "")),
                        }
                    )
                continue
            text = _assistant_text_from_blocks(content)
            if text:
                output.append({"role": "assistant", "content": text})
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    output.append(
                        {
                            "type": "function_call",
                            "call_id": str(block.get("id", "")),
                            "name": str(block.get("name", "")),
                            "arguments": json.dumps(block.get("input") if isinstance(block.get("input"), dict) else {}, ensure_ascii=False),
                        }
                    )
        return output

    async def _tool_to_responses_spec(self, tool: object) -> dict[str, Any]:
        """Map a kernel Tool to an OpenAI Responses function tool."""
        spec = await _kernel_tool_to_anthropic_spec(tool)
        return {
            "type": "function",
            "name": spec["name"],
            "description": spec.get("description", ""),
            "parameters": spec.get("input_schema") or {"type": "object", "properties": {}},
            "strict": False,
        }

    def _headers(self) -> dict[str, str]:
        """Build auth headers from env/config without logging secrets."""
        if not self.api_key:
            host = urlparse(self.base_url or "").netloc or "OpenAI-compatible"
            raise OpenAIAPIError(f"{host} API credentials are missing. Set AGENT_KERNEL_API_KEY or OPENAI_API_KEY.")
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _redacted_headers(self, headers: dict[str, str]) -> dict[str, str]:
        """Redact credentials in diagnostic call snapshots."""
        redacted = dict(headers)
        if "Authorization" in redacted:
            redacted["Authorization"] = "Bearer [REDACTED]"
        return redacted

    def _response_to_message(self, response: dict[str, Any]) -> AssistantMessage:
        """Normalize Responses output items to an AssistantMessage."""
        output = response.get("output")
        if not isinstance(output, list):
            raise OpenAIAPIError("OpenAI Responses response did not include output items.")
        content_blocks: list[ContentBlock] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "message":
                for block in item.get("content") or []:
                    if isinstance(block, dict) and isinstance(block.get("text"), str):
                        content_blocks.append({"type": "text", "text": block["text"]})
            elif item_type == "output_text" and isinstance(item.get("text"), str):
                content_blocks.append({"type": "text", "text": item["text"]})
            elif item_type == "function_call":
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": str(item.get("call_id") or item.get("id") or f"toolu_{uuid4().hex}"),
                        "name": str(item.get("name") or ""),
                        "input": _parse_tool_arguments(item.get("arguments"), provider_name="OpenAI Responses"),
                    }
                )
        if not content_blocks:
            output_text = response.get("output_text")
            if isinstance(output_text, str):
                content_blocks.append({"type": "text", "text": output_text})
        if not content_blocks:
            content_blocks.append({"type": "text", "text": ""})
        return create_assistant_message(
            content_blocks,
            message_id=response.get("id") if isinstance(response.get("id"), str) else None,
        )


async def _kernel_tool_to_anthropic_spec(tool: object) -> dict[str, Any]:
    """Build the kernel's canonical tool spec before provider-specific mapping."""
    if hasattr(tool, "to_api_spec"):
        return await _maybe_await(tool.to_api_spec())
    properties = {
        field_name: _json_schema_for_type(field_type)
        for field_name, field_type in getattr(tool, "input_schema", {}).items()
    }
    description_parts = []
    if hasattr(tool, "description"):
        description_parts.append(await tool.description(None))
    if hasattr(tool, "prompt"):
        description_parts.append(await tool.prompt())
    description = "\n\n".join(part for part in description_parts if part)
    return {
        "name": getattr(tool, "name"),
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": list(getattr(tool, "required_fields", ())),
            "additionalProperties": False,
        },
    }


def build_model_provider_from_env(*, require_credentials: bool = False) -> ModelProvider:
    """Build the v0.4 env-first model provider without changing the query loop."""
    provider_name = (os.environ.get("AGENT_KERNEL_PROVIDER") or "anthropic").strip().lower()
    if provider_name in {"anthropic", "anthropic-compatible"}:
        has_credentials = bool(
            os.environ.get("AGENT_KERNEL_API_KEY")
            or os.environ.get("ANTHROPIC_AUTH_TOKEN")
            or os.environ.get("ANTHROPIC_API_KEY")
        )
        if require_credentials and not has_credentials:
            raise AnthropicAPIError(
                "Anthropic-compatible API credentials are missing. Set AGENT_KERNEL_API_KEY, "
                "ANTHROPIC_AUTH_TOKEN, or ANTHROPIC_API_KEY."
            )
        if not has_credentials and not require_credentials:
            return FakeModelProvider()
        return AnthropicModelProvider(
            base_url=os.environ.get("AGENT_KERNEL_BASE_URL") or os.environ.get("ANTHROPIC_BASE_URL"),
            auth_token=os.environ.get("AGENT_KERNEL_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"),
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            model=os.environ.get("AGENT_KERNEL_MODEL") or os.environ.get("ANTHROPIC_MODEL"),
        )
    if provider_name == "openai-chat":
        has_credentials = bool(os.environ.get("AGENT_KERNEL_API_KEY") or os.environ.get("OPENAI_API_KEY"))
        if require_credentials and not has_credentials:
            raise OpenAIAPIError("OpenAI Chat API credentials are missing. Set AGENT_KERNEL_API_KEY or OPENAI_API_KEY.")
        if not has_credentials and not require_credentials:
            return FakeModelProvider()
        return OpenAIChatModelProvider.from_env()
    if provider_name in {"openai-responses", "openai-response"}:
        has_credentials = bool(os.environ.get("AGENT_KERNEL_API_KEY") or os.environ.get("OPENAI_API_KEY"))
        if require_credentials and not has_credentials:
            raise OpenAIAPIError("OpenAI Responses API credentials are missing. Set AGENT_KERNEL_API_KEY or OPENAI_API_KEY.")
        if not has_credentials and not require_credentials:
            return FakeModelProvider()
        return OpenAIResponsesModelProvider.from_env()
    raise OpenAIAPIError(
        f"Unsupported AGENT_KERNEL_PROVIDER '{provider_name}'. "
        "Use anthropic, openai-chat, or openai-responses."
    )


@dataclass
class FakeModelProvider:
    """确定性测试后端：每次调用消费 turns 中的下一项。"""
    turns: list[AssistantMessage | str | list[ContentBlock]] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)

    async def stream(
        self,
        *,
        messages: list[dict],
        system_prompt: list[str],
        tools: list[object],
        options: dict,
    ) -> AsyncIterator[AssistantMessage]:
        """异步产生当前模型后端的 AssistantMessage 消息流。"""
        self.calls.append(
            {
                "messages": messages,
                "system_prompt": system_prompt,
                "tools": tools,
                "options": options,
            }
        )
        if self.turns:
            turn = self.turns.pop(0)
        else:
            turn = "Done."
        if isinstance(turn, str):
            yield create_assistant_message(turn)
        elif isinstance(turn, list):
            yield create_assistant_message(turn)
        else:
            yield turn
