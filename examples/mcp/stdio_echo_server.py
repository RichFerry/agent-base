#!/usr/bin/env python3
"""Minimal local stdio MCP echo server for opt-in smoke tests.

This fixture implements only the tiny JSON-RPC subset needed by
tests/test_real_mcp_smoke.py. It does not access the network, filesystem, or
external processes.
"""

from __future__ import annotations

import json
import sys
from typing import Any


ECHO_TOOL = {
    "name": "echo",
    "description": "Echo a text value through a local stdio MCP smoke server.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to echo",
            }
        },
        "required": ["text"],
        "additionalProperties": False,
    },
    "annotations": {
        "readOnlyHint": True,
        "title": "Echo",
    },
}


def _send(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _result(message_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id, "result": result}


def _error(message_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id, "error": {"code": code, "message": message}}


def _call_tool(params: dict[str, Any]) -> dict[str, Any]:
    if params.get("name") != "echo":
        return {
            "isError": True,
            "content": [{"type": "text", "text": f"Unknown tool: {params.get('name')}"}],
        }
    arguments = params.get("arguments") or {}
    text = str(arguments.get("text") or "")
    payload = {"echo": text, "source": "stdio-mcp-smoke"}
    return {
        "structuredContent": payload,
        "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}],
    }


def _handle_request(message: dict[str, Any]) -> bool:
    method = message.get("method")
    message_id = message.get("id")
    params = message.get("params") if isinstance(message.get("params"), dict) else {}

    if method == "initialize":
        _send(
            _result(
                message_id,
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "stdio-echo", "version": "0.1.0"},
                },
            )
        )
        return True
    if method == "tools/list":
        _send(_result(message_id, {"tools": [ECHO_TOOL]}))
        return True
    if method == "tools/call":
        _send(_result(message_id, _call_tool(params)))
        return True
    if method == "shutdown":
        _send(_result(message_id, None))
        return True
    _send(_error(message_id, -32601, f"Method not found: {method}"))
    return True


def main() -> int:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            _send(_error(None, -32700, "Parse error"))
            continue
        if not isinstance(message, dict):
            _send(_error(None, -32600, "Invalid request"))
            continue
        method = message.get("method")
        if method in {"notifications/initialized", "exit"} and "id" not in message:
            if method == "exit":
                return 0
            continue
        if "id" not in message:
            continue
        _handle_request(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
