"""Opt-in real stdio MCP smoke tests.

Skipped by default. These tests start only the local echo server in
examples/mcp/stdio_echo_server.py and never access the network or a real model.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import select
import subprocess
import sys
from typing import Any

import pytest

from agent_kernel import KernelConfig, MCPClientConfig, QueryEngine
from agent_kernel.mcp import build_mcp_tool_name
from agent_kernel.model_provider import FakeModelProvider
from examples.local_agent import run_local_agent_once


RUN_REAL_MCP_SMOKE = os.environ.get("AGENT_KERNEL_RUN_REAL_MCP_SMOKE") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_REAL_MCP_SMOKE,
    reason="real MCP stdio smoke requires AGENT_KERNEL_RUN_REAL_MCP_SMOKE=1",
)


class StdioMcpSmokeClient:
    """Tiny stdio JSON-RPC client for the local echo smoke server."""

    def __init__(self, command: list[str], *, timeout_seconds: float = 5.0) -> None:
        self.command = command
        self.timeout_seconds = timeout_seconds
        self.process: subprocess.Popen[str] | None = None
        self.next_id = 1
        self.calls: list[dict[str, Any]] = []

    def start(self) -> None:
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        if self.process is None or self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("stdio MCP smoke client is not started")
        message_id = self.next_id
        self.next_id += 1
        message = {"jsonrpc": "2.0", "id": message_id, "method": method}
        if params is not None:
            message["params"] = params
        self.process.stdin.write(json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        while True:
            readable, _, _ = select.select([self.process.stdout], [], [], self.timeout_seconds)
            if not readable:
                raise TimeoutError(f"stdio MCP smoke server did not answer {method}")
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(f"stdio MCP smoke server exited before answering {method}")
            response = json.loads(line)
            if response.get("id") != message_id:
                continue
            if "error" in response:
                raise RuntimeError(response["error"].get("message") or "stdio MCP smoke server error")
            return response.get("result")

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        if self.process is None or self.process.stdin is None:
            return
        message = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        try:
            self.process.stdin.write(json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n")
            self.process.stdin.flush()
        except BrokenPipeError:
            return

    def initialize(self) -> dict[str, Any]:
        result = self.request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "agent-kernel-real-mcp-smoke", "version": "0.1.0"},
            },
        )
        self.notify("notifications/initialized")
        return result

    def list_tools(self) -> list[dict[str, Any]]:
        result = self.request("tools/list", {})
        tools = result.get("tools") if isinstance(result, dict) else None
        if not isinstance(tools, list):
            raise RuntimeError("stdio MCP smoke server returned no tools")
        return [tool for tool in tools if isinstance(tool, dict)]

    def call_tool(self, tool_name: str, args: dict[str, Any]) -> Any:
        self.calls.append({"tool_name": tool_name, "args": dict(args)})
        return self.request("tools/call", {"name": tool_name, "arguments": args})

    def close(self) -> None:
        process = self.process
        if process is None:
            return
        if process.poll() is None:
            try:
                self.request("shutdown", {})
            except Exception:
                pass
            self.notify("exit")
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    return repo


def _stdio_fixture_path() -> Path:
    return Path(__file__).parents[1] / "examples" / "mcp" / "stdio-mcp.json"


def _client_from_fixture() -> StdioMcpSmokeClient:
    repo_root = Path(__file__).parents[1]
    fixture = json.loads(_stdio_fixture_path().read_text(encoding="utf-8"))
    args = [str(repo_root / arg) if arg.startswith("examples/") else arg for arg in fixture.get("args", [])]
    command_name = str(fixture.get("command") or "")
    command = [sys.executable if command_name == "python3" else command_name, *args]
    return StdioMcpSmokeClient(command)


def _transcript_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _tool_result_blocks(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for event in events:
        if event.get("type") != "user":
            continue
        content = event.get("message", {}).get("content", [])
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                blocks.append(block)
    return blocks


def test_real_stdio_mcp_echo_tool_reaches_tool_result_and_transcript(tmp_path: Path) -> None:
    client = _client_from_fixture()
    client.start()
    process = client.process
    try:
        initialized = client.initialize()
        tools = client.list_tools()
        expected_tool_name = build_mcp_tool_name("stdio-echo", "echo")
        mcp_client = MCPClientConfig(
            name="stdio-echo",
            instructions="Local stdio MCP smoke server.",
            type="connected",
            tools=tuple(tools),
            call_tool_handler=client.call_tool,
        )
        provider = FakeModelProvider(
            [
                [
                    {
                        "type": "tool_use",
                        "id": "toolu_stdio_echo",
                        "name": expected_tool_name,
                        "input": {"text": "hello"},
                    }
                ],
                "stdio MCP smoke completed.",
            ]
        )
        engine = QueryEngine(
            model_provider=provider,
            config=KernelConfig(cwd=_repo(tmp_path), config_home=tmp_path / ".claude", mcp_clients=(mcp_client,)),
            session_id="real-stdio-mcp-smoke",
        )
        engine.tool_use_context.app_state.tool_permission_context.mode = "bypass"

        result = asyncio.run(run_local_agent_once("call the stdio echo MCP tool", engine=engine, max_turns=3))
    finally:
        client.close()

    tool_results = _tool_result_blocks(result.events)
    rows = _transcript_rows(result.transcript_path)

    assert process is not None
    assert process.poll() == 0
    assert initialized["serverInfo"]["name"] == "stdio-echo"
    assert tools[0]["name"] == "echo"
    assert result.events[0]["type"] == "system"
    assert result.events[0]["subtype"] == "init"
    assert expected_tool_name in result.events[0]["tools"]
    assert result.events[0]["mcp_servers"] == [{"name": "stdio-echo", "status": "connected"}]
    assert result.events[-1]["type"] == "result"
    assert result.events[-1]["is_error"] is False
    assert client.calls == [{"tool_name": "echo", "args": {"text": "hello"}}]
    assert len(tool_results) == 1
    assert tool_results[0]["tool_use_id"] == "toolu_stdio_echo"
    assert tool_results[0].get("is_error") is not True
    assert tool_results[0]["content"] == '{"echo":"hello","source":"stdio-mcp-smoke"}'
    assert result.final_response == "stdio MCP smoke completed."
    assert any("[tool_use] mcp__stdio-echo__echo" in line for line in result.logs)
    assert any("[tool_result:ok] toolu_stdio_echo" in line for line in result.logs)
    assert any(
        row.get("type") == "user"
        and row.get("message", {}).get("content", [{}])[0].get("tool_use_id") == "toolu_stdio_echo"
        for row in rows
    )
