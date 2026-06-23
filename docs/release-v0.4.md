# v0.4 Local CLI Readiness Notes

This document closes the v0.4 local CLI feature pass for Agent Base.

v0.4 keeps the Python kernel semantics stable while making the example runner
more useful for daily local work. It does not add a TUI, interactive permission
UI, hosted agents, remote MCP, forked skills, or default network behavior.

## Summary

- Provider selection now supports `anthropic`, `openai-chat`, and `openai-responses`.
- `ModelProvider.stream(...)` remains the internal kernel interface.
- WebSearch and WebFetch adapter construction moved into `agent_kernel/web_adapters.py`.
- WebSearch/WebFetch remain opt-in; default tests do not access the network.
- MCP fixture support remains, and local-only stdio config is available with `--mcp-config` or `AGENT_KERNEL_MCP_CONFIG`.
- Skills now have explicit discovery mode support and runner-level `--list-skills`.
- Local runner sessions can be listed, resumed, or continued.
- Memory can be inspected and explicitly read/written through safe relative paths.
- Real smokes remain manual and opt-in only.

## Provider Matrix

| Provider | Selection | Model env | Base URL env | Key env | Default network |
| --- | --- | --- | --- | --- | --- |
| Anthropic-compatible Messages | `AGENT_KERNEL_PROVIDER=anthropic` | `AGENT_KERNEL_MODEL` or `ANTHROPIC_MODEL` | `AGENT_KERNEL_BASE_URL` or `ANTHROPIC_BASE_URL` | `AGENT_KERNEL_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_API_KEY` | only when credentials are supplied and runner is invoked |
| OpenAI Chat Completions | `AGENT_KERNEL_PROVIDER=openai-chat` | `AGENT_KERNEL_MODEL` or `OPENAI_MODEL` | `AGENT_KERNEL_BASE_URL` or `OPENAI_BASE_URL` | `AGENT_KERNEL_API_KEY` or `OPENAI_API_KEY` | only when credentials are supplied and runner is invoked |
| OpenAI Responses | `AGENT_KERNEL_PROVIDER=openai-responses` | `AGENT_KERNEL_MODEL` or `OPENAI_MODEL` | `AGENT_KERNEL_BASE_URL` or `OPENAI_BASE_URL` | `AGENT_KERNEL_API_KEY` or `OPENAI_API_KEY` | only when credentials are supplied and runner is invoked |

## Runner Capability Matrix

| Capability | Runner flag / env | Default behavior | v0.4 status |
| --- | --- | --- | --- |
| WebSearch stub | `--enable-web-search`, `AGENT_KERNEL_WEB_SEARCH_PROVIDER=stub` | disabled | implemented / offline-tested |
| WebSearch HTTP JSON | `AGENT_KERNEL_WEB_SEARCH_PROVIDER=http-json` plus URL env | disabled | implemented / contract-tested |
| WebSearch Anthropic-compatible adapter | `AGENT_KERNEL_WEB_SEARCH_PROVIDER=anthropic-compatible` plus URL/key/model env | disabled | implemented / contract-tested / real endpoint smoke pending |
| WebFetch HTTP | `--enable-web-fetch`, `AGENT_KERNEL_WEB_FETCH_PROVIDER=http` | disabled | implemented / fake-tested |
| Local Skills | `--skills-dir PATH` | not loaded | implemented / explicit discovery |
| Skill inspection | `--skills-dir PATH --list-skills` | no model call | implemented |
| MCP fixture | `--mcp-fixture PATH` | not loaded | implemented |
| MCP stdio config | `--mcp-config PATH` or `AGENT_KERNEL_MCP_CONFIG` | not loaded | implemented / local-only |
| Sessions | `--list-sessions`, `--resume`, `--continue` | no resume unless requested | implemented |
| Memory | `--memory-status`, `--memory-read`, `--memory-write --memory-text` | no automatic extraction | implemented / explicit only |

## MCP Stdio Scope

The v0.4 MCP config loader accepts a local-only JSON shape:

```json
{
  "mcpServers": {
    "name": {
      "command": "python3",
      "args": ["examples/mcp/stdio_echo_server.py"],
      "env": {},
      "cwd": "."
    }
  }
}
```

It implements a minimal stdio JSON-RPC lifecycle with standard-library code:
`initialize`, `tools/list`, `resources/list`, `tools/call`, `resources/read`,
and shutdown/exit notification. Failed initialization does not intentionally
register partial tools.

Remote MCP, OAuth, SSE, third-party server defaults, and long-running product
process management are outside v0.4.

## Session and Memory Scope

Resume uses the existing JSONL transcript and `SessionStore` ordering. The
runner can list session ids, resume an explicit session id, or continue the
most recently modified local session.

Memory commands are explicit. v0.4 does not automatically extract memories from
conversation content. Memory file paths must be relative and cannot escape the
project memory directory.

## Verification Baseline

Default verification remains offline:

```bash
python3 -m pytest -q
python3 -m compileall agent_kernel tests
```

Focused v0.4 checks:

```bash
python3 -m pytest tests/test_permissions_tools_query.py -q
python3 -m pytest tests/test_local_agent_runner.py tests/test_mcp.py tests/test_skills.py -q
python3 -m pytest tests/test_capability_contracts.py -q
```

Real smokes remain opt-in and are documented in `docs/smoke-v0.3.md`.

## Explicitly Unsupported in v0.4

- Full product CLI/TUI.
- Interactive permission UI.
- Permission profiles beyond `ask` and `bypass`.
- Default real network search or fetch.
- Browser automation, JavaScript execution, cookies/sessions, PDF parsing, or deep HTML parsing.
- Remote MCP, OAuth, SSE, or default third-party MCP server startup.
- Forked skills.
- Remote agents, teams, and worktree isolation.
- Automatic memory extraction.
