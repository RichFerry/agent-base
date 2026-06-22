# Agent Base

**Language:** English | [简体中文](README.zh-CN.md)

Agent Base is a Python agent kernel with a local runner, deterministic tests,
and opt-in real smoke checks. It is built for people who want a readable,
testable base for local agent experiments without committing to a full product
shell.

## What It Provides

- A core async agent loop: user message, model turn, tool use, tool result, final response.
- A stable `QueryEngine.submit_message(...)` entry point.
- Fake and Anthropic-compatible model providers.
- Built-in tools for shell, files, search, todo, WebSearch, and WebFetch.
- Permission modes limited to `ask` and `bypass`.
- JSONL transcript, resume support, and SDK-style events.
- Optional local Skills and MCP fixture integration.
- Offline-first test suite.

## What It Is Not

- Not a full CLI/TUI product.
- Not an interactive permission UI.
- Not a browser automation system.
- Not a default search or fetch service.
- Not a hosted or remote agent platform.

## Install

```bash
git clone git@github.com:RichFerry/agent-base.git
cd agent-base
python3 -m pip install -e ".[test]"
```

Verify the local runner:

```bash
agent-kernel-local --help
```

Package facts:

- Package name: `agent-kernel`
- Current version: `0.3.0`
- Python: `>=3.11`
- Runtime dependencies: none

## Quick Start

Use an in-process fake model:

```python
import asyncio

from agent_kernel import FakeModelProvider, KernelConfig, QueryEngine


async def main() -> None:
    engine = QueryEngine(
        model_provider=FakeModelProvider(["Hello from a fake model."]),
        config=KernelConfig(),
    )

    async for event in engine.submit_message("hello", max_turns=1):
        print(event)


asyncio.run(main())
```

Run the local example runner:

```bash
agent-kernel-local "Reply with one short sentence: agent kernel smoke test."
```

Without model credentials, real model calls fail early with a clear error.

## Real Model Configuration

Agent Base reads credentials only from environment variables. Do not put API
keys in source files, fixtures, README examples, logs, or transcripts.

```bash
export ANTHROPIC_AUTH_TOKEN="..."
export ANTHROPIC_MODEL="..."
```

For a custom Anthropic-compatible endpoint:

```bash
export ANTHROPIC_BASE_URL="https://api.example.com/anthropic"
```

`ANTHROPIC_API_KEY` is also supported when `ANTHROPIC_AUTH_TOKEN` is not set.

## Local Runner

The runner is an example-layer entry point. It uses the existing kernel loop
rather than reimplementing one.

```text
user input
-> QueryEngine.submit_message()
-> model provider
-> tool_use
-> permission
-> tool execution
-> tool_result
-> assistant final
-> transcript / SDK events
```

Common commands:

```bash
agent-kernel-local --permission-mode ask "Summarize this project in one sentence."
agent-kernel-local --repl
```

Permission modes:

- `ask`: default. Tool calls that need approval are denied unless a callback or hook grants permission.
- `bypass`: explicit local verification mode. Structural path and safety checks still apply.

## Optional Capabilities

Optional capabilities are not loaded by default.

| Capability | How to enable | Default behavior |
| --- | --- | --- |
| WebSearch stub | `AGENT_KERNEL_WEB_SEARCH_PROVIDER=stub` + `--enable-web-search` | No network |
| WebSearch HTTP JSON adapter | `AGENT_KERNEL_WEB_SEARCH_PROVIDER=http-json` + endpoint env vars | Opt-in endpoint |
| WebSearch Anthropic-compatible adapter | `AGENT_KERNEL_WEB_SEARCH_PROVIDER=anthropic-compatible` + endpoint/model/key env vars | Opt-in endpoint |
| WebFetch HTTP handler | `AGENT_KERNEL_WEB_FETCH_PROVIDER=http` + `--enable-web-fetch` | Disabled |
| Local Skills | `--skills-dir examples/skills` | Not loaded |
| MCP fixture | `--mcp-fixture examples/mcp/echo-mcp.json` | Not loaded |
| MCP stdio smoke | `AGENT_KERNEL_RUN_REAL_MCP_SMOKE=1 python3 -m pytest tests/test_real_mcp_smoke.py -q` | Skipped |

### WebSearch Stub

```bash
AGENT_KERNEL_WEB_SEARCH_PROVIDER=stub \
agent-kernel-local \
  --enable-web-search \
  --permission-mode bypass \
  "Search for a stub result and summarize it briefly."
```

### WebFetch HTTP Handler

```bash
export AGENT_KERNEL_WEB_FETCH_PROVIDER=http
export AGENT_KERNEL_WEB_FETCH_TIMEOUT="10"
export AGENT_KERNEL_WEB_FETCH_MAX_BYTES="1000000"
export AGENT_KERNEL_WEB_FETCH_MAX_CHARS="100000"

agent-kernel-local \
  --enable-web-fetch \
  --permission-mode bypass \
  "Fetch https://example.com and summarize it briefly."
```

WebFetch validates URL scheme, applies timeout and size limits, does not execute
JavaScript, does not manage cookies or sessions, and does not parse PDFs or
complex HTML.

### Local Skills

```bash
agent-kernel-local \
  --skills-dir examples/skills \
  --permission-mode bypass \
  "Use the echo skill with hello."
```

The example skill lives at `examples/skills/echo/SKILL.md`.

### MCP Fixture

```bash
agent-kernel-local \
  --mcp-fixture examples/mcp/echo-mcp.json \
  --permission-mode bypass \
  "Call the local echo MCP tool with hello."
```

The fixture is local-only and deterministic. It is not a full MCP configuration
format and does not start a third-party MCP server.

### Combined Local Smoke

```bash
AGENT_KERNEL_WEB_SEARCH_PROVIDER=stub \
agent-kernel-local \
  --enable-web-search \
  --skills-dir examples/skills \
  --mcp-fixture examples/mcp/echo-mcp.json \
  --permission-mode bypass \
  "Search the stub result, use the echo skill, and call the echo MCP tool."
```

## Architecture

```text
QueryEngine.submit_message(prompt)
  -> record user message in session state and JSONL transcript
  -> PromptComposer.fetch_system_prompt_parts(...)
  -> query(QueryParams)
     -> context preparation / compaction
     -> ModelProvider.stream(...)
     -> assistant message / tool_use
     -> run_tools(...)
        -> schema validation
        -> input validation
        -> PreToolUse hooks
        -> permission resolver
        -> Tool.call(...)
        -> PostToolUse hooks
     -> user tool_result message
     -> next model turn
     -> terminal event
  -> optional SDK system/init and result wrappers
```

Important modules:

- `agent_kernel/query_engine.py`: session facade, dependency assembly, transcript writing, SDK event wrappers.
- `agent_kernel/query.py`: core async agent loop.
- `agent_kernel/model_provider.py`: fake and Anthropic-compatible providers.
- `agent_kernel/tool_execution.py`: tool lifecycle.
- `agent_kernel/permissions.py`: ask/bypass permission resolution.
- `agent_kernel/session.py`: JSONL transcript and resume.
- `agent_kernel/mcp.py`: MCP tool/resource wrappers.
- `agent_kernel/skills.py`: local Skill parsing and Skill tool.

## Built-In Tools

Default tools are registered by `agent_kernel.query_engine.default_tools()`:

- `Bash`
- `Glob`
- `Grep`
- `LS`
- `Read`
- `Write`
- `Edit`
- `MultiEdit`
- `NotebookEdit`
- `TodoWrite`
- `WebSearch`
- `WebFetch`

Agent, Skill, and MCP tools are appended only when their corresponding
configuration is present.

## Transcripts and SDK Events

`QueryEngine.submit_message(..., sdk_events=False)` yields the core event stream.

With `sdk_events=True`, the stream includes SDK-style lifecycle wrappers:

- `system/init`
- `result`
- error/status-shaped events where appropriate

Transcripts are written as JSONL under the configured session directory. Tool
results preserve pairing with their originating `tool_use`, and resume reloads
the ordered message chain from the transcript.

## Verification

Default verification is offline:

```bash
python3 -m pytest -q
python3 -m compileall agent_kernel tests
```

Focused checks:

```bash
python3 -m pytest tests/test_local_agent_runner.py -q
python3 -m pytest tests/test_packaging.py -q
```

Real smoke tests are manual and opt-in:

```bash
AGENT_KERNEL_RUN_REAL_SMOKE=1 python3 -m pytest tests/test_real_smoke.py -q
AGENT_KERNEL_RUN_REAL_E2E=1 python3 -m pytest tests/test_real_runner_e2e.py -q
AGENT_KERNEL_RUN_REAL_MCP_SMOKE=1 python3 -m pytest tests/test_real_mcp_smoke.py -q
```

See `docs/smoke-v0.3.md` for setup and safety details.

## Security Notes

- Do not commit API keys, `.env` files, transcripts containing secrets, or real provider responses.
- Default tests do not call real models or real network services.
- Real model and WebSearch/WebFetch checks are opt-in only.
- The runner defaults to `ask` permission mode.
- The example WebFetch handler is not a browser and does not execute JavaScript.

## Explicitly Out of Scope

- Full product CLI or TUI
- Interactive permission UI
- Default real network search/fetch
- Browser automation, JavaScript execution, cookies/sessions
- PDF or deep HTML parsing
- Full MCP product client or third-party MCP server startup in default tests
- Forked Skills
- Remote agents, teams, and worktree isolation
- Permission profiles beyond the ask/bypass kernel model

## Documentation

- `README.zh-CN.md`: Simplified Chinese README.
- `READING_GUIDE.md`: suggested source reading order.
- `docs/release-v0.3.md`: v0.3 release summary and readiness notes.
- `docs/smoke-v0.3.md`: manual real smoke setup.
- `CHANGELOG.md`: release changelog.

## Project Status

Current release: `v0.3.0`.

This repository is a kernel and example runner for local experimentation and
extension. It is designed to keep behavior observable and testable before adding
larger product surfaces.
