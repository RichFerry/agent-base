# Agent Base

**Language:** English | [简体中文](README.zh-CN.md)

Agent Base is a Python agent kernel with a local runner, deterministic tests,
and opt-in real smoke checks. It is built for people who want a readable,
testable base for local agent experiments without committing to a full product
shell.

## What It Provides

- A core async agent loop: user message, model turn, tool use, tool result, final response.
- A stable `QueryEngine.submit_message(...)` entry point.
- Fake, Anthropic-compatible, OpenAI Chat, and OpenAI Responses providers.
- Built-in tools for shell, files, search, todo, WebSearch, and WebFetch.
- Permission modes limited to `ask` and `bypass`.
- JSONL transcript, resume support, and SDK-style events.
- Optional local Skills, MCP fixture/config integration, and session/memory CLI helpers.
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
- Current version: `0.7.0`
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

## Model Provider Configuration

Agent Base reads credentials only from environment variables. Do not put API
keys in source files, fixtures, README examples, logs, or transcripts.

Select a provider with `AGENT_KERNEL_PROVIDER`:

| Provider | Value | Credential fallback |
| --- | --- | --- |
| Anthropic-compatible Messages | `anthropic` | `AGENT_KERNEL_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_API_KEY` |
| OpenAI Chat Completions | `openai-chat` | `AGENT_KERNEL_API_KEY`, `OPENAI_API_KEY` |
| OpenAI Responses | `openai-responses` | `AGENT_KERNEL_API_KEY`, `OPENAI_API_KEY` |

```bash
export AGENT_KERNEL_PROVIDER="anthropic"
export AGENT_KERNEL_API_KEY="..."
export AGENT_KERNEL_MODEL="..."
```

Provider-specific environment variables are still supported. For a custom
Anthropic-compatible endpoint:

```bash
export ANTHROPIC_BASE_URL="https://api.example.com/anthropic"
```

For OpenAI-compatible modes:

```bash
export AGENT_KERNEL_PROVIDER="openai-chat"
export OPENAI_API_KEY="..."
export OPENAI_MODEL="..."
```

`AGENT_KERNEL_BASE_URL` can override the provider base URL for compatible
endpoints.

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
agent-kernel-local --init-config
agent-kernel-local --doctor
agent-kernel-local workspace doctor
agent-kernel-local workspace doctor --json
agent-kernel-local config doctor
agent-kernel-local config effective
agent-kernel-local --list-sessions
agent-kernel-local sessions list
agent-kernel-local --resume SESSION_ID "Continue from here."
agent-kernel-local --continue "Continue the latest local session."
agent-kernel-local --memory-status
agent-kernel-local memory list
agent-kernel-local --memory-read
agent-kernel-local --memory-write notes/preference.md --memory-text "Prefer concise answers."
agent-kernel-local mcp doctor --start --json
agent-kernel-local sessions validate SESSION_ID --json
agent-kernel-local sessions timeline SESSION_ID
agent-kernel-local memory extract SESSION_ID --dry-run
```

Permission modes:

- `ask`: default. Tool calls that need approval are denied unless a callback or hook grants permission.
- `bypass`: explicit local verification mode. Structural path and safety checks still apply.

### Local Config

Agent Base uses `settings.json` as the official local runner config file:

```bash
agent-kernel-local --init-config
agent-kernel-local --doctor
agent-kernel-local --print-effective-config
```

Discovery order is user settings, project settings, then explicit
`--agent-config`; higher-precedence layers override lower ones. CLI flags and
`AGENT_KERNEL_*` environment variables override settings. API keys still come
only from environment variables.

Supported non-secret defaults include provider type/model/base URL, permission
mode, max turns, WebSearch/WebFetch opt-ins, Skills directories, MCP
fixture/config paths, session defaults, memory defaults, and debug flags.

Example:

```json
{
  "provider": {"type": "anthropic", "model": "", "baseUrl": ""},
  "runner": {"permissionMode": "ask", "maxTurns": 10},
  "skills": {"dirs": ["examples/skills"], "discoveryMode": "explicit"},
  "mcp": {"configs": ["examples/mcp/stdio-config.json"]}
}
```

Do not store API keys, tokens, passwords, or secrets in `settings.json`.

### v0.7 Workspace Runtime

v0.7.0 makes workspace identity explicit and shared across the kernel and local
runner. The agent now has one runtime view for:

- current `cwd`
- workspace root and whether it came from git discovery or the explicit cwd
- loaded `settings.json` sources
- configured Skills and MCP source scope
- project-scoped sessions, transcripts, memory, and artifacts
- act/bypass allowed working directories

Inspect it without credentials, network, or MCP startup:

```bash
agent-kernel-local workspace doctor
agent-kernel-local workspace doctor --json
agent-kernel-local --print-effective-config
```

Workspace storage is project-scoped under the config home:

```text
<config_home>/projects/<workspace-key>/
  *.jsonl
  memory/
  artifacts/
    bash-output/
    agent-output/
```

If `cwd` is inside a git repository, the workspace key is based on the git root
so sessions and memory are shared across subdirectories of the same project.
The act/bypass file boundary remains conservative: only the current `cwd` and
explicit additional working directories are considered allowed working paths.

### v0.6 MCP / Sessions / Memory Chain

v0.6.0 connects local MCP, JSONL sessions, and explicit memory extraction into
one auditable workflow:

```text
MCP fixture/config/stdin
-> QueryEngine tool registry
-> MCP tool/resource result
-> SDK events and JSONL transcript
-> sessions validate/inspect/timeline/export
-> memory extract dry-run
-> memory extract --yes writes memory files and MEMORY.md
-> resume keeps transcript ordering and loads the memory prompt
```

Useful commands:

```bash
agent-kernel-local mcp doctor --json
agent-kernel-local mcp doctor --start --json
agent-kernel-local mcp inspect local-echo --mcp-fixture examples/mcp/echo-mcp.json --json

agent-kernel-local sessions inspect SESSION_ID --json
agent-kernel-local sessions validate SESSION_ID --json
agent-kernel-local sessions timeline SESSION_ID
agent-kernel-local sessions export SESSION_ID --redacted
agent-kernel-local sessions gc --dry-run --older-than 30

agent-kernel-local memory extract SESSION_ID --dry-run --json
agent-kernel-local memory extract SESSION_ID --yes
agent-kernel-local memory validate --json
agent-kernel-local memory rebuild-index --dry-run
agent-kernel-local memory provenance reference/example.md --json
```

Memory extraction is manual only. Dry-run mutates nothing; `--yes` writes memory
files, updates `MEMORY.md`, stores provenance sidecars, and records a
session-visible extraction event. MCP resource memories store pointers, not raw
resource dumps.

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
| MCP stdio config | `--mcp-config examples/mcp/stdio-config.json` or `AGENT_KERNEL_MCP_CONFIG` | Not loaded |
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

Inspect local skills without calling a model:

```bash
agent-kernel-local --skills-dir examples/skills --list-skills
agent-kernel-local skills list --skills-dir examples/skills --json
agent-kernel-local skills validate --skills-dir examples/skills
agent-kernel-local skills info echo --skills-dir examples/skills
```

Multiple `--skills-dir` values are supported. Runner discovery is explicit by
default; ambient skill discovery is available only when configured. Forked
skills are still reported as not implemented.

### MCP Fixture

```bash
agent-kernel-local \
  --mcp-fixture examples/mcp/echo-mcp.json \
  --permission-mode bypass \
  "Call the local echo MCP tool with hello."
```

The fixture is local-only and deterministic. It is not a full MCP configuration
format and does not start a third-party MCP server.

### MCP Stdio Config

```bash
agent-kernel-local \
  --mcp-config examples/mcp/stdio-config.json \
  --permission-mode bypass \
  "Call the stdio echo MCP tool with hello."
```

The v0.5.0 config loader supports local-only stdio servers shaped as
`mcpServers.{name}.command`, `args`, `env`, and optional `cwd`. It uses stdlib
JSON-RPC over stdio and does not support remote MCP, OAuth, SSE, or default
third-party server startup.

v0.5.0 adds management diagnostics:

```bash
agent-kernel-local mcp list --mcp-fixture examples/mcp/echo-mcp.json --json
agent-kernel-local mcp doctor --mcp-config examples/mcp/stdio-config.json
agent-kernel-local mcp validate-config examples/mcp/stdio-config.json
```

### Sessions and Memory

```bash
agent-kernel-local --list-sessions
agent-kernel-local sessions info SESSION_ID --json
agent-kernel-local sessions export SESSION_ID
agent-kernel-local sessions delete SESSION_ID --yes
agent-kernel-local --resume SESSION_ID "Continue this session."
agent-kernel-local --continue "Continue the latest session."
agent-kernel-local --memory-status
agent-kernel-local memory list --json
agent-kernel-local --memory-read
agent-kernel-local --memory-write notes/preference.md --memory-text "Prefer concise answers."
agent-kernel-local memory remember --memory-type feedback --memory-name "Terse Replies" --memory-text "Prefer terse engineering summaries."
agent-kernel-local memory validate
```

Resume reads existing JSONL transcript ordering through `SessionStore`. Memory
management is explicit only; there is no automatic memory extraction in v0.5.
Memory paths must be relative and stay inside the project memory directory.

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
- `agent_kernel/model_provider.py`: fake, Anthropic-compatible, OpenAI Chat, and OpenAI Responses providers.
- `agent_kernel/web_adapters.py`: example-layer WebSearch and WebFetch adapters.
- `agent_kernel/tool_execution.py`: tool lifecycle.
- `agent_kernel/permissions.py`: ask/bypass permission resolution.
- `agent_kernel/session.py`: JSONL transcript and resume.
- `agent_kernel/memory.py`: project memory pathing and prompt helpers.
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
python3 -m pytest tests/test_workspace_runtime.py -q
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
- `docs/release-v0.4.md`: v0.4 local CLI readiness notes.
- `docs/config-v0.5.md`: v0.5 local settings and doctor notes.
- `docs/skills-v0.5.md`: v0.5 Skills management notes.
- `docs/session-memory-v0.5.md`: v0.5 session and memory management notes.
- `docs/mcp-v0.5.md`: v0.5 local MCP config hardening notes.
- `docs/cli-v0.5.md`: v0.5 local CLI daily-use notes.
- `docs/release-v0.5.md`: v0.5 release readiness notes.
- `docs/release-v0.6.md`: v0.6 MCP / Sessions / Memory chain notes.
- `docs/mcp-memory-session-v0.6.md`: v0.6 full-chain workflow notes.
- `docs/release-v0.7.md`: v0.7 Workspace Runtime notes.
- `docs/smoke-v0.3.md`: manual real smoke setup.
- `CHANGELOG.md`: release changelog.

## Project Status

Current release: `v0.7.0`.

This repository is a kernel and example runner for local experimentation and
extension. It is designed to keep behavior observable and testable before adding
larger product surfaces.
