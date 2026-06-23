# v0.2 Release Notes and Readiness Audit

This document closes the v0.2 local runner and capability hardening phase for the Python Agent Kernel in `main/`.

## Release Summary

v0.2 does not attempt to build a full product shell. It adds a minimal, testable local runner layer around the existing kernel so the current Python port can verify end-to-end capability flow through `QueryEngine.submit_message()`.

### Local Runner

- Entry point: `examples/local_agent.py`.
- Supports one-shot prompts and a simple REPL.
- Uses existing `QueryEngine`, default tools, prompt composer, permission context, session store, transcript, and SDK event wrappers.
- Reads Anthropic-compatible model configuration from environment variables for real local use.
- Fails clearly when model credentials are missing.
- Remains example-layer code, not a product CLI or TUI.

### WebSearch Provider Injection

- `WebSearch` remains a default tool, but no provider is configured by default.
- The local runner can opt in with `--enable-web-search` plus `AGENT_KERNEL_WEB_SEARCH_PROVIDER=stub` or `--web-search-provider stub`.
- The only v0.2 example provider is a deterministic no-network stub.
- Real network search, browser integration, and commercial search API defaults are out of scope.

### Local Skills

- The runner can opt in to local skills with `--skills-dir PATH`.
- Skills are loaded from child directories containing `SKILL.md`.
- Inline skills expand into the main conversation through the existing `Skill` tool.
- Forked skills return an explicit not implemented result.
- No skill marketplace, install UX, remote execution, or forked-skill runtime is included.

### MCP Local Fixture

- The runner can opt in to one local MCP smoke fixture with `--mcp-fixture PATH`.
- The fixture is converted into `MCPClientConfig` with local call/resource handlers.
- It does not represent a full MCP config standard.
- It does not start external processes, connect to a real MCP server, or use the network.

### Combined Flow

The v0.2 tests verify that WebSearch, local Skills, and MCP fixture tools can coexist in the same local runner and `QueryEngine` session:

```text
user input
-> QueryEngine.submit_message()
-> fake or configured model provider
-> WebSearch / Skill / MCP tool_use
-> permission
-> tool execution
-> tool_result
-> assistant final
-> transcript and SDK events
```

### Tool Registry Collision Policy

- Built-in tools are registered first.
- The internal `Agent` tool appears before `WebSearch` when agents are available; SDK init maps it to `Task` for compatibility.
- MCP tools are merged by name and do not replace existing tools.
- Normal MCP tools keep the `mcp__<server>__<tool>` prefix.
- MCP resource helpers, such as `ListMcpResourcesTool` and `ReadMcpResourceTool`, stay separate from normal MCP server tools.
- `Skill` is appended only when local skills are explicitly loaded.
- Duplicate names use a stable first registered wins policy.

### Transcript and SDK Contract

- SDK event streams start with `system/init` and end with `result`.
- JSONL transcript rows preserve user prompt, assistant `tool_use`, paired user `tool_result`, optional synthetic skill prompt, and assistant final ordering.
- `tool_result` rows keep `sourceToolAssistantUUID` and parent linkage to the assistant message that emitted the matching `tool_use`.
- Resume preserves message ordering and pairing.
- Compact boundaries are persisted as `system` rows with `subtype: compact_boundary` and stable `compactMetadata`.
- Model errors and recoverable tool errors use stable event/transcript paths.

### Permission Boundary

- The local runner exposes only `ask` and `bypass`.
- Default mode is `ask`.
- `bypass` is explicit pass-through for local verification.
- v0.2 does not implement permission profiles, `plan`, `acceptEdits`, or interactive permission UI.

## Explicitly Unsupported in v0.2

- Full product CLI or TUI.
- Real browser/search integration.
- Default real WebSearch provider.
- Real MCP server startup or full MCP config management.
- Forked skills.
- Remote agents.
- Agent teams.
- Worktree isolation.
- Permission profiles.
- Interactive permission UI.
- Runtime dependencies beyond the Python standard library.

## Readiness Audit

### Public API

No v0.2 example runner helpers were added to `agent_kernel.__init__`. The local runner helpers remain under `examples/local_agent.py`, and capability fixtures remain outside the core public package API.

Audit command:

```bash
git diff -- agent_kernel/__init__.py
```

Expected result: no diff.

### Runtime Dependencies

`pyproject.toml` still declares no runtime dependencies:

```toml
dependencies = []
```

The only optional test dependency remains `pytest`.

### Verification Baseline

Release readiness baseline:

```bash
python3 -m pytest -q
python3 -m compileall agent_kernel tests
```

Focused v0.2 contract coverage includes:

- `tests/test_local_agent_runner.py`
- `tests/test_local_agent_combined.py`
- `tests/test_capability_contracts.py`
- `tests/test_tool_registry_contracts.py`
- `tests/test_transcript_contracts.py`
- `tests/test_mcp.py`
- `tests/test_skills.py`
- `tests/test_sdk_transcript.py`

### Dirty Worktree Notes

`.DS_Store` and `agent_kernel/.DS_Store` may appear dirty in the local worktree. They are not part of the v0.2 release scope and should not be modified or staged as part of this release hardening pass.

The top-level `/Users/ferry/Desktop/agent/src` tree is the original TS source reference and remains outside the Python `main/` package release scope.

## v0.2 Exit Criteria

- Local runner exists and remains example-layer.
- WebSearch, Skills, and MCP fixture are explicit opt-in capabilities.
- Combined capability flow is covered without network or real model calls.
- Tool registry collision behavior is documented and tested.
- SDK/transcript event shape and ordering are documented and tested.
- Permission runner UX remains limited to `ask` and `bypass`.
- Unsupported product scopes are explicitly documented.
- Full tests and compileall pass.
