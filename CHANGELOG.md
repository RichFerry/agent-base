# Changelog

## 0.7.0

- Added an internal Workspace Runtime that resolves cwd, workspace root, config home, project store, sessions, memory, artifacts, settings sources, Skills sources, MCP sources, and act-mode allowed working directories.
- Anchored JSONL sessions, project memory, Bash background output, and Agent output under one workspace project store.
- Added `agent-kernel-local workspace doctor` and `workspace doctor --json` for credential-free workspace diagnostics.
- Added workspace metadata to SDK `system/init` events and model-facing environment context.
- Kept permission modes limited to `ask` and `bypass`; no permission profiles or interactive permission UI were added.

## 0.6.0

- Added MCP result metadata for local tool/resource calls so transcript and SDK events can be audited without changing the core tool loop.
- Added session validation, inspection, timeline, redacted export, and dry-run/confirmed GC commands.
- Added deterministic manual memory extraction from session transcripts, candidate JSON application, provenance sidecars, memory validation, and index rebuild commands.
- Added static MCP doctor, explicit `mcp doctor --start`, and MCP inspect diagnostics.
- Added v0.6 JSON event schema version metadata in the local runner.
- Added v0.6 docs for the MCP / Sessions / Memory full-chain workflow.

## 0.5.0

- Added layered `settings.json` local runner settings with secret rejection, redacted doctor output, and effective config diagnostics.
- Added daily-use CLI management aliases for config, skills, sessions, memory, and MCP.
- Added multi-directory Skills loading, validation, JSON listing, strict validation, and skill info commands.
- Added session info/export/delete/transcript-path commands while preserving JSONL resume ordering.
- Added explicit memory list/append/remember/forget/delete/validate commands with path safety.
- Hardened local MCP config loading with multiple configs, disabled servers, name collision checks, timeout validation, and bounded stderr diagnostics.
- Added v0.5 docs for settings, skills, session/memory, MCP, CLI, and release readiness.

## 0.4.0

- Added provider selection for Anthropic-compatible Messages, OpenAI Chat Completions, and OpenAI Responses while keeping `ModelProvider.stream(...)` as the internal kernel interface.
- Moved reusable WebSearch/WebFetch adapter construction into an internal adapter module and kept all network behavior explicit opt-in.
- Added local-only MCP stdio config loading with stdlib JSON-RPC lifecycle support.
- Added explicit skill discovery mode, deterministic local runner skill loading, duplicate skill reporting, and `--list-skills`.
- Added local runner session commands for listing, resume, and continue.
- Added explicit memory status/read/write CLI commands with relative-path safety checks and no automatic memory extraction.
- Updated EN/CH README and v0.4 release documentation.

## 0.3.0

- Added opt-in real smoke documentation and tests for real model, real runner E2E, WebSearch adapters, WebFetch handler, and MCP stdio smoke.
- Added `agent-kernel-local` console script for the example-layer local runner.
- Added packaging smoke tests for `pyproject.toml`, runner help, and editable install console script startup.
- Added minimal offline CI baseline.
- Added v0.3 release readiness documentation and repository hygiene guidance.
- Removed tracked generated `.DS_Store` and `agent_kernel.egg-info/` files from the git index for release hygiene.

## 0.2.0

- Added minimal local runner around `QueryEngine.submit_message()`.
- Added example-layer WebSearch provider injection, local Skills loading, and MCP fixture loading.
- Added combined capability, tool registry collision, and SDK/transcript contract tests.
- Documented v0.2 unsupported product scope.
