# v0.6.0 Release Notes

Agent Base v0.6.0 connects the previously separate MCP, Sessions, and Memory
surfaces into an explicit, auditable local workflow.

## Summary

- MCP tool/resource results now carry additive metadata in normal `tool_result`
  blocks: server, normalized names, operation, status, duration, content size,
  and truncation status.
- `mcp doctor` is static by default; `mcp doctor --start` explicitly starts
  local stdio servers, lists tools/resources, and shuts them down.
- `mcp inspect SERVER` shows local MCP tool/resource details without model
  calls.
- `sessions validate`, `sessions inspect`, `sessions timeline`, redacted
  `sessions export`, and `sessions gc` provide transcript audit workflows.
- `memory extract SESSION_ID --dry-run/--yes` proposes and applies deterministic
  memory candidates from transcript evidence.
- Memory writes include provenance sidecars; `memory validate`,
  `memory rebuild-index`, and `memory provenance` support maintenance.
- Local runner JSON events include additive schema version `"0.6"`.
- Package version bumped to `0.6.0`.

## Semantics Preserved

- `QueryEngine.submit_message(...)` remains the core agent loop entry point.
- MCP still enters through the existing tool registry and normal tool execution
  path.
- `tool_use` and `tool_result` ordering remains unchanged.
- Memory extraction is manual only; no after-run auto-write or background
  extraction was added.
- Memory prompt behavior remains compatible with earlier releases: sessions load
  the memory prompt/path, but memory content is not silently copied into every
  response path beyond existing prompt-composer behavior.
- Permission remains limited to `ask` and `bypass`.
- Default tests remain offline.

## Verification

```bash
python3 -m pytest -q
python3 -m compileall agent_kernel tests
agent-kernel-local --help
git diff --check
```

## Explicitly Unsupported

- Remote MCP, OAuth, SSE, or third-party MCP defaults.
- Automatic memory extraction or background memory writes.
- Full TUI or interactive permission UI.
- Permission profiles beyond `ask` and `bypass`.
- Browser automation, JavaScript execution, cookies/sessions, PDF/deep HTML
  parsing.
- Forked Skills.
- Remote agents, teams, and worktree isolation.
