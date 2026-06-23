# v0.5.0 Release Notes

Agent Base v0.5.0 turns the v0.4 local runner into a more complete daily-use
agent base while preserving the existing Python kernel semantics.

## Summary

- `settings.json` local runner settings with layered discovery and redacted
  diagnostics.
- Skills management commands for list, JSON list, validation, strict validation,
  and skill info.
- Session management commands for list, info, transcript path, export, delete,
  resume, and continue.
- Explicit memory management commands for list, read, write, append, remember,
  forget, delete, and validation.
- MCP local stdio config hardening with multiple configs, disabled servers,
  normalized collision checks, timeouts, stderr diagnostics, and management
  commands.
- Local CLI daily-use diagnostics: doctor JSON, effective config, JSON events,
  transcript path printing, debug provider, and debug tools.
- Package version bumped to `0.5.0`.

## Semantics Preserved

- `QueryEngine.submit_message(...)` remains the core runner entry point.
- `ModelProvider.stream(...)` remains the internal provider interface.
- WebSearch/WebFetch still execute only through injected handlers and normal
  tool_result flow.
- Skills remain inline context expansion; forked skills remain not implemented.
- MCP tools/resources still enter the existing tool registry and transcript path.
- Permission remains `ask` or `bypass`.
- Default tests remain offline.

## Verification

```bash
python3 -m pytest -q
python3 -m compileall agent_kernel tests
agent-kernel-local --help
git diff --check
```

## Explicitly Unsupported

- Full TUI.
- Interactive permission UI.
- Permission profiles beyond `ask` and `bypass`.
- Default real network/model calls.
- Browser automation, JavaScript execution, cookies/sessions, PDF/deep HTML
  parsing.
- Remote MCP, OAuth, SSE, or third-party MCP defaults.
- Forked Skills.
- Remote agents, teams, and worktree isolation.
- Automatic memory extraction.
