# v0.5 Local CLI Notes

`agent-kernel-local` remains an example-layer runner. v0.5.0 adds daily-use
management commands without introducing a TUI or changing the kernel loop.

## Direct Run

```bash
agent-kernel-local "Reply with one short sentence."
agent-kernel-local --repl
agent-kernel-local --permission-mode ask "Use safe defaults."
agent-kernel-local --permission-mode bypass "Run a local smoke."
```

## Diagnostics

```bash
agent-kernel-local --doctor
agent-kernel-local --doctor-json
agent-kernel-local --print-effective-config
agent-kernel-local --debug-provider
agent-kernel-local --debug-tools
agent-kernel-local --json-events "Reply briefly."
agent-kernel-local --print-transcript-path "Reply briefly."
```

`--doctor`, config commands, and management commands do not call a model or real
network service. `--debug-tools` uses a no-op internal provider.

## Management Aliases

```bash
agent-kernel-local config doctor
agent-kernel-local skills list
agent-kernel-local sessions list
agent-kernel-local memory list
agent-kernel-local mcp doctor
```

The older flag-based commands remain supported for compatibility with existing
tests and scripts.

## Boundaries

- No interactive permission UI.
- No permission profiles beyond `ask` and `bypass`.
- No TUI.
- No default real network or model calls.
