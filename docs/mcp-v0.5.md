# v0.5 MCP Local Config Hardening

MCP remains local-only in v0.5.0. The goal is reliable stdio config handling,
clear diagnostics, and clean process shutdown.

## Commands

```bash
agent-kernel-local mcp list --mcp-fixture examples/mcp/echo-mcp.json
agent-kernel-local mcp list --mcp-config examples/mcp/stdio-config.json --json
agent-kernel-local mcp doctor --mcp-config examples/mcp/stdio-config.json
agent-kernel-local mcp validate-config examples/mcp/stdio-config.json
```

Legacy flags still work during a run:

```bash
agent-kernel-local --mcp-config examples/mcp/stdio-config.json --permission-mode bypass "Call echo."
```

## Config Shape

```json
{
  "mcpServers": {
    "stdio-echo": {
      "command": "python3",
      "args": ["examples/mcp/stdio_echo_server.py"],
      "env": {},
      "cwd": ".",
      "startupTimeout": 5,
      "toolTimeout": 5
    }
  }
}
```

Supported fields are `command`, `args`, `env`, `cwd`, `disabled`,
`startupTimeout`, `toolTimeout`, and optional `instructions`.

## Hardening

- Empty or invalid `mcpServers` fails clearly.
- Non-object server entries fail with the server name in the error.
- Unsupported server types fail before startup.
- Disabled servers are ignored.
- Normalized server and tool name collisions fail clearly.
- Failed startup closes already-started stdio processes.
- Timeout/exit errors include bounded stderr diagnostics when available.

## Explicitly Not Included

- Remote MCP.
- OAuth.
- SSE.
- Third-party server defaults.
- Product-level background process management.
