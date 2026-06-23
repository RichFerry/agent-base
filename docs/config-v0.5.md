# v0.5 Local Settings and Doctor

v0.5.0 makes `settings.json` the official local runner settings file. Settings
store non-secret defaults only; API keys and tokens must remain in environment
variables.

## Discovery and Precedence

The runner loads settings in this order:

1. User settings: `<config_home>/settings.json`
2. Project settings: nearest `settings.json` from `cwd` up to the git root
3. Explicit settings: `--agent-config PATH`

Higher layers override lower layers. CLI flags and `AGENT_KERNEL_*` environment
variables override settings.

Unsupported top-level sections are ignored so existing local settings files from
other tools do not break the runner. Supported sections are validated and reject
secret-like fields such as API keys, tokens, passwords, and auth secrets.

## Create and Inspect

```bash
agent-kernel-local --init-config
agent-kernel-local --doctor
agent-kernel-local --doctor-json
agent-kernel-local --print-effective-config
agent-kernel-local --validate-config
```

Subcommand aliases are also supported:

```bash
agent-kernel-local config doctor
agent-kernel-local config effective
agent-kernel-local config validate
```

## Shape

```json
{
  "provider": {
    "type": "anthropic",
    "model": "",
    "baseUrl": "",
    "timeout": 60,
    "maxTokens": null
  },
  "runner": {
    "permissionMode": "ask",
    "maxTurns": 10,
    "quiet": false,
    "jsonEvents": false,
    "printTranscriptPath": false
  },
  "webSearch": {
    "enabled": false,
    "provider": "stub",
    "stubResults": "",
    "timeout": 10
  },
  "webFetch": {
    "enabled": false,
    "provider": "http",
    "timeout": 10,
    "maxBytes": 1000000,
    "maxChars": 100000
  },
  "skills": {
    "dirs": [],
    "discoveryMode": "explicit",
    "strictValidation": false
  },
  "mcp": {
    "fixtures": [],
    "configs": [],
    "startupTimeout": 5,
    "toolTimeout": 5
  },
  "session": {
    "defaultMode": "new"
  },
  "memory": {
    "enabled": true,
    "defaultPath": "MEMORY.md"
  },
  "debug": {
    "config": false,
    "tools": false,
    "provider": false,
    "redact": true
  }
}
```

Relative paths are resolved from the file that declared them.

## Explicitly Not Included

- Storing API keys or tokens in settings.
- Interactive configuration UI.
- Remote provider validation during doctor.
- Real model or network calls during doctor/config commands.
- Permission profiles beyond `ask` and `bypass`.
