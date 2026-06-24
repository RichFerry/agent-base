# Agent Base v0.7.0 Release Notes

v0.7.0 introduces Workspace Runtime: a shared internal view of where the agent
is running and where project-scoped state belongs.

## Workspace Runtime

The runtime resolves:

- `cwd`
- workspace root
- workspace root source: `git`, `cwd`, or `explicit`
- config home
- project store directory
- sessions/transcript directory
- memory scope and memory directory
- artifact directories
- loaded `settings.json` sources
- Skills source scopes
- MCP config/fixture source scopes
- act/bypass allowed working directories

Inspect the runtime without credentials, network calls, or MCP startup:

```bash
agent-kernel-local workspace doctor
agent-kernel-local workspace doctor --json
agent-kernel-local --print-effective-config
```

## Storage Layout

Project-scoped runtime state is anchored under:

```text
<config_home>/projects/<workspace-key>/
  *.jsonl
  memory/
  artifacts/
    bash-output/
    agent-output/
```

When `cwd` is inside a git repository, `<workspace-key>` is derived from the git
root. Otherwise it is derived from the current working directory.

## Runtime Integration

- `SessionStore.project_dir` uses the workspace project store.
- `MemoryLoader.get_auto_mem_path()` uses the workspace memory directory.
- Bash background output uses the workspace artifact directory.
- Agent output logs use the workspace artifact directory.
- SDK `system/init` includes additive `workspace` metadata.
- The model-facing environment section includes workspace root, transcript,
  memory, artifact, and act-mode path boundary information.

## Boundaries

- No new runtime dependency.
- No real model or network calls in default tests.
- No remote MCP, OAuth, SSE, browser automation, remote agents, teams, worktree
  isolation, or forked skills.
- Permission modes remain `ask` and `bypass`; v0.7 does not add permission
  profiles or an interactive permission UI.
- The act/bypass file boundary remains conservative: allowed working
  directories are the current `cwd` plus explicitly configured additional
  working directories.
