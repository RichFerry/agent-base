# MCP / Sessions / Memory Full Chain

v0.6.0 makes the local Agent Base chain inspectable from MCP execution through
session audit and explicit memory extraction.

## Chain

```text
MCP config/fixture/stdin
-> QueryEngine tool registry
-> model emits MCP tool_use/resource use
-> MCP result enters SDK events + JSONL transcript
-> sessions validate/inspect/export can audit the chain
-> memory extract SESSION_ID --dry-run proposes memory candidates
-> memory extract SESSION_ID --yes writes approved memory files + MEMORY.md
-> later resume keeps ordering and loads the memory prompt
```

## MCP Diagnostics

Static doctor does not start processes:

```bash
agent-kernel-local mcp doctor --json
```

Explicit start doctor starts local stdio servers and shuts them down:

```bash
agent-kernel-local mcp doctor --start --json
```

Inspect one server:

```bash
agent-kernel-local mcp inspect local-echo \
  --mcp-fixture examples/mcp/echo-mcp.json \
  --json
```

MCP result metadata is additive. Existing `tool_result.content` remains stable,
while `mcpMetadata` records server name, normalized names, operation, status,
duration, content size, and truncation state.

## Session Audit

```bash
agent-kernel-local sessions inspect SESSION_ID --json
agent-kernel-local sessions validate SESSION_ID --json
agent-kernel-local sessions timeline SESSION_ID
agent-kernel-local sessions export SESSION_ID --redacted
agent-kernel-local sessions gc --dry-run --older-than 30
```

Validation checks row shape, duplicate UUIDs, parent UUID ordering, tool
use/result pairing, compact boundaries, and MCP metadata shape.

Redacted export removes obvious secret-like tokens and truncates large payloads
while preserving row ordering and metadata.

## Memory Extraction

Dry-run proposes candidates without mutation:

```bash
agent-kernel-local memory extract SESSION_ID --dry-run --json
```

Apply candidates explicitly:

```bash
agent-kernel-local memory extract SESSION_ID --yes
```

Review/edit candidates before applying:

```bash
agent-kernel-local memory extract SESSION_ID --dry-run --json > candidates.json
agent-kernel-local memory extract SESSION_ID --candidate-json candidates.json --yes
```

Maintenance:

```bash
agent-kernel-local memory validate --json
agent-kernel-local memory rebuild-index --dry-run
agent-kernel-local memory rebuild-index --yes
agent-kernel-local memory provenance reference/example.md --json
```

The extractor is deterministic and conservative. It captures explicit
remember/preference user text and MCP references. MCP resource candidates store
source pointers, not raw resource dumps. It skips large, log-like, stack-trace,
or secret-like payloads.

## Boundaries

- No automatic memory writes.
- No real model or network calls in default tests.
- No remote MCP, OAuth, SSE, or third-party server defaults.
- No browser, JavaScript execution, cookie/session handling, or PDF parsing.
- No permission profiles beyond `ask` and `bypass`.
