# v0.5 Session and Memory Notes

v0.5.0 makes existing JSONL transcripts and explicit file-based memory easier to
use from the local runner. It does not add automatic memory extraction.

## Sessions

```bash
agent-kernel-local sessions list
agent-kernel-local sessions list --json
agent-kernel-local sessions info SESSION_ID
agent-kernel-local sessions info SESSION_ID --json
agent-kernel-local sessions transcript-path SESSION_ID
agent-kernel-local sessions export SESSION_ID
agent-kernel-local sessions delete SESSION_ID --yes
```

Resume behavior still uses `SessionStore` and append-only JSONL transcripts:

```bash
agent-kernel-local --resume SESSION_ID "Continue this session."
agent-kernel-local --continue "Continue the latest local session."
```

Session info reports transcript path, message count, last modified time, and
whether orphan `tool_result` rows were detected.

## Memory

```bash
agent-kernel-local memory status
agent-kernel-local memory list
agent-kernel-local memory list --json
agent-kernel-local memory read MEMORY.md
agent-kernel-local memory write notes/preference.md --memory-text "Prefer concise answers."
agent-kernel-local memory append notes/preference.md --memory-text "\nMore detail."
agent-kernel-local memory remember --memory-type feedback --memory-name "Terse Replies" --memory-text "Prefer terse engineering summaries."
agent-kernel-local memory forget feedback/terse-replies.md --yes
agent-kernel-local memory validate
```

Memory paths must be relative and must stay inside the project memory directory.
Symlink escapes are rejected by validation and path resolution.

## Explicitly Not Included

- Automatic memory extraction from conversation content.
- Background consolidation.
- Remote/team memory synchronization.
