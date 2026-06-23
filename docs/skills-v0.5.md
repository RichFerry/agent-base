# v0.5 Skills Notes

Skills remain inline instruction packages, not forked agents. The runner now
makes local skill management practical without changing `SkillTool` semantics.

## Commands

```bash
agent-kernel-local skills list --skills-dir examples/skills
agent-kernel-local skills list --skills-dir examples/skills --json
agent-kernel-local skills validate --skills-dir examples/skills
agent-kernel-local skills validate --skills-dir examples/skills --strict-skills
agent-kernel-local skills info echo --skills-dir examples/skills
```

The legacy flag form still works:

```bash
agent-kernel-local --skills-dir examples/skills --list-skills
```

## Discovery

- Runner discovery defaults to explicit skill dirs.
- Multiple `--skills-dir` values are supported.
- `settings.json` can provide `skills.dirs`.
- Ambient discovery is available only when settings explicitly choose it.

## Validation

Validation checks duplicate names, duplicate real paths, unsupported forked
context, and unsupported extra frontmatter keys. Non-strict validation reports
warnings. Strict validation fails on warnings.

Forked skills are still not implemented. If invoked, the Skill tool returns a
stable not-implemented result instead of starting a separate agent.
