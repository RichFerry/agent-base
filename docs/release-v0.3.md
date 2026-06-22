# v0.3 Release Notes and Readiness Audit

This document closes the v0.3 real-smoke and packaging hygiene phase for the Python Agent Kernel in `main/`.

## Release Summary

v0.3 does not add new agent capabilities beyond the v0.2 contract surface. It hardens the local runner and example-layer adapters with opt-in real smoke coverage, packaging metadata, installation smoke, CI baseline, and repository hygiene notes.

### Completed Scope

- Real model smoke remains opt-in and asserts only non-brittle SDK/transcript facts.
- Real runner E2E smoke remains opt-in and registers WebSearch stub, deterministic WebFetch, local Skills, and MCP fixture without requiring every real model run to call every tool.
- WebSearch supports example-layer `stub`, `http-json`, and `anthropic-compatible` adapters.
- WebFetch supports an example-layer opt-in HTTP handler with timeout and size limits, with no preflight gate.
- Local Skills remain explicit `--skills-dir` opt-in.
- MCP has local fixture coverage and opt-in stdio smoke using only the repository echo server.
- Packaging metadata now exposes the `agent-kernel-local` console script.
- CI runs only offline tests and compile checks.

## v0.3 Smoke Matrix

| Smoke | Entry point | Required opt-in | Default status | Network/model behavior | Release status |
| --- | --- | --- | --- | --- | --- |
| Real model smoke | `tests/test_real_smoke.py` | `AGENT_KERNEL_RUN_REAL_SMOKE=1` plus `ANTHROPIC_AUTH_TOKEN` or `ANTHROPIC_API_KEY` | skipped | real model only | implemented / opt-in |
| Real runner E2E | `tests/test_real_runner_e2e.py` | `AGENT_KERNEL_RUN_REAL_E2E=1` plus model credentials | skipped | real model; WebSearch stub; deterministic WebFetch; local Skills/MCP fixture | implemented / opt-in |
| WebSearch `http-json` | `tests/test_real_smoke.py` | `AGENT_KERNEL_RUN_REAL_SMOKE=1`, `AGENT_KERNEL_WEB_SEARCH_PROVIDER=http-json`, `AGENT_KERNEL_WEB_SEARCH_URL` | skipped | real caller-owned search endpoint only when configured | implemented / contract-tested / opt-in |
| WebSearch `anthropic-compatible` | `tests/test_real_smoke.py` | `AGENT_KERNEL_RUN_REAL_SMOKE=1`, provider/url/key/model env vars | skipped | real caller-owned Anthropic-compatible search endpoint only when configured | implemented / contract-tested / real endpoint smoke pending |
| WebFetch `http` | `examples/local_agent.py --enable-web-fetch` | `AGENT_KERNEL_WEB_FETCH_PROVIDER=http` | disabled | real HTTP(S) only in manual runner smoke | implemented / fake-tested / manual real smoke optional |
| MCP stdio | `tests/test_real_mcp_smoke.py` | `AGENT_KERNEL_RUN_REAL_MCP_SMOKE=1` | skipped | local stdio process only, no network, no real model | implemented / opt-in |
| Local Skills | `examples/local_agent.py --skills-dir examples/skills` | explicit `--skills-dir` | not loaded | local files only | implemented / offline-tested / manual real smoke optional |

Default `python3 -m pytest -q` remains fully offline. The opt-in real smokes must not be added to default CI.

## Packaging Baseline

`pyproject.toml` is the packaging source of truth:

- Package name: `agent-kernel`.
- Package version: `0.3.0`.
- Requires Python: `>=3.11`.
- Runtime dependencies: `[]`.
- Build dependencies: `setuptools>=61` and `wheel`.
- Test extra: `pytest>=8.0`.
- Console script: `agent-kernel-local = examples.local_agent:main`.

The console script targets the existing example-layer local runner. It does not add symbols to `agent_kernel.__init__` and does not expand the kernel public API.

Editable install smoke:

```bash
python3 -m pip install -e .
agent-kernel-local --help
```

For local test dependencies:

```bash
python3 -m pip install -e ".[test]"
```

## CI Baseline

The minimal GitHub Actions workflow in `.github/workflows/ci.yml` runs:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python -m compileall agent_kernel tests
```

CI intentionally does not run real model, real WebSearch, real WebFetch, or third-party MCP server checks.

## Repository Hygiene

`.gitignore` now ignores:

- `.DS_Store` files.
- Python bytecode and `.pytest_cache`.
- build artifacts, `dist/`, and `*.egg-info/`.
- local virtualenv and `.env` files.

The existing `.DS_Store` and `agent_kernel/.DS_Store` files were removed from the git index during the final v0.3 cleanup. Local file contents were left on disk and are now covered by `.gitignore`:

```bash
git rm --cached .DS_Store agent_kernel/.DS_Store
```

The generated `agent_kernel.egg-info/` metadata was also removed from the git index during the final v0.3 cleanup. Local file contents were left on disk and are now covered by `.gitignore`:

```bash
git rm -r --cached agent_kernel.egg-info
```

## Verification Baseline

Default release readiness commands:

```bash
python3 -m pytest -q
python3 -m compileall agent_kernel tests
```

Focused packaging smoke:

```bash
python3 -m pytest tests/test_packaging.py -q
```

## Explicitly Unsupported in v0.3

- Default real network search or fetch.
- Default real model calls in tests or CI.
- Browser automation, JavaScript execution, cookies/sessions, PDF or deep HTML parsing.
- Full MCP product client, third-party server startup in default tests, or networked MCP.
- Forked skills.
- Remote agents, teams, and worktree isolation.
- Permission profiles or interactive permission UI.
- Core runtime refactors or public API expansion.
