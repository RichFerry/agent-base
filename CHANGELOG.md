# Changelog

## 0.3.0

- Added opt-in real smoke documentation and tests for real model, real runner E2E, WebSearch adapters, WebFetch handler, and MCP stdio smoke.
- Added `agent-kernel-local` console script for the example-layer local runner.
- Added packaging smoke tests for `pyproject.toml`, runner help, and editable install console script startup.
- Added minimal offline CI baseline.
- Added v0.3 release readiness documentation and repository hygiene guidance.
- Removed tracked generated `.DS_Store` and `agent_kernel.egg-info/` files from the git index for release hygiene.

## 0.2.0

- Added minimal local runner around `QueryEngine.submit_message()`.
- Added example-layer WebSearch provider injection, local Skills loading, and MCP fixture loading.
- Added combined capability, tool registry collision, and SDK/transcript contract tests.
- Documented v0.2 unsupported product scope.
