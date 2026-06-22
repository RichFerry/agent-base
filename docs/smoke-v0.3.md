# v0.3 Manual Real Smoke Tests

This guide describes opt-in real smoke tests for the Python Agent Kernel. These checks are intentionally manual and are not part of the default offline test suite.

## Safety Rules

- Never commit API keys.
- Never paste real API keys into README examples, fixtures, tests, or logs.
- Read credentials only from environment variables.
- Keep `python3 -m pytest -q` offline by default.
- Run real smoke tests only with `AGENT_KERNEL_RUN_REAL_SMOKE=1`.
- Run real MCP stdio smoke only with `AGENT_KERNEL_RUN_REAL_MCP_SMOKE=1`.
- Run real runner E2E smoke only with `AGENT_KERNEL_RUN_REAL_E2E=1`.
- Do not use these smoke tests to evaluate model quality or exact wording.

## Environment Setup

Use environment variables. Replace `...` locally; do not write the value into tracked files.

```bash
cd /Users/ferry/Desktop/agent/main
export ANTHROPIC_AUTH_TOKEN="..."
export ANTHROPIC_MODEL="..."
```

If your provider requires a custom Anthropic-compatible endpoint, set:

```bash
export ANTHROPIC_BASE_URL="https://api.example.com/anthropic"
```

`ANTHROPIC_API_KEY` is also supported when `ANTHROPIC_AUTH_TOKEN` is not set.

If Python reports a local certificate verification error, point OpenSSL at a local CA bundle instead of disabling TLS verification:

```bash
export SSL_CERT_FILE="/etc/ssl/cert.pem"
```

## Automated Opt-In Smoke

Default tests remain fake/offline:

```bash
python3 -m pytest -q
```

To run real smoke tests explicitly:

```bash
AGENT_KERNEL_RUN_REAL_SMOKE=1 python3 -m pytest tests/test_real_smoke.py -q
```

Real model tests are skipped unless both conditions are true:

- `AGENT_KERNEL_RUN_REAL_SMOKE=1`
- `ANTHROPIC_AUTH_TOKEN` or `ANTHROPIC_API_KEY` is present

The real model tests assert only that the kernel does not crash, returns non-empty final text, emits SDK `system/init` and `result` events, and writes a transcript.

## v0.3 Smoke Matrix

| Smoke | Entry point | Required opt-in | Default status | Network/model behavior | Status |
| --- | --- | --- | --- | --- | --- |
| Real model smoke | `tests/test_real_smoke.py` | `AGENT_KERNEL_RUN_REAL_SMOKE=1` plus model credentials | skipped | real model only | implemented / opt-in |
| Real runner E2E | `tests/test_real_runner_e2e.py` | `AGENT_KERNEL_RUN_REAL_E2E=1` plus model credentials | skipped | real model with local WebSearch stub, deterministic WebFetch, local Skills, MCP fixture | implemented / opt-in |
| WebSearch `http-json` | `tests/test_real_smoke.py` | `AGENT_KERNEL_RUN_REAL_SMOKE=1`, `AGENT_KERNEL_WEB_SEARCH_PROVIDER=http-json`, `AGENT_KERNEL_WEB_SEARCH_URL` | skipped | caller-owned search endpoint only when configured | implemented / contract-tested / opt-in |
| WebSearch `anthropic-compatible` | `tests/test_real_smoke.py` | `AGENT_KERNEL_RUN_REAL_SMOKE=1`, provider/url/key/model env vars | skipped | caller-owned Anthropic-compatible search endpoint only when configured | implemented / contract-tested / real endpoint smoke pending |
| WebFetch `http` | `examples/local_agent.py --enable-web-fetch` | `AGENT_KERNEL_WEB_FETCH_PROVIDER=http` | disabled | real HTTP(S) only in manual runner smoke | implemented / fake-tested / manual real smoke optional |
| MCP stdio | `tests/test_real_mcp_smoke.py` | `AGENT_KERNEL_RUN_REAL_MCP_SMOKE=1` | skipped | local stdio process only; no network; no real model | implemented / opt-in |
| Local Skills | `examples/local_agent.py --skills-dir examples/skills` | explicit `--skills-dir` | not loaded | local files only | implemented / offline-tested / manual real smoke optional |

To run the opt-in real WebSearch adapter smoke, configure a caller-owned HTTP JSON endpoint:

```bash
AGENT_KERNEL_RUN_REAL_SMOKE=1 \
AGENT_KERNEL_WEB_SEARCH_PROVIDER=http-json \
AGENT_KERNEL_WEB_SEARCH_URL="..." \
AGENT_KERNEL_WEB_SEARCH_API_KEY="..." \
python3 -m pytest tests/test_real_smoke.py -q
```

This WebSearch smoke is skipped unless all of these are true:

- `AGENT_KERNEL_RUN_REAL_SMOKE=1`
- `AGENT_KERNEL_WEB_SEARCH_PROVIDER=http-json`
- `AGENT_KERNEL_WEB_SEARCH_URL` is present

`AGENT_KERNEL_WEB_SEARCH_API_KEY` is optional because some local or private providers may not require bearer auth. When present, the example adapter sends it as an `Authorization: Bearer ...` header and never prints or stores it.

To run the opt-in Anthropic-compatible WebSearch adapter smoke, configure a caller-owned Anthropic-compatible search endpoint:

```bash
AGENT_KERNEL_RUN_REAL_SMOKE=1 \
AGENT_KERNEL_WEB_SEARCH_PROVIDER=anthropic-compatible \
AGENT_KERNEL_WEB_SEARCH_URL="..." \
AGENT_KERNEL_WEB_SEARCH_API_KEY="..." \
AGENT_KERNEL_WEB_SEARCH_MODEL="..." \
python3 -m pytest tests/test_real_smoke.py -q
```

This verifies the Agent Kernel WebSearch chain:

```text
model emits WebSearch tool_use
-> Python WebSearchTool
-> anthropic-compatible search handler
-> tool_result
-> transcript / SDK events
-> assistant final
```

It does not verify provider-side search through the main model path. The smoke is skipped unless all of these are true:

- `AGENT_KERNEL_RUN_REAL_SMOKE=1`
- `AGENT_KERNEL_WEB_SEARCH_PROVIDER=anthropic-compatible`
- `AGENT_KERNEL_WEB_SEARCH_URL` is present
- `AGENT_KERNEL_WEB_SEARCH_API_KEY` is present
- `AGENT_KERNEL_WEB_SEARCH_MODEL` is present

Current status: `implemented / contract-tested / real endpoint smoke pending`.

WebFetch has an example-layer opt-in HTTP handler. It is disabled by default in the local runner and does not add a preflight/preview/confirm gate.

```bash
export AGENT_KERNEL_WEB_FETCH_PROVIDER=http
export AGENT_KERNEL_WEB_FETCH_TIMEOUT="..."
export AGENT_KERNEL_WEB_FETCH_MAX_BYTES="..."
export AGENT_KERNEL_WEB_FETCH_MAX_CHARS="..."
```

The WebFetch path remains:

```text
model emits WebFetch tool_use
-> Python WebFetchTool
-> injected fetch handler
-> tool_result
-> transcript / SDK events
```

The handler uses Python standard-library HTTP(S) GET, validates URL scheme, applies timeout and size limits, does not execute JavaScript, does not manage cookies/sessions, and does not perform browser/PDF/deep HTML processing. Real WebFetch endpoint smoke is not required for v0.3.6; default tests cover it with fake HTTP responses.

To run the opt-in local stdio MCP smoke:

```bash
AGENT_KERNEL_RUN_REAL_MCP_SMOKE=1 \
python3 -m pytest tests/test_real_mcp_smoke.py -q
```

This smoke starts only `examples/mcp/stdio_echo_server.py`, a deterministic local stdio process. It does not access the network, call a real model, or start a third-party MCP server. The test verifies:

- the stdio server initializes and lists an `echo` tool
- the tool is registered into `QueryEngine` as `mcp__stdio-echo__echo`
- a fake model emits that MCP `tool_use`
- the stdio handler returns `{"echo":"hello","source":"stdio-mcp-smoke"}`
- the result enters SDK events and the JSONL transcript as a normal `tool_result`
- the server process receives shutdown/exit and terminates

To run the opt-in real local runner E2E smoke:

```bash
AGENT_KERNEL_RUN_REAL_E2E=1 \
AGENT_KERNEL_WEB_SEARCH_PROVIDER=stub \
python3 -m pytest tests/test_real_runner_e2e.py -q
```

This smoke is skipped unless both conditions are true:

- `AGENT_KERNEL_RUN_REAL_E2E=1`
- `ANTHROPIC_AUTH_TOKEN` or `ANTHROPIC_API_KEY` is present

It runs the real model through `examples/local_agent.py` helper code while registering only local or deterministic capabilities:

- WebSearch uses the no-network stub handler.
- WebFetch uses a deterministic test handler, not a real HTTP request.
- Skills load from `examples/skills`.
- MCP loads `examples/mcp/echo-mcp.json`, not a real server.

The automated pytest smoke keeps permission mode at `ask`, so unexpected tool calls are denied as normal `tool_result` rows instead of executing privileged tools. It asserts only sanitized contract facts: SDK `system/init` and `result` events exist, enabled tool names are advertised, the final response is non-empty, and the transcript is written. It does not require the model to call every tool and does not persist the full model response into repository files. If the model chooses not to call all available tools, treat that as model behavior limitation; fake-controlled contract tests remain the source of truth for forced WebSearch + WebFetch + Skills + MCP tool paths.

## Manual Smoke Checklist

### Minimal Real Model Smoke

```bash
python3 examples/local_agent.py \
  --permission-mode ask \
  "Reply with one short sentence: agent kernel smoke test."
```

### WebSearch Stub + Real Model Smoke

This uses the local no-network WebSearch stub. It does not call a real search API.

```bash
AGENT_KERNEL_WEB_SEARCH_PROVIDER=stub \
python3 examples/local_agent.py \
  --enable-web-search \
  --permission-mode bypass \
  "Search for a stub result and summarize it briefly."
```

### WebSearch HTTP JSON Adapter Smoke

This uses an explicit example-layer adapter. It is not a default provider and does not live in the kernel public API.

```bash
export AGENT_KERNEL_WEB_SEARCH_PROVIDER=http-json
export AGENT_KERNEL_WEB_SEARCH_URL="..."
export AGENT_KERNEL_WEB_SEARCH_API_KEY="..."
```

The adapter sends a `POST` JSON body to `AGENT_KERNEL_WEB_SEARCH_URL`:

```json
{"query":"...","allowed_domains":["example.com"]}
```

It accepts provider responses shaped as `{"results":[...]}`, `{"items":[...]}`, `{"data":{"results":[...]}}`, or a top-level result list. Result items are mapped to the existing WebSearch `title`, `url`, and optional `snippet` contract.

Manual runner smoke:

```bash
python3 examples/local_agent.py \
  --enable-web-search \
  --permission-mode bypass \
  "Search for agent kernel smoke and summarize it briefly."
```

### WebSearch Anthropic-Compatible Adapter Smoke

This wraps a caller-provided Anthropic-compatible search backend as the existing `web_search_handler` contract. It is still an example-layer adapter: the model must emit a `WebSearch` tool_use, then `WebSearchTool` calls this handler and writes a normal `tool_result`.

```bash
export AGENT_KERNEL_WEB_SEARCH_PROVIDER=anthropic-compatible
export AGENT_KERNEL_WEB_SEARCH_URL="..."
export AGENT_KERNEL_WEB_SEARCH_API_KEY="..."
export AGENT_KERNEL_WEB_SEARCH_MODEL="..."
```

The adapter sends a non-streaming `/v1/messages` request to the configured URL, with a server-side `web_search_20250305` tool schema and the original WebSearch `query`, `allowed_domains`, and `blocked_domains` values. Structured `web_search_tool_result` blocks and text citations are mapped to WebSearch `title` / `url` / `snippet` results.

If the backend returns only natural-language text and no structured search results, the adapter creates one bounded synthetic result:

```json
{"title":"Anthropic-compatible search result","url":"","snippet":"<sanitized answer summary>"}
```

Do not commit real provider responses or API keys.

Manual runner smoke:

```bash
python3 examples/local_agent.py \
  --enable-web-search \
  --permission-mode bypass \
  "Search for agent kernel smoke and summarize it briefly."
```

### WebFetch HTTP Handler Smoke

This uses the example-layer WebFetch HTTP handler. It is opt-in and does not add a preflight gate:

```bash
export AGENT_KERNEL_WEB_FETCH_PROVIDER=http
export AGENT_KERNEL_WEB_FETCH_TIMEOUT="10"
export AGENT_KERNEL_WEB_FETCH_MAX_BYTES="1000000"
export AGENT_KERNEL_WEB_FETCH_MAX_CHARS="100000"
python3 examples/local_agent.py \
  --enable-web-fetch \
  --permission-mode bypass \
  "Fetch https://docs.python.org/3/ and summarize it briefly."
```

The default local runner installs a no-network WebFetch handler that returns a clear configuration error. Only explicit WebFetch provider configuration enables HTTP(S) fetching.

### Skills Local-Only + Real Model Smoke

```bash
python3 examples/local_agent.py \
  --skills-dir examples/skills \
  --permission-mode bypass \
  "Use the echo skill with hello."
```

### MCP Fixture + Real Model Smoke

This uses the local fixture only. It does not start a real MCP server.

```bash
python3 examples/local_agent.py \
  --mcp-fixture examples/mcp/echo-mcp.json \
  --permission-mode bypass \
  "Call the local echo MCP tool with hello."
```

### MCP Stdio Echo Smoke

This uses a local-only stdio MCP smoke server:

```text
examples/mcp/stdio_echo_server.py
examples/mcp/stdio-mcp.json
```

It exposes one deterministic tool:

```text
input: {"text":"hello"}
output: {"echo":"hello","source":"stdio-mcp-smoke"}
```

Run it only as an opt-in pytest smoke:

```bash
AGENT_KERNEL_RUN_REAL_MCP_SMOKE=1 \
python3 -m pytest tests/test_real_mcp_smoke.py -q
```

The smoke uses a fake model to force the MCP `tool_use`; it validates the real stdio process and the kernel MCP `tool_result` path without a real model call.

### Real Runner E2E Smoke

This smoke combines the real model, local runner, WebSearch stub, deterministic WebFetch handler, local Skills, MCP fixture, SDK events, and transcript writing. It is intentionally opt-in and does not depend on real search or real fetch endpoints.

```bash
AGENT_KERNEL_WEB_SEARCH_PROVIDER=stub \
AGENT_KERNEL_WEB_FETCH_PROVIDER=http \
AGENT_KERNEL_RUN_REAL_E2E=1 \
python3 examples/local_agent.py \
  --enable-web-search \
  --skills-dir examples/skills \
  --mcp-fixture examples/mcp/echo-mcp.json \
  --permission-mode bypass \
  "Use available tools briefly: search the stub result, use the echo skill with hello, and call the local echo MCP tool."
```

If the real model does not call every available tool, do not treat that as a kernel failure. Use the fake-controlled tests for strict tool ordering and tool_result assertions.

## Offline Tests

The normal tests continue to use fake providers, fake handlers, temporary fixtures, and local files:

- `tests/test_local_agent_runner.py`
- `tests/test_local_agent_combined.py`
- `tests/test_capability_contracts.py`
- `tests/test_tool_registry_contracts.py`
- `tests/test_transcript_contracts.py`
- `tests/test_mcp.py`
- `tests/test_skills.py`
- `tests/test_sdk_transcript.py`
- `tests/test_real_smoke.py`, `tests/test_real_mcp_smoke.py`, and `tests/test_real_runner_e2e.py` are collected but skipped unless explicitly enabled.

These tests must not require network access, real model calls, or real MCP servers.

## Secret Handling Checklist

- Use `export ANTHROPIC_AUTH_TOKEN="..."` or `export ANTHROPIC_API_KEY="..."` locally.
- Use `export AGENT_KERNEL_WEB_SEARCH_API_KEY="..."` locally when your WebSearch endpoint requires it.
- Do not place secrets in `README.md`, `docs/`, `tests/`, `examples/`, fixtures, transcripts, or shell snippets committed to git.
- Do not run real smoke tests with verbose logging that prints environment variables.
- Rotate any key that was pasted into chat, logs, or files.

## Troubleshooting

- `CERTIFICATE_VERIFY_FAILED`: set `SSL_CERT_FILE` to a valid local CA bundle, such as `/etc/ssl/cert.pem` on macOS.
- `401` or authentication errors: verify `ANTHROPIC_AUTH_TOKEN` or `ANTHROPIC_API_KEY` is present in the shell running the smoke test.
- `404` or model errors: verify `ANTHROPIC_BASE_URL` and `ANTHROPIC_MODEL` match the provider.
- WebSearch `401`: verify `AGENT_KERNEL_WEB_SEARCH_API_KEY` and the provider's expected auth header. The Anthropic-compatible adapter sends `Authorization: Bearer ...`.
- WebSearch `404`: verify `AGENT_KERNEL_WEB_SEARCH_URL`, including the base path expected by your provider.
- WebSearch TLS errors: set `SSL_CERT_FILE` to a valid CA bundle instead of disabling certificate verification.
- WebSearch timeout: verify the provider is reachable and consider a small `AGENT_KERNEL_WEB_SEARCH_TIMEOUT` adjustment.
- WebSearch invalid JSON: for `http-json`, verify the endpoint returns JSON matching `results`, `items`, `data.results`, or a top-level result list. For `anthropic-compatible`, verify the endpoint returns a Messages API JSON object with `content`.
- WebFetch invalid URL: verify the URL is absolute and uses `http` or `https`.
- WebFetch timeout: verify the endpoint is reachable and adjust `AGENT_KERNEL_WEB_FETCH_TIMEOUT`.
- WebFetch too large: lower the requested content size or raise `AGENT_KERNEL_WEB_FETCH_MAX_BYTES` / `AGENT_KERNEL_WEB_FETCH_MAX_CHARS` locally.
- Empty final response: keep the prompt simple and inspect only sanitized SDK `result` metadata, not secrets or full request headers.
