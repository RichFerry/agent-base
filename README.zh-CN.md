# Agent Base

**语言：** [English](README.md) | 简体中文

Agent Base 是一个 Python Agent Kernel，包含本地 runner、可复现测试，以及显式
opt-in 的真实 smoke 检查。它适合需要一个可阅读、可测试、可扩展的本地 Agent
基础项目，而不是完整产品外壳。

## 它提供什么

- 核心异步 agent loop：用户消息、模型回合、工具调用、工具结果、最终回复。
- 稳定的 `QueryEngine.submit_message(...)` 入口。
- Fake、Anthropic-compatible、OpenAI Chat 与 OpenAI Responses provider。
- 内置 shell、文件、搜索、todo、WebSearch、WebFetch 等工具。
- 权限模式保持在 `ask` 和 `bypass` 两种。
- JSONL transcript、resume 支持，以及 SDK-style events。
- 可选的本地 Skills、MCP fixture/config 集成，以及 session/memory CLI 辅助命令。
- 默认离线的测试套件。

## 它不是什么

- 不是完整 CLI/TUI 产品。
- 不包含交互式权限 UI。
- 不是浏览器自动化系统。
- 不内置默认搜索或抓取服务。
- 不是托管式或远程 Agent 平台。

## 安装

```bash
git clone git@github.com:RichFerry/agent-base.git
cd agent-base
python3 -m pip install -e ".[test]"
```

验证本地 runner：

```bash
agent-kernel-local --help
```

包信息：

- 包名：`agent-kernel`
- 当前版本：`0.7.0`
- Python 版本：`>=3.11`
- 运行时依赖：无

## 快速开始

使用进程内 fake model：

```python
import asyncio

from agent_kernel import FakeModelProvider, KernelConfig, QueryEngine


async def main() -> None:
    engine = QueryEngine(
        model_provider=FakeModelProvider(["Hello from a fake model."]),
        config=KernelConfig(),
    )

    async for event in engine.submit_message("hello", max_turns=1):
        print(event)


asyncio.run(main())
```

运行本地示例 runner：

```bash
agent-kernel-local "Reply with one short sentence: agent kernel smoke test."
```

如果没有模型凭据，真实模型调用会在启动阶段给出清晰错误，而不是静默发起网络请求。

## 模型 Provider 配置

Agent Base 只从环境变量读取凭据。不要把 API key 写入源码、fixture、README 示例、
日志或 transcript。

用 `AGENT_KERNEL_PROVIDER` 选择 provider：

| Provider | 值 | 凭据 fallback |
| --- | --- | --- |
| Anthropic-compatible Messages | `anthropic` | `AGENT_KERNEL_API_KEY`、`ANTHROPIC_AUTH_TOKEN`、`ANTHROPIC_API_KEY` |
| OpenAI Chat Completions | `openai-chat` | `AGENT_KERNEL_API_KEY`、`OPENAI_API_KEY` |
| OpenAI Responses | `openai-responses` | `AGENT_KERNEL_API_KEY`、`OPENAI_API_KEY` |

```bash
export AGENT_KERNEL_PROVIDER="anthropic"
export AGENT_KERNEL_API_KEY="..."
export AGENT_KERNEL_MODEL="..."
```

仍然支持 provider-specific 环境变量。如果使用自定义 Anthropic-compatible endpoint：

```bash
export ANTHROPIC_BASE_URL="https://api.example.com/anthropic"
```

如果使用 OpenAI-compatible 模式：

```bash
export AGENT_KERNEL_PROVIDER="openai-chat"
export OPENAI_API_KEY="..."
export OPENAI_MODEL="..."
```

`AGENT_KERNEL_BASE_URL` 可以覆盖兼容 endpoint 的 provider base URL。

## 本地 Runner

runner 属于 example 层入口。它复用现有 kernel loop，而不是重新实现一套 agent loop。

```text
user input
-> QueryEngine.submit_message()
-> model provider
-> tool_use
-> permission
-> tool execution
-> tool_result
-> assistant final
-> transcript / SDK events
```

常用命令：

```bash
agent-kernel-local --permission-mode ask "Summarize this project in one sentence."
agent-kernel-local --repl
agent-kernel-local --init-config
agent-kernel-local --doctor
agent-kernel-local workspace doctor
agent-kernel-local workspace doctor --json
agent-kernel-local config doctor
agent-kernel-local config effective
agent-kernel-local --list-sessions
agent-kernel-local sessions list
agent-kernel-local --resume SESSION_ID "Continue from here."
agent-kernel-local --continue "Continue the latest local session."
agent-kernel-local --memory-status
agent-kernel-local memory list
agent-kernel-local --memory-read
agent-kernel-local --memory-write notes/preference.md --memory-text "Prefer concise answers."
agent-kernel-local mcp doctor --start --json
agent-kernel-local sessions validate SESSION_ID --json
agent-kernel-local sessions timeline SESSION_ID
agent-kernel-local memory extract SESSION_ID --dry-run
```

权限模式：

- `ask`：默认模式。需要授权的工具调用，如果没有 callback 或 hook 批准，会被拒绝。
- `bypass`：显式本地验证模式。路径和结构性安全检查仍然生效。

### 本地 Config

Agent Base 使用 `settings.json` 作为官方本地 runner 配置文件：

```bash
agent-kernel-local --init-config
agent-kernel-local --doctor
agent-kernel-local --print-effective-config
```

发现顺序是用户 settings、项目 settings、显式 `--agent-config`；高优先级层覆盖低优先级层。
CLI 参数和 `AGENT_KERNEL_*` 环境变量会覆盖 settings。API key 仍然只能从环境变量读取。

可配置的非 secret 默认值包括 provider 类型/model/base URL、permission mode、max turns、
WebSearch/WebFetch opt-in、Skills 目录、MCP fixture/config 路径、session 默认行为、
memory 默认行为，以及 debug flags。

```json
{
  "provider": {"type": "anthropic", "model": "", "baseUrl": ""},
  "runner": {"permissionMode": "ask", "maxTurns": 10},
  "skills": {"dirs": ["examples/skills"], "discoveryMode": "explicit"},
  "mcp": {"configs": ["examples/mcp/stdio-config.json"]}
}
```

不要把 API key、token、password 或其它 secret 写进 `settings.json`。

### v0.7 Workspace Runtime

v0.7.0 把 workspace identity 做成 kernel 和 local runner 共享的统一运行时视图。
现在 agent 可以明确知道：

- 当前 `cwd`
- workspace root，以及它来自 git discovery 还是显式 cwd
- 已加载的 `settings.json` 来源
- Skills 与 MCP 配置来源属于 project / user / explicit 哪一类
- project 级 sessions、transcripts、memory、artifacts 的位置
- act/bypass 模式下允许的工作目录边界

这些信息不需要 credentials、网络或 MCP 启动即可检查：

```bash
agent-kernel-local workspace doctor
agent-kernel-local workspace doctor --json
agent-kernel-local --print-effective-config
```

Workspace 存储位于 config home 下的 project bucket：

```text
<config_home>/projects/<workspace-key>/
  *.jsonl
  memory/
  artifacts/
    bash-output/
    agent-output/
```

如果 `cwd` 位于 git 仓库中，workspace key 基于 git root，因此同一项目的不同子目录会
共享 sessions 和 memory。act/bypass 的文件边界仍然保持保守：默认只有当前 `cwd`
和显式 additional working directories 被视为允许工作路径。

### v0.6 MCP / Sessions / Memory 全链路

v0.6.0 把本地 MCP、JSONL sessions 和显式 memory extraction 串成一条可审计链路：

```text
MCP fixture/config/stdin
-> QueryEngine tool registry
-> MCP tool/resource result
-> SDK events and JSONL transcript
-> sessions validate/inspect/timeline/export
-> memory extract dry-run
-> memory extract --yes 写 memory files 和 MEMORY.md
-> resume 保持 transcript ordering 并加载 memory prompt
```

常用命令：

```bash
agent-kernel-local mcp doctor --json
agent-kernel-local mcp doctor --start --json
agent-kernel-local mcp inspect local-echo --mcp-fixture examples/mcp/echo-mcp.json --json

agent-kernel-local sessions inspect SESSION_ID --json
agent-kernel-local sessions validate SESSION_ID --json
agent-kernel-local sessions timeline SESSION_ID
agent-kernel-local sessions export SESSION_ID --redacted
agent-kernel-local sessions gc --dry-run --older-than 30

agent-kernel-local memory extract SESSION_ID --dry-run --json
agent-kernel-local memory extract SESSION_ID --yes
agent-kernel-local memory validate --json
agent-kernel-local memory rebuild-index --dry-run
agent-kernel-local memory provenance reference/example.md --json
```

Memory extraction 仍然只允许手动触发。dry-run 不修改文件；`--yes` 会写入 memory
文件、更新 `MEMORY.md`、保存 provenance sidecar，并在对应 session transcript 中记录
可见的 extraction event。MCP resource memory 默认保存引用指针，不复制原始 resource dump。

## 可选能力

可选能力默认不加载。

| 能力 | 如何启用 | 默认行为 |
| --- | --- | --- |
| WebSearch stub | `AGENT_KERNEL_WEB_SEARCH_PROVIDER=stub` + `--enable-web-search` | 不联网 |
| WebSearch HTTP JSON adapter | `AGENT_KERNEL_WEB_SEARCH_PROVIDER=http-json` + endpoint 环境变量 | 显式 endpoint |
| WebSearch Anthropic-compatible adapter | `AGENT_KERNEL_WEB_SEARCH_PROVIDER=anthropic-compatible` + endpoint/model/key 环境变量 | 显式 endpoint |
| WebFetch HTTP handler | `AGENT_KERNEL_WEB_FETCH_PROVIDER=http` + `--enable-web-fetch` | 默认关闭 |
| 本地 Skills | `--skills-dir examples/skills` | 默认不加载 |
| MCP fixture | `--mcp-fixture examples/mcp/echo-mcp.json` | 默认不加载 |
| MCP stdio config | `--mcp-config examples/mcp/stdio-config.json` 或 `AGENT_KERNEL_MCP_CONFIG` | 默认不加载 |
| MCP stdio smoke | `AGENT_KERNEL_RUN_REAL_MCP_SMOKE=1 python3 -m pytest tests/test_real_mcp_smoke.py -q` | 默认跳过 |

### WebSearch Stub

```bash
AGENT_KERNEL_WEB_SEARCH_PROVIDER=stub \
agent-kernel-local \
  --enable-web-search \
  --permission-mode bypass \
  "Search for a stub result and summarize it briefly."
```

### WebFetch HTTP Handler

```bash
export AGENT_KERNEL_WEB_FETCH_PROVIDER=http
export AGENT_KERNEL_WEB_FETCH_TIMEOUT="10"
export AGENT_KERNEL_WEB_FETCH_MAX_BYTES="1000000"
export AGENT_KERNEL_WEB_FETCH_MAX_CHARS="100000"

agent-kernel-local \
  --enable-web-fetch \
  --permission-mode bypass \
  "Fetch https://example.com and summarize it briefly."
```

WebFetch 会校验 URL scheme，应用 timeout 和大小限制；不会执行 JavaScript，不管理
cookie/session，也不解析 PDF 或复杂 HTML。

### 本地 Skills

```bash
agent-kernel-local \
  --skills-dir examples/skills \
  --permission-mode bypass \
  "Use the echo skill with hello."
```

示例 skill 位于 `examples/skills/echo/SKILL.md`。

不调用模型即可查看本地 skills：

```bash
agent-kernel-local --skills-dir examples/skills --list-skills
agent-kernel-local skills list --skills-dir examples/skills --json
agent-kernel-local skills validate --skills-dir examples/skills
agent-kernel-local skills info echo --skills-dir examples/skills
```

支持多个 `--skills-dir`。Runner 默认使用显式 discovery；ambient discovery 只有在
settings 中明确开启时才启用。Forked skills 仍然明确返回 not implemented。

### MCP Fixture

```bash
agent-kernel-local \
  --mcp-fixture examples/mcp/echo-mcp.json \
  --permission-mode bypass \
  "Call the local echo MCP tool with hello."
```

该 fixture 是本地 deterministic 示例。它不是完整 MCP 配置格式，也不会启动第三方
MCP server。

### MCP Stdio Config

```bash
agent-kernel-local \
  --mcp-config examples/mcp/stdio-config.json \
  --permission-mode bypass \
  "Call the stdio echo MCP tool with hello."
```

v0.5.0 config loader 支持本地 stdio server，格式为
`mcpServers.{name}.command`、`args`、`env` 和可选 `cwd`。它用标准库在 stdio
上跑 JSON-RPC，不支持 remote MCP、OAuth、SSE，默认也不会启动第三方 server。

v0.5.0 增加了管理诊断：

```bash
agent-kernel-local mcp list --mcp-fixture examples/mcp/echo-mcp.json --json
agent-kernel-local mcp doctor --mcp-config examples/mcp/stdio-config.json
agent-kernel-local mcp validate-config examples/mcp/stdio-config.json
```

### Session 与 Memory

```bash
agent-kernel-local --list-sessions
agent-kernel-local sessions info SESSION_ID --json
agent-kernel-local sessions export SESSION_ID
agent-kernel-local sessions delete SESSION_ID --yes
agent-kernel-local --resume SESSION_ID "Continue this session."
agent-kernel-local --continue "Continue the latest session."
agent-kernel-local --memory-status
agent-kernel-local memory list --json
agent-kernel-local --memory-read
agent-kernel-local --memory-write notes/preference.md --memory-text "Prefer concise answers."
agent-kernel-local memory remember --memory-type feedback --memory-name "Terse Replies" --memory-text "Prefer terse engineering summaries."
agent-kernel-local memory validate
```

Resume 通过 `SessionStore` 读取已有 JSONL transcript 顺序。Memory 写入必须显式触发；
v0.5 不做自动 memory extraction。Memory 路径必须是相对路径，并且只能位于项目
memory 目录内。

### 组合本地 Smoke

```bash
AGENT_KERNEL_WEB_SEARCH_PROVIDER=stub \
agent-kernel-local \
  --enable-web-search \
  --skills-dir examples/skills \
  --mcp-fixture examples/mcp/echo-mcp.json \
  --permission-mode bypass \
  "Search the stub result, use the echo skill, and call the echo MCP tool."
```

## 架构

```text
QueryEngine.submit_message(prompt)
  -> record user message in session state and JSONL transcript
  -> PromptComposer.fetch_system_prompt_parts(...)
  -> query(QueryParams)
     -> context preparation / compaction
     -> ModelProvider.stream(...)
     -> assistant message / tool_use
     -> run_tools(...)
        -> schema validation
        -> input validation
        -> PreToolUse hooks
        -> permission resolver
        -> Tool.call(...)
        -> PostToolUse hooks
     -> user tool_result message
     -> next model turn
     -> terminal event
  -> optional SDK system/init and result wrappers
```

重要模块：

- `agent_kernel/query_engine.py`：session 门面、依赖装配、transcript 写入、SDK event 包装。
- `agent_kernel/query.py`：核心异步 agent loop。
- `agent_kernel/model_provider.py`：fake、Anthropic-compatible、OpenAI Chat 与 OpenAI Responses provider。
- `agent_kernel/web_adapters.py`：example-layer WebSearch 与 WebFetch adapter。
- `agent_kernel/tool_execution.py`：工具生命周期。
- `agent_kernel/permissions.py`：ask/bypass 权限解析。
- `agent_kernel/session.py`：JSONL transcript 与 resume。
- `agent_kernel/memory.py`：项目 memory 路径与 prompt helper。
- `agent_kernel/mcp.py`：MCP tool/resource 包装。
- `agent_kernel/skills.py`：本地 Skill 解析与 Skill tool。

## 内置工具

默认工具由 `agent_kernel.query_engine.default_tools()` 注册：

- `Bash`
- `Glob`
- `Grep`
- `LS`
- `Read`
- `Write`
- `Edit`
- `MultiEdit`
- `NotebookEdit`
- `TodoWrite`
- `WebSearch`
- `WebFetch`

只有在对应配置存在时，才会追加 Agent、Skill 和 MCP 工具。

## Transcript 与 SDK Events

`QueryEngine.submit_message(..., sdk_events=False)` 会产出核心事件流。

设置 `sdk_events=True` 后，事件流会包含 SDK-style 生命周期包装：

- `system/init`
- `result`
- 适用场景下的 error/status 形态事件

Transcript 会以 JSONL 写入配置的 session 目录。工具结果会保留与原始 `tool_use`
的配对关系，resume 会从 transcript 重新加载有序消息链。

## 验证

默认验证是离线的：

```bash
python3 -m pytest -q
python3 -m compileall agent_kernel tests
```

定向检查：

```bash
python3 -m pytest tests/test_local_agent_runner.py -q
python3 -m pytest tests/test_workspace_runtime.py -q
python3 -m pytest tests/test_packaging.py -q
```

真实 smoke 测试是手动且 opt-in 的：

```bash
AGENT_KERNEL_RUN_REAL_SMOKE=1 python3 -m pytest tests/test_real_smoke.py -q
AGENT_KERNEL_RUN_REAL_E2E=1 python3 -m pytest tests/test_real_runner_e2e.py -q
AGENT_KERNEL_RUN_REAL_MCP_SMOKE=1 python3 -m pytest tests/test_real_mcp_smoke.py -q
```

配置和安全细节见 `docs/smoke-v0.3.md`。

## 安全说明

- 不要提交 API key、`.env` 文件、包含 secret 的 transcript，或真实 provider 响应。
- 默认测试不会调用真实模型或真实网络服务。
- 真实模型和 WebSearch/WebFetch 检查必须显式 opt-in。
- runner 默认使用 `ask` 权限模式。
- 示例 WebFetch handler 不是浏览器，也不会执行 JavaScript。

## 明确不在范围内

- 完整产品级 CLI 或 TUI。
- 交互式权限 UI。
- 默认真实联网搜索或抓取。
- 浏览器自动化、JavaScript 执行、cookie/session 管理。
- PDF 或复杂 HTML 解析。
- 完整 MCP 产品客户端，或默认测试启动第三方 MCP server。
- forked Skills。
- remote agents、teams、worktree isolation。
- 超出 ask/bypass kernel 模型的 permission profiles。

## 文档

- `README.md`：英文 README。
- `READING_GUIDE.md`：推荐源码阅读顺序。
- `docs/release-v0.3.md`：v0.3 release 摘要与 readiness 说明。
- `docs/release-v0.4.md`：v0.4 本地 CLI readiness 说明。
- `docs/config-v0.5.md`：v0.5 本地 settings 与 doctor 说明。
- `docs/skills-v0.5.md`：v0.5 Skills 管理说明。
- `docs/session-memory-v0.5.md`：v0.5 session 与 memory 管理说明。
- `docs/mcp-v0.5.md`：v0.5 本地 MCP config hardening 说明。
- `docs/cli-v0.5.md`：v0.5 本地 CLI 日常使用说明。
- `docs/release-v0.5.md`：v0.5 release readiness 说明。
- `docs/release-v0.6.md`：v0.6 MCP / Sessions / Memory 链路说明。
- `docs/mcp-memory-session-v0.6.md`：v0.6 全链路工作流说明。
- `docs/release-v0.7.md`：v0.7 Workspace Runtime 说明。
- `docs/smoke-v0.3.md`：手动真实 smoke 配置。
- `CHANGELOG.md`：发布变更记录。

## 项目状态

当前 release：`v0.7.0`。

本仓库是用于本地实验和扩展的 kernel 与 example runner。它的目标是在加入更大产品外壳之前，
保持行为可观察、可测试。
