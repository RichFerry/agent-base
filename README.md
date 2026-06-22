# Claude Code-style Agent Kernel Python Port

这是一个放在 `main/agent_kernel` 下的 Python Agent Kernel。它的目标不是复刻完整 Claude Code 产品外壳，而是把 Claude Code 核心 agent base 的运行语义翻译成可读、可测、可继续扩展的 Python 版本。

当前内核覆盖了 agent loop、message flow、tool protocol、ask/bypass permission resolver、prompt composer、session transcript、memory loading、context compaction、hooks、MCP、skills、subagent/task agent、SDK/transcript 外围事件，以及 Anthropic-compatible 真实模型 API 接入。

更细的逐模块阅读路线见 [READING_GUIDE.md](./READING_GUIDE.md)。

## v0.2 Release Summary

v0.2 adds a minimal local runner and freezes the capability contracts needed to validate the Python kernel without building a product shell. The release scope includes:

- `examples/local_agent.py` for local one-shot or REPL-style smoke runs.
- WebSearch provider injection with an example-layer no-network stub.
- Explicit `--skills-dir` loading for local inline skills.
- Explicit `--mcp-fixture` loading for local MCP smoke fixtures.
- Combined WebSearch + Skills + MCP flow through the normal `QueryEngine` loop.
- Tool registry collision policy for built-ins, Agent/Task, WebSearch, Skill, MCP tools, and MCP resource helpers.
- SDK event and JSONL transcript contracts for tool_use/tool_result pairing, resume, compact boundary, and error paths.
- Permission runner mode limited to `ask` and `bypass`.

Full release notes and readiness audit are in [docs/release-v0.2.md](./docs/release-v0.2.md).

## Real Smoke Tests

Real provider, MCP stdio, and local runner E2E smoke tests are manual and opt-in. Default `pytest` remains fake/offline. See [docs/smoke-v0.3.md](./docs/smoke-v0.3.md) for environment setup, local runner smoke commands, opt-in WebSearch adapter smokes, local stdio MCP smoke, real runner E2E smoke, and secret-handling rules.

## v0.3 Release Hygiene

v0.3 packaging and readiness notes are in [docs/release-v0.3.md](./docs/release-v0.3.md). The package exposes an example-layer console script after editable install:

```bash
python3 -m pip install -e .
agent-kernel-local --help
```

This console script points to `examples.local_agent:main`; it does not expand the `agent_kernel` public API.

## 当前定位

- 这是一个独立 Python 包，包名是 `agent-kernel`，源码目录是 `agent_kernel/`。
- v0.1 kernel release 的权威范围是这个 `main/` 包目录：`agent_kernel/`、`tests/`、`pyproject.toml`、`README.md` 和 `READING_GUIDE.md`。
- 工作区顶层若存在其他 `src/` 目录，它不属于本次 v0.1 kernel release scope，除非后续显式迁入这个 Python 包。
- 运行入口是 `QueryEngine.submit_message(...)`，底层核心 loop 是 `query(...)` async generator。
- 默认没有真实 UI；权限中的 `ask` 通过可注入 callback 或 hook 决策，没有决策器时安全降级为拒绝。
- 模型边界是 `ModelProvider` 协议，默认支持 `FakeModelProvider` 和 `AnthropicModelProvider`。
- 真实 API 使用 Anthropic-compatible `/v1/messages` 协议，支持 SSE streaming normalizer、tool use、abort/cancel。
- 内核只依赖 Python 标准库；测试依赖是 `pytest`。

## 架构总览

```text
QueryEngine.submit_message(prompt)
  -> 记录 user message 到 mutable_messages 与 JSONL transcript
  -> PromptComposer.fetch_system_prompt_parts(...)
  -> query(QueryParams)
     -> 上下文整理 / auto compact / partial compact / microcompact
     -> ModelProvider.stream(...)
     -> assistant message / tool_use
     -> run_tools(...)
        -> schema validate
        -> tool validate_input
        -> PreToolUse hooks
        -> permission resolver
        -> Tool.call(...)
        -> PostToolUse hooks
     -> user tool_result message 回灌
     -> 下一轮模型调用
     -> terminal event
  -> 可选 SDK system/init、result、error 包装
```

核心原则是：`query(...)` 只负责 agent loop 语义；`QueryEngine` 负责 session 级依赖装配、transcript 和 SDK 外围事件。

## 快速开始

进入包目录：

```bash
cd /Users/ferry/Desktop/agent/main
```

安装测试依赖：

```bash
python3 -m pip install -e '.[test]'
```

用 fake provider 跑一轮最小对话：

```python
import asyncio

from agent_kernel import FakeModelProvider, KernelConfig, QueryEngine


async def main() -> None:
    engine = QueryEngine(
        model_provider=FakeModelProvider(["你好，我是一个 fake model response。"]),
        config=KernelConfig(),
    )

    async for event in engine.submit_message("hello", max_turns=1):
        print(event)


asyncio.run(main())
```

## 本地 Runner 示例

`examples/local_agent.py` 是 v0.2 的最小本地运行入口，用来验证一条用户输入如何经过 `QueryEngine.submit_message()`、模型、工具、权限、tool_result、assistant final 和 transcript/SDK events。它不是完整 CLI，也不是 TUI。

真实模型调用仍通过 Anthropic-compatible 环境变量配置：

```bash
cd /Users/ferry/Desktop/agent/main
export ANTHROPIC_AUTH_TOKEN="你的 token"
export ANTHROPIC_MODEL="claude-opus-4-6"
python3 examples/local_agent.py "用一句话介绍这个 kernel"
```

如果没有 `ANTHROPIC_AUTH_TOKEN` 或 `ANTHROPIC_API_KEY`，runner 会打印清晰错误并退出，不会进入网络调用。默认权限模式保持安全的 `ask`；这个示例 runner 不实现交互式授权 UI，因此需要授权的写操作会被拒绝并作为正常 `tool_result` 进入 transcript。

运行时，关键事件日志写到 stderr，assistant final response 写到 stdout。REPL 模式可复用同一个 `QueryEngine` session：

```bash
python3 examples/local_agent.py --repl
```

### v0.2 Capability Matrix

| Capability | Runner flag / config | Provider shape | Default behavior | Test status | Explicitly unsupported scope |
| --- | --- | --- | --- | --- | --- |
| WebSearch / WebFetch provider injection | `--enable-web-search` plus `AGENT_KERNEL_WEB_SEARCH_PROVIDER=stub`, `http-json`, or `anthropic-compatible`; `--enable-web-fetch` plus `AGENT_KERNEL_WEB_FETCH_PROVIDER=http`; provider-specific env vars; or injected handlers in `build_local_engine()` | WebSearch receives `{"query": "..."}` and returns normalized result data. WebFetch receives a URL string and returns existing WebFetch response fields | `WebSearch` and `WebFetch` tools exist in the default registry, but runner network providers are not enabled by default. WebFetch defaults to a clear no-network configuration error in the local runner | `tests/test_local_agent_runner.py`, `tests/test_local_agent_combined.py`, `tests/test_capability_contracts.py`, `tests/test_real_smoke.py`; `anthropic-compatible` is implemented / contract-tested / real endpoint smoke pending | Default network search/fetch, commercial search API defaults, browser/JS/cookie/session/PDF/deep HTML processing, WebFetch preflight gate |
| Local Skills | `--skills-dir PATH` or `skills_dir=` in `build_local_engine()` | Directory of child folders containing `SKILL.md`; parsed into `SkillDefinition` and exposed through the existing `Skill` tool | No extra local skills are loaded unless `--skills-dir` is provided. Inline skills inject prompt context into the next model turn | `tests/test_skills.py`, `tests/test_local_agent_runner.py`, `tests/test_local_agent_combined.py`, `tests/test_capability_contracts.py` | Forked skill execution, skill marketplace/install UX, remote skill execution |
| MCP local fixture | `--mcp-fixture PATH` or `mcp_fixture=` in `build_local_engine()` | Local JSON fixture converted to `MCPClientConfig` with connected tools/resources plus local call/read handlers | No MCP fixture is loaded by default. Invalid fixture paths or shapes fail before model setup with clear errors | `tests/test_mcp.py`, `tests/test_local_agent_runner.py`, `tests/test_local_agent_combined.py`, `tests/test_capability_contracts.py`; opt-in local stdio smoke in `tests/test_real_mcp_smoke.py` | Third-party MCP server startup, full MCP config standard, networked MCP |
| Permission mode | `--permission-mode ask|bypass` or `permission_mode=` in `build_local_engine()` | Existing `ToolPermissionContext.mode`; no new runner permission profile | Defaults to `ask`. With no interactive permission UI, ask-only calls that need approval are safely denied as tool_result errors. `bypass` is explicit pass-through | `tests/test_local_agent_runner.py`, `tests/test_mcp.py`, `tests/test_skills.py`, `tests/test_capability_contracts.py` | Permission profiles, `plan` / `acceptEdits`, interactive permission UI |
| Runner opt-in, SDK events, transcript | Explicit runner flags and `run_local_agent_once(..., sdk_events=True)` | Existing `QueryEngine.submit_message()` event stream, SDK `system/init` and `result`, JSONL transcript rows | Optional capabilities are absent unless their flags/config are supplied. `tool_use` / `tool_result` ordering stays on the normal query loop | `tests/test_local_agent_combined.py`, `tests/test_capability_contracts.py`, `tests/test_sdk_transcript.py` | Full product CLI/TUI, remote agents, teams, worktree isolation |

### Tool Registry Collision Policy

The v0.2 runner treats the tool registry as a stable contract. Built-in tools are registered first, MCP tools are merged by name without replacing existing tools, and the `Skill` tool is appended only when local skills are explicitly loaded. Normal MCP tools keep the `mcp__<server>__<tool>` prefix, so MCP resource helper tools such as `ListMcpResourcesTool` and `ReadMcpResourceTool` remain separate from server tools.

If two optional capabilities produce the same tool name, the first registered tool wins and later duplicates are skipped. This is a stability policy for v0.2, not a product UX. Tests in `tests/test_tool_registry_contracts.py` lock the current behavior for built-in/WebSearch/Skill/MCP collisions, failed MCP clients, SDK init ordering, and transcript ordering.

### WebSearch 注入示例

`WebSearch` 不内置搜索后端。local runner 可以从 example 层注入一个 provider handler；默认推荐用无网络的 `stub` adapter 验证工具链路和本地测试。v0.3 额外提供 opt-in `http-json` 和 `anthropic-compatible` adapter smoke，用于调用调用方自己配置的搜索端点；它们不是内核 public API，也不是默认搜索服务。

```bash
cd /Users/ferry/Desktop/agent/main
export ANTHROPIC_AUTH_TOKEN="你的 token"
export AGENT_KERNEL_WEB_SEARCH_PROVIDER=stub
python3 examples/local_agent.py --enable-web-search --permission-mode bypass "search latest Python release"
```

也可以让 stub adapter 从本地 JSON 文件读取结果：

```bash
python3 examples/local_agent.py \
  --web-search-stub-results ./stub-search-results.json \
  --permission-mode bypass \
  "search Python release notes"
```

真实 WebSearch adapter 需要显式环境变量，示例和故障排查见 [docs/smoke-v0.3.md](./docs/smoke-v0.3.md)：

```bash
export AGENT_KERNEL_WEB_SEARCH_PROVIDER=http-json
export AGENT_KERNEL_WEB_SEARCH_URL="..."
export AGENT_KERNEL_WEB_SEARCH_API_KEY="..."
python3 examples/local_agent.py --enable-web-search --permission-mode bypass "search agent kernel smoke"
```

Anthropic-compatible search backend 也必须显式配置，并且仍然走 `WebSearchTool -> web_search_handler -> tool_result` 链路：

```bash
export AGENT_KERNEL_WEB_SEARCH_PROVIDER=anthropic-compatible
export AGENT_KERNEL_WEB_SEARCH_URL="..."
export AGENT_KERNEL_WEB_SEARCH_API_KEY="..."
export AGENT_KERNEL_WEB_SEARCH_MODEL="..."
python3 examples/local_agent.py --enable-web-search --permission-mode bypass "search agent kernel smoke"
```

默认权限模式仍是 `ask`。因为这个 runner 不实现交互式授权 UI，模型调用 `WebSearch` 时如果保持 `ask` 会被安全拒绝；需要本地验证 WebSearch 工具执行链路时，显式传 `--permission-mode bypass`。如果未配置 provider，WebSearch 会返回清晰的 unavailable 错误，而不是尝试真实联网。

### WebFetch 注入示例

local runner 默认不会真实 fetch。需要显式启用 example-layer HTTP handler：

```bash
export AGENT_KERNEL_WEB_FETCH_PROVIDER=http
export AGENT_KERNEL_WEB_FETCH_TIMEOUT="10"
export AGENT_KERNEL_WEB_FETCH_MAX_BYTES="1000000"
export AGENT_KERNEL_WEB_FETCH_MAX_CHARS="100000"
python3 examples/local_agent.py \
  --enable-web-fetch \
  --permission-mode bypass \
  "fetch https://docs.python.org/3/ and summarize it briefly"
```

这条链路不实现 preflight/preview/confirm 关卡：`WebFetch tool_use -> injected fetch handler -> tool_result -> transcript / SDK events`。handler 只做基础 HTTP(S) GET、URL scheme 校验、timeout 和大小限制；不执行 JS、不做 browser、cookie/session、PDF 或复杂 HTML 解析。

### Local Skills 示例

local runner 默认不要求传入额外 skills 目录。需要验证本地 skill tool 链路时，可以显式传 `--skills-dir`：

```bash
cd /Users/ferry/Desktop/agent/main
export ANTHROPIC_AUTH_TOKEN="你的 token"
python3 examples/local_agent.py \
  --skills-dir examples/skills \
  "use the echo skill to repeat hello"
```

`--skills-dir` 指向一个目录，目录下每个子目录可以包含一个 `SKILL.md`。仓库内置的最小示例是：

```text
examples/skills/echo/SKILL.md
```

如果 skills 目录不存在，或目录中没有任何可解析的 `*/SKILL.md`，runner 会在启动阶段给出清晰错误。inline skill 调用会通过现有 `Skill` tool 把 skill prompt 注入下一轮模型调用；`context: fork` 的 forked skill 仍会明确返回 not implemented，不会在 v0.2.2 中实现。

### MCP Fixture 示例

local runner 也可以显式加载一个本地 MCP smoke fixture。这个 fixture 只用于证明 MCP tool/resource wrapper 能进入现有 `QueryEngine` 工具链路，不是完整 MCP server 配置格式，也不会启动外部进程或联网。

```bash
cd /Users/ferry/Desktop/agent/main
export ANTHROPIC_AUTH_TOKEN="你的 token"
python3 examples/local_agent.py \
  --mcp-fixture examples/mcp/echo-mcp.json \
  --permission-mode bypass \
  "call the echo MCP tool with hello"
```

示例 fixture 位于：

```text
examples/mcp/echo-mcp.json
```

它注册一个 deterministic `mcp__local-echo__echo` 工具和一个只读 resource。默认权限模式仍是 `ask`；因为 runner 不实现交互式授权 UI，验证 MCP 工具执行链路时需要显式传 `--permission-mode bypass`。如果 fixture 路径不存在或 JSON 无效，runner 会在启动阶段给出清晰错误。

v0.3.5 还提供一个显式 opt-in 的本地 stdio MCP smoke。它只启动仓库内 `examples/mcp/stdio_echo_server.py`，用 fake model 触发 `mcp__stdio-echo__echo`，验证真实 stdio 进程的 `tool_result` 进入 SDK events 和 transcript：

```bash
AGENT_KERNEL_RUN_REAL_MCP_SMOKE=1 \
python3 -m pytest tests/test_real_mcp_smoke.py -q
```

### Combined Capability Smoke 示例

WebSearch、local Skills 和 MCP fixture 可以在同一个 local runner / `QueryEngine` 中显式共存。这个示例仍然只使用 example 层 adapter：WebSearch 使用 no-network stub，本地 skill 来自 `examples/skills`，MCP 来自本地 fixture，不启动真实 MCP server。

```bash
cd /Users/ferry/Desktop/agent/main
export ANTHROPIC_AUTH_TOKEN="你的 token"
export AGENT_KERNEL_WEB_SEARCH_PROVIDER=stub
python3 examples/local_agent.py \
  --enable-web-search \
  --skills-dir examples/skills \
  --mcp-fixture examples/mcp/echo-mcp.json \
  --permission-mode bypass \
  "search something, use echo skill, then call echo MCP"
```

如果去掉 `AGENT_KERNEL_WEB_SEARCH_PROVIDER=stub` 或 `--web-search-provider stub`，`--enable-web-search` 会在启动阶段给出 WebSearch 未配置的错误；runner 不会默认联网。去掉 `--skills-dir` 时不会加载额外 local skills；去掉 `--mcp-fixture` 时不会注册 local MCP fixture 工具。

## 接入真实 API

真实模型使用环境变量配置。不要把 token 写进代码或提交到仓库。

```bash
export ANTHROPIC_BASE_URL="https://api.deepseek.com/anthropic"
export ANTHROPIC_AUTH_TOKEN="你的 token"
export ANTHROPIC_MODEL="deepseek-chat"
```

配置好以后，如果没有显式传入 `model_provider`，`QueryEngine` 会在检测到 `ANTHROPIC_AUTH_TOKEN` 或 `ANTHROPIC_API_KEY` 时自动使用 `AnthropicModelProvider.from_env()`。

```python
import asyncio

from agent_kernel import KernelConfig, QueryEngine


async def main() -> None:
    engine = QueryEngine(config=KernelConfig())

    async for event in engine.submit_message("用一句话介绍这个 kernel", max_turns=2):
        print(event)


asyncio.run(main())
```

如果要显式构造 provider：

```python
import asyncio

from agent_kernel import AnthropicModelProvider, KernelConfig, QueryEngine


async def main() -> None:
    engine = QueryEngine(
        model_provider=AnthropicModelProvider.from_env(),
        config=KernelConfig(),
    )

    async for event in engine.submit_message("ping", max_turns=1):
        print(event)


asyncio.run(main())
```

## 权限模式

内核层面保留两个主要模式：

- `ask`：只读工具一般放行，写文件、危险 Bash 等操作进入 ask；没有 callback/hook 决策时拒绝。
- `bypass`：放行普通写操作和命令操作，但敏感路径、越界路径、危险结构约束仍然不可绕过。

为了兼容源码命名，`default`、`acceptEdits`、`bypassPermissions`、`plan`、`dontAsk` 等别名仍可解析，但内核公开心智模型保持为 ask/bypass。

交互式授权通过 `permission_callback` 注入：

```python
from agent_kernel import PermissionDecision, QueryEngine


def approve(tool, input, context, decision):
    return PermissionDecision.allow()


engine = QueryEngine(permission_callback=approve)
```

## 内置工具

默认工具由 `agent_kernel.query_engine.default_tools()` 注册，顺序稳定，也会进入 SDK `system/init` 的工具列表。

当前默认工具包括：

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

如果配置了 subagent、skill 或 MCP client，`QueryEngine` 会在初始化时按需追加 `Task`、`Skill` 和 MCP 动态工具。

## Prompt、Memory 与 Session

`PromptComposer` 负责拼接系统提示词、工具提示、memory、环境信息、MCP/skill/agent 等动态 section。提示词常量放在源码模块里，不在 README 中重复全文：

- `agent_kernel/prompt_composer.py`
- `agent_kernel/tools/prompts.py`

Memory 默认定位到 Claude Code-style 配置目录：

```text
<config_home>/projects/<sanitized-git-root-or-cwd>/memory/MEMORY.md
```

v0.1 的默认 `PromptComposer` 会把 memory 系统的路径、写入规则和使用规则放入系统提示，但不承诺每次默认内联 `MEMORY.md` 的当前内容。`MemoryLoader.build_memory_prompt_with_content()` 保留了带入口文件内容的包装能力，供显式 memory 流程使用。

Transcript 默认写入：

```text
<config_home>/projects/<sanitized-cwd>/<sessionId>.jsonl
```

`config_home` 默认是 `~/.claude`，也可以通过 `CLAUDE_CONFIG_DIR` 或 `KernelConfig(config_home=...)` 覆盖。

## SDK 事件面

默认 `submit_message(..., sdk_events=False)` 保持核心事件流，不注入 SDK 外围消息。

打开 `sdk_events=True` 后，会额外产生：

- `system/init`
- `result`
- `error`
- `status` 可通过 `QueryEngine.get_sdk_status_message()` 构造

这层只包在 `QueryEngine` 外围，不进入 `query(...)` 核心 loop。

## Transcript / SDK Contract

v0.2 freezes the observable event and transcript contract for the local runner and `QueryEngine.submit_message(..., sdk_events=True)`:

- SDK event streams start with `system/init` and end with `result` for successful or failed runs.
- Persistent JSONL rows keep the internal message order: user prompt, assistant `tool_use`, paired user `tool_result`, optional synthetic skill prompt, assistant final.
- `tool_result` rows keep `sourceToolAssistantUUID` and a `parentUuid` pointing at the assistant message that emitted the matching `tool_use`.
- Resume loads the same ordered message chain from JSONL; compact resume starts at the latest compact boundary.
- Compact boundaries are persisted as `system` rows with `subtype: compact_boundary` and stable `compactMetadata`.
- Model errors are persisted through a `system/api_error` row and an SDK error `result`; recoverable tool errors stay as normal error `tool_result` rows.

The contract is covered by `tests/test_sdk_transcript.py` and `tests/test_transcript_contracts.py`, plus combined capability tests for WebSearch, Skills, and MCP.

## Context 管理

`ContextCompactionConfig` 控制上下文压缩行为：

- auto compact
- partial compact
- prompt-too-long 多轮 retry
- compact 失败 fallback
- post-compact 文件状态恢复
- microcompact，也就是清理旧 `tool_result` 内容而不是整段总结

默认关闭，需要调用方在 `KernelConfig(context_compaction=...)` 中开启。

## 扩展能力

扩展层保持源码同形的“声明配置 + 初始化解析 + loop 外围接入”风格：

- Hooks：`PreToolUse`、`PostToolUse`、`PermissionRequest`、`Stop` 等生命周期事件。
- MCP：静态 client config、动态工具包装、resource read/list 工具。
- Skills：frontmatter 风格定义、预算内格式化、按需展开、skill tool。
- Subagent / Task agent：agent 定义解析、工具隔离、sidechain transcript、fork history。

这些能力都是 agent base 的外围骨架，不要求依赖真实 Claude Code UI。

## 项目结构

```text
main/
  agent_kernel/
    __init__.py              # 公共导出面
    query_engine.py          # 对外 session 门面
    query.py                 # 核心 agent loop
    messages.py              # 内部消息模型与 API 归一化
    model_provider.py        # Fake/Anthropic-compatible provider 与 streaming normalizer
    tool_execution.py        # 工具执行生命周期与并发调度
    permissions.py           # ask/bypass 权限解析
    path_validation.py       # 文件系统路径安全
    prompt_composer.py       # system prompt 拼接
    memory.py                # memory 路径、规则与 MEMORY.md 内容包装 helper
    session.py               # JSONL transcript 与 resume
    context_compaction.py    # compact / microcompact
    sdk.py                   # SDK/transcript 映射与 init/result/error
    hooks.py                 # hook registry 与 hook runner
    mcp.py                   # MCP 动态工具与资源
    skills.py                # skill 解析与 tool
    agents.py                # subagent / Task agent
    tools/
      bash.py                # Bash 工具
      file_tools.py          # Read/Write/Edit/MultiEdit/NotebookEdit
      search_tools.py        # Glob/Grep/LS
      web_tools.py           # WebSearch/WebFetch
      todo.py                # TodoWrite
      prompts.py             # 工具提示词常量
  tests/                     # 行为测试
  pyproject.toml
  READING_GUIDE.md
```

## Release Verification

Kernel baseline 最小验证命令：

```bash
cd /Users/ferry/Desktop/agent/main
python3 -m compileall agent_kernel tests
python3 -m pytest -q
```

如果只想快速验证本地 runner 切片：

```bash
cd /Users/ferry/Desktop/agent/main
python3 -m pytest tests/test_local_agent_runner.py -q
```

如果只想验证 packaging / console script hygiene：

```bash
cd /Users/ferry/Desktop/agent/main
python3 -m pytest tests/test_packaging.py -q
```

如果只想快速验证 v0.1 release-hardening 涉及的边界：

```bash
cd /Users/ferry/Desktop/agent/main
python3 -m pytest tests/test_prompt_memory_session.py tests/test_agents.py -q
```

`compileall` 会在本地生成 `__pycache__`；需要完全无副作用时，请在临时工作区或 CI 环境运行。

## 重要环境变量

| 变量 | 作用 |
|---|---|
| `ANTHROPIC_BASE_URL` | Anthropic-compatible API base URL |
| `ANTHROPIC_AUTH_TOKEN` | Bearer token 认证 |
| `ANTHROPIC_API_KEY` | `x-api-key` 认证，低于 `ANTHROPIC_AUTH_TOKEN` 优先级 |
| `ANTHROPIC_MODEL` | 默认模型名 |
| `CLAUDE_CONFIG_DIR` | 覆盖配置、memory、transcript 根目录 |
| `CLAUDE_CODE_OVERRIDE_DATE` | 覆盖 prompt 中的本地日期，便于测试 |
| `CLAUDE_CODE_SIMPLE` | 开启 simple mode |
| `USER_TYPE` | 进入配置与 prompt 的用户类型 |

## v0.1 Kernel Boundary

这个项目已经具备一个 agent base 的主干能力，但仍不是完整 Claude Code 产品实现：

- 没有真实终端 UI、交互授权 UI、analytics、完整 hooks 进程管理。
- MCP、skills、subagent 当前是可测骨架和协议翻译层，不绑定某个外部运行时。
- WebSearch/WebFetch 默认依赖可注入 handler；没有内置浏览器或搜索服务账号。
- Skill 的 forked execution、Agent Teams、remote agents、worktree isolation 不属于 v0.1 kernel release。
- v0.1 不扩大 `agent_kernel` 公共 API，也不引入新的 runtime dependency。
- 目标是保持核心行为语义和扩展形态可读、可验收，而不是追齐所有商业产品外围细节。

## 阅读建议

第一次读源码建议按这个顺序：

1. `agent_kernel/query_engine.py`
2. `agent_kernel/query.py`
3. `agent_kernel/messages.py`
4. `agent_kernel/tool_execution.py`
5. `agent_kernel/tools/base.py`
6. `agent_kernel/permissions.py`
7. `agent_kernel/prompt_composer.py`
8. `agent_kernel/context_compaction.py`
9. `agent_kernel/session.py`
10. `agent_kernel/sdk.py`

然后再进入 `tools/`、`hooks.py`、`mcp.py`、`skills.py` 和 `agents.py`。更详细的逐模块说明在 [READING_GUIDE.md](./READING_GUIDE.md)。
