# Claude Code-style Agent Kernel Python Port

这是一个放在 `main/agent_kernel` 下的 Python Agent Kernel。它的目标不是复刻完整 Claude Code 产品外壳，而是把 Claude Code 核心 agent base 的运行语义翻译成可读、可测、可继续扩展的 Python 版本。

当前内核覆盖了 agent loop、message flow、tool protocol、ask/bypass permission resolver、prompt composer、session transcript、memory loading、context compaction、hooks、MCP、skills、subagent/task agent、SDK/transcript 外围事件，以及 Anthropic-compatible 真实模型 API 接入。

更细的逐模块阅读路线见 [READING_GUIDE.md](./READING_GUIDE.md)。

## 当前定位

- 这是一个独立 Python 包，包名是 `agent-kernel`，源码目录是 `agent_kernel/`。
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

Memory 默认从 Claude Code-style 配置目录读取：

```text
<config_home>/projects/<sanitized-git-root-or-cwd>/memory/MEMORY.md
```

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
    memory.py                # MEMORY.md 加载
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

## 验证命令

运行测试：

```bash
cd /Users/ferry/Desktop/agent/main
python3 -m pytest -q
```

编译检查：

```bash
cd /Users/ferry/Desktop/agent/main
python3 -m compileall agent_kernel tests
```

如果使用 Codex desktop 绑定的 Python runtime，可以使用：

```bash
cd /Users/ferry/Desktop/agent/main
PYTHONPATH=/tmp/agent-kernel-testdeps:. /Users/ferry/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest -q
```

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

## 当前边界

这个项目已经具备一个 agent base 的主干能力，但仍不是完整 Claude Code 产品实现：

- 没有真实终端 UI、交互授权 UI、analytics、完整 hooks 进程管理。
- MCP、skills、subagent 当前是可测骨架和协议翻译层，不绑定某个外部运行时。
- WebSearch/WebFetch 默认依赖可注入 handler；没有内置浏览器或搜索服务账号。
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
