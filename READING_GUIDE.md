# Agent Kernel 阅读导览

这份导览用于配合源码内注释阅读。建议不要从工具实现开始，而是沿着一次请求的真实调用链向下看。

## 推荐顺序

1. `agent_kernel/query_engine.py`
   - 对外入口 `QueryEngine.submit_message()`。
   - 负责组装模型、工具、memory、prompt、session 和 SDK 事件层。
2. `agent_kernel/query.py`
   - 核心 agent loop。
   - 一轮的顺序是：上下文整理、模型调用、收集 tool use、执行工具、回灌 tool result、进入下一轮。
3. `agent_kernel/messages.py`
   - 内部消息模型以及 API 前的消息归一化。
   - 重点看 `normalize_messages_for_api()` 和 `ensure_tool_result_pairing()`。
4. `agent_kernel/model_provider.py`
   - 模型抽象、Anthropic-compatible HTTP/SSE、stream event normalizer。
5. `agent_kernel/tool_execution.py`
   - 单工具执行管线和并发调度。
   - 顺序是 schema、业务校验、PreToolUse hook、权限、call、PostToolUse hook。
6. `agent_kernel/tools/base.py` 与 `agent_kernel/permissions.py`
   - 工具协议和 ask/bypass 权限决策。
7. `agent_kernel/context_compaction.py`
   - auto compact、partial compact、prompt-too-long retry、microcompact、文件状态恢复。
8. `agent_kernel/prompt_composer.py` 与 `agent_kernel/memory.py`
   - 系统提示词拼接顺序、CLAUDE.md、MEMORY.md 和环境信息。
9. `agent_kernel/session.py` 与 `agent_kernel/sdk.py`
   - JSONL transcript、resume、parentUuid 链和 SDK 外围映射。
10. `agent_kernel/agents.py`、`skills.py`、`mcp.py`、`hooks.py`
    - extension 层：subagent、skill、MCP 和生命周期 hook。
11. `agent_kernel/tools/`
    - Bash、文件、搜索、Web、Todo 等具体工具。

## 逐模块索引

### Agent 主干

| 模块 | 职责 | 阅读重点 |
|---|---|---|
| `agent_kernel/__init__.py` | 公共 API 汇总 | `__all__` 展示内核对外承诺的能力 |
| `agent_kernel/query_engine.py` | session 门面与依赖装配 | `__post_init__()`、`submit_message()`、事件持久化 |
| `agent_kernel/query.py` | 核心 agent loop | compact、模型、tool use、tool result、Stop hook、terminal 的顺序 |
| `agent_kernel/messages.py` | 消息类型、构造、API 归一化 | pairing 修复、attachment 保留、流式分片合并 |
| `agent_kernel/model_provider.py` | 模型抽象和 Anthropic-compatible API | 请求体、SSE parser、tool JSON delta、认证脱敏 |
| `agent_kernel/abort.py` | 跨层协作式取消 | controller/signal 分工和资源清理 callback |
| `agent_kernel/config.py` | 根配置和子系统配置 | 默认值来源、feature flags、compact/extension 声明 |

### 上下文与持久化

| 模块 | 职责 | 阅读重点 |
|---|---|---|
| `agent_kernel/context_compaction.py` | full/partial/micro compact | 安全 split、PTL retry、boundary metadata、文件状态恢复 |
| `agent_kernel/prompt_composer.py` | system prompt 与动态上下文拼接 | section 顺序、dynamic boundary、override/append 优先级 |
| `agent_kernel/memory.py` | 跨 session 项目知识 | memory 路径、MEMORY.md 截断、daily log 开关 |
| `agent_kernel/session.py` | JSONL transcript 与 resume | parentUuid、去重写入、compact/microcompact replay |
| `agent_kernel/sdk.py` | SDK 外围消息映射 | init/status/result/error 与 compact metadata 双向兼容 |

### 工具生命周期与安全

| 模块 | 职责 | 阅读重点 |
|---|---|---|
| `agent_kernel/tools/base.py` | Tool 协议和 ToolUseContext | 五阶段工具接口、共享 session 状态 |
| `agent_kernel/tool_execution.py` | 单工具生命周期和批次调度 | validate → hook → permission → call → hook |
| `agent_kernel/permissions.py` | ask/bypass 决策 | mode alias、callback、bypass-immune、hook 覆盖 |
| `agent_kernel/path_validation.py` | 文件路径安全策略 | cwd 边界、额外目录、敏感路径、shell/glob 语法 |
| `agent_kernel/path_utils.py` | 无策略路径 helper | expand、sanitize、git root、mtime |
| `agent_kernel/hooks.py` | 生命周期扩展点 | event input、matcher、返回形态归一化、控制流影响 |

### Extension

| 模块 | 职责 | 阅读重点 |
|---|---|---|
| `agent_kernel/agents.py` | Subagent/Task agent | 定义覆盖、工具隔离、fork history、sidechain transcript |
| `agent_kernel/skills.py` | 按需指令包 | frontmatter、预算索引、正文展开、skill hooks |
| `agent_kernel/mcp.py` | MCP 动态工具与资源 | 稳定命名、原始 JSON Schema、result content 转换 |

### 内置工具

| 模块 | 职责 | 阅读重点 |
|---|---|---|
| `agent_kernel/tools/__init__.py` | 工具导出面 | 与 `default_tools()` 的区别 |
| `agent_kernel/tools/bash.py` | Shell 执行 | 命令拆分、路径提取、进程组、timeout/background、大输出 |
| `agent_kernel/tools/file_tools.py` | Read/Write/Edit/Notebook | 多媒体读取、read-before-write、structured patch、原子编辑 |
| `agent_kernel/tools/search_tools.py` | Glob/Grep/LS | 只读并发、分页、二进制过滤、稳定输出 |
| `agent_kernel/tools/web_tools.py` | WebSearch/WebFetch | handler 注入、URL/redirect 安全、HTML 转换、cache |
| `agent_kernel/tools/todo.py` | TodoWrite 状态 | 输入校验、agent/session 隔离、完成后清理 |
| `agent_kernel/tools/prompts.py` | 工具模型提示词 | 正文精确度与运行时 schema 的分工 |

## 一次请求的数据流

```text
submit_message(user text)
  -> PromptComposer.fetch_system_prompt_parts
  -> query(QueryParams)
     -> compact/microcompact（必要时）
     -> ModelProvider.stream
     -> assistant message / tool_use
     -> run_tools
        -> validate -> hooks -> permission -> call -> hooks
     -> user message / tool_result
     -> 下一轮 ModelProvider.stream
     -> terminal
  -> SessionStore.record_transcript
  -> 可选 SDK init/result/error 包装
```

## 测试阅读入口

- `tests/test_permissions_tools_query.py`：主 loop、权限、真实 provider 形状、工具和 compact。
- `tests/test_prompt_memory_session.py`：prompt、memory、transcript、resume。
- `tests/test_agents.py`：subagent/fork。
- `tests/test_hooks.py`：hook 生命周期。
- `tests/test_mcp.py`、`tests/test_skills.py`：扩展协议。

测试是行为规范的一部分。源码注释解释“为什么”，测试展示“输入输出长什么样”。
