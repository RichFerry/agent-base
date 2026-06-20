"""贯穿模型请求、compact 和工具执行的协作式取消原语。

主要对象：
- ``AbortController``：拥有取消权，通常由 QueryEngine/ToolUseContext 持有。
- ``AbortSignal``：只读观察面，传给 provider、compact 和具体工具。

取消流程：调用 ``controller.abort(reason)`` 后，signal 记录状态、唤醒 ``wait()``、
同步调用已注册 callback；各异步阶段在边界调用 ``throw_if_aborted()``，统一转成
``asyncio.CancelledError``。真实 SSE 在工作线程读取，Bash 又拥有子进程组，因此只
取消顶层 task 不够，callback 还负责关闭 HTTP response 或清理外部资源。

同一个 controller 只服务一次 submit；下一 turn 会创建新的 controller，避免旧取消
状态污染后续请求。重复 abort 是幂等操作，第一条 reason 保留。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class AbortSignal:
    """只读取消状态；业务代码持有 signal，不应直接触发取消。"""
    aborted: bool = False
    reason: Any | None = None
    _event: asyncio.Event = field(default_factory=asyncio.Event, init=False, repr=False)
    # callback 用于关闭不受 asyncio task 直接控制的外部资源，例如 urllib response。
    _callbacks: list[Callable[[Any | None], None]] = field(default_factory=list, init=False, repr=False)

    def throw_if_aborted(self) -> None:
        """在流程边界把已记录的取消转成 asyncio.CancelledError。"""
        if self.aborted:
            raise asyncio.CancelledError(str(self.reason or "Request was aborted."))

    async def wait(self) -> Any | None:
        """异步等待取消信号触发，并返回取消原因。"""
        await self._event.wait()
        return self.reason

    def add_callback(self, callback: Callable[[Any | None], None]) -> Callable[[], None]:
        """注册资源清理函数，并返回解除注册的函数。"""
        if self.aborted:
            # 晚注册者也必须立即观察到取消，不能等待一个已经发生过的事件。
            callback(self.reason)
            return lambda: None
        self._callbacks.append(callback)

        def remove() -> None:
            """从当前取消信号中移除先前注册的回调。"""
            try:
                self._callbacks.remove(callback)
            except ValueError:
                pass

        return remove

    def _abort(self, reason: Any | None = None) -> None:
        """幂等地记录取消状态、唤醒等待者并执行清理回调。"""
        if self.aborted:
            # 第一条取消原因最有诊断价值；后续 abort 保持幂等。
            return
        self.aborted = True
        self.reason = reason
        self._event.set()
        # 先复制并清空，避免 callback 内重新注册/解除导致迭代列表变异。
        callbacks = list(self._callbacks)
        self._callbacks.clear()
        for callback in callbacks:
            callback(reason)


@dataclass
class AbortController:
    """拥有触发权的包装器；同一次 submit 的所有子流程共享它。"""
    signal: AbortSignal = field(default_factory=AbortSignal)

    def abort(self, reason: Any | None = "Request was aborted.") -> None:
        """使用给定原因触发本 controller 的取消信号。"""
        self.signal._abort(reason)
