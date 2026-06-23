"""Bash 工具：命令分析、路径权限、前后台进程和输出生命周期。

输入支持 command、description、timeout、run_in_background。权限阶段使用 shell lexer
拆分复合命令，剥离 env/sudo/timeout 等 wrapper，识别危险命令、重定向、process
substitution、cd 组合及 rm/mv/cp/touch/mkdir/tee/sed 等路径参数；提取出的路径统一交给
path_validation，不能借 shell 绕过文件安全。

执行阶段通过 asyncio subprocess 建立独立进程组。前台命令收集 stdout/stderr 和 exit
code；timeout/cancel 先 TERM 后 KILL 整组进程。后台命令立即返回 task id，异步把输出
写入 session 任务目录。超大输出也持久化，只在 tool_result 中保留预览，避免撑爆上下文。

结果 mapper 处理普通文本、错误状态和图片数据。Bash 是否只读由命令分析决定，因此
``is_concurrency_safe`` 不盲目并发未知命令。
"""

from __future__ import annotations

import asyncio
import base64
import fnmatch
import os
import re
import signal
import shlex
from pathlib import Path
from uuid import uuid4

from ..messages import AssistantMessage, ToolResultBlock
from ..path_validation import validate_path_for_operation
from ..permissions import PermissionDecision
from .base import Tool, ToolResult, ToolUseContext, ValidationResult
from .prompts import bash_tool_prompt


READ_ONLY_COMMAND_PREFIXES = (
    "ls",
    "pwd",
    "git status",
    "git diff",
    "git log",
    "python --version",
    "python3 --version",
    "node --version",
    "pytest --version",
)

DEFAULT_TIMEOUT_MS = 120_000
MAX_TIMEOUT_MS = 600_000
PREVIEW_SIZE_CHARS = 20_000


SHELL_SPLIT_RE = re.compile(r"\s*(?:&&|\|\||;|\|)\s*")
OUTPUT_REDIRECTION_RE = re.compile(r"""(?:^|\s)(?:[12]?>>?|&>)\s*("[^"]+"|'[^']+'|[^\s]+)""")
LEADING_ENV_RE = re.compile(
    r"""^(?:[A-Za-z_][A-Za-z0-9_]*(?:\[[^\]]*\])?\+?=(?:"[^"$`\\\n\r]*(?:\\.[^"$`\\\n\r]*)*"|'[^'\n\r]*'|[^ \t\n\r$`;|&()<>\\'"]*)[ \t]+)+"""
)
SLEEP_RE = re.compile(r"""(?:^|[;&|]\s*)sleep\s+(\d+(?:\.\d+)?)([smhd]?)\b""")
DURATION_RE = re.compile(r"""^\d+(?:\.\d+)?[smhd]?$""")


def _parse_shell_permission_rule(pattern: str) -> tuple[str, str]:
    """解析shell 权限 rule，供Bash 工具流程使用。"""
    # ``cmd:*`` 是前缀规则；普通 ``*`` 则按 shell 风格 wildcard 处理。
    if pattern.endswith(":*"):
        return "prefix", pattern[:-2]
    if _has_unescaped_star(pattern):
        return "wildcard", pattern
    return "exact", pattern


def _has_unescaped_star(pattern: str) -> bool:
    """判断是否具有unescaped star，供Bash 工具流程使用。"""
    if pattern.endswith(":*"):
        return False
    for index, char in enumerate(pattern):
        if char != "*":
            continue
        backslashes = 0
        cursor = index - 1
        while cursor >= 0 and pattern[cursor] == "\\":
            backslashes += 1
            cursor -= 1
        if backslashes % 2 == 0:
            return True
    return False


def _split_shell_subcommands(command: str) -> list[str]:
    """使用 shell lexer 拆分 &&、||、;、管道等顶层子命令。"""
    return [part.strip() for part in SHELL_SPLIT_RE.split(command) if part.strip()]


def _strip_output_redirections(command: str) -> str:
    """移除输出 redirections，供Bash 工具流程使用。"""
    return re.sub(r"\s*(?:>>?|2>|&>)\s*\S+\s*$", "", command).strip()


def _extract_output_redirections(command: str) -> list[str]:
    """提取输出 redirections，供Bash 工具流程使用。"""
    return [match.group(1).strip() for match in OUTPUT_REDIRECTION_RE.finditer(command)]


def _compound_command_has_cd(command: str) -> bool:
    """完成 ``_compound_command_has_cd`` 对应的Bash 工具内部步骤。"""
    subcommands = _split_shell_subcommands(command)
    return len(subcommands) > 1 and any(sub == "cd" or sub.startswith("cd ") for sub in subcommands)


def _strip_all_leading_env_vars(command: str) -> str:
    """移除all leading env vars，供Bash 工具流程使用。"""
    stripped = command.strip()
    while True:
        match = LEADING_ENV_RE.match(stripped)
        if not match:
            return stripped
        stripped = stripped[match.end() :].strip()


def _command_candidates(command: str, *, strip_env: bool = False) -> list[str]:
    """完成 ``_command_candidates`` 对应的Bash 工具内部步骤。"""
    command = _strip_output_redirections(command.strip())
    candidates = [command]
    if strip_env:
        env_stripped = _strip_all_leading_env_vars(command)
        if env_stripped not in candidates:
            candidates.append(env_stripped)
    return candidates


def _non_option_args(args: list[str]) -> list[str]:
    """完成 ``_non_option_args`` 对应的Bash 工具内部步骤。"""
    result: list[str] = []
    after_double_dash = False
    for arg in args:
        if after_double_dash:
            result.append(arg)
            continue
        if arg == "--":
            after_double_dash = True
            continue
        if arg.startswith("-"):
            continue
        result.append(arg)
    return result


def _command_name_and_args(subcommand: str) -> tuple[str, list[str]] | None:
    """完成 ``_command_name_and_args`` 对应的Bash 工具内部步骤。"""
    stripped = _strip_all_leading_env_vars(_strip_output_redirections(subcommand))
    # 无法可靠 shlex 解析时返回 None，权限层会保守地要求 ask。
    try:
        parts = shlex.split(stripped)
    except ValueError:
        return None
    if not parts:
        return None
    parts = _strip_safe_wrappers(parts)
    if not parts:
        return None
    name = parts[0].split("/")[-1]
    return name, parts[1:]


def _strip_safe_wrappers(parts: list[str]) -> list[str]:
    """移除safe wrappers，供Bash 工具流程使用。"""
    parts = list(parts)
    changed = True
    # 多层 wrapper 需要反复剥离，例如 ``sudo timeout 5 env X=1 rm file``。
    while changed and len(parts) > 1:
        changed = False
        name = parts[0].split("/")[-1]
        if name in {"sudo", "command", "env", "nohup"}:
            parts = parts[1:]
            changed = True
        elif name in {"time"}:
            while len(parts) > 1 and parts[1].startswith("-"):
                parts.pop(1)
            parts = parts[1:]
            changed = True
        elif name == "nice":
            index = 1
            while index < len(parts) and (parts[index].startswith("-") or parts[index].lstrip("+-").isdigit()):
                index += 1
            if index < len(parts):
                parts = parts[index:]
                changed = True
        elif name == "timeout":
            index = 1
            while index < len(parts):
                token = parts[index]
                if token == "--":
                    index += 1
                    break
                if DURATION_RE.match(token):
                    index += 1
                    break
                if token in {"--foreground", "--preserve-status", "--verbose", "-v"}:
                    index += 1
                    continue
                if token in {"--kill-after", "--signal", "-k", "-s"} and index + 1 < len(parts):
                    index += 2
                    continue
                if token.startswith(("--kill-after=", "--signal=", "-k", "-s")):
                    index += 1
                    continue
                break
            if index < len(parts):
                parts = parts[index:]
                changed = True
    return parts


def _extract_path_command_operations(command: str) -> list[tuple[str, str]]:
    """提取 rm/mv/cp/touch/mkdir/tee/sed 等命令涉及的目标路径。"""
    operations: list[tuple[str, str]] = []
    for subcommand in _split_shell_subcommands(command):
        parsed = _command_name_and_args(subcommand)
        if parsed is None:
            continue
        name, args = parsed
        path_args = _non_option_args(args)
        if name == "rm":
            operations.extend(("write", path) for path in path_args)
        elif name in {"touch", "mkdir"}:
            operations.extend(("create", path) for path in path_args)
        elif name in {"cp", "mv"} and path_args:
            operations.append(("write", path_args[-1]))
        elif name in {"chmod", "chown", "chgrp"} and len(path_args) > 1:
            operations.extend(("write", path) for path in path_args[1:])
        elif name == "ln" and path_args:
            operations.append(("create", path_args[-1]))
        elif name == "install" and path_args:
            operations.append(("create", path_args[-1]))
        elif name == "tee":
            operations.extend(("write", path) for path in path_args)
        elif name == "sed" and any(arg == "-i" or arg.startswith("-i") for arg in args) and path_args:
            operations.append(("write", path_args[-1]))
        elif name == "dd":
            for arg in args:
                if arg.startswith("of=") and len(arg) > 3:
                    operations.append(("write", arg[3:]))
    return operations


def _wildcard_match(pattern: str, command: str) -> bool:
    """完成 ``_wildcard_match`` 对应的Bash 工具内部步骤。"""
    pattern = pattern.strip()
    if pattern.endswith(" *") and pattern.count("*") == 1:
        bare = pattern[:-2]
        return command == bare or fnmatch.fnmatchcase(command, pattern)
    return fnmatch.fnmatchcase(command, pattern)


def _command_matches_rule(command: str, pattern: str, *, strip_env: bool = False) -> bool:
    """完成 ``_command_matches_rule`` 对应的Bash 工具内部步骤。"""
    rule_type, value = _parse_shell_permission_rule(pattern)
    # 宽松规则不能跨复合命令匹配，否则安全前缀后可拼接任意第二条命令。
    if rule_type in {"prefix", "wildcard"} and len(_split_shell_subcommands(command)) > 1:
        return False
    for candidate in _command_candidates(command, strip_env=strip_env):
        if rule_type == "exact":
            if candidate == value:
                return True
        elif rule_type == "prefix":
            if candidate == value or candidate.startswith(value + " "):
                return True
            xargs_prefix = "xargs " + value
            if candidate == xargs_prefix or candidate.startswith(xargs_prefix + " "):
                return True
        elif rule_type == "wildcard" and _wildcard_match(value, candidate):
            return True
    return False


def _detect_blocked_sleep_pattern(command: str) -> str | None:
    """完成 ``_detect_blocked_sleep_pattern`` 对应的Bash 工具内部步骤。"""
    for match in SLEEP_RE.finditer(command):
        amount = float(match.group(1))
        unit = match.group(2) or "s"
        seconds = amount * {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
        if seconds > 2:
            return f"sleep {match.group(1)}{match.group(2)}"
    return None


def _background_output_path(context: ToolUseContext, task_id: str) -> Path:
    """完成 ``_background_output_path`` 对应的Bash 工具内部步骤。"""
    output_dir = context.config.config_home / "bash-output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{task_id}.log"


async def _write_background_output(proc: asyncio.subprocess.Process, output_path: Path) -> dict:
    """写入后台任务 输出，供Bash 工具流程使用。"""
    try:
        # 后台 task 独立消费管道，避免子进程因 pipe buffer 填满而阻塞。
        stdout_b, stderr_b = await proc.communicate()
        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")
        output_path.write_text(
            "\n".join(part for part in [stdout.rstrip(), stderr.rstrip()] if part),
            encoding="utf-8",
        )
        return {"stdout": stdout, "stderr": stderr, "returnCode": proc.returncode}
    except asyncio.CancelledError:
        await _terminate_process_group(proc, kill=True)
        raise
    except Exception as exc:
        output_path.write_text(f"Background command failed: {type(exc).__name__}: {exc}", encoding="utf-8")
        return {"stdout": "", "stderr": str(exc), "returnCode": proc.returncode}


async def _terminate_process_group(proc: asyncio.subprocess.Process, *, kill: bool = False) -> None:
    """终止整个进程组，而不只终止 shell 父进程。"""
    if proc.returncode is not None:
        return
    # 正常 timeout 先 TERM 给清理机会；cancel 或二次超时直接 KILL。
    sig = signal.SIGKILL if kill else signal.SIGTERM
    try:
        if os.name != "nt" and proc.pid is not None:
            os.killpg(proc.pid, sig)
        elif kill:
            proc.kill()
        else:
            proc.terminate()
    except ProcessLookupError:
        return
    except Exception:
        try:
            proc.kill()
        except Exception:
            return
    try:
        await asyncio.wait_for(proc.wait(), timeout=1.0)
    except asyncio.TimeoutError:
        if not kill:
            await _terminate_process_group(proc, kill=True)


def _persist_large_output(context: ToolUseContext, stdout: str, stderr: str) -> tuple[str | None, int | None]:
    """持久化large 输出，供Bash 工具流程使用。"""
    combined_size = len((stdout + stderr).encode("utf-8"))
    if combined_size <= BashTool.max_result_size_chars:
        return None, None
    # 完整输出写磁盘，模型上下文只接收有限预览和可再次 Read 的路径。
    task_id = f"output-{uuid4().hex}"
    output_path = _background_output_path(context, task_id)
    output_path.write_text("\n".join(part for part in [stdout.rstrip(), stderr.rstrip()] if part), encoding="utf-8")
    return str(output_path), combined_size


def _large_output_message(filepath: str, original_size: int, preview: str) -> str:
    """完成 ``_large_output_message`` 对应的Bash 工具内部步骤。"""
    has_more = len(preview) >= PREVIEW_SIZE_CHARS
    return (
        f'<persisted-output filepath="{filepath}" originalSize="{original_size}">\n'
        f"{preview}"
        f"\n{'[Output truncated. Read the file above for the full output.]' if has_more else ''}\n"
        "</persisted-output>"
    )


def _image_tool_result(stdout: str, tool_use_id: str) -> ToolResultBlock | None:
    """完成 ``_image_tool_result`` 对应的Bash 工具内部步骤。"""
    match = re.match(r"^data:(image/(?:png|jpeg|jpg|gif|webp));base64,([A-Za-z0-9+/=\s]+)$", stdout.strip())
    if not match:
        return None
    media_type = "image/jpeg" if match.group(1) == "image/jpg" else match.group(1)
    data = "".join(match.group(2).split())
    try:
        base64.b64decode(data, validate=True)
    except Exception:
        return None
    return {
        "tool_use_id": tool_use_id,
        "type": "tool_result",
        "content": [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": data},
            }
        ],
    }


class BashTool(Tool):
    """可流式报告进度、支持后台任务的 shell 执行工具。"""
    name = "Bash"
    search_hint = "execute shell commands"
    max_result_size_chars = 30_000
    input_schema = {"command": str, "description": str, "timeout": int, "run_in_background": bool}
    required_fields = ("command",)

    async def description(self, input: dict | None = None) -> str:
        """返回提供给模型和调用方展示的工具简短说明。"""
        if input and isinstance(input.get("description"), str):
            return input["description"]
        return "Run shell command"

    async def prompt(self) -> str:
        """返回该工具按源码保留的模型使用提示词。"""
        return bash_tool_prompt()

    def is_read_only(self, input: dict) -> bool:
        """判断当前输入是否只读取外部状态而不产生修改。"""
        command = str(input.get("command", "")).strip()
        if re.search(r"\b(rm|mv|cp|chmod|chown|chgrp|mkdir|touch|tee|curl|wget|ln|install|dd|git\s+push|git\s+reset|git\s+checkout|git\s+commit)\b", command):
            return False
        return any(command == prefix or command.startswith(prefix + " ") for prefix in READ_ONLY_COMMAND_PREFIXES)

    def is_destructive(self, input: dict) -> bool:
        """判断当前输入是否可能产生破坏性副作用。"""
        return not self.is_read_only(input)

    def prepare_permission_matcher(self, input: dict):
        """为当前输入构造可选的权限规则匹配函数。"""
        command = str(input.get("command", ""))
        return lambda pattern: _command_matches_rule(command, pattern)

    async def validate_input(self, input: dict, context: ToolUseContext) -> ValidationResult:
        """执行依赖上下文的业务输入校验。"""
        if not input.get("command"):
            return ValidationResult(False, "command is required.", 1)
        timeout = input.get("timeout")
        if timeout is not None and (not isinstance(timeout, int) or timeout <= 0):
            return ValidationResult(False, "timeout must be a positive integer of milliseconds.", 2)
        sleep_pattern = _detect_blocked_sleep_pattern(str(input.get("command", "")))
        if sleep_pattern is not None and not input.get("run_in_background"):
            return ValidationResult(
                False,
                f"Blocked: {sleep_pattern}. Run blocking commands in the background with run_in_background: true.",
                10,
            )
        return ValidationResult(True)

    async def check_permissions(self, input: dict, context: ToolUseContext) -> PermissionDecision:
        """返回该工具针对当前输入的初步权限建议。"""
        command = str(input.get("command", "")).strip()
        permission_context = context.get_app_state().tool_permission_context
        # process substitution 可以隐藏任意子命令，自动静态分析不可靠。
        if re.search(r">\s*>\s*\(|<\s*\(", command):
            return PermissionDecision.ask(
                "Process substitution (>(...) or <(...)) can execute arbitrary commands and requires manual approval",
                bypass_immune=True,
            )
        redirection_paths = _extract_output_redirections(command)
        if redirection_paths and _compound_command_has_cd(command):
            return PermissionDecision.ask(
                "Commands that change directories and write via output redirection require explicit approval to ensure paths are evaluated correctly. For security, Agent Base cannot automatically determine the final working directory when 'cd' is used in compound commands.",
                bypass_immune=True,
            )
        path_operations = _extract_path_command_operations(command)
        if path_operations and _compound_command_has_cd(command):
            return PermissionDecision.ask(
                "Commands that change directories and modify filesystem paths require explicit approval to ensure paths are evaluated correctly.",
                bypass_immune=True,
            )
        # 重定向和路径型命令必须复用文件工具的同一安全边界。
        for redirection_path in redirection_paths:
            path_result = validate_path_for_operation(
                path=redirection_path,
                cwd=context.config.cwd,
                permission_context=permission_context,
                operation_type="write",
                tool_name="Write",
            )
            if path_result.decision and path_result.decision.behavior != "allow":
                return path_result.decision
        for operation_type, target_path in path_operations:
            path_result = validate_path_for_operation(
                path=target_path,
                cwd=context.config.cwd,
                permission_context=permission_context,
                operation_type=operation_type,
                tool_name="Write",
            )
            if path_result.decision and path_result.decision.behavior != "allow":
                return path_result.decision
        if self.is_read_only(input):
            return PermissionDecision.allow()
        return PermissionDecision.ask("This Bash command requires approval.")

    async def call(self, args: dict, context: ToolUseContext, can_use_tool, parent_message: AssistantMessage, on_progress=None) -> ToolResult:
        """执行命令，并把 exit code、timeout、background 元数据标准化。"""
        timeout = args.get("timeout")
        timeout_ms = min(int(timeout) if timeout else DEFAULT_TIMEOUT_MS, MAX_TIMEOUT_MS)
        timeout_seconds = timeout_ms / 1000
        if on_progress:
            on_progress({"message": f"Running command: {args['command']}"})
        subprocess_kwargs = {}
        if os.name != "nt":
            # 独立 session 让 timeout/cancel 能终止整个进程树。
            subprocess_kwargs["start_new_session"] = True
        proc = await asyncio.create_subprocess_shell(
            args["command"],
            cwd=str(context.config.cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **subprocess_kwargs,
        )
        if args.get("run_in_background"):
            # 后台模式不等待退出；task 和输出路径存入 context 供后续查询。
            task_id = f"bash-{uuid4().hex[:12]}"
            output_path = _background_output_path(context, task_id)
            task = asyncio.create_task(_write_background_output(proc, output_path))
            context.background_tasks[task_id] = {
                "command": args["command"],
                "task": task,
                "outputPath": str(output_path),
                "pid": proc.pid,
            }
            if on_progress:
                on_progress({"message": f"Command running in background with ID: {task_id}", "taskId": task_id})
            return ToolResult(
                {
                    "stdout": "",
                    "stderr": "",
                    "interrupted": False,
                    "returnCode": None,
                    "backgroundTaskId": task_id,
                    "backgroundedByUser": True,
                    "outputPath": str(output_path),
                }
            )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
            interrupted = False
        except asyncio.TimeoutError:
            # timeout 返回 interrupted ToolResult，让模型看到已有输出并决定下一步。
            await _terminate_process_group(proc)
            stdout_b, stderr_b = await proc.communicate()
            interrupted = True
        except asyncio.CancelledError:
            await _terminate_process_group(proc, kill=True)
            raise
        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")
        persisted_output_path, persisted_output_size = _persist_large_output(context, stdout, stderr)
        if on_progress:
            on_progress({"message": "Command completed" if not interrupted else "Command interrupted"})
        return ToolResult(
            {
                "stdout": stdout,
                "stderr": stderr,
                "interrupted": interrupted,
                "returnCode": proc.returncode,
                "persistedOutputPath": persisted_output_path,
                "persistedOutputSize": persisted_output_size,
                "timeoutMs": timeout_ms,
            }
        )

    def map_tool_result_to_tool_result_block_param(self, content: dict, tool_use_id: str) -> ToolResultBlock:
        """把工具内部结果映射为 Anthropic tool_result block。"""
        if content.get("structuredContent"):
            return {"tool_use_id": tool_use_id, "type": "tool_result", "content": content["structuredContent"]}
        processed_stdout = str(content.get("stdout", ""))
        image_block = _image_tool_result(processed_stdout, tool_use_id)
        if image_block is not None:
            return image_block
        if processed_stdout:
            processed_stdout = re.sub(r"^(\s*\n)+", "", processed_stdout).rstrip()
        if content.get("persistedOutputPath"):
            # persisted-output XML 标记提示模型使用 Read 获取完整内容。
            preview = processed_stdout[:PREVIEW_SIZE_CHARS]
            processed_stdout = _large_output_message(
                str(content["persistedOutputPath"]),
                int(content.get("persistedOutputSize") or len(processed_stdout.encode("utf-8"))),
                preview,
            )
        error_message = str(content.get("stderr", "")).strip()
        if content.get("interrupted"):
            if error_message:
                error_message += "\n"
            error_message += "<error>Command was aborted before completion</error>"
        elif content.get("returnCode") not in {None, 0} and not error_message:
            error_message = f"<error>Command failed with exit code {content['returnCode']}</error>"
        background_info = ""
        if content.get("backgroundTaskId"):
            background_info = (
                f"Command running in background with ID: {content['backgroundTaskId']}. "
                f"Output is being written to: {content.get('outputPath')}"
            )
        result = "\n".join(part for part in [processed_stdout, error_message, background_info] if part)
        return {
            "tool_use_id": tool_use_id,
            "type": "tool_result",
            "content": result,
            "is_error": bool(content.get("interrupted")),
        }
