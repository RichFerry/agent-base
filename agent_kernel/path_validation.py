"""File/Bash/Search 工具共享的路径安全和工作目录边界。

输入包括原始路径、cwd、ToolPermissionContext 和操作类型 read/write/create。校验先
拒绝 glob、shell expansion 等不能安全解析的原始语法，再规范化绝对路径，判断其是否
位于 cwd 或 additional_working_directories，最后检查 ``.git``、Claude settings、IDE
配置等敏感目标。

输出 ``PathPermissionResult`` 同时携带 resolved path 与 PermissionDecision：普通项目内
读取可自动允许；写入通常 ask；敏感路径返回 bypass-immune deny/ask。具体 Tool 可以
添加业务校验，但不应复制这套目录与敏感路径规则。

该模块不访问文件正文，也不执行 shell。Bash 先从命令中提取路径操作，再复用这里的
同一策略，从而避免通过重定向、rm/mv/cp 等绕过文件工具权限。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re

from .permissions import PermissionDecision, ToolPermissionContext


DANGEROUS_FILES = {
    ".gitconfig",
    ".gitmodules",
    ".bashrc",
    ".bash_profile",
    ".zshrc",
    ".zprofile",
    ".profile",
    ".ripgreprc",
    ".mcp.json",
    ".claude.json",
}

DANGEROUS_DIRECTORIES = {".git", ".vscode", ".idea", ".claude"}
# 写工具必须使用精确路径；这些字符表示调用方可能试图批量展开。
GLOB_PATTERN_RE = re.compile(r"[*?[\]{}]")


@dataclass(frozen=True)
class PathPermissionResult:
    """封装 ``PathPermissionResult`` 产生的结构化结果。"""
    allowed: bool
    resolved_path: Path
    decision: PermissionDecision | None = None


def resolve_for_permission(path: str | Path, cwd: Path) -> Path:
    """解析并确定for 权限，供路径安全流程使用。"""
    # 只去掉包裹引号；不执行 shell expansion，避免权限检查与实际目标不一致。
    raw = Path(str(path).strip("\"'"))
    expanded = Path(os.path.expanduser(str(raw)))
    if not expanded.is_absolute():
        expanded = cwd / expanded
    return expanded.resolve(strict=False)


def _normalize(path: Path) -> str:
    """完成 ``_normalize`` 对应的路径安全内部步骤。"""
    return str(path.resolve(strict=False)).casefold()


def path_in_directory(path: Path, directory: Path) -> bool:
    """完成 ``path_in_directory`` 对应的路径安全内部步骤。"""
    # 使用目录边界分隔符，防止 /repo2 被误判为 /repo 的子目录。
    normalized_path = _normalize(path)
    normalized_dir = _normalize(directory)
    return normalized_path == normalized_dir or normalized_path.startswith(normalized_dir.rstrip(os.sep) + os.sep)


def allowed_working_directories(cwd: Path, context: ToolPermissionContext) -> list[Path]:
    """完成 ``allowed_working_directories`` 对应的路径安全内部步骤。"""
    directories = [cwd]
    for value in context.additional_working_directories.keys():
        directories.append(Path(value))
    return directories


def path_in_allowed_working_path(path: Path, cwd: Path, context: ToolPermissionContext) -> bool:
    """完成 ``path_in_allowed_working_path`` 对应的路径安全内部步骤。"""
    return any(path_in_directory(path, directory) for directory in allowed_working_directories(cwd, context))


def _raw_path_requires_manual_approval(path: str) -> str | None:
    """完成 ``_raw_path_requires_manual_approval`` 对应的路径安全内部步骤。"""
    clean = path.strip("\"'")
    if clean.startswith("~") and clean not in {"~"} and not clean.startswith("~/") and not clean.startswith("~\\"):
        return "Tilde expansion variants (~user, ~+, ~-) in paths require manual approval"
    if "$" in clean or "%" in clean or clean.startswith("="):
        return "Shell expansion syntax in paths requires manual approval"
    return None


def is_dangerous_file_path(path: Path) -> bool:
    """判断dangerous 文件 路径，供路径安全流程使用。"""
    parts = [part.casefold() for part in path.parts]
    for index, part in enumerate(parts):
        if part in DANGEROUS_DIRECTORIES:
            # Claude 自己创建的 worktree 数据目录是已知安全例外。
            if part == ".claude" and index + 1 < len(parts) and parts[index + 1] == "worktrees":
                continue
            return True
    return path.name.casefold() in DANGEROUS_FILES


def check_path_safety_for_auto_edit(path: Path) -> PermissionDecision | None:
    """检查路径 safety for auto edit，供路径安全流程使用。"""
    if is_dangerous_file_path(path):
        return PermissionDecision.ask(
            f"Claude requested permissions to edit {path} which is a sensitive file.",
            bypass_immune=True,
        )
    return None


def validate_path_for_operation(
    *,
    path: str,
    cwd: Path,
    permission_context: ToolPermissionContext,
    operation_type: str,
    tool_name: str,
) -> PathPermissionResult:
    """解析原始路径并返回规范路径、允许状态及拒绝原因。"""
    # 原始语法必须在 resolve 前检查，否则 expand 后会丢失风险来源。
    raw_rejection = _raw_path_requires_manual_approval(path)
    resolved_path = resolve_for_permission(path, cwd)
    if raw_rejection:
        return PathPermissionResult(False, resolved_path, PermissionDecision.ask(raw_rejection, bypass_immune=True))

    if operation_type in {"write", "create"} and GLOB_PATTERN_RE.search(path.strip("\"'")):
        return PathPermissionResult(
            False,
            resolved_path,
            PermissionDecision.ask(
                "Glob patterns are not allowed in write operations. Please specify an exact file path.",
                bypass_immune=True,
            ),
        )

    if operation_type in {"write", "create"}:
        # 敏感路径保护优先于普通工作目录允许规则，并且对 bypass 免疫。
        safety_decision = check_path_safety_for_auto_edit(resolved_path)
        if safety_decision is not None:
            return PathPermissionResult(False, resolved_path, safety_decision)

    in_working_path = path_in_allowed_working_path(resolved_path, cwd, permission_context)
    # 项目内读取自动允许；项目内写入仍返回未允许，交由具体工具请求 ask。
    if operation_type == "read" and in_working_path:
        return PathPermissionResult(True, resolved_path)

    if not in_working_path:
        return PathPermissionResult(
            False,
            resolved_path,
            PermissionDecision.ask(
                "Path is outside allowed working directories.",
            ),
        )

    return PathPermissionResult(False, resolved_path)
