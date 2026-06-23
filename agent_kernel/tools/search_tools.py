"""本地文件发现与内容搜索工具：Glob、Grep、LS。

三者均为只读且并发安全，但仍通过 path_validation 限制搜索根目录。Glob 递归匹配文件
pattern 并按 mtime 排序；Grep 编译正则，支持 content、files_with_matches、count 模式、
上下文行、glob/type filter 和分页；LS 返回单层目录项及类型。

搜索读取会跳过明显二进制内容并容错编码问题。``_apply_limit`` 与结果 mapper 明确记录
offset/limit/truncation，防止大型仓库把无限结果灌入上下文。路径显示尽量相对 cwd，
但权限和真实读取始终使用规范绝对路径。

实现使用标准库以保持包零依赖；它追求 Agent Base 工具协议和输出形状，不试图替代
完整 ripgrep 的所有语法优化。
"""

from __future__ import annotations

import fnmatch
import os
import re
import time
from pathlib import Path
from typing import Any

from ..messages import AssistantMessage, ToolResultBlock
from ..path_validation import resolve_for_permission, validate_path_for_operation
from ..permissions import PermissionDecision
from .base import Tool, ToolResult, ToolUseContext, ValidationResult


GLOB_DESCRIPTION = """- Fast file pattern matching tool that works with any codebase size
- Supports glob patterns like "**/*.js" or "src/**/*.ts"
- Returns matching file paths sorted by modification time
- Use this tool when you need to find files by name patterns
- When you are doing an open ended search that may require multiple rounds of globbing and grepping, use the Agent tool instead"""

GREP_DESCRIPTION = """A powerful search tool built on ripgrep

  Usage:
  - ALWAYS use Grep for search tasks. NEVER invoke `grep` or `rg` as a Bash command. The Grep tool has been optimized for correct permissions and access.
  - Supports full regex syntax (e.g., "log.*Error", "function\\s+\\w+")
  - Filter files with glob parameter (e.g. "*.js", "**/*.tsx") or type parameter (e.g. "js", "py", "rust")
  - Output modes: "content" shows matching lines, "files_with_matches" shows only file paths (default), "count" shows match counts
  - Use Agent tool for open-ended searches requiring multiple rounds
  - Pattern syntax: Uses ripgrep (not grep) - literal braces need escaping (use `interface\\{\\}` to find `interface{}` in Go code)
  - Multiline matching: By default patterns match within single lines only. For cross-line patterns like `struct \\{[\\s\\S]*?field`, use `multiline: true`
"""

LS_DESCRIPTION = """List files and directories in a given path. Use this tool instead of `ls` when you need to inspect directory contents."""

VCS_DIRECTORIES_TO_EXCLUDE = {".git", ".svn", ".hg", ".bzr", ".jj", ".sl"}
DEFAULT_HEAD_LIMIT = 250
MAX_READ_BYTES = 2_000_000

TYPE_EXTENSIONS = {
    "js": {".js", ".jsx", ".mjs", ".cjs"},
    "ts": {".ts", ".tsx", ".mts", ".cts"},
    "py": {".py", ".pyi"},
    "rust": {".rs"},
    "go": {".go"},
    "java": {".java"},
    "json": {".json"},
    "md": {".md", ".markdown"},
    "html": {".html", ".htm"},
    "css": {".css"},
    "yaml": {".yaml", ".yml"},
    "sh": {".sh", ".bash", ".zsh"},
}


def _relative(path: Path, cwd: Path) -> str:
    """完成 ``_relative`` 对应的搜索工具内部步骤。"""
    try:
        return str(path.resolve(strict=False).relative_to(cwd.resolve(strict=False)))
    except ValueError:
        return str(path.resolve(strict=False))


def _safe_mtime(path: Path) -> float:
    """完成 ``_safe_mtime`` 对应的搜索工具内部步骤。"""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0


def _iter_files(root: Path) -> list[Path]:
    """遍历文件集合，供搜索工具流程使用。"""
    if root.is_file():
        return [root]
    files: list[Path] = []
    for current, dirs, names in os.walk(root):
        dirs[:] = [name for name in dirs if name not in VCS_DIRECTORIES_TO_EXCLUDE]
        current_path = Path(current)
        for name in names:
            files.append(current_path / name)
    return files


def _looks_binary(raw: bytes) -> bool:
    """完成 ``_looks_binary`` 对应的搜索工具内部步骤。"""
    return b"\x00" in raw[:4096]


def _read_search_text(path: Path) -> str | None:
    """读取search 文本，供搜索工具流程使用。"""
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if len(raw) > MAX_READ_BYTES or _looks_binary(raw):
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace")


def _apply_limit(items: list[Any], limit: int | None, offset: int = 0) -> tuple[list[Any], int | None]:
    """应用分页并返回实际 limit，供结果文本说明是否截断。"""
    if limit == 0:
        return items[offset:], None
    effective = DEFAULT_HEAD_LIMIT if limit is None else limit
    sliced = items[offset : offset + effective]
    return sliced, effective if len(items) - offset > effective else None


def _format_limit(applied_limit: int | None, offset: int | None) -> str:
    """格式化limit，供搜索工具流程使用。"""
    parts = []
    if applied_limit is not None:
        parts.append(f"limit: {applied_limit}")
    if offset:
        parts.append(f"offset: {offset}")
    return ", ".join(parts)


def _matches_glob(path: Path, root: Path, pattern: str | None) -> bool:
    """完成 ``_matches_glob`` 对应的搜索工具内部步骤。"""
    if not pattern:
        return True
    rel = _relative(path, root)
    patterns: list[str] = []
    for raw in pattern.split():
        if "{" in raw and "}" in raw:
            patterns.append(raw)
        else:
            patterns.extend(part for part in raw.split(",") if part)
    return any(fnmatch.fnmatch(rel, item) or fnmatch.fnmatch(path.name, item) for item in patterns)


def _matches_type(path: Path, type_name: str | None) -> bool:
    """完成 ``_matches_type`` 对应的搜索工具内部步骤。"""
    if not type_name:
        return True
    extensions = TYPE_EXTENSIONS.get(type_name)
    if extensions is None:
        return True
    return path.suffix in extensions


class GlobTool(Tool):
    """按 glob pattern 查找文件，并按 mtime 排序。"""
    name = "Glob"
    search_hint = "find files by name pattern or wildcard"
    max_result_size_chars = 100_000
    input_schema = {"pattern": str, "path": str}
    required_fields = ("pattern",)

    async def description(self, input: dict | None = None) -> str:
        """返回提供给模型和调用方展示的工具简短说明。"""
        return GLOB_DESCRIPTION

    async def prompt(self) -> str:
        """返回该工具按源码保留的模型使用提示词。"""
        return GLOB_DESCRIPTION

    def is_concurrency_safe(self, input: dict) -> bool:
        """判断当前输入是否可以与相邻安全工具并发执行。"""
        return True

    def is_read_only(self, input: dict) -> bool:
        """判断当前输入是否只读取外部状态而不产生修改。"""
        return True

    def get_path(self, input: dict) -> str:
        """从工具输入中提取用于权限检查的目标路径。"""
        return str(resolve_for_permission(input.get("path") or ".", Path.cwd()))

    async def validate_input(self, input: dict, context: ToolUseContext) -> ValidationResult:
        """执行依赖上下文的业务输入校验。"""
        base = resolve_for_permission(input.get("path") or ".", context.config.cwd)
        if input.get("path") and not base.exists():
            return ValidationResult(False, f"Directory does not exist: {input['path']}. Current working directory: {context.config.cwd}.", 1)
        if input.get("path") and not base.is_dir():
            return ValidationResult(False, f"Path is not a directory: {input['path']}", 2)
        return ValidationResult(True)

    async def check_permissions(self, input: dict, context: ToolUseContext) -> PermissionDecision:
        """返回该工具针对当前输入的初步权限建议。"""
        path = input.get("path") or "."
        path_result = validate_path_for_operation(
            path=path,
            cwd=context.config.cwd,
            permission_context=context.get_app_state().tool_permission_context,
            operation_type="read",
            tool_name=self.name,
        )
        if path_result.decision is not None:
            return path_result.decision
        return PermissionDecision.allow() if path_result.allowed else PermissionDecision.ask("Glob requires approval.")

    async def call(self, args: dict, context: ToolUseContext, can_use_tool, parent_message: AssistantMessage, on_progress=None) -> ToolResult:
        """执行工具核心逻辑，并返回标准 ToolResult。"""
        start = time.monotonic()
        base = resolve_for_permission(args.get("path") or ".", context.config.cwd)
        if on_progress:
            on_progress({"message": f"Finding files matching {args['pattern']}"})
        matches = [path for path in base.glob(args["pattern"]) if path.is_file()]
        matches.sort(key=_safe_mtime, reverse=True)
        limit = 100
        truncated = len(matches) > limit
        filenames = [_relative(path, context.config.cwd) for path in matches[:limit]]
        return ToolResult(
            {
                "durationMs": int((time.monotonic() - start) * 1000),
                "numFiles": len(filenames),
                "filenames": filenames,
                "truncated": truncated,
            }
        )

    def map_tool_result_to_tool_result_block_param(self, content: dict, tool_use_id: str) -> ToolResultBlock:
        """把工具内部结果映射为 Anthropic tool_result block。"""
        filenames = content.get("filenames", [])
        if not filenames:
            return {"tool_use_id": tool_use_id, "type": "tool_result", "content": "No files found"}
        result = "\n".join([*filenames, *(
            ["(Results are truncated. Consider using a more specific path or pattern.)"] if content.get("truncated") else []
        )])
        return {"tool_use_id": tool_use_id, "type": "tool_result", "content": result}


class GrepTool(Tool):
    """支持 content/files/count 三种模式的正则内容搜索。"""
    name = "Grep"
    search_hint = "search file contents with regex (ripgrep)"
    max_result_size_chars = 20_000
    input_schema = {
        "pattern": str,
        "path": str,
        "glob": str,
        "output_mode": str,
        "-B": int,
        "-A": int,
        "-C": int,
        "context": int,
        "-n": bool,
        "-i": bool,
        "type": str,
        "head_limit": int,
        "offset": int,
        "multiline": bool,
    }
    required_fields = ("pattern",)

    async def description(self, input: dict | None = None) -> str:
        """返回提供给模型和调用方展示的工具简短说明。"""
        return GREP_DESCRIPTION

    async def prompt(self) -> str:
        """返回该工具按源码保留的模型使用提示词。"""
        return GREP_DESCRIPTION

    def is_concurrency_safe(self, input: dict) -> bool:
        """判断当前输入是否可以与相邻安全工具并发执行。"""
        return True

    def is_read_only(self, input: dict) -> bool:
        """判断当前输入是否只读取外部状态而不产生修改。"""
        return True

    def get_path(self, input: dict) -> str:
        """从工具输入中提取用于权限检查的目标路径。"""
        return str(input.get("path") or ".")

    async def validate_input(self, input: dict, context: ToolUseContext) -> ValidationResult:
        """执行依赖上下文的业务输入校验。"""
        output_mode = input.get("output_mode", "files_with_matches")
        if output_mode not in {"content", "files_with_matches", "count"}:
            return ValidationResult(False, "output_mode must be content, files_with_matches, or count.", 2)
        path = resolve_for_permission(input.get("path") or ".", context.config.cwd)
        if input.get("path") and not path.exists():
            return ValidationResult(False, f"Path does not exist: {input['path']}. Current working directory: {context.config.cwd}.", 1)
        try:
            re.compile(input["pattern"], re.DOTALL if input.get("multiline") else 0)
        except re.error as exc:
            return ValidationResult(False, f"Invalid regular expression: {exc}", 3)
        return ValidationResult(True)

    async def check_permissions(self, input: dict, context: ToolUseContext) -> PermissionDecision:
        """返回该工具针对当前输入的初步权限建议。"""
        path = input.get("path") or "."
        path_result = validate_path_for_operation(
            path=path,
            cwd=context.config.cwd,
            permission_context=context.get_app_state().tool_permission_context,
            operation_type="read",
            tool_name=self.name,
        )
        if path_result.decision is not None:
            return path_result.decision
        return PermissionDecision.allow() if path_result.allowed else PermissionDecision.ask("Grep requires approval.")

    async def call(self, args: dict, context: ToolUseContext, can_use_tool, parent_message: AssistantMessage, on_progress=None) -> ToolResult:
        """执行工具核心逻辑，并返回标准 ToolResult。"""
        root = resolve_for_permission(args.get("path") or ".", context.config.cwd)
        if on_progress:
            on_progress({"message": f"Searching for {args['pattern']}"})
        flags = re.IGNORECASE if args.get("-i") else 0
        if args.get("multiline"):
            flags |= re.DOTALL
        regex = re.compile(args["pattern"], flags)
        output_mode = args.get("output_mode", "files_with_matches")
        offset = int(args.get("offset") or 0)
        head_limit = args.get("head_limit")
        head_limit = int(head_limit) if head_limit is not None else None
        files = [
            path
            for path in _iter_files(root)
            if _matches_glob(path, root if root.is_dir() else root.parent, args.get("glob"))
            and _matches_type(path, args.get("type"))
        ]
        files.sort(key=lambda path: _relative(path, context.config.cwd))

        if output_mode == "content":
            before = int(args.get("context") if args.get("context") is not None else args.get("-C") if args.get("-C") is not None else args.get("-B") or 0)
            after = int(args.get("context") if args.get("context") is not None else args.get("-C") if args.get("-C") is not None else args.get("-A") or 0)
            show_numbers = args.get("-n", True)
            lines_out: list[str] = []
            for path in files:
                text = _read_search_text(path)
                if text is None:
                    continue
                lines = text.splitlines()
                matched_indexes = [index for index, line in enumerate(lines) if regex.search(line)]
                for index in matched_indexes:
                    start = max(0, index - before)
                    end = min(len(lines), index + after + 1)
                    for line_index in range(start, end):
                        prefix = f"{_relative(path, context.config.cwd)}:"
                        if show_numbers:
                            prefix += f"{line_index + 1}:"
                        lines_out.append(prefix + lines[line_index][:500])
            limited, applied_limit = _apply_limit(lines_out, head_limit, offset)
            return ToolResult(
                {
                    "mode": "content",
                    "numFiles": 0,
                    "filenames": [],
                    "content": "\n".join(limited),
                    "numLines": len(limited),
                    "appliedLimit": applied_limit,
                    "appliedOffset": offset if offset else None,
                }
            )

        if output_mode == "count":
            rows: list[str] = []
            total = 0
            for path in files:
                text = _read_search_text(path)
                if text is None:
                    continue
                count = len(regex.findall(text))
                if count:
                    total += count
                    rows.append(f"{_relative(path, context.config.cwd)}:{count}")
            limited, applied_limit = _apply_limit(rows, head_limit, offset)
            return ToolResult(
                {
                    "mode": "count",
                    "numFiles": len(limited),
                    "filenames": [],
                    "content": "\n".join(limited),
                    "numMatches": total,
                    "appliedLimit": applied_limit,
                    "appliedOffset": offset if offset else None,
                }
            )

        filenames: list[str] = []
        for path in files:
            text = _read_search_text(path)
            if text is not None and regex.search(text):
                filenames.append(_relative(path, context.config.cwd))
        limited, applied_limit = _apply_limit(filenames, head_limit, offset)
        return ToolResult(
            {
                "mode": "files_with_matches",
                "numFiles": len(limited),
                "filenames": limited,
                "appliedLimit": applied_limit,
                "appliedOffset": offset if offset else None,
            }
        )

    def map_tool_result_to_tool_result_block_param(self, content: dict, tool_use_id: str) -> ToolResultBlock:
        """把工具内部结果映射为 Anthropic tool_result block。"""
        mode = content.get("mode", "files_with_matches")
        limit_info = _format_limit(content.get("appliedLimit"), content.get("appliedOffset"))
        if mode == "content":
            result = content.get("content") or "No matches found"
            if limit_info:
                result += f"\n\n[Showing results with pagination = {limit_info}]"
            return {"tool_use_id": tool_use_id, "type": "tool_result", "content": result}
        if mode == "count":
            raw = content.get("content") or "No matches found"
            matches = int(content.get("numMatches") or 0)
            files = int(content.get("numFiles") or 0)
            summary = f"\n\nFound {matches} total {'occurrence' if matches == 1 else 'occurrences'} across {files} {'file' if files == 1 else 'files'}."
            if limit_info:
                summary = summary[:-1] + f" with pagination = {limit_info}."
            return {"tool_use_id": tool_use_id, "type": "tool_result", "content": raw + summary}
        filenames = content.get("filenames", [])
        if not filenames:
            return {"tool_use_id": tool_use_id, "type": "tool_result", "content": "No files found"}
        result = f"Found {content.get('numFiles', len(filenames))} {'file' if len(filenames) == 1 else 'files'}"
        if limit_info:
            result += f" {limit_info}"
        result += "\n" + "\n".join(filenames)
        return {"tool_use_id": tool_use_id, "type": "tool_result", "content": result}


class LSTool(Tool):
    """返回单层目录项及基础类型信息。"""
    name = "LS"
    aliases = ("List",)
    search_hint = "list directory contents"
    input_schema = {"path": str, "ignore": list}
    required_fields = ("path",)

    async def description(self, input: dict | None = None) -> str:
        """返回提供给模型和调用方展示的工具简短说明。"""
        return LS_DESCRIPTION

    async def prompt(self) -> str:
        """返回该工具按源码保留的模型使用提示词。"""
        return LS_DESCRIPTION

    def is_concurrency_safe(self, input: dict) -> bool:
        """判断当前输入是否可以与相邻安全工具并发执行。"""
        return True

    def is_read_only(self, input: dict) -> bool:
        """判断当前输入是否只读取外部状态而不产生修改。"""
        return True

    def get_path(self, input: dict) -> str:
        """从工具输入中提取用于权限检查的目标路径。"""
        return str(input.get("path") or ".")

    async def validate_input(self, input: dict, context: ToolUseContext) -> ValidationResult:
        """执行依赖上下文的业务输入校验。"""
        path = resolve_for_permission(input["path"], context.config.cwd)
        if not path.exists():
            return ValidationResult(False, f"Path does not exist: {input['path']}", 1)
        if not path.is_dir():
            return ValidationResult(False, f"Path is not a directory: {input['path']}", 2)
        return ValidationResult(True)

    async def check_permissions(self, input: dict, context: ToolUseContext) -> PermissionDecision:
        """返回该工具针对当前输入的初步权限建议。"""
        path_result = validate_path_for_operation(
            path=input["path"],
            cwd=context.config.cwd,
            permission_context=context.get_app_state().tool_permission_context,
            operation_type="read",
            tool_name=self.name,
        )
        if path_result.decision is not None:
            return path_result.decision
        return PermissionDecision.allow() if path_result.allowed else PermissionDecision.ask("LS requires approval.")

    async def call(self, args: dict, context: ToolUseContext, can_use_tool, parent_message: AssistantMessage, on_progress=None) -> ToolResult:
        """执行工具核心逻辑，并返回标准 ToolResult。"""
        path = resolve_for_permission(args["path"], context.config.cwd)
        ignore = set(args.get("ignore") or [])
        entries = []
        for child in sorted(path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
            rel = _relative(child, context.config.cwd)
            if any(fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(child.name, pattern) for pattern in ignore):
                continue
            entries.append({"name": child.name + ("/" if child.is_dir() else ""), "path": rel, "type": "directory" if child.is_dir() else "file"})
        return ToolResult({"path": _relative(path, context.config.cwd), "entries": entries})

    def map_tool_result_to_tool_result_block_param(self, content: dict, tool_use_id: str) -> ToolResultBlock:
        """把工具内部结果映射为 Anthropic tool_result block。"""
        entries = content.get("entries", [])
        if not entries:
            return {"tool_use_id": tool_use_id, "type": "tool_result", "content": f"{content.get('path', '.')}/ is empty"}
        return {"tool_use_id": tool_use_id, "type": "tool_result", "content": "\n".join(entry["name"] for entry in entries)}
