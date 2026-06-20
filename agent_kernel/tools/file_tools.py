"""Read、Write、Edit、MultiEdit 和 NotebookEdit 的文件语义实现。

FileRead 输入 file_path/offset/limit。它识别文本编码和 LF/CRLF，文本结果带行号；图片
返回 base64 source 与尺寸；PDF 尝试提取文本；ipynb 转成 cell 可读视图；未知二进制
返回明确提示。成功读取会写入 ``context.read_file_state``，记录内容、mtime 与 partial
范围，供后续编辑校验和 compact 后恢复。

FileWrite 完整创建/覆盖文件；Edit 做唯一 old_string 替换并兼容直/弯引号、尾换行和
原换行风格；MultiEdit 先在内存顺序应用全部修改，任何一步失败都不写盘；NotebookEdit
按 cell id/index 执行 replace/insert/delete。所有写工具都走共享路径权限并返回
structured patch，成功后更新 ReadFileStateEntry。

关键不变量是 read-before-write 与 mtime/content 一致性：模型不能根据旧视图覆盖用户
刚刚修改的文件。该模块不决定 ask/bypass，权限建议交给 permissions resolver。
"""

from __future__ import annotations

import base64
import difflib
import json
import mimetypes
import re
from pathlib import Path
from uuid import uuid4

from ..messages import AssistantMessage, ToolResultBlock
from ..path_utils import file_mtime_ms
from ..path_validation import resolve_for_permission, validate_path_for_operation
from ..permissions import PermissionDecision
from .base import ReadFileStateEntry, Tool, ToolResult, ToolUseContext, ValidationResult, ensure_absolute
from .prompts import CYBER_RISK_MITIGATION_REMINDER, FILE_UNCHANGED_STUB, MAX_LINES_TO_READ, edit_tool_prompt, read_tool_prompt, write_tool_prompt


def add_line_numbers(content: str, start_line: int) -> str:
    """添加行 numbers，供文件工具流程使用。"""
    if not content:
        return ""
    lines = content.splitlines()
    return "\n".join(f"{index + start_line}\t{line}" for index, line in enumerate(lines))


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
MAX_EDIT_FILE_SIZE = 10 * 1024 * 1024
NOTEBOOK_EDIT_DESCRIPTION = "Replace the contents of a specific cell in a Jupyter notebook."
NOTEBOOK_EDIT_PROMPT = "Completely replaces the contents of a specific cell in a Jupyter notebook (.ipynb file) with new source. Jupyter notebooks are interactive documents that combine code, text, and visualizations, commonly used for data analysis and scientific computing. The notebook_path parameter must be an absolute path, not a relative path. The cell_number is 0-indexed. Use edit_mode=insert to add a new cell at the index specified by cell_number. Use edit_mode=delete to delete the cell at the index specified by cell_number."

LEFT_DOUBLE_CURLY_QUOTE = "\u201c"
RIGHT_DOUBLE_CURLY_QUOTE = "\u201d"
LEFT_SINGLE_CURLY_QUOTE = "\u2018"
RIGHT_SINGLE_CURLY_QUOTE = "\u2019"
QUOTE_NORMALIZATION = str.maketrans(
    {
        LEFT_DOUBLE_CURLY_QUOTE: '"',
        RIGHT_DOUBLE_CURLY_QUOTE: '"',
        LEFT_SINGLE_CURLY_QUOTE: "'",
        RIGHT_SINGLE_CURLY_QUOTE: "'",
    }
)


def _looks_binary(data: bytes) -> bool:
    """完成 ``_looks_binary`` 对应的文件工具内部步骤。"""
    return b"\x00" in data[:4096]


def _decode_text(raw: bytes) -> tuple[str, str]:
    """完成 ``_decode_text`` 对应的文件工具内部步骤。"""
    if raw.startswith(b"\xff\xfe"):
        return raw.decode("utf-16le"), "utf-16le"
    if raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16be"), "utf-16be"
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig"), "utf-8-sig"
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace"), "utf-8"


def _read_text_with_metadata(path: Path) -> tuple[str, str, str]:
    """读取文本并同时识别编码与 LF/CRLF，供写回时保真。"""
    raw = path.read_bytes()
    text, encoding = _decode_text(raw)
    line_endings = "CRLF" if "\r\n" in text else "LF"
    return text.replace("\r\n", "\n"), encoding, line_endings


def _write_text_with_metadata(path: Path, content: str, encoding: str = "utf-8", line_endings: str = "LF") -> None:
    """写入文本 with metadata，供文件工具流程使用。"""
    output = content.replace("\n", "\r\n") if line_endings == "CRLF" else content
    path.write_text(output, encoding=encoding)


def _normalize_quotes(value: str) -> str:
    """规范化quotes，供文件工具流程使用。"""
    return value.translate(QUOTE_NORMALIZATION)


def _find_actual_string(file_content: str, search_string: str) -> str | None:
    """查找actual string，供文件工具流程使用。"""
    if search_string in file_content:
        return search_string
    normalized_file = _normalize_quotes(file_content)
    normalized_search = _normalize_quotes(search_string)
    index = normalized_file.find(normalized_search)
    if index == -1:
        return None
    return file_content[index : index + len(search_string)]


def _is_opening_quote_context(chars: list[str], index: int) -> bool:
    """判断opening quote 上下文，供文件工具流程使用。"""
    if index == 0:
        return True
    return chars[index - 1] in {" ", "\t", "\n", "\r", "(", "[", "{", "\u2014", "\u2013"}


def _apply_curly_double_quotes(value: str) -> str:
    """应用curly double quotes，供文件工具流程使用。"""
    chars = list(value)
    result: list[str] = []
    for index, char in enumerate(chars):
        if char == '"':
            result.append(LEFT_DOUBLE_CURLY_QUOTE if _is_opening_quote_context(chars, index) else RIGHT_DOUBLE_CURLY_QUOTE)
        else:
            result.append(char)
    return "".join(result)


def _apply_curly_single_quotes(value: str) -> str:
    """应用curly single quotes，供文件工具流程使用。"""
    chars = list(value)
    result: list[str] = []
    for index, char in enumerate(chars):
        if char != "'":
            result.append(char)
            continue
        previous_char = chars[index - 1] if index > 0 else ""
        next_char = chars[index + 1] if index < len(chars) - 1 else ""
        if previous_char.isalpha() and next_char.isalpha():
            result.append(RIGHT_SINGLE_CURLY_QUOTE)
        else:
            result.append(LEFT_SINGLE_CURLY_QUOTE if _is_opening_quote_context(chars, index) else RIGHT_SINGLE_CURLY_QUOTE)
    return "".join(result)


def _preserve_quote_style(old_string: str, actual_old_string: str, new_string: str) -> str:
    """完成 ``_preserve_quote_style`` 对应的文件工具内部步骤。"""
    if old_string == actual_old_string:
        return new_string
    result = new_string
    if LEFT_DOUBLE_CURLY_QUOTE in actual_old_string or RIGHT_DOUBLE_CURLY_QUOTE in actual_old_string:
        result = _apply_curly_double_quotes(result)
    if LEFT_SINGLE_CURLY_QUOTE in actual_old_string or RIGHT_SINGLE_CURLY_QUOTE in actual_old_string:
        result = _apply_curly_single_quotes(result)
    return result


def _apply_edit_to_file(original: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """应用精确替换；默认要求 old_string 唯一，降低误编辑风险。"""
    if old_string == "":
        return new_string
    if new_string != "":
        return original.replace(old_string, new_string) if replace_all else original.replace(old_string, new_string, 1)
    strip_trailing_newline = not old_string.endswith("\n") and (old_string + "\n") in original
    search = old_string + "\n" if strip_trailing_newline else old_string
    return original.replace(search, new_string) if replace_all else original.replace(search, new_string, 1)


def _structured_patch(old: str | None, new: str, file_path: str) -> list[dict]:
    """完成 ``_structured_patch`` 对应的文件工具内部步骤。"""
    old_lines = [] if old is None else old.splitlines()
    new_lines = new.splitlines()
    patch = "\n".join(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm="",
        )
    )
    return [
        {
            "oldStart": 1,
            "oldLines": old_lines,
            "newStart": 1,
            "newLines": new_lines,
            "patch": patch,
        }
    ]


def _image_dimensions(raw: bytes, suffix: str) -> dict[str, int] | None:
    """完成 ``_image_dimensions`` 对应的文件工具内部步骤。"""
    if suffix == ".png" and len(raw) >= 24 and raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return {"width": int.from_bytes(raw[16:20], "big"), "height": int.from_bytes(raw[20:24], "big")}
    if suffix in {".jpg", ".jpeg"}:
        index = 2
        while index + 9 < len(raw):
            if raw[index] != 0xFF:
                index += 1
                continue
            marker = raw[index + 1]
            length = int.from_bytes(raw[index + 2 : index + 4], "big")
            if marker in {0xC0, 0xC2} and index + 8 < len(raw):
                return {
                    "height": int.from_bytes(raw[index + 5 : index + 7], "big"),
                    "width": int.from_bytes(raw[index + 7 : index + 9], "big"),
                }
            index += 2 + max(length, 1)
    return None


def _extract_pdf_text(raw: bytes, path: Path) -> str:
    """提取PDF 文本，供文件工具流程使用。"""
    try:
        try:
            from pypdf import PdfReader  # type: ignore
        except Exception:
            from PyPDF2 import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        extracted = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
        if extracted:
            return extracted
    except Exception:
        pass
    text = raw.decode("latin-1", errors="ignore")
    snippets = re.findall(r"\(([^()]{1,200})\)", text)
    cleaned = "\n".join(" ".join(snippet.split()) for snippet in snippets if snippet.strip())
    return cleaned[:20_000]


def _notebook_text(raw: str) -> str:
    """完成 ``_notebook_text`` 对应的文件工具内部步骤。"""
    notebook = json.loads(raw)
    lines: list[str] = []
    for index, cell in enumerate(notebook.get("cells", []), start=1):
        cell_type = cell.get("cell_type", "cell")
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        lines.append(f"# Cell {index} ({cell_type})")
        lines.append(str(source).rstrip())
        outputs = cell.get("outputs") or []
        if outputs:
            lines.append("# Outputs")
            for output in outputs:
                data = output.get("data", {}) if isinstance(output.get("data", {}), dict) else {}
                text = output.get("text") or output.get("ename") or data.get("text/plain") or data.get("text/markdown")
                if isinstance(text, list):
                    text = "".join(text)
                if text:
                    lines.append(str(text).rstrip())
                image_outputs = [key for key in data if isinstance(key, str) and key.startswith("image/")]
                if image_outputs:
                    lines.append(f"[Output includes {', '.join(image_outputs)} data]")
        lines.append("")
    return "\n".join(lines).strip()


class FileReadTool(Tool):
    """只读文件工具，同时建立后续 Edit 使用的 ReadFileStateEntry。"""
    name = "Read"
    search_hint = "read files, images, PDFs, notebooks"
    max_result_size_chars = 2**63 - 1
    input_schema = {"file_path": str, "offset": int, "limit": int, "pages": str}
    required_fields = ("file_path",)

    async def description(self, input: dict | None = None) -> str:
        """返回提供给模型和调用方展示的工具简短说明。"""
        return "Read a file from the local filesystem."

    async def prompt(self) -> str:
        """返回该工具按源码保留的模型使用提示词。"""
        return read_tool_prompt()

    def is_concurrency_safe(self, input: dict) -> bool:
        """判断当前输入是否可以与相邻安全工具并发执行。"""
        return True

    def is_read_only(self, input: dict) -> bool:
        """判断当前输入是否只读取外部状态而不产生修改。"""
        return True

    def get_path(self, input: dict) -> str:
        """从工具输入中提取用于权限检查的目标路径。"""
        return str(ensure_absolute(input.get("file_path", ".")))

    async def check_permissions(self, input: dict, context: ToolUseContext) -> PermissionDecision:
        """返回该工具针对当前输入的初步权限建议。"""
        path_result = validate_path_for_operation(
            path=input["file_path"],
            cwd=context.config.cwd,
            permission_context=context.get_app_state().tool_permission_context,
            operation_type="read",
            tool_name=self.name,
        )
        if path_result.decision is not None:
            return path_result.decision
        if path_result.allowed:
            return PermissionDecision.allow()
        return PermissionDecision.ask("File read requires approval.")

    async def validate_input(self, input: dict, context: ToolUseContext) -> ValidationResult:
        """执行依赖上下文的业务输入校验。"""
        file_path = input.get("file_path")
        if not isinstance(file_path, str) or not file_path:
            return ValidationResult(False, "file_path is required.", 1)
        return ValidationResult(True)

    async def call(self, args: dict, context: ToolUseContext, can_use_tool, parent_message: AssistantMessage, on_progress=None) -> ToolResult:
        """按媒体类型分派读取逻辑，并在文本路径上支持 offset/limit。"""
        full_path = resolve_for_permission(args["file_path"], context.config.cwd)
        if on_progress:
            on_progress({"message": f"Reading {full_path}"})
        offset = int(args.get("offset") or 1)
        limit = args.get("limit")
        limit = int(limit) if limit is not None else None
        if full_path.is_dir():
            raise IsADirectoryError("This tool can only read files, not directories.")
        existing_state = context.read_file_state.get(str(full_path))
        if (
            existing_state
            and not existing_state.is_partial_view
            and existing_state.offset is not None
            and existing_state.offset == offset
            and existing_state.limit == limit
            and file_mtime_ms(full_path) == existing_state.timestamp
        ):
            return ToolResult({"type": "file_unchanged", "file": {"filePath": args["file_path"]}})
        raw_bytes = full_path.read_bytes()
        suffix = full_path.suffix.casefold()
        mime_type = mimetypes.guess_type(full_path.name)[0] or "application/octet-stream"
        if suffix in IMAGE_EXTENSIONS:
            data = base64.b64encode(raw_bytes).decode("ascii") if len(raw_bytes) <= 5_000_000 else None
            dimensions = _image_dimensions(raw_bytes, suffix)
            return ToolResult(
                {
                    "type": "image",
                    "file": {
                        "filePath": args["file_path"],
                        "mimeType": mime_type,
                        "sizeBytes": len(raw_bytes),
                        "base64": data,
                        "dimensions": dimensions,
                    },
                }
            )
        if suffix == ".pdf":
            pdf_text = _extract_pdf_text(raw_bytes, full_path)
            return ToolResult(
                {
                    "type": "pdf",
                    "file": {
                        "filePath": args["file_path"],
                        "mimeType": mime_type,
                        "sizeBytes": len(raw_bytes),
                        "content": pdf_text,
                    },
                }
            )
        if _looks_binary(raw_bytes):
            return ToolResult(
                {
                    "type": "binary",
                    "file": {
                        "filePath": args["file_path"],
                        "mimeType": mime_type,
                        "sizeBytes": len(raw_bytes),
                    },
                }
            )
        content, _ = _decode_text(raw_bytes)
        content = content.replace("\r\n", "\n")
        if suffix == ".ipynb":
            content = _notebook_text(content)
        all_lines = content.splitlines()
        start_index = max(offset - 1, 0)
        selected = all_lines[start_index : start_index + (limit or MAX_LINES_TO_READ)]
        selected_content = "\n".join(selected)
        is_partial_view = start_index > 0 or (start_index + len(selected)) < len(all_lines)
        context.read_file_state[str(full_path)] = ReadFileStateEntry(
            content=content,
            timestamp=file_mtime_ms(full_path),
            offset=offset,
            limit=limit,
            is_partial_view=is_partial_view,
        )
        return ToolResult(
            {
                "type": "text",
                "file": {
                    "filePath": args["file_path"],
                    "content": selected_content,
                    "numLines": len(selected),
                    "startLine": offset,
                    "totalLines": len(all_lines),
                },
            }
        )

    def map_tool_result_to_tool_result_block_param(self, content: dict, tool_use_id: str) -> ToolResultBlock:
        """把工具内部结果映射为 Anthropic tool_result block。"""
        if content["type"] == "file_unchanged":
            return {"tool_use_id": tool_use_id, "type": "tool_result", "content": FILE_UNCHANGED_STUB}
        if content["type"] == "image":
            file = content["file"]
            description = f"Image file: {file['filePath']} ({file['mimeType']}, {file['sizeBytes']} bytes)"
            if file.get("dimensions"):
                description += f", {file['dimensions']['width']}x{file['dimensions']['height']}"
            if file.get("base64"):
                return {
                    "tool_use_id": tool_use_id,
                    "type": "tool_result",
                    "content": [
                        {"type": "text", "text": description},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": file["mimeType"],
                                "data": file["base64"],
                            },
                        },
                    ],
            }
            return {"tool_use_id": tool_use_id, "type": "tool_result", "content": description}
        if content["type"] == "pdf":
            file = content["file"]
            text = f"\nExtracted text:\n{file['content']}" if file.get("content") else "\nNo extractable text found in this PDF."
            return {"tool_use_id": tool_use_id, "type": "tool_result", "content": f"PDF file read: {file['filePath']} ({file['sizeBytes']} bytes){text}"}
        if content["type"] in {"binary", "document"}:
            file = content["file"]
            extra = f"\n{file.get('content')}" if file.get("content") else ""
            return {"tool_use_id": tool_use_id, "type": "tool_result", "content": f"{content['type'].title()} file: {file['filePath']} ({file['mimeType']}, {file['sizeBytes']} bytes){extra}"}
        file = content["file"]
        if file["content"]:
            result = add_line_numbers(file["content"], file["startLine"]) + CYBER_RISK_MITIGATION_REMINDER
            end_line = file["startLine"] + file["numLines"] - 1
            if end_line < file["totalLines"]:
                result += f"\n\n<system-reminder>File has more lines after line {end_line}. Use offset and limit to read more.</system-reminder>"
        elif file["totalLines"] == 0:
            result = "<system-reminder>Warning: the file exists but the contents are empty.</system-reminder>"
        else:
            result = f"<system-reminder>Warning: the file exists but is shorter than the provided offset ({file['startLine']}). The file has {file['totalLines']} lines.</system-reminder>"
        return {"tool_use_id": tool_use_id, "type": "tool_result", "content": result}


class FileWriteTool(Tool):
    """创建或完全覆盖文件，返回统一 diff/patch 元数据。"""
    name = "Write"
    search_hint = "create or overwrite files"
    input_schema = {"file_path": str, "content": str}
    required_fields = ("file_path", "content")

    async def description(self, input: dict | None = None) -> str:
        """返回提供给模型和调用方展示的工具简短说明。"""
        return "Write a file to the local filesystem."

    async def prompt(self) -> str:
        """返回该工具按源码保留的模型使用提示词。"""
        return write_tool_prompt()

    def get_path(self, input: dict) -> str:
        """从工具输入中提取用于权限检查的目标路径。"""
        return str(ensure_absolute(input.get("file_path", ".")))

    def is_destructive(self, input: dict) -> bool:
        """判断当前输入是否可能产生破坏性副作用。"""
        return True

    async def check_permissions(self, input: dict, context: ToolUseContext) -> PermissionDecision:
        """返回该工具针对当前输入的初步权限建议。"""
        full_path = resolve_for_permission(input["file_path"], context.config.cwd)
        operation_type = "write" if full_path.exists() else "create"
        path_result = validate_path_for_operation(
            path=input["file_path"],
            cwd=context.config.cwd,
            permission_context=context.get_app_state().tool_permission_context,
            operation_type=operation_type,
            tool_name=self.name,
        )
        if path_result.decision is not None:
            return path_result.decision
        return PermissionDecision.ask("File write requires approval.")

    async def validate_input(self, input: dict, context: ToolUseContext) -> ValidationResult:
        """执行依赖上下文的业务输入校验。"""
        full_path = resolve_for_permission(input["file_path"], context.config.cwd)
        if full_path.exists():
            state = context.read_file_state.get(str(full_path))
            if not state or state.is_partial_view:
                return ValidationResult(False, "File has not been read yet. Read it first before writing to it.", 2)
            current_content, _, _ = _read_text_with_metadata(full_path)
            if file_mtime_ms(full_path) > state.timestamp and current_content != state.content:
                return ValidationResult(False, "File has been modified since read, either by the user or by a linter. Read it again before attempting to write it.", 3)
        return ValidationResult(True)

    async def call(self, args: dict, context: ToolUseContext, can_use_tool, parent_message: AssistantMessage, on_progress=None) -> ToolResult:
        """执行工具核心逻辑，并返回标准 ToolResult。"""
        full_path = resolve_for_permission(args["file_path"], context.config.cwd)
        if on_progress:
            on_progress({"message": f"Writing {full_path}"})
        old_content = _read_text_with_metadata(full_path)[0] if full_path.exists() else None
        full_path.parent.mkdir(parents=True, exist_ok=True)
        _write_text_with_metadata(full_path, args["content"])
        context.read_file_state[str(full_path)] = ReadFileStateEntry(
            content=args["content"],
            timestamp=file_mtime_ms(full_path),
        )
        return ToolResult(
            {
                "type": "update" if old_content is not None else "create",
                "filePath": args["file_path"],
                "content": args["content"],
                "originalFile": old_content,
                "structuredPatch": _structured_patch(old_content, args["content"], args["file_path"]),
            }
        )

    def map_tool_result_to_tool_result_block_param(self, content: dict, tool_use_id: str) -> ToolResultBlock:
        """把工具内部结果映射为 Anthropic tool_result block。"""
        if content["type"] == "create":
            return {"tool_use_id": tool_use_id, "type": "tool_result", "content": f"File created successfully at: {content['filePath']}"}
        return {"tool_use_id": tool_use_id, "type": "tool_result", "content": f"The file {content['filePath']} has been updated successfully."}


class EditTool(Tool):
    """对已读取文件执行单个精确字符串替换。"""
    name = "Edit"
    search_hint = "modify file contents in place"
    input_schema = {"file_path": str, "old_string": str, "new_string": str, "replace_all": bool}
    required_fields = ("file_path", "old_string", "new_string")

    async def description(self, input: dict | None = None) -> str:
        """返回提供给模型和调用方展示的工具简短说明。"""
        return "A tool for editing files"

    async def prompt(self) -> str:
        """返回该工具按源码保留的模型使用提示词。"""
        return edit_tool_prompt()

    def get_path(self, input: dict) -> str:
        """从工具输入中提取用于权限检查的目标路径。"""
        return str(ensure_absolute(input.get("file_path", ".")))

    def is_destructive(self, input: dict) -> bool:
        """判断当前输入是否可能产生破坏性副作用。"""
        return True

    async def check_permissions(self, input: dict, context: ToolUseContext) -> PermissionDecision:
        """返回该工具针对当前输入的初步权限建议。"""
        full_path = resolve_for_permission(input["file_path"], context.config.cwd)
        operation_type = "create" if input.get("old_string") == "" and not full_path.exists() else "write"
        path_result = validate_path_for_operation(
            path=input["file_path"],
            cwd=context.config.cwd,
            permission_context=context.get_app_state().tool_permission_context,
            operation_type=operation_type,
            tool_name=self.name,
        )
        if path_result.decision is not None:
            return path_result.decision
        return PermissionDecision.ask("File edit requires approval.")

    async def validate_input(self, input: dict, context: ToolUseContext) -> ValidationResult:
        """执行依赖上下文的业务输入校验。"""
        if input["old_string"] == input["new_string"]:
            return ValidationResult(False, "No changes to make: old_string and new_string are exactly the same.", 1)
        full_path = resolve_for_permission(input["file_path"], context.config.cwd)
        if full_path.exists() and full_path.stat().st_size > MAX_EDIT_FILE_SIZE:
            return ValidationResult(False, f"File is too large to edit ({full_path.stat().st_size} bytes). Maximum editable file size is {MAX_EDIT_FILE_SIZE} bytes.", 10)
        if not full_path.exists():
            if input["old_string"] == "":
                return ValidationResult(True)
            return ValidationResult(False, f"File does not exist. Current working directory: {context.config.cwd}.", 4)
        if full_path.suffix.casefold() == ".ipynb":
            return ValidationResult(False, "File is a Jupyter Notebook. Use the NotebookEdit tool to edit this file.", 5)
        content, _, _ = _read_text_with_metadata(full_path)
        if input["old_string"] == "":
            if content.strip() != "":
                return ValidationResult(False, "Cannot create new file - file already exists.", 3)
            return ValidationResult(True)
        state = context.read_file_state.get(str(full_path))
        if not state or state.is_partial_view:
            return ValidationResult(False, "File has not been read yet. Read it first before writing to it.", 6)
        if file_mtime_ms(full_path) > state.timestamp and content != state.content:
            return ValidationResult(False, "File has been modified since read, either by the user or by a linter. Read it again before attempting to write it.", 7)
        actual_old_string = _find_actual_string(content, input["old_string"])
        if actual_old_string is None:
            return ValidationResult(False, f"String to replace not found in file.\nString: {input['old_string']}", 8)
        matches = content.count(actual_old_string)
        if matches > 1 and not input.get("replace_all", False):
            return ValidationResult(False, f"Found {matches} matches of the string to replace, but replace_all is false. To replace all occurrences, set replace_all to true. To replace only one occurrence, please provide more context to uniquely identify the instance.\nString: {input['old_string']}", 9)
        return ValidationResult(True, meta={"actualOldString": actual_old_string})

    async def call(self, args: dict, context: ToolUseContext, can_use_tool, parent_message: AssistantMessage, on_progress=None) -> ToolResult:
        """执行工具核心逻辑，并返回标准 ToolResult。"""
        full_path = resolve_for_permission(args["file_path"], context.config.cwd)
        if on_progress:
            on_progress({"message": f"Editing {full_path}"})
        old_string = args["old_string"]
        new_string = args["new_string"]
        replace_all = bool(args.get("replace_all", False))
        if full_path.exists():
            original, encoding, line_endings = _read_text_with_metadata(full_path)
        else:
            original, encoding, line_endings = "", "utf-8", "LF"
        actual_old_string = _find_actual_string(original, old_string) or old_string
        actual_new_string = _preserve_quote_style(old_string, actual_old_string, new_string)
        updated = _apply_edit_to_file(original, actual_old_string, actual_new_string, replace_all)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        _write_text_with_metadata(full_path, updated, encoding=encoding, line_endings=line_endings)
        context.read_file_state[str(full_path)] = ReadFileStateEntry(
            content=updated,
            timestamp=file_mtime_ms(full_path),
        )
        return ToolResult(
            {
                "filePath": args["file_path"],
                "oldString": actual_old_string,
                "newString": new_string,
                "originalFile": original,
                "structuredPatch": _structured_patch(original, updated, args["file_path"]),
                "userModified": False,
                "replaceAll": replace_all,
            }
        )

    def map_tool_result_to_tool_result_block_param(self, content: dict, tool_use_id: str) -> ToolResultBlock:
        """把工具内部结果映射为 Anthropic tool_result block。"""
        modified_note = ".  The user modified your proposed changes before accepting them. " if content.get("userModified") else ""
        if content.get("replaceAll"):
            return {"tool_use_id": tool_use_id, "type": "tool_result", "content": f"The file {content['filePath']} has been updated{modified_note}. All occurrences were successfully replaced."}
        return {"tool_use_id": tool_use_id, "type": "tool_result", "content": f"The file {content['filePath']} has been updated successfully{modified_note}."}


class MultiEditTool(Tool):
    """在内存中顺序验证全部编辑，成功后一次写盘以保证原子性。"""
    name = "MultiEdit"
    search_hint = "make multiple edits to a single file atomically"
    input_schema = {"file_path": str, "edits": list}
    required_fields = ("file_path", "edits")

    async def description(self, input: dict | None = None) -> str:
        """返回提供给模型和调用方展示的工具简短说明。"""
        return "Apply multiple string replacements to one file atomically."

    async def prompt(self) -> str:
        """返回该工具按源码保留的模型使用提示词。"""
        return "This is a tool for making multiple edits to a single file in one operation. It is built on the same exact-match editing semantics as Edit."

    def get_path(self, input: dict) -> str:
        """从工具输入中提取用于权限检查的目标路径。"""
        return str(ensure_absolute(input.get("file_path", ".")))

    def is_destructive(self, input: dict) -> bool:
        """判断当前输入是否可能产生破坏性副作用。"""
        return True

    async def check_permissions(self, input: dict, context: ToolUseContext) -> PermissionDecision:
        """返回该工具针对当前输入的初步权限建议。"""
        path_result = validate_path_for_operation(
            path=input["file_path"],
            cwd=context.config.cwd,
            permission_context=context.get_app_state().tool_permission_context,
            operation_type="write",
            tool_name=self.name,
        )
        if path_result.decision is not None:
            return path_result.decision
        return PermissionDecision.ask("MultiEdit requires approval.")

    async def validate_input(self, input: dict, context: ToolUseContext) -> ValidationResult:
        """执行依赖上下文的业务输入校验。"""
        edits = input.get("edits")
        if not isinstance(edits, list) or not edits:
            return ValidationResult(False, "edits must be a non-empty list.", 1)
        full_path = resolve_for_permission(input["file_path"], context.config.cwd)
        if not full_path.exists():
            return ValidationResult(False, f"File does not exist. Current working directory: {context.config.cwd}.", 2)
        if full_path.suffix.casefold() == ".ipynb":
            return ValidationResult(False, "File is a Jupyter Notebook. Use the NotebookEdit tool to edit this file.", 3)
        state = context.read_file_state.get(str(full_path))
        if not state or state.is_partial_view:
            return ValidationResult(False, "File has not been read yet. Read it first before writing to it.", 4)
        content, _, _ = _read_text_with_metadata(full_path)
        if file_mtime_ms(full_path) > state.timestamp and content != state.content:
            return ValidationResult(False, "File has been modified since read, either by the user or by a linter. Read it again before attempting to write it.", 5)
        staged = content
        for index, edit in enumerate(edits):
            if not isinstance(edit, dict):
                return ValidationResult(False, f"Edit at index {index} must be an object.", 6)
            old_string = edit.get("old_string")
            new_string = edit.get("new_string")
            if not isinstance(old_string, str) or not isinstance(new_string, str):
                return ValidationResult(False, f"Edit at index {index} must include old_string and new_string.", 7)
            if old_string == new_string:
                return ValidationResult(False, f"No changes to make in edit {index}: old_string and new_string are exactly the same.", 8)
            actual = _find_actual_string(staged, old_string)
            if actual is None:
                return ValidationResult(False, f"String to replace not found in file for edit {index}.\nString: {old_string}", 9)
            matches = staged.count(actual)
            if matches > 1 and not edit.get("replace_all", False):
                return ValidationResult(False, f"Found {matches} matches of the string to replace in edit {index}, but replace_all is false.", 10)
            new_preserved = _preserve_quote_style(old_string, actual, new_string)
            staged = _apply_edit_to_file(staged, actual, new_preserved, bool(edit.get("replace_all", False)))
        return ValidationResult(True)

    async def call(self, args: dict, context: ToolUseContext, can_use_tool, parent_message: AssistantMessage, on_progress=None) -> ToolResult:
        """执行工具核心逻辑，并返回标准 ToolResult。"""
        full_path = resolve_for_permission(args["file_path"], context.config.cwd)
        if on_progress:
            on_progress({"message": f"Applying {len(args['edits'])} edits to {full_path}"})
        original, encoding, line_endings = _read_text_with_metadata(full_path)
        updated = original
        applied = []
        for edit in args["edits"]:
            old_string = edit["old_string"]
            actual = _find_actual_string(updated, old_string) or old_string
            new_preserved = _preserve_quote_style(old_string, actual, edit["new_string"])
            updated = _apply_edit_to_file(updated, actual, new_preserved, bool(edit.get("replace_all", False)))
            applied.append({"oldString": actual, "newString": edit["new_string"], "replaceAll": bool(edit.get("replace_all", False))})
        _write_text_with_metadata(full_path, updated, encoding=encoding, line_endings=line_endings)
        context.read_file_state[str(full_path)] = ReadFileStateEntry(content=updated, timestamp=file_mtime_ms(full_path))
        return ToolResult(
            {
                "filePath": args["file_path"],
                "edits": applied,
                "originalFile": original,
                "structuredPatch": _structured_patch(original, updated, args["file_path"]),
            }
        )

    def map_tool_result_to_tool_result_block_param(self, content: dict, tool_use_id: str) -> ToolResultBlock:
        """把工具内部结果映射为 Anthropic tool_result block。"""
        return {"tool_use_id": tool_use_id, "type": "tool_result", "content": f"Applied {len(content['edits'])} edits to {content['filePath']} successfully."}


def _cell_source_to_string(source) -> str:
    """完成 ``_cell_source_to_string`` 对应的文件工具内部步骤。"""
    if isinstance(source, list):
        return "".join(str(item) for item in source)
    return str(source or "")


def _set_cell_source(cell: dict, source: str) -> None:
    """设置单元格 source，供文件工具流程使用。"""
    cell["source"] = source


def _find_cell_index(cells: list[dict], cell_id: str | None, edit_mode: str) -> int | None:
    """查找单元格 index，供文件工具流程使用。"""
    if not cell_id:
        return 0 if edit_mode == "insert" else None
    for index, cell in enumerate(cells):
        if cell.get("id") == cell_id:
            return index + 1 if edit_mode == "insert" else index
    match = re.fullmatch(r"cell-(\d+)", cell_id)
    if match:
        index = int(match.group(1))
        return index + 1 if edit_mode == "insert" else index
    if cell_id.isdigit():
        index = int(cell_id)
        return index + 1 if edit_mode == "insert" else index
    return None


class NotebookEditTool(Tool):
    """按 cell id/index 对 ipynb cell 执行 replace/insert/delete。"""
    name = "NotebookEdit"
    search_hint = "edit Jupyter notebook cells (.ipynb)"
    max_result_size_chars = 100_000
    input_schema = {"notebook_path": str, "cell_id": str, "new_source": str, "cell_type": str, "edit_mode": str}
    required_fields = ("notebook_path", "new_source")

    async def description(self, input: dict | None = None) -> str:
        """返回提供给模型和调用方展示的工具简短说明。"""
        return NOTEBOOK_EDIT_DESCRIPTION

    async def prompt(self) -> str:
        """返回该工具按源码保留的模型使用提示词。"""
        return NOTEBOOK_EDIT_PROMPT

    def get_path(self, input: dict) -> str:
        """从工具输入中提取用于权限检查的目标路径。"""
        return str(input.get("notebook_path", ""))

    def is_destructive(self, input: dict) -> bool:
        """判断当前输入是否可能产生破坏性副作用。"""
        return True

    async def check_permissions(self, input: dict, context: ToolUseContext) -> PermissionDecision:
        """返回该工具针对当前输入的初步权限建议。"""
        path_result = validate_path_for_operation(
            path=input["notebook_path"],
            cwd=context.config.cwd,
            permission_context=context.get_app_state().tool_permission_context,
            operation_type="write",
            tool_name=self.name,
        )
        if path_result.decision is not None:
            return path_result.decision
        return PermissionDecision.ask("Notebook edit requires approval.")

    async def validate_input(self, input: dict, context: ToolUseContext) -> ValidationResult:
        """执行依赖上下文的业务输入校验。"""
        full_path = resolve_for_permission(input["notebook_path"], context.config.cwd)
        edit_mode = input.get("edit_mode") or "replace"
        if full_path.suffix.casefold() != ".ipynb":
            return ValidationResult(False, "File must be a Jupyter notebook (.ipynb file). For editing other file types, use the Edit tool.", 2)
        if edit_mode not in {"replace", "insert", "delete"}:
            return ValidationResult(False, "Edit mode must be replace, insert, or delete.", 4)
        if edit_mode == "insert" and input.get("cell_type") not in {"code", "markdown"}:
            return ValidationResult(False, "Cell type is required when using edit_mode=insert.", 5)
        state = context.read_file_state.get(str(full_path))
        if not state:
            return ValidationResult(False, "File has not been read yet. Read it first before writing to it.", 9)
        if not full_path.exists():
            return ValidationResult(False, "Notebook file does not exist.", 1)
        if file_mtime_ms(full_path) > state.timestamp:
            return ValidationResult(False, "File has been modified since read, either by the user or by a linter. Read it again before attempting to write it.", 10)
        try:
            notebook = json.loads(full_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return ValidationResult(False, "Notebook is not valid JSON.", 6)
        cells = notebook.get("cells")
        if not isinstance(cells, list):
            return ValidationResult(False, "Notebook is not valid JSON.", 6)
        cell_index = _find_cell_index(cells, input.get("cell_id"), edit_mode)
        if cell_index is None:
            return ValidationResult(False, "Cell ID must be specified when not inserting a new cell.", 7)
        if edit_mode != "insert" and not (0 <= cell_index < len(cells)):
            return ValidationResult(False, f"Cell with ID \"{input.get('cell_id')}\" not found in notebook.", 8)
        if edit_mode == "insert" and not (0 <= cell_index <= len(cells)):
            return ValidationResult(False, f"Cell with ID \"{input.get('cell_id')}\" not found in notebook.", 8)
        return ValidationResult(True)

    async def call(self, args: dict, context: ToolUseContext, can_use_tool, parent_message: AssistantMessage, on_progress=None) -> ToolResult:
        """执行工具核心逻辑，并返回标准 ToolResult。"""
        full_path = resolve_for_permission(args["notebook_path"], context.config.cwd)
        edit_mode = args.get("edit_mode") or "replace"
        if on_progress:
            on_progress({"message": f"Editing notebook {full_path}"})
        original = full_path.read_text(encoding="utf-8")
        notebook = json.loads(original)
        cells = notebook.setdefault("cells", [])
        cell_index = _find_cell_index(cells, args.get("cell_id"), edit_mode)
        language = ((notebook.get("metadata") or {}).get("language_info") or {}).get("name") or "python"
        new_cell_id = args.get("cell_id")
        cell_type = args.get("cell_type")
        if edit_mode == "replace" and cell_index == len(cells):
            edit_mode = "insert"
            cell_type = cell_type or "code"
        if edit_mode == "delete":
            cells.pop(cell_index)
        elif edit_mode == "insert":
            new_cell_id = uuid4().hex[:12] if notebook.get("nbformat", 4) >= 4 else None
            if cell_type == "markdown":
                cell = {"cell_type": "markdown", "metadata": {}, "source": args["new_source"]}
            else:
                cell = {"cell_type": "code", "metadata": {}, "source": args["new_source"], "execution_count": None, "outputs": []}
            if new_cell_id:
                cell["id"] = new_cell_id
            cells.insert(cell_index, cell)
        else:
            target = cells[cell_index]
            _set_cell_source(target, args["new_source"])
            if target.get("cell_type") == "code":
                target["execution_count"] = None
                target["outputs"] = []
            if cell_type in {"code", "markdown"}:
                target["cell_type"] = cell_type
            cell_type = target.get("cell_type", cell_type or "code")
        updated = json.dumps(notebook, ensure_ascii=False, indent=1)
        full_path.write_text(updated, encoding="utf-8")
        context.read_file_state[str(full_path)] = ReadFileStateEntry(content=updated, timestamp=file_mtime_ms(full_path))
        return ToolResult(
            {
                "new_source": args["new_source"],
                "cell_id": new_cell_id,
                "cell_type": cell_type or "code",
                "language": language,
                "edit_mode": edit_mode,
                "error": "",
                "notebook_path": str(full_path),
                "original_file": original,
                "updated_file": updated,
            }
        )

    def map_tool_result_to_tool_result_block_param(self, content: dict, tool_use_id: str) -> ToolResultBlock:
        """把工具内部结果映射为 Anthropic tool_result block。"""
        if content.get("error"):
            return {"tool_use_id": tool_use_id, "type": "tool_result", "content": content["error"], "is_error": True}
        if content["edit_mode"] == "replace":
            return {"tool_use_id": tool_use_id, "type": "tool_result", "content": f"Updated cell {content.get('cell_id')} with {content['new_source']}"}
        if content["edit_mode"] == "insert":
            return {"tool_use_id": tool_use_id, "type": "tool_result", "content": f"Inserted cell {content.get('cell_id')} with {content['new_source']}"}
        if content["edit_mode"] == "delete":
            return {"tool_use_id": tool_use_id, "type": "tool_result", "content": f"Deleted cell {content.get('cell_id')}"}
        return {"tool_use_id": tool_use_id, "type": "tool_result", "content": "Unknown edit mode"}
