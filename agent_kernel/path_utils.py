"""无策略的路径基础工具：展开、sanitize、git root 与 mtime。

``expand_path`` 处理 ``~``、环境变量和相对路径；``sanitize_path`` 把绝对项目路径编码
为 Claude Code projects 目录使用的稳定键；``find_git_root`` 从给定目录向上寻找
``.git``。``file_mtime_ms`` 为 Read/Edit 和 compact 文件恢复提供毫秒时间戳。

本模块只做机械转换，不判断路径能否读写，也不处理敏感文件。所有安全策略集中在
``path_validation.py``，避免普通路径 helper 在不同工具中产生不一致权限语义。
"""

from __future__ import annotations

from pathlib import Path
import os
import re


def expand_path(path: str | Path) -> Path:
    """完成 ``expand_path`` 对应的路径处理内部步骤。"""
    return Path(path).expanduser().resolve()


def sanitize_path(path: str | Path) -> str:
    """把绝对路径变为 Claude Code projects 目录使用的稳定键。"""
    raw = str(Path(path).expanduser())
    raw = raw.replace("\\", "/")
    raw = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-")
    return raw or "root"


def find_git_root(start: Path) -> Path | None:
    """查找git root，供路径处理流程使用。"""
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists():
            return candidate
    return None


def is_git_repo(path: Path) -> bool:
    """判断git repo，供路径处理流程使用。"""
    return find_git_root(path) is not None


def file_mtime_ms(path: Path) -> int:
    """完成 ``file_mtime_ms`` 对应的路径处理内部步骤。"""
    return int(os.stat(path).st_mtime * 1000)
