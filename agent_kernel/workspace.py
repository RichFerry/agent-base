"""Workspace runtime boundary discovery and diagnostics.

The runtime is intentionally descriptive: it tells the kernel and local runner
where the current workspace starts, which persistent directories belong to that
workspace, and which configured extension paths are project/user/explicit
scoped. It does not load settings, start MCP servers, or decide permissions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from .path_utils import find_git_root, sanitize_path


WorkspaceScope = Literal["project", "user", "explicit", "disabled", "unknown"]
SETTINGS_FILE_NAME = "settings.json"


@dataclass(frozen=True)
class WorkspacePathSource:
    """A path plus the runtime scope it belongs to."""

    kind: str
    path: Path
    scope: WorkspaceScope
    source: str | None = None

    def as_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "path": str(self.path),
            "scope": self.scope,
            "source": self.source,
            "exists": self.path.exists(),
        }


@dataclass(frozen=True)
class WorkspaceRuntime:
    """Resolved workspace facts shared by sessions, memory, artifacts, and CLI diagnostics."""

    cwd: Path
    workspace_root: Path
    workspace_root_source: str
    config_home: Path
    project_store_dir: Path
    sessions_dir: Path
    memory_dir: Path | None
    memory_scope: WorkspaceScope
    artifacts_dir: Path
    bash_output_dir: Path
    agent_output_dir: Path
    allowed_working_directories: tuple[Path, ...]
    settings_sources: tuple[WorkspacePathSource, ...] = ()
    skill_sources: tuple[WorkspacePathSource, ...] = ()
    mcp_sources: tuple[WorkspacePathSource, ...] = ()

    @property
    def transcript_dir(self) -> Path:
        return self.sessions_dir

    def as_json(self) -> dict[str, Any]:
        return {
            "cwd": str(self.cwd),
            "workspaceRoot": str(self.workspace_root),
            "workspaceRootSource": self.workspace_root_source,
            "configHome": str(self.config_home),
            "projectStoreDir": str(self.project_store_dir),
            "sessions": {
                "scope": "project",
                "dir": str(self.sessions_dir),
                "transcriptDir": str(self.transcript_dir),
            },
            "memory": {
                "scope": self.memory_scope,
                "dir": str(self.memory_dir) if self.memory_dir is not None else None,
            },
            "artifacts": {
                "scope": "project",
                "dir": str(self.artifacts_dir),
                "bashOutputDir": str(self.bash_output_dir),
                "agentOutputDir": str(self.agent_output_dir),
            },
            "actMode": {
                "allowedWorkingDirectories": [str(path) for path in self.allowed_working_directories],
            },
            "settingsSources": [source.as_json() for source in self.settings_sources],
            "skillSources": [source.as_json() for source in self.skill_sources],
            "mcpSources": [source.as_json() for source in self.mcp_sources],
        }


def _resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _path_in(path: Path, root: Path) -> bool:
    resolved = _resolve_path(path)
    resolved_root = _resolve_path(root)
    return resolved == resolved_root or resolved_root in resolved.parents


def _scope_for_path(path: Path, *, workspace_root: Path, config_home: Path) -> WorkspaceScope:
    if _path_in(path, workspace_root):
        return "project"
    if _path_in(path, config_home):
        return "user"
    return "explicit"


def _path_sources(
    kind: str,
    paths: Iterable[str | Path],
    *,
    workspace_root: Path,
    config_home: Path,
    source: str | None = None,
) -> tuple[WorkspacePathSource, ...]:
    result: list[WorkspacePathSource] = []
    seen: set[str] = set()
    for raw in paths:
        path = Path(raw).expanduser()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(
            WorkspacePathSource(
                kind=kind,
                path=path,
                scope=_scope_for_path(path, workspace_root=workspace_root, config_home=config_home),
                source=source,
            )
        )
    return tuple(result)


def build_workspace_runtime(
    *,
    cwd: str | Path,
    config_home: str | Path,
    workspace_root: str | Path | None = None,
    settings_paths: Iterable[str | Path] = (),
    skill_paths: Iterable[str | Path] = (),
    mcp_config_paths: Iterable[str | Path] = (),
    mcp_fixture_paths: Iterable[str | Path] = (),
    mcp_server_names: Iterable[str] = (),
    memory_enabled: bool = True,
    allowed_working_directories: Iterable[str | Path] = (),
) -> WorkspaceRuntime:
    """Resolve workspace-local storage and configured extension scopes."""

    resolved_cwd = _resolve_path(cwd)
    resolved_config_home = Path(config_home).expanduser()
    if workspace_root is None:
        git_root = find_git_root(resolved_cwd)
        resolved_workspace_root = _resolve_path(git_root or resolved_cwd)
        root_source = "git" if git_root is not None else "cwd"
    else:
        resolved_workspace_root = _resolve_path(workspace_root)
        root_source = "explicit"

    project_store_dir = resolved_config_home / "projects" / sanitize_path(resolved_workspace_root)
    artifacts_dir = project_store_dir / "artifacts"
    allowed_dirs = (resolved_cwd, *tuple(_resolve_path(path) for path in allowed_working_directories))

    settings_sources = _path_sources(
        "settings",
        settings_paths,
        workspace_root=resolved_workspace_root,
        config_home=resolved_config_home,
    )
    skill_sources = _path_sources(
        "skills",
        skill_paths,
        workspace_root=resolved_workspace_root,
        config_home=resolved_config_home,
    )
    mcp_sources = (
        *_path_sources(
            "mcp_config",
            mcp_config_paths,
            workspace_root=resolved_workspace_root,
            config_home=resolved_config_home,
        ),
        *_path_sources(
            "mcp_fixture",
            mcp_fixture_paths,
            workspace_root=resolved_workspace_root,
            config_home=resolved_config_home,
        ),
        *(
            WorkspacePathSource(kind="mcp_server", path=Path(name), scope="unknown", source="runtime")
            for name in mcp_server_names
        ),
    )

    return WorkspaceRuntime(
        cwd=resolved_cwd,
        workspace_root=resolved_workspace_root,
        workspace_root_source=root_source,
        config_home=resolved_config_home,
        project_store_dir=project_store_dir,
        sessions_dir=project_store_dir,
        memory_dir=(project_store_dir / "memory") if memory_enabled else None,
        memory_scope="project" if memory_enabled else "disabled",
        artifacts_dir=artifacts_dir,
        bash_output_dir=artifacts_dir / "bash-output",
        agent_output_dir=artifacts_dir / "agent-output",
        allowed_working_directories=allowed_dirs,
        settings_sources=settings_sources,
        skill_sources=skill_sources,
        mcp_sources=mcp_sources,
    )
