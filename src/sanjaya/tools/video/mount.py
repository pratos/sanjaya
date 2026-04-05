"""Controlled OSAccess mount for exposing workspace artifacts to Monty."""

from __future__ import annotations

from pathlib import Path

from pydantic_monty import MemoryFile, OSAccess


class WorkspaceMount:
    """Materialize workspace files into Monty's virtual filesystem."""

    def __init__(self, workspace_root: str, mount_root: str = "/workspace"):
        self.workspace_root = Path(workspace_root)
        self.mount_root = mount_root.rstrip("/") or "/workspace"

    def build_os_access(self) -> OSAccess:
        files: list[MemoryFile] = []
        if self.workspace_root.exists():
            for disk_path in sorted(self.workspace_root.rglob("*")):
                if not disk_path.is_file():
                    continue
                rel = disk_path.relative_to(self.workspace_root).as_posix()
                virtual_path = f"{self.mount_root}/{rel}"
                try:
                    content = disk_path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    content = disk_path.read_bytes()
                files.append(MemoryFile(path=virtual_path, content=content))

        return OSAccess(files=files, environ={"SANJAYA_WORKSPACE": self.mount_root})
