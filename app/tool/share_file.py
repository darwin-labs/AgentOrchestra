import base64
import mimetypes
import os
import tempfile
from pathlib import Path
from typing import Optional

from pydantic import Field

from app.config import SandboxSettings, config
from app.sandbox.client import SANDBOX_CLIENT
from app.tool.base import BaseTool, ToolResult


DEFAULT_MAX_BYTES = 5 * 1024 * 1024  # 5 MB


class ShareFileTool(BaseTool):
    name: str = "share_file"
    description: str = (
        "Share a file with the API user by sending its contents as base64. "
        "Use this to return files created in the workspace (reports, data, images, etc.). "
        "Only files inside the workspace are allowed."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file (absolute or relative to the workspace)",
            },
            "display_name": {
                "type": "string",
                "description": "Optional name shown to the user (defaults to filename)",
            },
            "mime_type": {
                "type": "string",
                "description": "Optional MIME type override (e.g. text/csv)",
            },
            "max_bytes": {
                "type": "integer",
                "description": "Optional max size limit for the shared file in bytes",
            },
        },
        "required": ["path"],
    }

    max_bytes: int = Field(default=DEFAULT_MAX_BYTES, exclude=True)

    def _resolve_path(self, path: str) -> Path:
        if os.path.isabs(path):
            return Path(path)
        if config.sandbox and config.sandbox.use_sandbox:
            work_dir = (
                config.sandbox.work_dir if config.sandbox else SandboxSettings().work_dir
            )
            return Path(work_dir) / path
        return Path(config.workspace_root) / path

    def _validate_path(self, path: Path) -> Optional[str]:
        if config.sandbox and config.sandbox.use_sandbox:
            work_dir = (
                config.sandbox.work_dir if config.sandbox else SandboxSettings().work_dir
            )
            if not str(path).startswith(work_dir.rstrip("/") + "/") and str(path) != work_dir:
                return f"Path must be within sandbox workspace: {work_dir}"
            return None

        workspace_root = Path(config.workspace_root).resolve()
        try:
            path.resolve().relative_to(workspace_root)
        except ValueError:
            return f"Path must be within workspace: {workspace_root}"
        return None

    async def _read_local_bytes(self, path: Path) -> bytes:
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        return path.read_bytes()

    async def _read_sandbox_bytes(self, path: Path) -> bytes:
        if not SANDBOX_CLIENT.sandbox:
            await SANDBOX_CLIENT.create(config=config.sandbox or SandboxSettings())
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp_path = tmp.name
            await SANDBOX_CLIENT.copy_from(str(path), tmp_path)
            return Path(tmp_path).read_bytes()
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    async def execute(
        self,
        *,
        path: str,
        display_name: Optional[str] = None,
        mime_type: Optional[str] = None,
        max_bytes: Optional[int] = None,
        **kwargs,
    ) -> ToolResult:
        resolved_path = self._resolve_path(path)
        validation_error = self._validate_path(resolved_path)
        if validation_error:
            return ToolResult(error=validation_error)

        try:
            if config.sandbox and config.sandbox.use_sandbox:
                data = await self._read_sandbox_bytes(resolved_path)
            else:
                data = await self._read_local_bytes(resolved_path)
        except Exception as e:
            return ToolResult(error=f"Failed to read file: {str(e)}")

        size = len(data)
        limit = max_bytes if max_bytes is not None else self.max_bytes
        if limit and size > limit:
            return ToolResult(
                error=(
                    f"File too large to share ({size} bytes). "
                    f"Max allowed is {limit} bytes."
                )
            )

        encoded = base64.b64encode(data).decode("utf-8")
        name = display_name or resolved_path.name
        detected_mime = mime_type or mimetypes.guess_type(name)[0] or "application/octet-stream"

        return ToolResult(
            output=f"Shared file '{name}' ({size} bytes)",
            base64_file=encoded,
            file_name=name,
            file_path=str(resolved_path),
            mime_type=detected_mime,
            file_size=size,
        )
