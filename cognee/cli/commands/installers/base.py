from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path


class HarnessInstaller(ABC):
    """Base class for per-harness MCP config installers."""

    @abstractmethod
    def config_path(self, scope: str = "user") -> Path:
        """Return the absolute path to the harness config file."""
        ...

    @abstractmethod
    def _read_config(self, path: Path) -> dict:
        """Read and parse the config file. Return {} if it doesn't exist."""
        ...

    @abstractmethod
    def _write_config(self, path: Path, cfg: dict) -> None:
        """Atomically write *cfg* to *path* (via a .tmp sibling + os.replace)."""
        ...

    @abstractmethod
    def _has_entry(self, cfg: dict) -> bool:
        """Return True if cognee is already registered in *cfg*."""
        ...

    @abstractmethod
    def _add_entry(self, cfg: dict, mcp_block: dict) -> dict:
        """Return a new cfg dict with the cognee MCP entry merged in."""
        ...

    @abstractmethod
    def _remove_entry(self, cfg: dict) -> dict:
        """Return a new cfg dict with the cognee MCP entry removed."""
        ...

    @staticmethod
    def build_mcp_block(mcp_dir: str | None = None) -> dict:
        """Return the JSON-serialisable MCP server block for cognee-mcp.

        If *mcp_dir* is given (developer / source-checkout mode), produce:
            {"command": "uv", "args": ["--directory", mcp_dir, "run", "cognee-mcp"]}

        If ``cognee-mcp`` is on PATH (pip/pipx install), produce:
            {"command": "cognee-mcp", "args": []}

        Raises RuntimeError if neither condition is met.
        """
        if mcp_dir:
            mcp_dir = os.path.abspath(mcp_dir)
            return {"command": "uv", "args": ["--directory", mcp_dir, "run", "cognee-mcp"]}

        if shutil.which("cognee-mcp"):
            return {"command": "cognee-mcp", "args": []}

        raise RuntimeError(
            "cognee-mcp is not on PATH. "
            "Either install it (pip install cognee-mcp) or pass --mcp-dir "
            "pointing to your cognee-mcp source checkout."
        )

    def install(
        self,
        *,
        mcp_dir: str | None = None,
        scope: str = "user",
        dry_run: bool = False,
    ) -> str:
        """Install cognee into the harness config.

        Returns a human-readable summary of what was (or would be) done.
        """
        path = self.config_path(scope=scope)
        mcp_block = self.build_mcp_block(mcp_dir)
        cfg = self._read_config(path)

        if self._has_entry(cfg):
            return f"Already installed — {path} unchanged."

        new_cfg = self._add_entry(cfg, mcp_block)

        if dry_run:
            return (
                f"[dry-run] Would write to {path}:\n"
                f"  command: {mcp_block['command']!r}\n"
                f"  args:    {mcp_block['args']!r}"
            )

        self._backup(path)
        self._write_config(path, new_cfg)
        return f"Installed cognee into {path}"

    def uninstall(self, *, scope: str = "user", dry_run: bool = False) -> str:
        """Remove the cognee entry from the harness config.

        Returns a human-readable summary of what was (or would be) done.
        """
        path = self.config_path(scope=scope)
        cfg = self._read_config(path)

        if not self._has_entry(cfg):
            return f"cognee is not registered in {path} — nothing to do."

        new_cfg = self._remove_entry(cfg)

        if dry_run:
            return f"[dry-run] Would remove cognee entry from {path}"

        self._backup(path)
        self._write_config(path, new_cfg)
        return f"Removed cognee from {path}"

    @staticmethod
    def _backup(path: Path) -> None:
        """Copy *path* to *path*.cognee.bak on first touch (non-destructive)."""
        backup = path.with_suffix(path.suffix + ".cognee.bak")
        if path.exists() and not backup.exists():
            import shutil as _shutil

            _shutil.copy2(path, backup)
