from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from cognee.cli.commands.installers.base import HarnessInstaller


class ClaudeCodeInstaller(HarnessInstaller):
    """Installer for Claude Code (claude.ai desktop / claude CLI)."""

    def config_path(self, scope: str = "user") -> Path:
        if scope == "project":
            # Project-level: .mcp.json in the current working directory —
            # this is where `claude mcp add --scope project` writes.
            return Path.cwd() / ".mcp.json"
        # User-level (default)
        if os.name == "nt":
            base = Path(os.environ.get("USERPROFILE", Path.home()))
        else:
            base = Path.home()
        return base / ".claude.json"

    @staticmethod
    def _claude_binary() -> str | None:
        return shutil.which("claude")

    def install(
        self,
        *,
        mcp_dir: str | None = None,
        scope: str = "user",
        dry_run: bool = False,
    ) -> str:
        """Install cognee via `claude mcp add` if available, else JSON edit."""
        claude_bin = self._claude_binary()

        if claude_bin and not dry_run:
            return self._native_install(claude_bin, mcp_dir=mcp_dir, scope=scope)

        if dry_run:
            mcp_block = self.build_mcp_block(mcp_dir)
            if claude_bin:
                cmd = self._build_native_args(claude_bin, mcp_block, scope=scope)
                return f"[dry-run] Would run: {' '.join(cmd)}"
            # Fallback: show JSON edit plan
            path = self.config_path(scope=scope)
            return (
                f"[dry-run] Would write to {path}:\n"
                f"  command: {mcp_block['command']!r}\n"
                f"  args:    {mcp_block['args']!r}"
            )

        # No claude binary — fall back to JSON edit
        return super().install(mcp_dir=mcp_dir, scope=scope, dry_run=dry_run)

    def _build_native_args(
        self, claude_bin: str, mcp_block: dict, scope: str = "user"
    ) -> list[str]:
        """Build the argument list for `claude mcp add`."""
        if mcp_block["args"]:
            # uv --directory <dir> run cognee-mcp
            # claude mcp add --scope user cognee -- uv --directory <dir> run cognee-mcp
            cmd = [
                claude_bin,
                "mcp",
                "add",
                "--scope",
                scope,
                "cognee",
                "--",
                mcp_block["command"],
            ] + mcp_block["args"]
        else:
            # cognee-mcp directly
            cmd = [
                claude_bin,
                "mcp",
                "add",
                "--scope",
                scope,
                "cognee",
                mcp_block["command"],
            ]
        return cmd

    def _native_install(self, claude_bin: str, *, mcp_dir: str | None, scope: str) -> str:
        mcp_block = self.build_mcp_block(mcp_dir)

        # Check idempotency via JSON first (faster and reliable)
        path = self.config_path(scope=scope)
        cfg = self._read_config(path)
        if self._has_entry(cfg):
            return f"Already installed — {path} unchanged."

        cmd = self._build_native_args(claude_bin, mcp_block, scope=scope)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            # Native command failed — fall back to JSON edit
            self._backup(path)
            new_cfg = self._add_entry(cfg, mcp_block)
            self._write_config(path, new_cfg)
            return (
                f"Installed cognee into {path} "
                f"(native `claude mcp add` failed: {result.stderr.strip()!r}; used JSON fallback)"
            )
        return f"Installed cognee via `claude mcp add --scope {scope}`"

    def _read_config(self, path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except OSError:
            return {}
        except json.JSONDecodeError:
            raise RuntimeError(
                f"{path} exists but is not valid JSON — fix or remove it, then re-run."
            )

    def _write_config(self, path: Path, cfg: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def _has_entry(self, cfg: dict) -> bool:
        return "cognee" in cfg.get("mcpServers", {})

    def _add_entry(self, cfg: dict, mcp_block: dict) -> dict:
        cfg = dict(cfg)
        mcp_servers = dict(cfg.get("mcpServers", {}))
        mcp_servers["cognee"] = mcp_block
        cfg["mcpServers"] = mcp_servers
        return cfg

    def _remove_entry(self, cfg: dict) -> dict:
        cfg = dict(cfg)
        mcp_servers = dict(cfg.get("mcpServers", {}))
        mcp_servers.pop("cognee", None)
        cfg["mcpServers"] = mcp_servers
        return cfg
