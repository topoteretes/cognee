from __future__ import annotations

import json
import os
from pathlib import Path

from cognee.cli.commands.installers.base import HarnessInstaller


class CursorInstaller(HarnessInstaller):
    """Installer for Cursor (cursor.sh) — edits ~/.cursor/mcp.json."""

    def config_path(self, scope: str = "user") -> Path:
        if scope == "project":
            # Project-level: .cursor/mcp.json relative to cwd
            return Path.cwd() / ".cursor" / "mcp.json"

        # User-level (default)
        if os.name == "nt":
            base = Path(os.environ.get("USERPROFILE", Path.home()))
        else:
            base = Path.home()
        return base / ".cursor" / "mcp.json"

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
