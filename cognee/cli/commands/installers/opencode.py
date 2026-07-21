from __future__ import annotations

import json
import os
from pathlib import Path

from cognee.cli.commands.installers.base import HarnessInstaller


class OpenCodeInstaller(HarnessInstaller):
    """Installer for OpenCode (opencode.ai) — edits ~/.config/opencode/opencode.json.

    MCP entries live under the 'mcp' key (not 'mcpServers'), with type='local'
    and command as an array. Verified against opencode.ai/docs/mcp-servers.
    """

    def config_path(self, scope: str = "user") -> Path:
        if os.name == "nt":
            appdata = os.environ.get("APPDATA", "")
            base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
            return base / "opencode" / "opencode.json"
        else:
            xdg = os.environ.get("XDG_CONFIG_HOME", "")
            base = Path(xdg) if xdg else Path.home() / ".config"
            return base / "opencode" / "opencode.json"

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
        return "cognee" in cfg.get("mcp", {})

    def _add_entry(self, cfg: dict, mcp_block: dict) -> dict:
        """Merge the cognee entry under the opencode 'mcp' key.

        mcp_block from build_mcp_block() has shape:
            {"command": "<binary>", "args": [...]}

        opencode expects:
            {"type": "local", "command": ["<binary>", ...args], "enabled": true}
        """
        cfg = dict(cfg)
        mcp = dict(cfg.get("mcp", {}))

        # Convert base class shape → opencode array shape
        command_list = [mcp_block["command"]] + list(mcp_block.get("args", []))
        mcp["cognee"] = {
            "type": "local",
            "command": command_list,
            "enabled": True,
        }
        cfg["mcp"] = mcp
        return cfg

    def _remove_entry(self, cfg: dict) -> dict:
        cfg = dict(cfg)
        mcp = dict(cfg.get("mcp", {}))
        mcp.pop("cognee", None)
        cfg["mcp"] = mcp
        return cfg
