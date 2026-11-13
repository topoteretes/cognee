"""Identify installations managed by the Hermes plugin catalog."""

from __future__ import annotations

import json
import re
from pathlib import Path


def catalog_name(plugin_root: Path | None = None) -> str | None:
    """Return the catalog key, or None for an installation without a marker.

    A damaged marker still protects the installed copy from pip overwrites.
    Only accept catalog keys safe to display as part of an update command.
    """
    root = plugin_root if plugin_root is not None else Path(__file__).resolve().parents[1]
    marker = root / ".hermes-catalog.json"
    if not marker.exists() and not marker.is_symlink():
        return None
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
        name = data.get("catalog_name") if isinstance(data, dict) else None
        if isinstance(name, str) and re.fullmatch(r"[a-z0-9_-]{1,64}", name):
            return name
    except (OSError, ValueError):
        pass
    return "cognee"
