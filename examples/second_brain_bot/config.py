"""Environment bootstrap for running the bot against real cognee.

Loads a local (gitignored) ``.env`` for the LLM/embedding provider config, then
sets two cognee knobs that must be in place BEFORE the first ``import cognee``
(cognee reads them at import time; with access control left on it would require
an authenticated user and the bot's calls would fail):

    ENABLE_BACKEND_ACCESS_CONTROL=false   single-user: no auth, one shared store
    CACHING=false                         disable session-memory side calls

Both use ``setdefault`` (an explicit env var or .env entry wins). Idempotent.
"""

from __future__ import annotations

import os
from pathlib import Path

_BOT_DIR = Path(__file__).resolve().parent
_loaded = False


def load_cognee_env() -> None:
    global _loaded
    if _loaded:
        return

    env_path = _BOT_DIR / ".env"
    if env_path.exists():
        for raw in env_path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

    # Must be set before cognee is imported.
    os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")
    os.environ.setdefault("CACHING", "false")
    _loaded = True
