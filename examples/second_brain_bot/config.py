"""Environment bootstrap for running the bot against real cognee.

Two things have to happen before cognee is imported:

1. The LLM / embedding provider config has to be in the environment. We keep it
   in a local ``.env`` (gitignored) so the key lives in exactly one place.
2. Two cognee knobs must be set BEFORE the first ``import cognee``:

       ENABLE_BACKEND_ACCESS_CONTROL=false   single-user: no auth, one shared store
       CACHING=false                         disable session-memory side calls

   cognee reads these at import time. With access control left on (the default),
   cognee requires an authenticated user and the bot's calls fail.

Both knobs use ``setdefault``, so an explicit environment variable or a ``.env``
entry always wins. ``load_cognee_env`` is idempotent and is called from run.py
and from the cognee adapter just before it imports cognee.
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
