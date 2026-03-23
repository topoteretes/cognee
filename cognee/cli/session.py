"""CLI session state persistence for interactive and resume modes."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional


SESSION_DIR = Path.home() / ".cognee" / "cli"
SESSION_FILE = SESSION_DIR / "session.json"


def load_session() -> dict:
    """Load the last saved session state."""
    if SESSION_FILE.exists():
        try:
            return json.loads(SESSION_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_session(
    dataset: str = "main_dataset",
    query_type: str = "GRAPH_COMPLETION",
    **extra: object,
) -> None:
    """Persist session state to disk."""
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "dataset": dataset,
        "query_type": query_type,
        "last_active": datetime.now().isoformat(),
        **extra,
    }
    SESSION_FILE.write_text(json.dumps(data, indent=2) + "\n")
