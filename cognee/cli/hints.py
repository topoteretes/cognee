"""Value-moment hint seams (ships dormant — no hints are registered yet).

This is the single choke point a future cloud value-moment hint (Bet 6) will
flow through. The suppression matrix, frequency caps, and state file exist
now so the hint later becomes a one-line registration, not a layout rework.

Rules (from the CLI-etiquette research that shaped this design):
- stderr only, after the real output, everything dimmed, no boxes
- lifetime-once per hint id, at most one hint per 24h globally
- suppressed whenever any of: CI, stdout or stderr not a TTY, --quiet/--json,
  COGNEE_NO_HINTS=1, TERM=dumb, state file unwritable (fail closed)
"""

import json
import os
import time
from pathlib import Path
from typing import List, Optional

from cognee.cli.ui import TermCaps, Style, detect_caps

_GLOBAL_CAP_SECONDS = 24 * 3600


def _state_path() -> Path:
    return Path(os.environ.get("COGNEE_CLI_STATE", str(Path.home() / ".cognee"))) / (
        "cli_state.json"
    )


def load_state() -> dict:
    try:
        with open(_state_path(), encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def save_state(state: dict) -> bool:
    try:
        path = _state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2)
        return True
    except Exception:
        return False


def record_event(counter: str) -> dict:
    """Bump a named counter (e.g. 'cognify_success') so future value-moment
    triggers become a counter read. Never raises; returns the state."""
    state = load_state()
    state.setdefault("first_run_at", int(time.time()))
    counters = state.setdefault("counters", {})
    counters[counter] = int(counters.get(counter, 0)) + 1
    save_state(state)
    return state


def hints_enabled(caps: Optional[TermCaps] = None, quiet: bool = False) -> bool:
    caps = caps or detect_caps()
    if quiet:
        return False
    if os.environ.get("COGNEE_NO_HINTS"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    if caps.ci or not caps.stderr_tty or not caps.stdout_tty:
        return False
    return True


def emit_hint(hint_id: str, lines: List[str], caps: Optional[TermCaps] = None) -> bool:
    """Show a hint at most once per id, at most one hint per day, or not at all.

    Returns True only if the hint was actually shown. Fails closed: if state
    can't be persisted, the hint is not shown (a hint that can't remember it
    was shown would repeat — the one unforgivable hint behavior).
    """
    caps = caps or detect_caps()
    if not hints_enabled(caps):
        return False

    state = load_state()
    shown = state.setdefault("hints_shown", {})
    if hint_id in shown:
        return False
    last_shown = max([0, *shown.values()]) if shown else 0
    now = int(time.time())
    if now - last_shown < _GLOBAL_CAP_SECONDS:
        return False

    shown[hint_id] = now
    if not save_state(state):
        return False

    import sys

    style = Style(caps.color)
    sys.stderr.write("\n")
    for line in lines:
        sys.stderr.write(style.dim(f"  {line}") + "\n")
    return True
