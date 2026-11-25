"""PyPI update check — rate-limited, cached, fail-silent.

The counterpart of the other plugins' registry checks (claude-code/codex poll
the marketplace manifest, openclaw the npm registry; same env names:
``COGNEE_UPDATE_CHECK``, ``COGNEE_UPDATE_CHECK_INTERVAL``). Deliberately
CLI-only: ``hermes cognee status`` / ``version`` may reach for the network
once per interval, the session path never does.

The verdict is cached in ``~/.cognee-plugin/hermes/update-check.json``::

    {"checked_at": <epoch>, "latest": "1.2.0"}

Every failure mode returns the best answer available (the cached one, else "")
— an update nudge must never break a status command.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

from .config import SHARED_PLUGIN_STATE_DIR

PYPI_PACKAGE = "cognee-integration-hermes-agent"
_PYPI_URL = f"https://pypi.org/pypi/{PYPI_PACKAGE}/json"

_CACHE_PATH = SHARED_PLUGIN_STATE_DIR / "hermes" / "update-check.json"


def _read_cache(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_cache(path: Path, latest: str, now: float) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"checked_at": now, "latest": latest}), encoding="utf-8")
    except OSError:
        pass


def _fetch_latest(timeout: float, opener: Optional[Callable[..., Any]]) -> str:
    request = urllib.request.Request(_PYPI_URL, headers={"Accept": "application/json"})
    open_fn = opener or urllib.request.urlopen
    with open_fn(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return str(((payload or {}).get("info") or {}).get("version") or "")


def latest_published_version(
    *,
    interval: float = 3600.0,
    force: bool = False,
    timeout: float = 3.0,
    now: Optional[float] = None,
    cache_path: Optional[Path] = None,
    opener: Optional[Callable[..., Any]] = None,
) -> str:
    """The newest version on PyPI, or "" when it cannot be determined.

    Within ``interval`` of the last check the cached answer is returned without
    touching the network; ``force`` skips that gate. Never raises.
    """
    path = cache_path or _CACHE_PATH
    current_time = time.time() if now is None else now
    cache = _read_cache(path)
    cached_latest = str(cache.get("latest") or "")
    try:
        checked_at = float(cache.get("checked_at") or 0.0)
    except (TypeError, ValueError):
        checked_at = 0.0
    if not force and cached_latest and current_time - checked_at < interval:
        return cached_latest
    try:
        latest = _fetch_latest(timeout, opener)
    except Exception:
        return cached_latest
    if latest:
        _write_cache(path, latest, current_time)
    return latest or cached_latest


def _version_tuple(version: str) -> tuple:
    parts: list[Any] = []
    for token in version.split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def is_newer(candidate: str, current: str) -> bool:
    """True when ``candidate`` is a strictly newer release than ``current``.

    Numeric best-effort comparison — good enough for the X.Y.Z scheme this
    package publishes; anything unparseable compares as 0 and never nags.
    """
    if not candidate or not current:
        return False
    return _version_tuple(candidate) > _version_tuple(current)
