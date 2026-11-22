"""Capture writes the server never received, replayed once it answers again.

The Hermes counterpart of openclaw's ``write-buffer.ts`` and the claude-code /
codex warm-up buffer. A session write (a turn or a tool trace) that cannot reach
the server is queued in ``~/.cognee-plugin/hermes/pending-entries.json`` and
replayed after the next successful write, when a session starts, and, for the
ending session, before its session-end improve. Without it every turn captured
while the local server is still booting, or briefly down, is lost.

Only failures where the request certainly never reached the server are queued:
the connection could not be opened (refused, unreachable, DNS failure) or the
server answered 503 while warming up. A timeout, a reset mid-request or another
error can happen after the server stored the entry, so those are not replayed:
a lost turn is better than a duplicated one.

The file is shared by every Hermes process on the machine; each
read-modify-write holds an exclusive ``flock`` on a sidecar lock file where the
platform has one, and a process-wide lock otherwise. A drain claims each entry
before sending it, taking it out of the file under that lock, so two processes
draining at once never send the same entry; an entry goes back only when the
server is still away. A crash mid-send therefore loses at most the entry in
flight rather than sending it twice.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from .config import SHARED_PLUGIN_STATE_DIR

try:  # POSIX only; Windows falls back to the in-process lock.
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows only
    fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

WRITE_BUFFER_PATH = SHARED_PLUGIN_STATE_DIR / "hermes" / "pending-entries.json"
MAX_BUFFERED_ENTRIES = 500
MAX_BUFFERED_AGE_SECS = 7 * 24 * 60 * 60

#: ``send(item, timeout)``: store one entry within ``timeout`` seconds, the part
#: of the drain's budget that is left.
SendFn = Callable[[dict[str, Any], float], None]
#: How long a session-scoped drain (session end, pre-compress) waits for a drain
#: already running in this process. That drain is sending the same entries, so
#: waiting longer only delays the caller.
SESSION_DRAIN_LOCK_WAIT_SECS = 5.0


def is_bufferable_write_error(exc: BaseException) -> bool:
    """True when a failed write certainly never reached the server."""
    status = getattr(exc, "status", None)
    if isinstance(status, int):
        return status == 503
    return bool(getattr(exc, "not_connected", False))


class WriteBuffer:
    def __init__(self, path: Path = WRITE_BUFFER_PATH) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._drain_lock = threading.Lock()
        # False only once a drain left the file empty and nothing was enqueued
        # since. True at start: an earlier run may have left entries behind.
        self._maybe_non_empty = True

    @contextlib.contextmanager
    def _locked(self):
        with self._lock:
            handle = None
            if fcntl is not None:
                try:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    handle = open(f"{self.path}.lock", "a")  # noqa: SIM115
                    fcntl.flock(handle, fcntl.LOCK_EX)
                except OSError:
                    if handle is not None:
                        handle.close()
                    handle = None
            try:
                yield
            finally:
                if handle is not None:
                    with contextlib.suppress(OSError):
                        fcntl.flock(handle, fcntl.LOCK_UN)
                    handle.close()

    def _load(self) -> list[dict[str, Any]]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        entries = data.get("entries") if isinstance(data, dict) else None
        return [e for e in entries if isinstance(e, dict)] if isinstance(entries, list) else []

    def _save(self, entries: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"entries": entries}), encoding="utf-8")
        os.replace(tmp, self.path)
        with contextlib.suppress(OSError):
            os.chmod(self.path, 0o600)

    @staticmethod
    def _prune(entries: list[dict[str, Any]], now: Optional[float] = None) -> list[dict[str, Any]]:
        """Keep the newest ``MAX_BUFFERED_ENTRIES``, none older than a week."""
        now = time.time() if now is None else now
        fresh = [
            e for e in entries if now - float(e.get("queued_at") or 0) <= MAX_BUFFERED_AGE_SECS
        ]
        return fresh[-MAX_BUFFERED_ENTRIES:]

    @property
    def maybe_non_empty(self) -> bool:
        return self._maybe_non_empty

    def enqueue(self, *, dataset: str, session_id: str, kind: str, entry: dict[str, Any]) -> None:
        self._maybe_non_empty = True
        item = {
            "dataset": dataset,
            "session_id": session_id,
            "kind": kind,
            "entry": entry,
            "queued_at": time.time(),
        }
        with self._locked():
            entries = self._load()
            entries.append(item)
            self._save(self._prune(entries))

    def size(self, session_id: str = "") -> int:
        with self._locked():
            entries = self._load()
        if session_id:
            return sum(1 for e in entries if e.get("session_id") == session_id)
        return len(entries)

    def _claim(self, session_id: str) -> Optional[dict[str, Any]]:
        """Take the oldest eligible entry out of the file, or None when there is none.

        Under the file lock, so an entry is claimed by exactly one drain, in
        whichever process; nobody else can send it once it has left the file.
        """
        with self._locked():
            entries = self._prune(self._load())
            for index, entry in enumerate(entries):
                if session_id and entry.get("session_id") != session_id:
                    continue
                del entries[index]
                self._save(entries)
                return entry
            return None

    def _restore(self, item: dict[str, Any]) -> None:
        """Put a claimed entry back in its place (by ``queued_at``): the server is still away."""
        self._maybe_non_empty = True
        with self._locked():
            entries = self._load()
            at = len(entries)
            for index, entry in enumerate(entries):
                if float(entry.get("queued_at") or 0) > float(item.get("queued_at") or 0):
                    at = index
                    break
            entries.insert(at, item)
            self._save(self._prune(entries))

    def drain(
        self,
        send: SendFn,
        *,
        budget_secs: float,
        session_id: str = "",
        on_drop: Optional[Callable[[dict[str, Any], BaseException], None]] = None,
    ) -> int:
        """Replay queued entries oldest-first. Returns how many were stored.

        With ``session_id`` only that session's entries are sent (session end);
        others stay queued. Each entry is claimed (taken out of the file) before
        it is sent, so a drain running in another process never sends it too. A
        send that fails with a bufferable error puts the entry back and ends the
        drain; any other failure drops it, since replaying it can never succeed
        (a 4xx) or may duplicate it (a timeout). Each send gets what is left of
        ``budget_secs``, so the budget bounds the whole drain. Single-flight per
        process: a drain already running is not joined, and a session-scoped one
        waits for it at most ``SESSION_DRAIN_LOCK_WAIT_SECS``.
        """
        acquired = (
            self._drain_lock.acquire(timeout=SESSION_DRAIN_LOCK_WAIT_SECS)
            if session_id
            else self._drain_lock.acquire(blocking=False)
        )
        if not acquired:
            return 0
        try:
            deadline = time.monotonic() + budget_secs
            stored = 0
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                item = self._claim(session_id)
                if item is None:
                    break
                try:
                    send(item, remaining)
                    stored += 1
                except Exception as exc:
                    if is_bufferable_write_error(exc):
                        self._restore(item)
                        break
                    if on_drop is not None:
                        on_drop(item, exc)
            with self._locked():
                if not self._load():
                    self._maybe_non_empty = False
            return stored
        finally:
            self._drain_lock.release()


_BUFFERS: dict[str, WriteBuffer] = {}
_BUFFERS_LOCK = threading.Lock()


def write_buffer_for(path: Path = WRITE_BUFFER_PATH) -> WriteBuffer:
    """The process-wide buffer for ``path``, shared by every provider instance."""
    with _BUFFERS_LOCK:
        key = str(path)
        if key not in _BUFFERS:
            _BUFFERS[key] = WriteBuffer(Path(path))
        return _BUFFERS[key]
