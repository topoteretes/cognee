"""Append records to a JSONL file on a background thread, so a caller never waits on disk.

One writer per process writes ``llm_calls-<start time>-<pid>.jsonl`` in the
capture directory; a forked child gets its own writer and file. A write error
is logged and the writer stops (records after it are discarded and counted in
the log), since a disk that failed once is not retried per record.
"""

import atexit
import os
import queue
import threading
from datetime import datetime, timezone

from cognee.shared.logging_utils import get_logger

logger = get_logger("llm_call_capture")

_STOP = object()


class JsonlWriter:
    """A background thread that appends one JSON line per ``put()`` to its file."""

    def __init__(self, directory: str):
        started = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.pid = os.getpid()
        self.path = os.path.join(directory, f"llm_calls-{started}-{self.pid}.jsonl")
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._failed = False
        self._discarded = 0
        self._thread = threading.Thread(
            target=self._run, name="cognee-llm-call-capture", daemon=True
        )
        self._thread.start()
        atexit.register(self.close)

    def put(self, line: str) -> None:
        if self._failed:
            self._discarded += 1
            return
        self._queue.put(line)

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait until every line put so far is on disk. Returns False on timeout."""
        done = threading.Event()
        self._queue.put(done)
        return done.wait(timeout)

    def close(self, timeout: float = 5.0) -> None:
        """Write what is queued and stop the thread (also run at interpreter exit)."""
        if self._thread.is_alive():
            self._queue.put(_STOP)
            self._thread.join(timeout)
        if self._discarded:
            logger.warning(
                "LLM call capture discarded %d records after a write error", self._discarded
            )

    def _run(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as file:
                while True:
                    item = self._queue.get()
                    if item is _STOP:
                        return
                    if isinstance(item, threading.Event):
                        file.flush()
                        item.set()
                        continue
                    file.write(item + "\n")
                    if self._queue.empty():
                        file.flush()
        except OSError:
            self._failed = True
            logger.warning("LLM call capture stopped: cannot write %s", self.path, exc_info=True)
            self._release_waiters()

    def _release_waiters(self) -> None:
        """After a failure, unblock flush() callers and drop what is still queued."""
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if isinstance(item, threading.Event):
                item.set()
            elif item is not _STOP:
                self._discarded += 1
