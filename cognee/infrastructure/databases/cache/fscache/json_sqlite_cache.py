"""A minimal JSON-only, SQLite-backed key/value store.

This replaces the ``diskcache`` dependency for the filesystem session cache.
``diskcache`` is flagged by the GitHub Advisory Database as GHSA-w8v5-vhqr-4h9v
because its default serialization can fall back to ``pickle``. Cognee's
filesystem session adapter only ever stores JSON strings, so it does not need
DiskCache's general object-serialization surface.

This store intentionally exposes the small subset of the DiskCache API that the
``FSCacheAdapter`` relies on:

- ``get(key)`` / ``set(key, value, expire=...)`` / ``delete(key)``
- ``clear()`` / ``expire()``
- ``transact()`` (context manager providing an atomic write boundary)
- ``close()``

Values are stored verbatim as TEXT. The adapter only ever passes JSON strings,
so no pickling or arbitrary-object deserialization ever occurs.
"""

import os
import sqlite3
import threading
import time
from contextlib import contextmanager

__all__ = ["JsonSqliteCache"]

_DB_FILENAME = "cache.sqlite3"


class JsonSqliteCache:
    """A small SQLite-backed string key/value store with TTL support.

    The store keeps a single connection guarded by a re-entrant lock so that
    ``transact()`` blocks (which the adapter uses to make read-modify-write
    sequences atomic) execute as a single SQLite transaction.
    """

    def __init__(self, directory: str):
        """Create (or open) a SQLite-backed cache rooted at ``directory``."""
        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)
        self._db_path = os.path.join(self.directory, _DB_FILENAME)
        # check_same_thread=False so the single guarded connection can be used
        # from any thread; concurrency is serialized by self._lock.
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            "key TEXT PRIMARY KEY, "
            "value TEXT NOT NULL, "
            "expire_at REAL"  # absolute epoch seconds, or NULL for no expiry
            ")"
        )
        self._conn.commit()
        self._lock = threading.RLock()
        # Depth of nested transact() contexts; only the outermost commits.
        self._txn_depth = 0

    @staticmethod
    def _now() -> float:
        return time.time()

    def get(self, key: str):
        """Return the stored value for ``key`` or ``None`` if absent/expired."""
        with self._lock:
            row = self._conn.execute(
                "SELECT value, expire_at FROM cache WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            value, expire_at = row
            if expire_at is not None and expire_at <= self._now():
                self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                self._maybe_commit()
                return None
            return value

    def set(self, key: str, value, expire: float | None = None) -> None:
        """Store ``value`` under ``key`` with an optional TTL (seconds)."""
        expire_at = self._now() + expire if expire else None
        with self._lock:
            self._conn.execute(
                "INSERT INTO cache (key, value, expire_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
                "expire_at = excluded.expire_at",
                (key, value, expire_at),
            )
            self._maybe_commit()

    def delete(self, key: str) -> bool:
        """Delete ``key``. Returns True if a row was removed."""
        with self._lock:
            cur = self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            self._maybe_commit()
            return cur.rowcount > 0

    def clear(self) -> None:
        """Remove all entries from the cache."""
        with self._lock:
            self._conn.execute("DELETE FROM cache")
            self._maybe_commit()

    def expire(self) -> int:
        """Evict all entries whose TTL has elapsed. Returns the count removed."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM cache WHERE expire_at IS NOT NULL AND expire_at <= ?",
                (self._now(),),
            )
            self._maybe_commit()
            return cur.rowcount

    @contextmanager
    def transact(self):
        """Run a block as a single atomic SQLite transaction.

        Supports nesting: only the outermost ``transact()`` boundary commits or
        rolls back, mirroring DiskCache's nestable transaction semantics.
        """
        with self._lock:
            self._txn_depth += 1
            try:
                yield
            except Exception:
                if self._txn_depth == 1:
                    self._conn.rollback()
                raise
            finally:
                self._txn_depth -= 1
            if self._txn_depth == 0:
                self._conn.commit()

    def _maybe_commit(self) -> None:
        """Commit immediately unless inside an open ``transact()`` block."""
        if self._txn_depth == 0:
            self._conn.commit()

    def close(self) -> None:
        """Flush and close the underlying SQLite connection."""
        with self._lock:
            if self._conn is not None:
                self._conn.commit()
                self._conn.close()
                self._conn = None
