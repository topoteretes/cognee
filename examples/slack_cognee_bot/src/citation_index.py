"""Adapter-owned side index for Slack message citations (issue #3609).

Why this exists — the metadata round-trip, honestly:

Recon proved that arbitrary per-message metadata (channel / ts / permalink) does
**not** survive into what ``search(SearchType.CHUNKS)`` returns. The chunk
``metadata`` dict and the source Document's ``external_metadata`` are both
stripped from the vector payload
(``LanceDBAdapter._build_data_point_schema`` excludes ``metadata`` and nested
models). The *only* field that survives and that we can control is
``document_id`` — because we assign a deterministic ``DataItem.data_id`` at
ingest, and cognee reuses it as the ``Data`` id → ``Document`` id →
``DocumentChunk.document_id``.

So citations are produced by a **join**, not by smuggled metadata: this index
maps ``document_id`` → the human-readable ``{channel, ts, permalink, author,
snippet}``. ``ingest`` writes a row; ``answer`` reads ``document_id`` off each
returned chunk and looks the row up here.

Backed by the standard-library ``sqlite3`` (no new dependency), matching
cognee's SQLite-by-default convention.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass

SNIPPET_MAX_CHARS = 200


@dataclass(frozen=True)
class CitationRecord:
    """A stored citation row, joined back by ``document_id``."""

    document_id: str
    channel_id: str
    ts: str
    permalink: str
    author: str
    snippet: str


class CitationIndex:
    """Persistent ``document_id`` → citation map backing cited answers.

    A single shared connection is reused across calls so an in-memory database
    (``":memory:"``, used by tests) keeps its state for the object's lifetime.
    Access is guarded by a lock since the async adapter may touch it from more
    than one task.
    """

    def __init__(self, db_path: str = "slack_citations.db"):
        # check_same_thread=False so the index is usable regardless of which
        # worker thread an async callback happens to run on.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS citations (
                    document_id TEXT PRIMARY KEY,
                    channel_id  TEXT NOT NULL,
                    ts          TEXT NOT NULL,
                    permalink   TEXT NOT NULL,
                    author      TEXT NOT NULL,
                    snippet     TEXT NOT NULL
                )
                """
            )
            self._conn.commit()

    def put(
        self,
        document_id: str,
        *,
        channel_id: str,
        ts: str,
        permalink: str,
        author: str,
        snippet: str,
    ) -> None:
        """Upsert the citation row for a message (keyed by its ``document_id``)."""
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO citations
                    (document_id, channel_id, ts, permalink, author, snippet)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    channel_id = excluded.channel_id,
                    ts         = excluded.ts,
                    permalink  = excluded.permalink,
                    author     = excluded.author,
                    snippet    = excluded.snippet
                """,
                (document_id, channel_id, ts, permalink, author, snippet[:SNIPPET_MAX_CHARS]),
            )
            self._conn.commit()

    def get(self, document_id: str) -> CitationRecord | None:
        """Look up the citation row for a chunk's ``document_id`` (or ``None``)."""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT document_id, channel_id, ts, permalink, author, snippet
                FROM citations WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()
        if row is None:
            return None
        return CitationRecord(*row)

    def delete_channel(self, channel_id: str) -> int:
        """Delete every citation row for a channel (used by ``forget``). Returns row count."""
        with self._lock:
            cur = self._conn.execute("DELETE FROM citations WHERE channel_id = ?", (channel_id,))
            self._conn.commit()
            return cur.rowcount

    def close(self) -> None:
        with self._lock:
            self._conn.close()
