"""Settings shared by the three Turso (rewrite engine) backends."""

from functools import lru_cache

import pydantic
from pydantic_settings import BaseSettings, SettingsConfigDict

JOURNAL_MODES = ("wal", "mvcc")


class TursoConfig(BaseSettings):
    """Knobs of the Turso rewrite engine that apply to every Turso-backed store.

    ``turso_journal_mode``
        ``wal`` (default): SQLite-compatible write-ahead log; writers serialize on
        one lock and wait up to ``turso_busy_timeout_ms``. Files stay readable by
        stock SQLite.
        ``mvcc``: Turso's multi-version concurrency control. Writes run as
        ``BEGIN CONCURRENT`` transactions that commit in parallel and only fail
        when two transactions write the same row (``Write-write conflict``), in
        which case cognee's own write paths retry. The database file gains a
        ``-log`` companion and is no longer readable by stock SQLite. Experimental
        upstream; opt in per deployment. Applies to the graph and vector stores
        only: the relational and session-cache engines always run in ``wal``
        (see :meth:`wal_only`).
    """

    turso_journal_mode: str = "wal"
    turso_busy_timeout_ms: int = 120000
    turso_conflict_retries: int = 5

    model_config = SettingsConfigDict(extra="allow")

    @pydantic.model_validator(mode="after")
    def normalize(self):
        mode = (self.turso_journal_mode or "wal").strip().lower()
        if mode not in JOURNAL_MODES:
            raise ValueError(
                f"TURSO_JOURNAL_MODE must be one of {', '.join(JOURNAL_MODES)}; got {mode!r}."
            )
        self.turso_journal_mode = mode
        if self.turso_busy_timeout_ms < 0:
            raise ValueError("TURSO_BUSY_TIMEOUT_MS must be >= 0.")
        if self.turso_conflict_retries < 0:
            raise ValueError("TURSO_CONFLICT_RETRIES must be >= 0.")
        return self

    @property
    def concurrent_writes(self) -> bool:
        return self.turso_journal_mode == "mvcc"

    def wal_only(self) -> "TursoConfig":
        """This config with the journal mode pinned to ``wal``.

        For engines whose write transactions cannot be re-run on a conflict: the
        relational and session-cache engines serve ``get_async_session()`` callers
        that run arbitrary code between statements, so an ``mvcc`` write-write
        conflict there would be a lost write, not a retry. Under ``wal`` the same
        collision waits on ``busy_timeout`` instead.
        """
        return self.model_copy(update={"turso_journal_mode": "wal"})


@lru_cache
def get_turso_config() -> TursoConfig:
    return TursoConfig()
