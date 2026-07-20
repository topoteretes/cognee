"""Engine-agnostic SQL cache adapter (Postgres via asyncpg, SQLite via aiosqlite)."""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from typing import List, Optional

from pydantic import ValidationError
from sqlalchemy import create_engine, delete, event, func, insert, or_, select, text, update
from sqlalchemy.engine import make_url
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cognee.infrastructure.databases.cache.cache_db_interface import CacheDBInterface
from cognee.infrastructure.databases.cache.models import SessionAgentTraceEntry, SessionQAEntry
from cognee.infrastructure.databases.exceptions.exceptions import (
    CacheConnectionError,
    SessionQAEntryValidationError,
    SharedLadybugLockRequiresRedisError,
)
from cognee.infrastructure.databases.relational import get_relational_config
from cognee.modules.storage.utils import JSONEncoder
from cognee.shared.logging_utils import get_logger

from .tables import (
    cache_kv,
    cache_metadata,
    cache_qa_entries,
    cache_session_context,
    cache_trace_entries,
    cache_usage_logs,
)

logger = get_logger("SqlCacheAdapter")

# Attempts for write transactions that can deadlock under concurrent workers.
_DEADLOCK_ATTEMPTS = 3

# Advisory-lock id guarding the throttled global TTL sweep on Postgres.
_PURGE_LOCK_ID = int.from_bytes(sha256(b"cognee_cache_ttl_sweep").digest()[:8], "big", signed=True)


def _is_deadlock_error(error: Exception) -> bool:
    """Best-effort detection of a Postgres deadlock without importing asyncpg."""
    if not isinstance(error, DBAPIError):
        return False
    orig = getattr(error, "orig", None)
    seen = set()
    while orig is not None and id(orig) not in seen:
        seen.add(id(orig))
        if type(orig).__name__ == "DeadlockDetectedError":
            return True
        if getattr(orig, "sqlstate", None) == "40P01":
            return True
        orig = getattr(orig, "__cause__", None)
    return "deadlock detected" in str(error).lower()


class _SqlAdvisoryLockHandle:
    """Handle returned by acquire_lock; owns one checked-out sync connection."""

    def __init__(self, connection, lock_id: int):
        self.connection = connection
        self.lock_id = lock_id
        self._released = False

    def release(self) -> None:
        """Unlock the advisory lock and return the connection to the pool."""
        if self._released:
            return
        self._released = True
        try:
            self.connection.execute(
                text("SELECT pg_advisory_unlock(:lock_id)"), {"lock_id": self.lock_id}
            )
        except Exception as error:
            logger.debug("Error releasing Postgres advisory lock: %s", error)
        finally:
            try:
                self.connection.close()
            except Exception as error:
                logger.debug("Error closing advisory lock connection: %s", error)


class SqlCacheAdapter(CacheDBInterface):
    """SQL-backed cache adapter for session QA, trace, usage-log, and KV storage.

    Runs on any SQLAlchemy async URL — production Postgres (``postgresql+asyncpg``)
    and serverless SQLite (``sqlite+aiosqlite``) share the same code paths; Postgres
    extras (``FOR UPDATE``, advisory locks) degrade gracefully on SQLite.

    Note: the factory caches one adapter per ``lock_key`` (Ladybug per-db lock_key
    instantiation pattern), so several instances may share one database.
    """

    def __init__(
        self,
        connection_string: str,
        lock_key: str = "default_lock",
        log_key: str = "usage_logs",
        session_ttl_seconds: Optional[int] = 604800,
        agentic_lock_expire: int = 240,
        agentic_lock_timeout: int = 300,
        purge_interval_seconds: int = 900,
    ):
        """Create the async engine lazily-validated on first use (no eager connect)."""
        super().__init__(host="", port=0, lock_key=lock_key, log_key=log_key)

        self.db_uri = connection_string
        self.session_ttl_seconds = session_ttl_seconds
        self.agentic_lock_expire = agentic_lock_expire
        self.agentic_lock_timeout = agentic_lock_timeout
        self.purge_interval_seconds = purge_interval_seconds

        try:
            url = make_url(connection_string)
            self._is_postgres = url.get_backend_name() == "postgresql"

            is_sqlite = url.get_backend_name() == "sqlite"

            relational_config = get_relational_config()
            pool_args: dict = (
                dict(relational_config.pool_args) if relational_config.pool_args else {}
            )
            if is_sqlite:
                # Concurrency tuning: wait out writer locks instead of failing
                # with SQLITE_BUSY when several processes share one cache.db.
                connect_args = dict(pool_args.pop("connect_args", None) or {})
                connect_args.setdefault("timeout", 30)
                pool_args["connect_args"] = connect_args

            self.engine = create_async_engine(
                connection_string,
                json_serializer=lambda obj: json.dumps(obj, cls=JSONEncoder),
                **pool_args,
            )
            if is_sqlite:

                @event.listens_for(self.engine.sync_engine, "connect")
                def _set_sqlite_pragmas(dbapi_connection, connection_record):
                    cursor = dbapi_connection.cursor()
                    cursor.execute("PRAGMA journal_mode=WAL")
                    cursor.execute("PRAGMA busy_timeout=30000")
                    cursor.close()

            self.sessionmaker = async_sessionmaker(bind=self.engine, expire_on_commit=False)
        except ModuleNotFoundError as error:
            raise CacheConnectionError(
                "SQL cache backend driver is not installed "
                "(CACHE_BACKEND=postgres requires cognee[postgres]): " + str(error)
            ) from error
        except Exception as error:
            raise CacheConnectionError(
                f"Failed to initialize SQL cache engine for {connection_string}: {error}"
            ) from error

        self._sync_lock_engine = None
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._last_purge = 0.0

    # --------------------------------------------------------------------- #
    # Initialization / shared helpers
    # --------------------------------------------------------------------- #

    async def _ensure_initialized(self) -> None:
        """Create cache tables on first use; wrap first-connect failures."""
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            try:
                async with self.engine.begin() as connection:
                    await connection.run_sync(cache_metadata.create_all, checkfirst=True)
            except Exception as error:
                error_msg = f"Failed to connect to SQL cache database: {error}"
                logger.error(error_msg)
                raise CacheConnectionError(error_msg) from error
            self._initialized = True

    @staticmethod
    def _now() -> datetime:
        """Current UTC time, timezone-aware (used for all expiry math)."""
        return datetime.now(timezone.utc)

    def _ttl_enabled(self) -> bool:
        """Whether session-scoped sliding TTL is active."""
        return bool(self.session_ttl_seconds and self.session_ttl_seconds > 0)

    def _session_expiry(self) -> Optional[datetime]:
        """Expiry timestamp for session-scoped rows, or None when TTL is disabled."""
        if not self._ttl_enabled():
            return None
        return self._now() + timedelta(seconds=self.session_ttl_seconds)

    def _not_expired(self, table):
        """Read-time expiry filter shared by every SELECT."""
        return or_(table.c.expires_at.is_(None), table.c.expires_at > self._now())

    def _session_filter(self, table, user_id: str, session_id: str):
        """WHERE clause for one session's rows."""
        return (table.c.user_id == user_id) & (table.c.session_id == session_id)

    async def _refresh_session_ttl(self, session, table, user_id: str, session_id: str) -> None:
        """Slide the whole session's expiry forward (Redis EXPIRE-on-write parity)."""
        if not self._ttl_enabled():
            return
        await session.execute(
            update(table)
            .where(self._session_filter(table, user_id, session_id))
            .values(expires_at=self._session_expiry())
        )

    async def _purge_session_expired(self, session, table, user_id: str, session_id: str) -> None:
        """Scoped lazy purge: drop this session's already-expired rows on write."""
        await session.execute(
            delete(table).where(
                self._session_filter(table, user_id, session_id),
                table.c.expires_at.isnot(None),
                table.c.expires_at <= self._now(),
            )
        )

    async def _maybe_purge_expired(self) -> None:
        """Throttled global TTL sweep — at most once per purge_interval_seconds.

        Guarded by a transaction-scoped Postgres advisory lock so concurrent
        workers don't stampede; it auto-releases at COMMIT/ROLLBACK (even when
        the transaction aborts) so it can never leak on a pooled connection.
        The guard is skipped on SQLite. Failures are swallowed: correctness
        never depends on purging (reads filter on expires_at).
        """
        if self.purge_interval_seconds <= 0:
            return
        now = time.monotonic()
        if self._last_purge and (now - self._last_purge) < self.purge_interval_seconds:
            return
        self._last_purge = now

        try:
            async with self.sessionmaker() as session, session.begin():
                acquired = True
                if self._is_postgres:
                    acquired = (
                        await session.execute(
                            text("SELECT pg_try_advisory_xact_lock(:lock_id)"),
                            {"lock_id": _PURGE_LOCK_ID},
                        )
                    ).scalar()
                if not acquired:
                    return
                cutoff = self._now()
                for table in (
                    cache_qa_entries,
                    cache_trace_entries,
                    cache_session_context,
                    cache_usage_logs,
                    cache_kv,
                ):
                    await session.execute(
                        delete(table).where(
                            table.c.expires_at.isnot(None), table.c.expires_at <= cutoff
                        )
                    )
        except Exception as error:
            logger.debug("SQL cache TTL sweep failed (will retry next interval): %s", error)

    @staticmethod
    def _build_qa_entry_dump(
        question: str,
        context: str,
        answer: str,
        qa_id: Optional[str] = None,
        feedback_text: Optional[str] = None,
        feedback_score: Optional[int] = None,
        used_graph_element_ids: Optional[dict] = None,
        memify_metadata: Optional[dict] = None,
        used_session_context_ids: Optional[list] = None,
    ) -> dict:
        """Serialize one QA entry into the normalized cache payload shape."""
        entry = SessionQAEntry(
            time=datetime.utcnow().isoformat(),
            question=question,
            context=context,
            answer=answer,
            qa_id=qa_id or str(uuid.uuid4()),
            feedback_text=feedback_text,
            feedback_score=feedback_score,
            used_graph_element_ids=used_graph_element_ids,
            memify_metadata=memify_metadata,
            used_session_context_ids=used_session_context_ids,
        )
        return entry.model_dump()

    @staticmethod
    def _build_agent_trace_entry_dump(
        trace_id: str,
        origin_function: str,
        status: str,
        memory_query: str = "",
        memory_context: str = "",
        method_params: Optional[dict] = None,
        method_return_value=None,
        error_message: str = "",
        session_feedback: str = "",
    ) -> dict:
        """Serialize one agent-trace step into the normalized cache payload shape."""
        entry = SessionAgentTraceEntry(
            trace_id=trace_id,
            origin_function=origin_function,
            status=status,
            memory_query=memory_query,
            memory_context=memory_context,
            method_params=method_params or {},
            method_return_value=method_return_value,
            error_message=error_message,
            session_feedback=session_feedback,
        )
        return entry.model_dump()

    @staticmethod
    def _merge_entry_update(
        entry: dict,
        question: Optional[str] = None,
        context: Optional[str] = None,
        answer: Optional[str] = None,
        feedback_text: Optional[str] = None,
        feedback_score: Optional[int] = None,
        used_graph_element_ids: Optional[dict] = None,
        memify_metadata: Optional[dict] = None,
        used_session_context_ids: Optional[list] = None,
    ) -> dict:
        """Merge partial QA updates into an existing payload; None preserves values."""
        merged = {**entry}
        if question is not None:
            merged["question"] = question
        if context is not None:
            merged["context"] = context
        if answer is not None:
            merged["answer"] = answer
        if feedback_text is not None:
            merged["feedback_text"] = feedback_text
        if feedback_score is not None:
            merged["feedback_score"] = feedback_score
        if used_graph_element_ids is not None:
            merged["used_graph_element_ids"] = used_graph_element_ids
        if used_session_context_ids is not None:
            merged["used_session_context_ids"] = used_session_context_ids
        if memify_metadata is not None:
            existing_metadata = merged.get("memify_metadata")
            if isinstance(existing_metadata, dict):
                merged["memify_metadata"] = {**existing_metadata, **memify_metadata}
            else:
                merged["memify_metadata"] = memify_metadata
        return merged

    @staticmethod
    def _merge_entry_clear_feedback(entry: dict) -> dict:
        """Return a copy of the entry with feedback fields cleared."""
        return {**entry, "feedback_text": None, "feedback_score": None}

    @staticmethod
    def _validate_entry_dict(entry_dict: dict) -> dict:
        """Validate one serialized QA entry and return its normalized dump."""
        try:
            return SessionQAEntry.model_validate(entry_dict).model_dump()
        except ValidationError as error:
            raise SessionQAEntryValidationError(
                message=f"Session QA entry validation failed: {error!s}"
            ) from error

    # --------------------------------------------------------------------- #
    # Locks (sync — called via asyncio.to_thread by the Ladybug graph adapter)
    # --------------------------------------------------------------------- #

    def _get_sync_lock_engine(self):
        """Lazy sync engine (psycopg2, pool_size=2) used only for advisory locks."""
        if self._sync_lock_engine is None:
            sync_url = make_url(self.db_uri).set(drivername="postgresql+psycopg2")
            self._sync_lock_engine = create_engine(
                sync_url, pool_size=2, isolation_level="AUTOCOMMIT"
            )
        return self._sync_lock_engine

    def acquire_lock(self):
        """Acquire a Postgres advisory lock keyed by lock_key. (Sync because of Ladybug)

        Returns a handle owning the checked-out connection; the lock lives for the
        connection's lifetime (released on release_lock or connection death — no
        Redis-style auto-expiry watchdog). Raises RuntimeError on timeout and
        SharedLadybugLockRequiresRedisError on non-Postgres URLs.
        """
        if not self._is_postgres:
            logger.error("Shared Ladybug lock requires Redis or Postgres cache backend.")
            raise SharedLadybugLockRequiresRedisError()

        lock_id = int.from_bytes(sha256(self.lock_key.encode()).digest()[:8], "big", signed=True)
        deadline = time.monotonic() + self.agentic_lock_timeout

        connection = self._get_sync_lock_engine().connect()
        try:
            while True:
                acquired = connection.execute(
                    text("SELECT pg_try_advisory_lock(:lock_id)"), {"lock_id": lock_id}
                ).scalar()
                if acquired:
                    handle = _SqlAdvisoryLockHandle(connection, lock_id)
                    self.lock = handle
                    return handle
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Could not acquire Postgres advisory lock: {self.lock_key}")
                time.sleep(0.1)
        except BaseException:
            try:
                connection.close()
            except Exception as error:
                logger.debug("Error closing advisory lock connection: %s", error)
            raise

    def release_lock(self, lock=None):
        """Release the passed advisory-lock handle, if held. (Sync because of Ladybug)"""
        if not self._is_postgres:
            logger.error("Shared Ladybug lock requires Redis or Postgres cache backend.")
            raise SharedLadybugLockRequiresRedisError()

        handle = lock if lock is not None else self.lock
        if handle is None:
            return
        try:
            handle.release()
        except Exception as error:
            logger.debug("Error releasing Postgres advisory lock: %s", error)
        finally:
            if handle is self.lock:
                self.lock = None

    # --------------------------------------------------------------------- #
    # QA entries
    # --------------------------------------------------------------------- #

    async def create_qa_entry(
        self,
        user_id: str,
        session_id: str,
        question: str,
        context: str,
        answer: str,
        qa_id: Optional[str] = None,
        feedback_text: Optional[str] = None,
        feedback_score: Optional[int] = None,
        used_graph_element_ids: Optional[dict] = None,
        memify_metadata: Optional[dict] = None,
        used_session_context_ids: Optional[list] = None,
    ) -> None:
        """Append one QA entry to the session. Creates the session if it doesn't exist."""
        await self._ensure_initialized()
        try:
            qa_entry = self._build_qa_entry_dump(
                question,
                context,
                answer,
                qa_id,
                feedback_text,
                feedback_score,
                used_graph_element_ids=used_graph_element_ids,
                memify_metadata=memify_metadata,
                used_session_context_ids=used_session_context_ids,
            )
            async with self.sessionmaker() as session, session.begin():
                await self._purge_session_expired(session, cache_qa_entries, user_id, session_id)
                await session.execute(
                    insert(cache_qa_entries).values(
                        user_id=user_id,
                        session_id=session_id,
                        qa_id=qa_entry["qa_id"],
                        payload=qa_entry,
                        expires_at=self._session_expiry(),
                    )
                )
                await self._refresh_session_ttl(session, cache_qa_entries, user_id, session_id)
        except Exception as error:
            error_msg = f"Unexpected error while adding Q&A to SQL cache: {error}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from error
        await self._maybe_purge_expired()

    async def get_latest_qa_entries(
        self, user_id: str, session_id: str, last_n: int = 5
    ) -> List[SessionQAEntry]:
        """Return the most recent QA entries (chronological); [] when none, for all last_n."""
        if last_n <= 0:
            return []

        await self._ensure_initialized()
        try:
            async with self.sessionmaker() as session:
                result = await session.execute(
                    select(cache_qa_entries.c.payload)
                    .where(
                        self._session_filter(cache_qa_entries, user_id, session_id),
                        self._not_expired(cache_qa_entries),
                    )
                    .order_by(cache_qa_entries.c.seq.desc())
                    .limit(last_n)
                )
                rows = result.scalars().all()
            return [SessionQAEntry.model_validate(payload) for payload in reversed(rows)]
        except Exception as error:
            error_msg = f"Unexpected error while reading Q&A from SQL cache: {error}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from error

    async def get_all_qa_entries(self, user_id: str, session_id: str) -> List[SessionQAEntry]:
        """Return all QA entries stored for the given session, oldest first."""
        await self._ensure_initialized()
        try:
            async with self.sessionmaker() as session:
                result = await session.execute(
                    select(cache_qa_entries.c.payload)
                    .where(
                        self._session_filter(cache_qa_entries, user_id, session_id),
                        self._not_expired(cache_qa_entries),
                    )
                    .order_by(cache_qa_entries.c.seq.asc())
                )
                rows = result.scalars().all()
            return [SessionQAEntry.model_validate(payload) for payload in rows]
        except Exception as error:
            error_msg = f"Unexpected error while reading Q&A from SQL cache: {error}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from error

    async def get_qa_entries_by_ids(
        self,
        user_id: str,
        session_id: str,
        qa_ids: List[str],
    ) -> List[SessionQAEntry]:
        """Return matching QA entries for the given session, oldest first."""
        if not qa_ids:
            return []

        await self._ensure_initialized()
        try:
            async with self.sessionmaker() as session:
                result = await session.execute(
                    select(cache_qa_entries.c.payload)
                    .where(
                        self._session_filter(cache_qa_entries, user_id, session_id),
                        cache_qa_entries.c.qa_id.in_(qa_ids),
                        self._not_expired(cache_qa_entries),
                    )
                    .order_by(cache_qa_entries.c.seq.asc())
                )
                rows = result.scalars().all()
            return [SessionQAEntry.model_validate(payload) for payload in rows]
        except Exception as error:
            error_msg = f"Unexpected error while reading Q&A by ids from SQL cache: {error}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from error

    async def _update_qa_payload(self, user_id: str, session_id: str, qa_id: str, merge_fn) -> bool:
        """Shared FOR UPDATE read-merge-write transaction for QA updates."""
        attempt = 0
        while True:
            try:
                async with self.sessionmaker() as session, session.begin():
                    result = await session.execute(
                        select(cache_qa_entries.c.payload)
                        .where(
                            self._session_filter(cache_qa_entries, user_id, session_id),
                            cache_qa_entries.c.qa_id == qa_id,
                            self._not_expired(cache_qa_entries),
                        )
                        .with_for_update()
                    )
                    payload = result.scalar_one_or_none()
                    if payload is None:
                        return False
                    validated = self._validate_entry_dict(merge_fn(dict(payload)))
                    await session.execute(
                        update(cache_qa_entries)
                        .where(
                            self._session_filter(cache_qa_entries, user_id, session_id),
                            cache_qa_entries.c.qa_id == qa_id,
                        )
                        .values(payload=validated)
                    )
                    await self._refresh_session_ttl(session, cache_qa_entries, user_id, session_id)
                    return True
            except DBAPIError as error:
                attempt += 1
                if _is_deadlock_error(error) and attempt < _DEADLOCK_ATTEMPTS:
                    await asyncio.sleep(0.05 * (2**attempt))
                    continue
                raise

    async def update_qa_entry(
        self,
        user_id: str,
        session_id: str,
        qa_id: str,
        question: Optional[str] = None,
        context: Optional[str] = None,
        answer: Optional[str] = None,
        feedback_text: Optional[str] = None,
        feedback_score: Optional[int] = None,
        used_graph_element_ids: Optional[dict] = None,
        memify_metadata: Optional[dict] = None,
        used_session_context_ids: Optional[list] = None,
    ) -> bool:
        """
        Update a QA entry by qa_id. Same QA fields as create_qa_entry.
        Only passed fields are updated; None preserves existing values.
        Returns True if updated, False if qa_id not found.
        """
        await self._ensure_initialized()
        try:
            return await self._update_qa_payload(
                user_id,
                session_id,
                qa_id,
                lambda entry: self._merge_entry_update(
                    entry,
                    question,
                    context,
                    answer,
                    feedback_text,
                    feedback_score,
                    used_graph_element_ids=used_graph_element_ids,
                    memify_metadata=memify_metadata,
                    used_session_context_ids=used_session_context_ids,
                ),
            )
        except SessionQAEntryValidationError:
            raise
        except Exception as error:
            error_msg = f"Unexpected error while updating Q&A in SQL cache: {error}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from error

    async def delete_feedback(self, user_id: str, session_id: str, qa_id: str) -> bool:
        """Set feedback_text and feedback_score to None for a QA entry."""
        await self._ensure_initialized()
        try:
            return await self._update_qa_payload(
                user_id, session_id, qa_id, self._merge_entry_clear_feedback
            )
        except SessionQAEntryValidationError:
            raise
        except Exception as error:
            error_msg = f"Unexpected error while clearing feedback in SQL cache: {error}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from error

    async def delete_qa_entry(self, user_id: str, session_id: str, qa_id: str) -> bool:
        """
        Delete a single QA entry by qa_id (single atomic DELETE).
        Returns True if deleted, False if qa_id not found.
        """
        await self._ensure_initialized()
        try:
            async with self.sessionmaker() as session, session.begin():
                result = await session.execute(
                    delete(cache_qa_entries).where(
                        self._session_filter(cache_qa_entries, user_id, session_id),
                        cache_qa_entries.c.qa_id == qa_id,
                        self._not_expired(cache_qa_entries),
                    )
                )
                deleted = result.rowcount > 0
                if deleted:
                    await self._refresh_session_ttl(session, cache_qa_entries, user_id, session_id)
                return deleted
        except Exception as error:
            error_msg = f"Unexpected error while deleting Q&A from SQL cache: {error}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from error

    async def delete_session(self, user_id: str, session_id: str) -> bool:
        """
        Delete the entire session (QA entries + agent traces).
        Returns True if any live session data existed, False otherwise.
        """
        await self._ensure_initialized()
        try:
            async with self.sessionmaker() as session, session.begin():
                deleted_rows = 0
                for table in (cache_qa_entries, cache_trace_entries, cache_session_context):
                    # Expired rows are invisible — drop them first so they don't
                    # count toward "session existed".
                    await self._purge_session_expired(session, table, user_id, session_id)
                    result = await session.execute(
                        delete(table).where(self._session_filter(table, user_id, session_id))
                    )
                    deleted_rows += result.rowcount
                return deleted_rows > 0
        except Exception as error:
            error_msg = f"Unexpected error while deleting session from SQL cache: {error}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from error

    # --------------------------------------------------------------------- #
    # Agent traces
    # --------------------------------------------------------------------- #

    async def append_agent_trace_step(
        self,
        user_id: str,
        session_id: str,
        trace_id: str,
        origin_function: str,
        status: str,
        memory_query: str = "",
        memory_context: str = "",
        method_params: Optional[dict] = None,
        method_return_value=None,
        error_message: str = "",
        session_feedback: str = "",
    ) -> None:
        """Append one trace step to the stored trace list for this session."""
        await self._ensure_initialized()
        try:
            trace_entry = self._build_agent_trace_entry_dump(
                trace_id=trace_id,
                origin_function=origin_function,
                status=status,
                memory_query=memory_query,
                memory_context=memory_context,
                method_params=method_params,
                method_return_value=method_return_value,
                error_message=error_message,
                session_feedback=session_feedback,
            )
            async with self.sessionmaker() as session, session.begin():
                await self._purge_session_expired(session, cache_trace_entries, user_id, session_id)
                await session.execute(
                    insert(cache_trace_entries).values(
                        user_id=user_id,
                        session_id=session_id,
                        payload=trace_entry,
                        expires_at=self._session_expiry(),
                    )
                )
                await self._refresh_session_ttl(session, cache_trace_entries, user_id, session_id)
        except Exception as error:
            error_msg = f"Unexpected error while appending agent trace step to SQL cache: {error}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from error
        await self._maybe_purge_expired()

    async def get_agent_trace_session(
        self, user_id: str, session_id: str, last_n: Optional[int] = None
    ) -> List[SessionAgentTraceEntry]:
        """Retrieve stored trace steps for the given session (reads don't refresh TTL)."""
        if last_n is not None and last_n <= 0:
            return []

        await self._ensure_initialized()
        try:
            async with self.sessionmaker() as session:
                query = select(cache_trace_entries.c.payload).where(
                    self._session_filter(cache_trace_entries, user_id, session_id),
                    self._not_expired(cache_trace_entries),
                )
                if last_n is not None:
                    result = await session.execute(
                        query.order_by(cache_trace_entries.c.seq.desc()).limit(last_n)
                    )
                    rows = list(reversed(result.scalars().all()))
                else:
                    result = await session.execute(query.order_by(cache_trace_entries.c.seq.asc()))
                    rows = result.scalars().all()
            return [SessionAgentTraceEntry.model_validate(payload) for payload in rows]
        except Exception as error:
            error_msg = f"Unexpected error while reading agent trace from SQL cache: {error}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from error

    async def get_agent_trace_feedback(
        self, user_id: str, session_id: str, last_n: Optional[int] = None
    ) -> List[str]:
        """Retrieve ordered per-step feedback for the given trace session."""
        entries = await self.get_agent_trace_session(user_id, session_id, last_n=last_n)
        return [entry.session_feedback for entry in entries]

    async def get_agent_trace_count(self, user_id: str, session_id: str) -> int:
        """Return the number of stored trace steps for the given session."""
        await self._ensure_initialized()
        try:
            async with self.sessionmaker() as session:
                result = await session.execute(
                    select(func.count())
                    .select_from(cache_trace_entries)
                    .where(
                        self._session_filter(cache_trace_entries, user_id, session_id),
                        self._not_expired(cache_trace_entries),
                    )
                )
                return result.scalar_one()
        except Exception as error:
            error_msg = f"Unexpected error while counting agent trace steps in SQL cache: {error}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from error

    # --------------------------------------------------------------------- #
    # Session context (active guidance: goals, rules, preferences, lessons)
    # --------------------------------------------------------------------- #

    async def create_session_context_entry(
        self, user_id: str, session_id: str, entry_dump: dict
    ) -> None:
        """Append one session-context entry (kind-discriminated dict) to the session.

        The caller validates the payload; we only promote its ``id`` to the
        ``entry_id`` column so updates can target a single row directly.
        """
        await self._ensure_initialized()
        # Redis/FS append regardless of "id" (an id-less entry is simply never
        # targetable by update). Mirror that: fall back to a synthetic entry_id
        # only to satisfy the NOT NULL column; the stored payload is untouched.
        entry_id = entry_dump.get("id") or str(uuid.uuid4())
        try:
            async with self.sessionmaker() as session, session.begin():
                await self._purge_session_expired(
                    session, cache_session_context, user_id, session_id
                )
                await session.execute(
                    insert(cache_session_context).values(
                        user_id=user_id,
                        session_id=session_id,
                        entry_id=entry_id,
                        payload=entry_dump,
                        expires_at=self._session_expiry(),
                    )
                )
                await self._refresh_session_ttl(session, cache_session_context, user_id, session_id)
        except Exception as error:
            error_msg = f"Unexpected error while adding session context to SQL cache: {error}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from error
        await self._maybe_purge_expired()

    async def get_session_context_entries(self, user_id: str, session_id: str) -> list[dict]:
        """Return all stored session-context entries for the session, oldest first."""
        await self._ensure_initialized()
        try:
            async with self.sessionmaker() as session:
                result = await session.execute(
                    select(cache_session_context.c.payload)
                    .where(
                        self._session_filter(cache_session_context, user_id, session_id),
                        self._not_expired(cache_session_context),
                    )
                    .order_by(cache_session_context.c.seq.asc())
                )
                return [dict(payload) for payload in result.scalars().all()]
        except Exception as error:
            error_msg = f"Unexpected error while reading session context from SQL cache: {error}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from error

    async def update_session_context_entry(
        self, user_id: str, session_id: str, entry_id: str, merge: dict
    ) -> bool:
        """Shallow-merge ``merge`` into the entry whose id is ``entry_id``.

        Returns True if a matching entry was updated, False otherwise.
        """
        await self._ensure_initialized()
        attempt = 0
        while True:
            try:
                async with self.sessionmaker() as session, session.begin():
                    result = await session.execute(
                        select(cache_session_context.c.payload)
                        .where(
                            self._session_filter(cache_session_context, user_id, session_id),
                            cache_session_context.c.entry_id == entry_id,
                            self._not_expired(cache_session_context),
                        )
                        .with_for_update()
                    )
                    payload = result.scalar_one_or_none()
                    if payload is None:
                        return False
                    merged = {**dict(payload), **merge}
                    await session.execute(
                        update(cache_session_context)
                        .where(
                            self._session_filter(cache_session_context, user_id, session_id),
                            cache_session_context.c.entry_id == entry_id,
                        )
                        .values(payload=merged)
                    )
                    await self._refresh_session_ttl(
                        session, cache_session_context, user_id, session_id
                    )
                    return True
            except DBAPIError as error:
                attempt += 1
                if _is_deadlock_error(error) and attempt < _DEADLOCK_ATTEMPTS:
                    await asyncio.sleep(0.05 * (2**attempt))
                    continue
                error_msg = f"Unexpected error while updating session context in SQL cache: {error}"
                logger.error(error_msg)
                raise CacheConnectionError(error_msg) from error
            except Exception as error:
                error_msg = f"Unexpected error while updating session context in SQL cache: {error}"
                logger.error(error_msg)
                raise CacheConnectionError(error_msg) from error

    async def delete_session_context(self, user_id: str, session_id: str) -> bool:
        """Delete all session-context entries for the session. True if any existed."""
        await self._ensure_initialized()
        try:
            async with self.sessionmaker() as session, session.begin():
                await self._purge_session_expired(
                    session, cache_session_context, user_id, session_id
                )
                result = await session.execute(
                    delete(cache_session_context).where(
                        self._session_filter(cache_session_context, user_id, session_id)
                    )
                )
                return result.rowcount > 0
        except Exception as error:
            error_msg = f"Unexpected error while deleting session context from SQL cache: {error}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from error

    # --------------------------------------------------------------------- #
    # Usage logs
    # --------------------------------------------------------------------- #

    async def log_usage(
        self,
        user_id: str,
        log_entry: dict,
        ttl: Optional[int] = 604800,
    ):
        """
        Log usage information (API endpoint calls, MCP tool invocations) to SQL cache.

        Args:
            user_id: The user ID.
            log_entry: Dictionary containing usage log information.
            ttl: Optional time-to-live (seconds). If provided, the whole per-user
                log list expires after this time (Redis EXPIREs the whole list,
                so every existing row's expiry is refreshed too).

        Raises:
            CacheConnectionError: If the cache connection fails.
        """
        await self._ensure_initialized()
        try:
            expires_at = self._now() + timedelta(seconds=ttl) if ttl else None
            async with self.sessionmaker() as session, session.begin():
                await session.execute(
                    insert(cache_usage_logs).values(
                        log_key=self.log_key,
                        user_id=user_id,
                        payload=log_entry,
                        expires_at=expires_at,
                    )
                )
                if expires_at is not None:
                    await session.execute(
                        update(cache_usage_logs)
                        .where(
                            cache_usage_logs.c.log_key == self.log_key,
                            cache_usage_logs.c.user_id == user_id,
                        )
                        .values(expires_at=expires_at)
                    )
        except Exception as error:
            error_msg = f"Unexpected error while logging usage to SQL cache: {error}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from error
        await self._maybe_purge_expired()

    async def get_usage_logs(self, user_id: str, limit: int = 100):
        """
        Retrieve usage logs for a given user.

        Args:
            user_id: The user ID.
            limit: Maximum number of logs to retrieve (default: 100).

        Returns:
            List of usage log entries, most recent first.
        """
        await self._ensure_initialized()
        try:
            async with self.sessionmaker() as session:
                result = await session.execute(
                    select(cache_usage_logs.c.payload)
                    .where(
                        cache_usage_logs.c.log_key == self.log_key,
                        cache_usage_logs.c.user_id == user_id,
                        self._not_expired(cache_usage_logs),
                    )
                    .order_by(cache_usage_logs.c.seq.desc())
                    .limit(limit)
                )
                return list(result.scalars().all())
        except Exception as error:
            error_msg = f"Unexpected error while retrieving usage logs from SQL cache: {error}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from error

    # --------------------------------------------------------------------- #
    # Key/value storage (small exact-key cache values)
    # --------------------------------------------------------------------- #

    async def get_value(self, key: str) -> Optional[str]:
        """Return the string value stored under key, or None if absent/expired."""
        await self._ensure_initialized()
        try:
            async with self.sessionmaker() as session:
                result = await session.execute(
                    select(cache_kv.c.value).where(
                        cache_kv.c.key == key, self._not_expired(cache_kv)
                    )
                )
                return result.scalar_one_or_none()
        except Exception as error:
            error_msg = f"Unexpected error while reading key/value from SQL cache: {error}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from error

    async def set_value(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        """Upsert a string value under key; ttl=None stores it without expiry."""
        await self._ensure_initialized()
        try:
            if self._is_postgres:
                from sqlalchemy.dialects.postgresql import insert as upsert_insert
            else:
                from sqlalchemy.dialects.sqlite import insert as upsert_insert

            expires_at = self._now() + timedelta(seconds=ttl) if ttl else None
            statement = upsert_insert(cache_kv).values(key=key, value=value, expires_at=expires_at)
            statement = statement.on_conflict_do_update(
                index_elements=[cache_kv.c.key],
                set_={
                    "value": statement.excluded.value,
                    "expires_at": statement.excluded.expires_at,
                },
            )
            async with self.sessionmaker() as session, session.begin():
                await session.execute(statement)
        except Exception as error:
            error_msg = f"Unexpected error while writing key/value to SQL cache: {error}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from error
        await self._maybe_purge_expired()

    async def delete_value(self, key: str) -> None:
        """Delete the value stored under key, if present."""
        await self._ensure_initialized()
        try:
            async with self.sessionmaker() as session, session.begin():
                await session.execute(delete(cache_kv).where(cache_kv.c.key == key))
        except Exception as error:
            error_msg = f"Unexpected error while deleting key/value from SQL cache: {error}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from error

    # --------------------------------------------------------------------- #
    # Maintenance
    # --------------------------------------------------------------------- #

    async def prune(self) -> None:
        """
        Empty the cache. Scoped to the four cognee cache tables only — a deliberate,
        safer divergence from Redis FLUSHDB (which nukes co-tenant keys).
        """
        await self._ensure_initialized()
        try:
            async with self.sessionmaker() as session, session.begin():
                for table in (
                    cache_qa_entries,
                    cache_trace_entries,
                    cache_session_context,
                    cache_usage_logs,
                    cache_kv,
                ):
                    await session.execute(delete(table))
        except Exception as error:
            error_msg = f"Unexpected error while pruning SQL cache: {error}"
            logger.error(error_msg)
            raise CacheConnectionError(error_msg) from error

    async def close(self):
        """Dispose engines. Idempotent; a reused instance lazily re-initializes."""
        try:
            await self.engine.dispose(close=True)
        except Exception as error:
            logger.debug("Error closing SQL cache async engine: %s", error)
        if self._sync_lock_engine is not None:
            try:
                self._sync_lock_engine.dispose(close=True)
            except Exception as error:
                logger.debug("Error closing SQL cache sync lock engine: %s", error)
            self._sync_lock_engine = None
        self._initialized = False
