"""SQLAlchemy models for session lifecycle."""

from datetime import datetime, timezone

from sqlalchemy import UUID, Boolean, Column, DateTime, Float, Integer, String, Text, false

from cognee.infrastructure.databases.relational import Base


class SessionRecord(Base):
    """One row per (user, session_id).

    Narrow by design: lifecycle + aggregate counters only. The QA
    entries and trace steps themselves live in the session cache
    (Redis / FS) and are referenced here only indirectly via
    ``session_id`` (which is a string, not a FK — cache IDs aren't
    FK-able).

    Status is stored as a string but treated as an enum at the app
    layer (see ``SessionStatus``). The ``abandoned`` transition is
    inferred at read time from ``last_activity_at`` rather than
    written — a running session that's been idle past the threshold
    is reported as abandoned without any sweeper touching the row.
    """

    __tablename__ = "session_records"

    # Session ID is a caller-provided string (e.g. "cc_myproj_ab12cd34ef56"
    # from the Claude Code plugin). Scoped per user — same string from
    # two users is two different sessions.
    session_id = Column(String, primary_key=True)
    user_id = Column(UUID, primary_key=True, index=True)

    dataset_id = Column(UUID, nullable=True, index=True)

    # Agent-connection id (registry string like "my-agent-ab12cd34ef56",
    # not a UUID) — attributes the session to the agent that drove it.
    # Stamped best-effort when an agent connection registers with this
    # session_id; null for sessions no agent claimed.
    agent_id = Column(String, nullable=True, index=True)

    # Stored status. "abandoned" is the only value inferred at read
    # time instead of being stored — everything else (running,
    # completed, failed) is explicitly set.
    status = Column(String, nullable=False, default="running", index=True)

    started_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    last_activity_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )
    ended_at = Column(DateTime(timezone=True), nullable=True)

    # Aggregate counters accumulated by LLMGateway / SessionManager hooks.
    tokens_in = Column(Integer, nullable=False, default=0)
    tokens_out = Column(Integer, nullable=False, default=0)
    cost_usd = Column(Float, nullable=False, default=0.0)

    # Cheap per-session counters useful for the dashboard without
    # scanning the cache.
    error_count = Column(Integer, nullable=False, default=0)

    # Last-seen model string — informational only. Per-model cost
    # aggregates live in ``SessionModelUsage`` so mixed-model sessions
    # attribute correctly.
    last_model = Column(Text, nullable=True)

    # Cross-worker ``improve()`` lock (SDK-593). ``improve_lock_token`` is the
    # opaque token of the current holder (NULL = free); a lock whose
    # ``improve_lock_acquired_at`` is older than ``IMPROVE_LOCK_TTL_SECONDS``
    # counts as expired and may be taken over. ``improve_rerun_requested`` is
    # set by a caller that found the lock busy, so the holder runs one more
    # watermark-driven pass before releasing. See
    # ``cognee/infrastructure/locks/session_lock.py``.
    improve_lock_token = Column(String, nullable=True)
    improve_lock_acquired_at = Column(DateTime(timezone=True), nullable=True)
    improve_rerun_requested = Column(Boolean, nullable=False, default=False, server_default=false())

    def to_dict(self) -> dict:
        started = getattr(self, "started_at", None)
        last_act = getattr(self, "last_activity_at", None)
        ended = getattr(self, "ended_at", None)
        dataset = getattr(self, "dataset_id", None)
        return {
            "session_id": self.session_id,
            "user_id": str(self.user_id),
            "dataset_id": str(dataset) if dataset is not None else None,
            "agent_id": getattr(self, "agent_id", None),
            "status": self.status,
            "started_at": started.isoformat() if started is not None else None,
            "last_activity_at": last_act.isoformat() if last_act is not None else None,
            "ended_at": ended.isoformat() if ended is not None else None,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": self.cost_usd,
            "error_count": self.error_count,
            "last_model": self.last_model,
            "improve_lock_held": getattr(self, "improve_lock_token", None) is not None,
            "improve_rerun_requested": bool(getattr(self, "improve_rerun_requested", False)),
        }


class SessionModelUsage(Base):
    """Per-(session, user, model) token + cost aggregate.

    Populated by ``accumulate_usage`` when an LLM call fires inside a
    tracked session scope. Normalizing this out of ``SessionRecord``
    lets mixed-model sessions (e.g. embedding calls + completion calls
    on different models) attribute cost correctly in
    ``GET /api/v1/sessions/cost-by-model``.
    """

    __tablename__ = "session_model_usage"

    session_id = Column(String, primary_key=True)
    user_id = Column(UUID, primary_key=True, index=True)
    model = Column(Text, primary_key=True)

    tokens_in = Column(Integer, nullable=False, default=0)
    tokens_out = Column(Integer, nullable=False, default=0)
    cost_usd = Column(Float, nullable=False, default=0.0)

    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    def to_dict(self) -> dict:
        updated = getattr(self, "updated_at", None)
        return {
            "session_id": self.session_id,
            "user_id": str(self.user_id),
            "model": self.model,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": self.cost_usd,
            "updated_at": updated.isoformat() if updated is not None else None,
        }
