"""SQLAlchemy models for session lifecycle."""

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, UUID

from cognee.infrastructure.databases.relational import Base
from cognee.infrastructure.session.session_scope import parse_storage_session_id


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

    # Unscoped/legacy rows store the caller-provided ID directly. New
    # dataset-scoped rows store a deterministic internal ID here, allowing the
    # same public ID to exist once per dataset without changing the existing PK.
    session_id = Column(String, primary_key=True)
    user_id = Column(UUID, primary_key=True, index=True)

    # Security marker as well as display value. Only rows with this marker may
    # be shared through dataset permissions. Existing rows remain NULL even if
    # an older release populated dataset_id, because their cache may be mixed.
    public_session_id = Column(String, nullable=True, index=True)
    dataset_id = Column(UUID, nullable=True, index=True)

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

    def to_dict(self) -> dict:
        started = getattr(self, "started_at", None)
        last_act = getattr(self, "last_activity_at", None)
        ended = getattr(self, "ended_at", None)
        public_session_id = getattr(self, "public_session_id", None)
        exposed_session_id = public_session_id or self.session_id
        # A legacy row's sticky dataset_id is not trustworthy: older cache keys
        # were not dataset-scoped. Hide it so callers cannot mistake that row
        # for a safely shareable dataset session.
        dataset = getattr(self, "dataset_id", None) if public_session_id is not None else None
        return {
            "session_id": exposed_session_id,
            "public_session_id": exposed_session_id,
            "user_id": str(self.user_id),
            "dataset_id": str(dataset) if dataset is not None else None,
            "dataset_scoped": public_session_id is not None,
            "status": self.status,
            "started_at": started.isoformat() if started is not None else None,
            "last_activity_at": last_act.isoformat() if last_act is not None else None,
            "ended_at": ended.isoformat() if ended is not None else None,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": self.cost_usd,
            "error_count": self.error_count,
            "last_model": self.last_model,
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
        public_session_id = parse_storage_session_id(self.session_id).session_id
        return {
            "session_id": public_session_id,
            "user_id": str(self.user_id),
            "model": self.model,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": self.cost_usd,
            "updated_at": updated.isoformat() if updated is not None else None,
        }
