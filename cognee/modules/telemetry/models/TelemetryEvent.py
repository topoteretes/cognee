from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import JSON, Column, DateTime, String, UUID

from cognee.infrastructure.databases.relational import Base


class TelemetryEvent(Base):
    """One product-telemetry event, stored locally instead of (or alongside) being
    POSTed to the analytics proxy.

    Column names mirror the warehouse table the HTTP sink ultimately feeds
    (``analytics.main.pipeline_events``) so queries port between the two without
    a reshape:

    - ``event_name`` here == ``tracking_event`` there (the *cognee* event name;
      the warehouse's own ``event_name`` holds the Segment call type, which is an
      artifact of that transport and has no meaning locally).
    - ``properties`` is the same payload dict either way.

    Rows are pruned to a retention window by the sink (see
    ``TELEMETRY_RETENTION_DAYS``); this table is a rolling window, not an archive.
    """

    __tablename__ = "telemetry_events"
    # A process that prunes metadata and re-runs migrations (the performance
    # benchmark does this between runs) executes this declaration more than once
    # against the same MetaData, which is an error by default. Same reason
    # PGVectorAdapter sets it on its dynamically-built tables.
    __table_args__ = {"extend_existing": True}

    id = Column(UUID, primary_key=True, default=uuid4)

    event_name = Column(String, index=True)
    user_id = Column(UUID, index=True, nullable=True)
    tenant_id = Column(UUID, index=True, nullable=True)
    # Promoted out of ``properties`` so an activity view can be filtered per
    # dataset without a JSON scan. Null for events with no dataset (or with
    # several — a multi-dataset recall keeps the full list in ``properties``).
    dataset_id = Column(UUID, index=True, nullable=True)
    # Machine/install identity as sent to the proxy. Kept for parity with the
    # warehouse rows; on a per-tenant deployment it is constant per pod.
    anonymous_id = Column(String, nullable=True)

    properties = Column(JSON, nullable=True)

    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )
