from datetime import datetime, timezone
from uuid import uuid4
from sqlalchemy import Column, DateTime, JSON, UUID

from cognee.modules.data.models.metrics_base import MetricsBase


class Metrics(MetricsBase):
    __tablename__ = "eval_metrics"

    id = Column(UUID, primary_key=True, default=uuid4)

    payload = Column(JSON, nullable=False)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
