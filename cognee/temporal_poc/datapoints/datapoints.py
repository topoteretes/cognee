from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models.EntityType import EntityType
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class Interval(DataPoint):
    time_from: int = Field(..., ge=0)
    time_to: int = Field(..., ge=0)


class Timestamp(DataPoint):
    time_at: int = Field(..., ge=0)


class Event(DataPoint):
    name: str
    description: Optional[str] = None
    at: Optional[Timestamp] = None
    during: Optional[Interval] = None
    location: Optional[str] = None

    metadata: dict = {"index_fields": ["name"]}
