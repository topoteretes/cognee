from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models.EntityType import EntityType
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class Timestamp(DataPoint):
    time_at: int = Field(...)


class Interval(DataPoint):
    time_from: Timestamp = Field(...)
    time_to: Timestamp = Field(...)


class Event(DataPoint):
    name: str
    description: Optional[str] = None
    at: Optional[Timestamp] = None
    during: Optional[Interval] = None
    location: Optional[str] = None

    metadata: dict = {"index_fields": ["name"]}
