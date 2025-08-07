from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models.EntityType import EntityType
from typing import Optional, List, Any
from pydantic import BaseModel, Field, ConfigDict, SkipValidation
from cognee.infrastructure.engine.models.Edge import Edge
from cognee.modules.engine.models.Entity import Entity


class Timestamp(DataPoint):
    time_at: int = Field(...)
    year: int = Field(...)
    month: int = Field(...)
    day: int = Field(...)
    hour: int = Field(...)
    minute: int = Field(...)
    second: int = Field(...)
    timestamp_str: str = Field(...)


class Interval(DataPoint):
    time_from: Timestamp = Field(...)
    time_to: Timestamp = Field(...)


class Event(DataPoint):
    name: str
    description: Optional[str] = None
    at: Optional[Timestamp] = None
    during: Optional[Interval] = None
    location: Optional[str] = None
    attributes: SkipValidation[Any] = None  # (Edge, list[Entity])

    metadata: dict = {"index_fields": ["name"]}
