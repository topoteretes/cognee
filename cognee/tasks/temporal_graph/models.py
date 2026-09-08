from typing import List, Optional

from pydantic import BaseModel, Field


class Timestamp(BaseModel):
    year: int = Field(
        ...,
        ge=1,
        le=9999,
        description="Always required. If only a year is known, use it.",
    )
    month: int = Field(1, ge=1, le=12, description="If unknown, default to 1")
    day: int = Field(1, ge=1, le=31, description="If unknown, default to 1")
    hour: int = Field(0, ge=0, le=23, description="If unknown, default to 0")
    minute: int = Field(0, ge=0, le=59, description="If unknown, default to 0")
    second: int = Field(0, ge=0, le=59, description="If unknown, default to 0")


class Interval(BaseModel):
    starts_at: Timestamp
    ends_at: Timestamp


class QueryInterval(BaseModel):
    starts_at: Timestamp | None = None
    ends_at: Timestamp | None = None


class Event(BaseModel):
    name: str
    description: str | None = None
    time_from: Timestamp | None = None
    time_to: Timestamp | None = None
    location: str | None = None


class EventList(BaseModel):
    events: list[Event]


class EntityAttribute(BaseModel):
    entity: str
    entity_type: str
    relationship: str


class EventWithEntities(BaseModel):
    event_name: str
    description: str | None = None
    attributes: list[EntityAttribute] = []


class EventEntityList(BaseModel):
    events: list[EventWithEntities]
