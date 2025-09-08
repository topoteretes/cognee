from typing import Optional, List
from pydantic import BaseModel, Field


class Timestamp(BaseModel):
    year: int = Field(..., ge=1, le=9999)
    month: int = Field(..., ge=1, le=12)
    day: int = Field(..., ge=1, le=31)
    hour: int = Field(..., ge=0, le=23)
    minute: int = Field(..., ge=0, le=59)
    second: int = Field(..., ge=0, le=59)


class Interval(BaseModel):
    starts_at: Timestamp
    ends_at: Timestamp


class QueryInterval(BaseModel):
    starts_at: Optional[Timestamp] = None
    ends_at: Optional[Timestamp] = None


class Event(BaseModel):
    name: str
    description: Optional[str] = None
    time_from: Optional[Timestamp] = None
    time_to: Optional[Timestamp] = None
    location: Optional[str] = None


class EventList(BaseModel):
    events: List[Event]


class EntityAttribute(BaseModel):
    entity: str
    entity_type: str
    relationship: str


class EventWithEntities(BaseModel):
    event_name: str
    description: Optional[str] = None
    attributes: List[EntityAttribute] = []


class EventEntityList(BaseModel):
    events: List[EventWithEntities]
