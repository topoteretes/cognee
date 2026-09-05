from typing import Literal, Optional, List
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
    precision: Optional[Literal["year", "month", "day", "hour", "minute", "second"]] = Field(
        None,
        description=(
            "The most specific unit actually stated in the text: 'year' for '1996', "
            "'month' for 'March 1996', 'day' for '5 March 1996'. Leave null if unsure."
        ),
    )


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
