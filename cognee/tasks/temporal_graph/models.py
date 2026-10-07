from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field, model_validator


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

    @model_validator(mode="before")
    @classmethod
    def coerce_strings_to_timestamps(cls, data: Any) -> Any:
        def _parse(val: Any) -> Any:
            if val is None or isinstance(val, dict) or isinstance(val, Timestamp):
                return val
            if isinstance(val, str):
                try:
                    val = val.replace("Z", "+00:00")
                    dt = datetime.fromisoformat(val)
                    return {
                        "year": dt.year,
                        "month": dt.month,
                        "day": dt.day,
                        "hour": dt.hour,
                        "minute": dt.minute,
                        "second": dt.second,
                    }
                except ValueError as e:
                    raise ValueError(f"Could not parse ISO datetime string {val!r}: {e}") from e
            raise ValueError(
                f"Input should be an object or an ISO date string, got {type(val).__name__}"
            )

        if isinstance(data, str):
            data = {"starts_at": data}

        if isinstance(data, dict):
            new_data = data.copy()
            for field in ("starts_at", "ends_at"):
                if field in new_data:
                    new_data[field] = _parse(new_data[field])
            return new_data

        return data


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
