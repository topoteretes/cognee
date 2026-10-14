import re
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

_YEAR_MONTH = re.compile(r"^(\d{1,4})(?:-(\d{1,2}))?$")
_EMPTY_VALUES = {"", "none", "null"}


def _timestamp_fields(moment: datetime) -> dict:
    return {
        "year": moment.year,
        "month": moment.month,
        "day": moment.day,
        "hour": moment.hour,
        "minute": moment.minute,
        "second": moment.second,
    }


# LLMs do not always answer a Timestamp with the nested object: on the
# prompted-JSON path some return an ISO string ("2026-10-03T00:00:00"), a bare
# year (2009) or "now" — the extraction prompt itself uses those spellings.
# Those are coerced instead of failing validation (#5375). Kept as a comment,
# not a docstring, so the JSON schema sent to the LLM is unchanged.
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

    @model_validator(mode="before")
    @classmethod
    def _coerce_flat_value(cls, value: Any) -> Any:
        if isinstance(value, datetime):
            return _timestamp_fields(value)
        if isinstance(value, int) and not isinstance(value, bool):
            return {"year": value}
        if not isinstance(value, str):
            return value

        text = value.strip()
        if text.lower() in {"now", "today"}:
            return _timestamp_fields(datetime.now(timezone.utc))

        year_month = _YEAR_MONTH.match(text)
        if year_month:
            year, month = year_month.groups()
            return {"year": int(year), "month": int(month or 1)}

        try:
            # Python 3.10's fromisoformat rejects a trailing "Z". The wall-clock
            # values are kept as written, so a dated answer keeps its date.
            moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return value
        return _timestamp_fields(moment)


class Interval(BaseModel):
    starts_at: Timestamp
    ends_at: Timestamp


def _empty_to_none(value: Any) -> Any:
    if isinstance(value, str) and value.strip().lower() in _EMPTY_VALUES:
        return None
    return value


class QueryInterval(BaseModel):
    starts_at: Timestamp | None = None
    ends_at: Timestamp | None = None

    _coerce_empty = field_validator("starts_at", "ends_at", mode="before")(_empty_to_none)


class Event(BaseModel):
    name: str
    description: str | None = None
    time_from: Timestamp | None = None
    time_to: Timestamp | None = None
    location: str | None = None

    _coerce_empty = field_validator("time_from", "time_to", mode="before")(_empty_to_none)


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
