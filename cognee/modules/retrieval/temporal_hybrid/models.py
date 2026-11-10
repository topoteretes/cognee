"""The time window the LLM extracts from a TEMPORAL question (``extract_query_interval``)."""

from pydantic import BaseModel, Field


class QueryTime(BaseModel):
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


class QueryInterval(BaseModel):
    starts_at: QueryTime | None = None
    ends_at: QueryTime | None = None
