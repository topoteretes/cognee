from pydantic import Field
from cognee.infrastructure.engine import DataPoint


class Timestamp(DataPoint):
    time_at: int = Field(...)
    year: int = Field(...)
    month: int = Field(...)
    day: int = Field(...)
    hour: int = Field(...)
    minute: int = Field(...)
    second: int = Field(...)
    timestamp_str: str = Field(...)
