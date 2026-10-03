from typing import Any, Literal

from pydantic import Field, model_validator

from cognee.infrastructure.engine import DataPoint

TimestampPrecision = Literal["year", "month", "day", "second"]


class Timestamp(DataPoint):
    """A point in time that facts anchor to through ``*_at`` edges.

    ``timestamp_str`` is the normalized form the source stated — ``YYYY``,
    ``YYYY-MM``, ``YYYY-MM-DD`` or ``YYYY-MM-DD HH:MM:SS`` — and ``precision``
    records how much of it was stated, so ``1950`` and ``1950-01-01`` stay
    distinguishable although their calendar fields are the same. The calendar
    fields and ``time_at`` (milliseconds since the epoch, UTC) hold the lower
    bound of that period, with unstated parts filled by the earliest value.
    ``name`` repeats ``timestamp_str``: it is what graph renderers and the
    hybrid retrieval context label a node by. It is filled from
    ``timestamp_str`` when a caller does not pass it.
    """

    name: str = ""
    timestamp_str: str = Field(...)
    precision: TimestampPrecision = "second"
    time_at: int = Field(...)
    year: int = Field(...)
    month: int = Field(...)
    day: int = Field(...)
    hour: int = Field(...)
    minute: int = Field(...)
    second: int = Field(...)
    # Not embedded: a bare time string carries no meaning for vector search.
    # Deterministic id from the normalized string, so every mention of the same
    # time across chunks and documents resolves to one node.
    metadata: dict = {"index_fields": [], "identity_fields": ["timestamp_str"]}

    @model_validator(mode="before")
    @classmethod
    def _name_defaults_to_timestamp_str(cls, data: Any) -> Any:
        if isinstance(data, dict) and not data.get("name") and data.get("timestamp_str"):
            return {**data, "name": data["timestamp_str"]}
        return data
