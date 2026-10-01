from typing import Any, Literal

from pydantic import Field, model_validator

from cognee.infrastructure.engine import DataPoint

# "span" is a period with a stated start and end (``1803/1815``): its
# calendar fields and ``time_at`` are the start, ``time_until`` the end.
TimestampPrecision = Literal["year", "month", "day", "second", "span"]


class Timestamp(DataPoint):
    """A point in time that facts anchor to through ``*_at`` edges.

    ``timestamp_str`` is the normalized form the source stated — ``YYYY``,
    ``YYYY-MM``, ``YYYY-MM-DD``, ``YYYY-MM-DD HH:MM:SS``, or ``<start>/<end>``
    for a period with both bounds stated — and ``precision``
    records how much of it was stated, so ``1950`` and ``1950-01-01`` stay
    distinguishable although their calendar fields are the same. The calendar
    fields and ``time_at`` (milliseconds since the epoch, UTC) hold the lower
    bound of that period, with unstated parts filled by the earliest value;
    ``time_until`` is the period's exclusive upper bound, so a time window
    overlaps this timestamp exactly when ``time_at < window_end`` and
    ``time_until > window_start`` — the test the graph adapters run.
    ``name`` repeats ``timestamp_str``: it is what graph renderers and the
    hybrid retrieval context label a node by. Both ``name`` and ``time_until``
    are filled in when a caller does not pass them.
    """

    name: str = ""
    timestamp_str: str = Field(...)
    precision: TimestampPrecision = "second"
    time_at: int = Field(...)
    # Exclusive upper bound in ms; defaults to one second after ``time_at``, the
    # period a full ``YYYY-MM-DD HH:MM:SS`` timestamp names.
    time_until: int = 0
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
    def _fill_defaults(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        filled = dict(data)
        if not filled.get("name") and filled.get("timestamp_str"):
            filled["name"] = filled["timestamp_str"]
        if not filled.get("time_until") and isinstance(filled.get("time_at"), int):
            filled["time_until"] = filled["time_at"] + 1000
        return filled
