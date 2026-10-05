from collections.abc import Iterator
from typing import Any
from uuid import UUID


def iter_ingestion_entries(data_ingestion_info: Any) -> Iterator[tuple[str, UUID | None]]:
    """Yield ``(status, data_id)`` for each per-item entry of a run.

    Entries are ``{"run_info": ..., "data_id": ...}`` dicts; ``run_info`` may be
    a run info model or, once the result went through JSON, a plain dict. The
    status is ``""`` when the entry has none, and ``data_id`` is ``None`` when it
    is missing or not a UUID. Nothing is filtered or deduplicated: callers
    decide which statuses count.
    """
    if not isinstance(data_ingestion_info, list):
        return
    for entry in data_ingestion_info:
        if isinstance(entry, dict):
            yield _entry_status(entry.get("run_info")), _as_uuid(entry.get("data_id"))


def _entry_status(run_info: Any) -> str:
    if isinstance(run_info, dict):
        status = run_info.get("status")
    else:
        status = getattr(run_info, "status", None)
    return status if isinstance(status, str) else ""


def _as_uuid(value: Any) -> UUID | None:
    if value is None or isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except ValueError:
        return None
