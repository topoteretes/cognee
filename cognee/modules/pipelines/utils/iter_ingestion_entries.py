from collections.abc import Iterator
from typing import Any
from uuid import UUID


def iter_ingestion_entries(data_ingestion_info: Any) -> Iterator[tuple[str, list[UUID]]]:
    """Yield ``(status, data_ids)`` for each per-item entry of a run.

    Entries are ``{"run_info": ..., "data_id": ...}`` dicts; ``run_info`` may be
    a run info model or, once the result went through JSON, a plain dict. An
    item stored as several data items lists them all under ``data_ids``. The
    status is ``""`` when the entry has none; ids that are missing or not UUIDs
    are left out. Nothing is filtered or deduplicated: callers decide which
    statuses count.
    """
    if not isinstance(data_ingestion_info, list):
        return
    for entry in data_ingestion_info:
        if isinstance(entry, dict):
            yield _entry_status(entry.get("run_info")), _entry_data_ids(entry)


def _entry_status(run_info: Any) -> str:
    if isinstance(run_info, dict):
        status = run_info.get("status")
    else:
        status = getattr(run_info, "status", None)
    return status if isinstance(status, str) else ""


def _entry_data_ids(entry: dict) -> list[UUID]:
    raw_ids = entry.get("data_ids")
    if not isinstance(raw_ids, list):
        raw_ids = [entry.get("data_id")]
    return [data_id for data_id in map(_as_uuid, raw_ids) if data_id is not None]


def _as_uuid(value: Any) -> UUID | None:
    if value is None or isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except ValueError:
        return None
