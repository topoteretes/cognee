"""Local (relational-DB) telemetry sink.

Writes product-telemetry events into the deployment's own relational database
instead of shipping them to the analytics proxy. Selected via
``TELEMETRY_SINK=postgres`` (see ``cognee.shared.utils.send_telemetry``); the
default remains the HTTP sink, so open-source behaviour is unchanged.

Why it exists: a hosted, multi-tenant deployment wants each tenant to see its
own recent activity, live, without that data leaving the tenant's boundary. The
table is a rolling ``TELEMETRY_RETENTION_DAYS`` window, not an archive.

Design notes:

- **Buffered.** Events accumulate in a bounded in-memory deque and are inserted
  in batches by a background flusher. One INSERT per event would put the
  telemetry write path in contention with the workload that produced it, and
  pipelines emit in bursts.
- **Lossy under pressure, never blocking.** The buffer has a hard cap and drops
  the oldest events when full. Telemetry must never slow down or fail the
  operation it observes.
- **Best-effort.** Every DB interaction is wrapped; failures are logged at debug
  and dropped.
"""

import asyncio
import os
from collections import deque
from datetime import datetime, timedelta, timezone

from cognee.shared.logging_utils import get_logger
from cognee.shared.utils import as_uuid

logger = get_logger(__name__)

# Bounded so a DB outage can't turn telemetry into a memory leak.
_BUFFER_MAX = int(os.getenv("TELEMETRY_BUFFER_MAX", "5000"))
_FLUSH_INTERVAL_SECONDS = float(os.getenv("TELEMETRY_FLUSH_INTERVAL", "5"))
_FLUSH_BATCH = int(os.getenv("TELEMETRY_FLUSH_BATCH", "200"))
_RETENTION_DAYS = int(os.getenv("TELEMETRY_RETENTION_DAYS", "30"))
# Prune runs on a flush multiple rather than a timer — one cheap indexed DELETE
# every ~N flushes is plenty for a rolling window.
_PRUNE_EVERY_N_FLUSHES = int(os.getenv("TELEMETRY_PRUNE_EVERY_N_FLUSHES", "120"))

_buffer: deque = deque(maxlen=_BUFFER_MAX)
_dropped = 0
_flush_count = 0

_flusher_task: asyncio.Task | None = None
_flusher_loop: asyncio.AbstractEventLoop | None = None


def enqueue(payload: dict) -> None:
    """Queue one telemetry payload for insertion. Sync, non-blocking, never raises.

    Also (re)starts the background flusher when called from inside a running
    loop. With no running loop the event is still buffered — it will be written
    by whichever later call does have one, and dropped if none ever does.
    """
    global _dropped

    if len(_buffer) == _BUFFER_MAX:
        # deque(maxlen=...) evicts silently; count it so the loss is visible.
        _dropped += 1
        if _dropped % 1000 == 1:
            logger.debug("Telemetry buffer full; dropped %s event(s) so far", _dropped)

    _buffer.append(payload)

    try:
        _ensure_flusher(asyncio.get_running_loop())
    except RuntimeError:
        pass


def _ensure_flusher(loop: asyncio.AbstractEventLoop) -> None:
    """Start the background flusher if it isn't running on this loop.

    The task is loop-bound, so a loop change (tests, ``asyncio.run`` boundaries)
    requires a fresh task.
    """
    global _flusher_task, _flusher_loop

    if _flusher_task is not None and not _flusher_task.done() and _flusher_loop is loop:
        return

    _flusher_task = loop.create_task(_flusher())
    _flusher_loop = loop


async def _flusher() -> None:
    """Drain the buffer on an interval until cancelled."""
    while True:
        try:
            await asyncio.sleep(_FLUSH_INTERVAL_SECONDS)
            if _buffer:
                await flush()
        except asyncio.CancelledError:
            await flush()
            raise
        except Exception as error:
            logger.debug("Telemetry flusher iteration failed: %s", error)


async def flush() -> None:
    """Insert everything currently buffered. Best-effort; never raises.

    Call directly on shutdown to avoid losing the tail of the buffer.
    """
    global _flush_count

    if not _buffer:
        return

    batch = [_buffer.popleft() for _ in range(min(len(_buffer), _FLUSH_BATCH))]

    try:
        from cognee.infrastructure.databases.relational import get_relational_engine
        from cognee.modules.telemetry.models.TelemetryEvent import TelemetryEvent

        engine = get_relational_engine()
        async with engine.get_async_session() as session:
            session.add_all([_to_row(TelemetryEvent, payload) for payload in batch])
            await session.commit()
    except Exception as error:
        # Dropped, not requeued: a persistent failure would otherwise spin the
        # same batch forever and starve fresh events out of a bounded buffer.
        logger.debug("Telemetry flush of %s event(s) failed: %s", len(batch), error)
        return

    _flush_count += 1
    if _flush_count % _PRUNE_EVERY_N_FLUSHES == 0:
        await _prune()


def _dataset_id(properties: dict):
    """Pull a single dataset UUID out of a payload's properties, if there is one.

    Call sites spell it several ways — ``dataset_id`` (remember, forget),
    ``dataset_ids`` as a comma-joined list (recall), and ``dataset`` which may be
    a name or an id (improve). Normalising here rather than at each call site
    keeps the emitters untouched and means a new one only has to use any of the
    existing spellings.

    Returns None when there is no dataset or when the event legitimately spans
    several; the full list is still in ``properties`` either way.
    """
    single = as_uuid(properties.get("dataset_id"))
    if single:
        return single

    joined = properties.get("dataset_ids") or ""
    if isinstance(joined, str):
        candidates = [part for part in (p.strip() for p in joined.split(",")) if part]
        if len(candidates) == 1:
            return as_uuid(candidates[0])

    # ``dataset`` is usually a name; keep it only when it happens to be an id.
    return as_uuid(properties.get("dataset"))


def _to_row(model, payload: dict):
    """Map a proxy-shaped telemetry payload onto a ``TelemetryEvent`` row."""
    properties = payload.get("properties") or {}
    return model(
        event_name=payload.get("event_name"),
        user_id=as_uuid(properties.get("user_id")),
        tenant_id=as_uuid(properties.get("tenant_id")),
        dataset_id=_dataset_id(properties),
        pipeline_run_id=as_uuid(properties.get("pipeline_run_id")),
        origin=properties.get("origin"),
        anonymous_id=payload.get("anonymous_id"),
        properties=properties,
    )


async def _prune() -> None:
    """Delete events older than the retention window. Best-effort."""
    try:
        from sqlalchemy import delete

        from cognee.infrastructure.databases.relational import get_relational_engine
        from cognee.modules.telemetry.models.TelemetryEvent import TelemetryEvent

        cutoff = datetime.now(timezone.utc) - timedelta(days=_RETENTION_DAYS)
        engine = get_relational_engine()
        async with engine.get_async_session() as session:
            await session.execute(delete(TelemetryEvent).where(TelemetryEvent.created_at < cutoff))
            await session.commit()
    except Exception as error:
        logger.debug("Telemetry prune failed: %s", error)
