"""Server push for one dataset's visualization: live events, growth, heartbeat.

The polling replacement behind ``WS /visualize/subscribe/{dataset_id}``. A
client watching a graph otherwise has to poll ``GET /visualize/live-events``
every second or two *and* refetch the whole graph payload periodically just to
notice a cognify run finished. Both polls move here, to one connection per
viewer that stays quiet until something actually happens.

Deliberately not built on the cognify pipeline-run queue that
``WS /cognify/subscribe`` reads: ``get_from_queue`` pops, so a second consumer
would steal run updates from any client attached to that endpoint. Growth is
detected by reading the dataset's latest completed run instead, which no other
reader can be starved by.
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from fastapi import WebSocket

from cognee.api.v1.visualize.visualize import get_live_events
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.data.methods.get_datasets_graph_counts import COGNIFY_PIPELINE_NAME
from cognee.modules.pipelines.methods import get_pipeline_run_by_dataset
from cognee.modules.pipelines.models import PipelineRunStatus
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger

logger = get_logger()

# The loop wakes on a fixed tick and does each job every N ticks, so the three
# cadences stay in step instead of drifting apart on three timers. Tests
# shorten the tick.
TICK_SECONDS = 1.0
LIVE_EVENTS_TICKS = 2
GRAPH_POLL_TICKS = 5
HEARTBEAT_TICKS = 15


def _parse_cursor(value: Optional[str]) -> Optional[datetime]:
    """A cursor string back into the datetime ``get_live_events`` expects.

    Cursors come from event timestamps, which are written as naive UTC ISO
    strings. An unparsable one keeps the previous cursor rather than resetting
    the stream to "everything", which would replay the whole timeline.
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        logger.warning("Ignoring unparsable live-events cursor: %r", value)
        return None


async def _latest_completed_run_id(dataset_id: UUID) -> Optional[UUID]:
    """The dataset's newest cognify run id, but only once that run completed.

    None while a run is in flight (or when none ever ran), so the caller can
    treat "the id changed" as "the graph grew" without also firing when a run
    merely started.
    """
    run = await get_pipeline_run_by_dataset(dataset_id, COGNIFY_PIPELINE_NAME)

    if run is None or run.status != PipelineRunStatus.DATASET_PROCESSING_COMPLETED:
        return None

    return run.pipeline_run_id


# Shared across every /subscribe/{dataset_id} connection for the same
# dataset: without it, N concurrent viewers each run this window-function
# query every GRAPH_POLL_TICKS, multiplying DB load by the viewer count for a
# result that is identical for all of them at any given moment. Keyed by
# dataset_id only (the result does not depend on the caller), TTL'd to one
# poll interval so a real graph_grew signal is never delayed by more than
# that. Not single-flighted: a handful of connections racing a cache refresh
# within the same tick still collapses to a small, bounded number of queries,
# not the exact number of viewers, and correctness is unaffected either way.
#
# A plain dict rather than cognee's shared CacheDBInterface (get_cache_engine)
# on purpose: that store is string-keyed, so a (UUID, Optional[UUID]) value
# would need serializing, get_cache_engine() can return None when caching is
# disabled, and its default backend is itself a SQL round-trip — on that
# backend it would trade this dataset's DB query for a cache-table one, not
# remove it. It would only pay off under CACHE_BACKEND=redis, which this
# feature does not otherwise require.
_latest_run_cache: Dict[UUID, Tuple[float, Optional[UUID]]] = {}


def _reset_latest_run_cache() -> None:
    """Test-only: drop every cached entry so a test doesn't see another
    test's cached run id for the same dataset_id."""
    _latest_run_cache.clear()


# A dataset nobody has polled in this many TTL windows almost certainly has
# no more active subscribers, so its entry is dropped rather than kept
# forever. The sweep only runs from a live miss, so it only fires while some
# dataset somewhere is still being actively polled — a server with zero
# active subscriptions across every dataset stops sweeping too, but at that
# point the cache is idle and not growing either. Wide margin: this bounds
# the cache to "datasets watched recently, as of the last time anything was
# watched", not a hard cap enforced on a fixed clock.
_STALE_ENTRY_TTL_MULTIPLE = 10


async def _cached_latest_completed_run_id(dataset_id: UUID) -> Optional[UUID]:
    now = time.monotonic()
    # Read module globals at call time, not module-import time, so a test
    # that speeds up TICK_SECONDS via monkeypatch also speeds up this TTL.
    ttl_seconds = GRAPH_POLL_TICKS * TICK_SECONDS
    cached = _latest_run_cache.get(dataset_id)
    if cached is not None and now - cached[0] < ttl_seconds:
        return cached[1]

    run_id = await _latest_completed_run_id(dataset_id)
    _latest_run_cache[dataset_id] = (now, run_id)

    stale_before = now - ttl_seconds * _STALE_ENTRY_TTL_MULTIPLE
    for stale_id in [k for k, v in _latest_run_cache.items() if v[0] < stale_before]:
        del _latest_run_cache[stale_id]

    return run_id


async def _watch_for_disconnect(websocket: WebSocket) -> None:
    """Return when the client goes away.

    A push-only loop never notices a closed connection: nothing raises until
    something is sent, and a send after the peer left is silently dropped. So
    the socket is read in parallel purely to observe the disconnect — which
    also drains whatever frames a client sends (a deployment that replaces the
    authentication dependency may have its clients send some), instead of
    letting them queue for a reader that never comes.
    """
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return


async def _push_updates(
    websocket: WebSocket,
    dataset_id: UUID,
    user: User,
    cursor: Optional[datetime],
) -> None:
    """Poll and push until the connection drops or a poll refuses."""
    # Uncached: this runs once per connection, not once per tick, so it
    # cannot contribute to the per-viewer polling load the cache exists for.
    # Using the cache here would let a brand-new connection inherit another
    # viewer's stale poll as its baseline, which can misreport an already-
    # complete run as newly grown (see stream_dataset_updates's docstring).
    last_run_id = await _latest_completed_run_id(dataset_id)
    tick = 0

    while True:
        await asyncio.sleep(TICK_SECONDS)
        tick += 1

        if tick % LIVE_EVENTS_TICKS == 0:
            # Re-authorizes on every poll, being the same call the HTTP
            # endpoint serves: read access revoked mid-connection ends the
            # stream within one poll instead of at the next reconnect.
            payload = await get_live_events(dataset_id, since=cursor, user=user)
            events: List[Dict[str, Any]] = payload["events"]
            if events:
                await websocket.send_json(
                    {"kind": "live_events", "events": events, "cursor": payload["cursor"]}
                )
                cursor = _parse_cursor(payload["cursor"]) or cursor

        if tick % GRAPH_POLL_TICKS == 0:
            # Re-authorizes here too, same as the live_events tick above: this
            # lane used to rely on that tick's check to catch a revocation,
            # which left up to one GRAPH_POLL_TICKS window where a growth
            # signal could still go out after access was pulled.
            authorized = await get_authorized_existing_datasets([dataset_id], "read", user)
            if not authorized:
                raise PermissionDeniedError(message="Not authorized to read this dataset")

            run_id = await _cached_latest_completed_run_id(dataset_id)
            if run_id is not None and run_id != last_run_id:
                last_run_id = run_id
                await websocket.send_json({"kind": "graph_grew", "pipeline_run_id": str(run_id)})

        if tick % HEARTBEAT_TICKS == 0:
            await websocket.send_json(
                {"kind": "heartbeat", "time": datetime.now(timezone.utc).isoformat()}
            )


async def stream_dataset_updates(
    websocket: WebSocket,
    dataset_id: UUID,
    user: User,
    since: Optional[datetime] = None,
) -> None:
    """Push this dataset's updates over an accepted, authorized WebSocket.

    Sends a ``ready`` frame, then pushes until the client disconnects. Frames:

    - ``{"kind": "ready", "dataset_id": str, "cursor": str | null}`` — once,
      echoing the cursor the stream starts from.
    - ``{"kind": "live_events", "events": [...], "cursor": str}`` — every ~2s,
      but only when the delta is non-empty. ``cursor`` is what to reconnect
      with; the stream advances its own copy from the same value.
    - ``{"kind": "graph_grew", "pipeline_run_id": str}`` — within ~5s of a
      cognify run for this dataset completing. A run that completed before the
      connection opened is the baseline and is not announced.
    - ``{"kind": "heartbeat", "time": str}`` — every ~15s, so a client can tell
      a quiet stream from a dead one.

    Args:
        websocket: An already accepted connection. Callers own accept/close so
            a rejection can carry a close code the client will see.
        dataset_id: The dataset to follow. Already authorized by the caller.
        user: The authenticated caller, whose session events are streamed.
        since: Resume point from a previous connection's last cursor. None
            starts from every event currently available.

    Raises:
        PermissionDeniedError: read access to the dataset was lost while the
            stream was running.
    """
    await websocket.send_json(
        {
            "kind": "ready",
            "dataset_id": str(dataset_id),
            "cursor": since.isoformat() if since else None,
        }
    )

    push_task = asyncio.create_task(_push_updates(websocket, dataset_id, user, since))
    disconnect_task = asyncio.create_task(_watch_for_disconnect(websocket))

    done, pending = await asyncio.wait(
        {push_task, disconnect_task}, return_when=asyncio.FIRST_COMPLETED
    )

    # Cancelled but deliberately not awaited: awaiting a cancelled child would
    # also absorb a cancellation aimed at *this* task — a server shutting the
    # connection down — and carry on serving a socket that was told to stop.
    # The child observes its own cancellation on the next pass regardless.
    for task in pending:
        task.cancel()

    # Surfaces whatever ended the push loop (a lost permission, a failed poll)
    # to the caller, which owns the close code. A finished disconnect watcher
    # carries no result, so a client that simply left ends the stream quietly.
    for task in done:
        task.result()
