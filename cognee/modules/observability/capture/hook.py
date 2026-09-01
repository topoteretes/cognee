"""Non-blocking eval capture hook (SDK-529).

Captures what the pipeline produces and chooses (per-chunk graphs, merge
decisions, summaries, retrieval candidate pools, run manifests) for offline
evals, at ZERO cost when disabled and near-zero on the hot path when enabled.
Capture never breaks, slows (beyond the snapshot cost), or blocks the
operation it observes.

Hot-path contract
-----------------
* ``emit()`` starts with ``sink = _sink; if sink is None: return`` — the OFF
  cost is one global read, no allocation. Only after that is the
  ``CaptureEvent`` built and appended to the deque. On a full buffer the
  newest event is dropped and ``_dropped`` incremented. No serialization, no
  I/O, no await, no logging, no config read, no exception path — ever.
* ``is_active()`` is one global read in steady state (a flag-guarded one-time
  ``_ensure_initialized()`` on first call). Callers hoist it out of loops.
* Sinks run only from the background flusher or ``drain()``/``shutdown()``.
* Snapshot rule: ``emit()`` never serializes, so an emit point whose payload is
  mutated afterwards MUST pass a snapshot — for pydantic graphs
  ``payload=graph.model_dump(mode="json"), payload_kind="json"`` (~32 µs per
  40n/60e KnowledgeGraph). Not ``model_copy(deep=True)`` (8x slower) and not a
  shallow ``model_copy`` (shares Node instances). Immutable payloads may be
  passed by reference.

Buffering
---------
``collections.deque``: append/popleft are atomic under the GIL and loop- and
thread-agnostic — emit is reached from ``to_thread``'d sync code, run_sync's
thread, the dataset-queue reaper thread, and across ``asyncio.run``
boundaries. ``asyncio.Queue`` is loop-bound and its ``put_nowait`` from a
foreign thread wakes getters via loop-owned futures, so it is rejected. The
``len() >= QUEUE_SIZE`` check is a soft bound across threads (harmless). Run
manifests bypass the bound (``_emit_unbounded``): the manifest is the record
that carries ``dropped_events``, so drop-newest must never eat it; the number
of manifests in flight is bounded by the number of concurrent runs.

Flushing
--------
One flusher task per event loop, started by the first on-loop emit or eagerly
at ``run_scope`` entry (so worker-thread emits during a run are picked up by
the interval tick even when nothing emits on-loop), and woken either by the
``FLUSH_INTERVAL_S`` tick or by ``emit()`` when the buffer reaches
``BATCH_SIZE`` (``call_soon_threadsafe`` — the only loop primitive safe from
any thread/loop). Serialization runs in a worker thread because it is pure CPU
(~15 ms per 64-graph batch) and would otherwise freeze concurrent recall
coroutines; inside atexit handlers the executor is gone and it runs inline.
``_in_flight`` counts batches from pop to delivery, so ``drain()`` waits for a
batch the flusher is still serializing. Each sink write is bounded by
``SINK_TIMEOUT_S`` so a wedged sink cannot pin the flusher (and every later
``drain()``) forever. A cancelled flush pushes its popped batch back so
``asyncio.run`` teardown loses nothing; the atexit hook or next ``drain()``
picks it up. Known limitation: a loop closed WITHOUT cancelling (pytest-asyncio
0.21) loses an in-flight batch — tests drain explicitly.
"""

from __future__ import annotations

import asyncio
import atexit
import collections
import random
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID

from cognee.shared.logging_utils import get_logger

from .config import CaptureConfig, get_capture_config
from .events import RETRIEVAL_KIND_PREFIX, CaptureEvent
from .sinks import CaptureSink, StorageSink, run_off_loop

if TYPE_CHECKING:
    from .manifest import RunScope

logger = get_logger("eval_capture")

# Knobs. Defaults mirror CaptureConfig; _load_knobs() overwrites them from the
# environment once per process, _configure() from tests.
_FIELDS = CaptureConfig.model_fields
QUEUE_SIZE: int = _FIELDS["cognee_capture_queue_size"].default
BATCH_SIZE: int = _FIELDS["cognee_capture_batch_size"].default
FLUSH_INTERVAL_S: float = _FIELDS["cognee_capture_flush_interval_s"].default
SAMPLE_RATE: float = _FIELDS["cognee_capture_sample_rate"].default
SINK_TIMEOUT_S: float = _FIELDS["cognee_capture_sink_timeout_s"].default

# Module state.
_sink: CaptureSink | None = None
_initialized: bool = False
_buffer: collections.deque[CaptureEvent] = collections.deque()
# Process-global, monotonic. Reported once per manifest as a delta, never logged per drop.
_dropped: int = 0
# Batches popped from the buffer but not yet delivered (serializing or awaiting
# the sink), on any loop. drain() waits on it.
_in_flight: int = 0
_atexit_registered: bool = False

# The active manifest scope (owned here so manifest.py can import it without a
# runtime cycle; manifest.py owns RunScope and run_scope()).
_current_scope: ContextVar[RunScope | None] = ContextVar("cognee_capture_scope", default=None)


@dataclass(slots=True)
class _Flusher:
    loop: asyncio.AbstractEventLoop
    wake: asyncio.Event
    # True from the moment a wake.set() is scheduled until the flusher has
    # consumed it, so a synchronous burst past BATCH_SIZE schedules ONE loop
    # callback instead of one per emit.
    wake_pending: bool = False
    # Bound right after creation: the flush task needs the flusher, the flusher the task.
    task: asyncio.Task = field(init=False)


_flushers: dict[asyncio.AbstractEventLoop, _Flusher] = {}


def _running_loop_fallback() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


# asyncio._get_running_loop returns None instead of raising: no exception path per emit.
_get_running_loop = getattr(asyncio, "_get_running_loop", _running_loop_fallback)


# ---------------------------------------------------------------------------
# Initialization and registration
# ---------------------------------------------------------------------------


def _register_atexit() -> None:
    """Flush leftovers at interpreter exit (mirrors ``_register_telemetry_session_atexit``).

    Covers the CLI (``asyncio.run`` per command cancels the flusher, the
    cancelled batch is pushed back, atexit flushes it) and uvicorn (loop closed
    before interpreter exit).
    """
    global _atexit_registered
    if _atexit_registered:
        return
    _atexit_registered = True
    atexit.register(_flush_at_exit)


def _flush_at_exit() -> None:
    if _sink is None or not (_buffer or _in_flight):
        return
    if _get_running_loop() is not None:
        return
    try:
        # threading._shutdown() has already run here, so the default executor
        # refuses new work; run_off_loop() falls back to inline serialization.
        asyncio.run(drain(1.0))
    except Exception:
        pass


def _load_knobs(config: CaptureConfig) -> None:
    global QUEUE_SIZE, BATCH_SIZE, FLUSH_INTERVAL_S, SAMPLE_RATE, SINK_TIMEOUT_S
    QUEUE_SIZE = config.cognee_capture_queue_size
    BATCH_SIZE = config.cognee_capture_batch_size
    FLUSH_INTERVAL_S = config.cognee_capture_flush_interval_s
    SAMPLE_RATE = config.cognee_capture_sample_rate
    SINK_TIMEOUT_S = config.cognee_capture_sink_timeout_s


def _ensure_initialized() -> None:
    """Read the config once per process and auto-register the storage sink.

    Guarded by ``_initialized`` (NOT by ``_sink``). This is the ONLY place
    ``get_capture_config()`` is consulted at runtime, and it is reached from
    ``is_active()`` — never from ``emit()``. A failing initialization (bad
    env, unreachable storage) logs once and leaves capture off: capture must
    never break the operation it observes.
    """
    global _initialized, _sink
    if _initialized:
        return
    _initialized = True
    try:
        config = get_capture_config()
        _load_knobs(config)
        if config.cognee_capture_enabled and _sink is None:
            # Lazy: keeps this module free of the storage/base_config import chain.
            from cognee.infrastructure.files.storage import get_file_storage

            _sink = StorageSink(get_file_storage(config.cognee_capture_dir))
            _register_atexit()
    except Exception as exc:
        logger.warning("eval capture disabled: initialization failed (%s)", exc)


def is_active() -> bool:
    """True when a sink is registered. Hoist out of per-node loops."""
    if not _initialized:
        _ensure_initialized()
    return _sink is not None


def register_capture_sink(sink: CaptureSink | None) -> None:
    """Process-wide sink registration; last wins, ``None`` clears.

    An explicit registration overrides env auto-registration of the SINK
    (mirrors ``register_activity_sink`` semantics); the tuning knobs
    (``COGNEE_CAPTURE_QUEUE_SIZE`` etc.) still come from the environment.
    """
    global _sink, _initialized
    if not _initialized:
        _initialized = True
        try:
            _load_knobs(get_capture_config())
        except Exception as exc:
            logger.debug("capture config unavailable, keeping default knobs (%s)", exc)
    _sink = sink
    if sink is not None:
        _register_atexit()


# ---------------------------------------------------------------------------
# Emit (hot path)
# ---------------------------------------------------------------------------


def emit(
    kind: str,
    payload: Any,
    *,
    payload_kind: str = "pydantic",
    stage: str | None = None,
    run_id: UUID | str | None = None,
    dataset_id: UUID | str | None = None,
) -> None:
    """Buffer one observation. Never serializes, awaits, logs, or blocks.

    ``payload`` is kept by reference — see the snapshot rule in the module
    docstring for payloads mutated after emit. ``run_id``/``dataset_id`` are
    explicit overrides; unset values are resolved from the active
    ``run_scope`` at serialization time.
    """
    global _dropped
    sink = _sink
    if sink is None:
        return

    if len(_buffer) >= QUEUE_SIZE:
        # Drop-NEWEST: the buffer is a bounded observation window, not a queue
        # the pipeline waits on.
        _dropped += 1
        return

    _enqueue(
        CaptureEvent(
            kind=kind,
            payload=payload,
            payload_kind=payload_kind,
            run_id=run_id,
            dataset_id=dataset_id,
            scope=_current_scope.get(),
            stage=stage,
            ts=time.time(),
        )
    )


def _emit_unbounded(
    kind: str,
    payload: Any,
    *,
    payload_kind: str,
    run_id: UUID | str | None,
    dataset_id: UUID | str | None,
) -> None:
    """``emit()`` without the ``QUEUE_SIZE`` bound — for run manifests only.

    A run that overflowed the buffer must still land its manifest: it is the
    record carrying ``dropped_events`` and the authoritative dataset
    attribution, so drop-newest applying to it would make overflow
    unobservable offline. Bounded by the number of concurrent runs.
    """
    if _sink is None:
        return
    _enqueue(
        CaptureEvent(
            kind=kind,
            payload=payload,
            payload_kind=payload_kind,
            run_id=run_id,
            dataset_id=dataset_id,
            scope=_current_scope.get(),
            stage=None,
            ts=time.time(),
        )
    )


def _enqueue(event: CaptureEvent) -> None:
    """Append, make sure this loop has a flusher, fire the BATCH_SIZE wake."""
    buffer = _buffer
    buffer.append(event)

    loop = _get_running_loop()
    flusher = _ensure_flusher(loop) if loop is not None else None
    # else: a worker-thread emit; the deque is picked up by whichever flusher runs next.

    if len(buffer) >= BATCH_SIZE:
        if flusher is None:
            flusher = _any_live_flusher()
        if flusher is not None and not flusher.wake_pending and not flusher.loop.is_closed():
            # wake_pending: one callback per batch, even in a synchronous burst
            # where the scheduled wake.set() has not run yet.
            flusher.wake_pending = True
            try:
                flusher.loop.call_soon_threadsafe(flusher.wake.set)
            except RuntimeError:
                flusher.wake_pending = False  # loop closing


def _ensure_flusher(loop: asyncio.AbstractEventLoop) -> _Flusher | None:
    """Return this loop's live flusher, starting one if absent or done()."""
    flusher = _flushers.get(loop)
    if flusher is None or flusher.task.done():
        # A done() flusher (crashed, cancelled) is treated as absent.
        flusher = _start_flusher(loop)
    return flusher


def ensure_flusher() -> None:
    """Start the running loop's flusher eagerly (called at ``run_scope`` entry).

    Emits from plain threads cannot start a flusher (no running loop in the
    emitting thread), so a run whose emit points all execute in worker threads
    would otherwise accumulate events until QUEUE_SIZE and drop the rest. Both
    wirings open their scope on-loop at run start, so this gives every run a
    live flusher before its first worker-thread emit. Off the hot path.
    """
    if _sink is None:
        return
    loop = _get_running_loop()
    if loop is not None:
        _ensure_flusher(loop)


def _start_flusher(loop: asyncio.AbstractEventLoop) -> _Flusher | None:
    # Prune explicitly: add_done_callback never fires for a task parked on a
    # closed loop. Snapshot the items — another thread may be mutating the dict.
    for stale_loop, stale in list(_flushers.items()):
        if stale_loop.is_closed() or stale.task.done():
            _flushers.pop(stale_loop, None)

    flusher = _Flusher(loop=loop, wake=asyncio.Event())
    flush_coroutine = _flush_loop(flusher)
    try:
        flusher.task = loop.create_task(flush_coroutine)
    except RuntimeError:
        # Loop closing — same posture as _TELEMETRY_TASKS in cognee/shared/utils.py.
        flush_coroutine.close()
        return None

    _flushers[loop] = flusher

    def _discard(_task: asyncio.Task) -> None:
        if _flushers.get(loop) is flusher:
            _flushers.pop(loop, None)

    flusher.task.add_done_callback(_discard)
    return flusher


def _any_live_flusher() -> _Flusher | None:
    for flusher in list(_flushers.values()):
        if not flusher.loop.is_closed() and not flusher.task.done():
            return flusher
    return None


# ---------------------------------------------------------------------------
# Flusher
# ---------------------------------------------------------------------------


async def _flush_loop(flusher: _Flusher) -> None:
    wake = flusher.wake
    while True:
        try:
            try:
                await asyncio.wait_for(wake.wait(), FLUSH_INTERVAL_S)
            except asyncio.TimeoutError:
                pass  # interval tick
            wake.clear()
            # Re-arm BEFORE flushing so an emit that lands mid-flush can wake us again.
            flusher.wake_pending = False
            # Burst absorption: after a full batch keep flushing until the buffer
            # is below the threshold; sleep only when quiet.
            while _buffer:
                await _flush_one_batch()
                if len(_buffer) < BATCH_SIZE:
                    break
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # A bad iteration must never kill the task.
            logger.debug("capture flusher iteration failed (%s)", exc)


def _serialize_payload(event: CaptureEvent) -> Any:
    payload_kind = event.payload_kind
    if payload_kind == "pydantic":
        return event.payload.model_dump(mode="json")
    if payload_kind in ("json", "text"):
        return event.payload
    raise ValueError(f"unknown payload_kind {payload_kind!r}")


def _serialize_batch(batch: list[CaptureEvent]) -> list[dict]:
    """Turn events into sink records. Runs in a worker thread (pure CPU)."""
    records: list[dict] = []
    for event in batch:
        try:
            payload = _serialize_payload(event)
        except Exception as exc:
            # Keep the record so the run's event count stays honest.
            logger.debug("capture payload serialization failed (%s)", exc)
            payload = {"error": repr(exc)}

        scope = event.scope
        run_id = event.run_id or (scope.run_id if scope is not None else None)
        dataset_id = event.dataset_id or (scope.dataset_id if scope is not None else None)
        records.append(
            {
                "kind": event.kind,
                "run_id": None if run_id is None else str(run_id),
                "dataset_id": None if dataset_id is None else str(dataset_id),
                "stage": event.stage,
                "ts": event.ts,
                "payload": payload,
            }
        )
    return records


async def _flush_one_batch() -> int:
    """Pop up to BATCH_SIZE events, serialize off-loop, hand groups to the sink.

    Returns the number of events popped. ``_in_flight`` is held for the whole
    pop→deliver window so ``drain()`` can wait for a batch that another
    consumer (the flusher) is still serializing.
    """
    global _in_flight, _dropped
    buffer = _buffer
    batch: list[CaptureEvent] = []
    try:
        while buffer and len(batch) < BATCH_SIZE:
            batch.append(buffer.popleft())
    except IndexError:
        pass  # raced with another consumer (drain + flusher split the work)
    if not batch:
        return 0

    sink = _sink
    if sink is None:
        return len(batch)  # sink cleared after these were queued: nothing to deliver to

    # Events not yet handed to the sink; on cancellation these go back.
    pending: list[CaptureEvent | None] = list(batch)
    _in_flight += 1
    try:
        records = await run_off_loop(_serialize_batch, batch)

        groups: dict[tuple[str | None, str], list[int]] = {}
        for index, record in enumerate(records):
            groups.setdefault((record["run_id"], record["kind"]), []).append(index)

        for indexes in groups.values():
            group = [records[index] for index in indexes]
            try:
                # Bounded: a wedged sink (S3 under partition) must not pin the
                # flusher — and every later drain() — forever.
                await asyncio.wait_for(sink(group), SINK_TIMEOUT_S)
            except Exception as exc:
                logger.debug("capture sink failed (%s)", exc)
            for index in indexes:
                pending[index] = None
            # Let other coroutines interleave between sink writes.
            await asyncio.sleep(0)
    except asyncio.CancelledError:
        # asyncio.run teardown / uvicorn shutdown: do not lose the popped batch.
        buffer.extendleft(reversed([event for event in pending if event is not None]))
        raise
    except Exception as exc:
        # Not re-queued: a deterministic failure would pin the head of the
        # buffer forever. Account for the loss instead of hiding it.
        lost = sum(1 for event in pending if event is not None)
        _dropped += lost
        logger.debug("capture flush failed, %d event(s) dropped (%s)", lost, exc)
    finally:
        _in_flight -= 1
    return len(batch)


# ---------------------------------------------------------------------------
# Drain / shutdown
# ---------------------------------------------------------------------------


async def _drain_until(deadline: float) -> None:
    # (a) Inline on the calling loop: flush what is buffered NOW. The deque is
    # the coordination point — popleft is atomic, so a concurrently running
    # flusher and this drain simply split the work (blob names are
    # collision-free). Budgeted by the entry count so a producer faster than
    # the sink cannot pin the caller: the flusher owns whatever arrives later.
    budget = len(_buffer)
    while budget > 0 and _buffer and time.monotonic() < deadline:
        budget -= await _flush_one_batch()
    # (b) Wait for batches other consumers hold (popped, serializing or
    # awaiting their sink): a plain int counter, no foreign-loop futures.
    while _in_flight > 0 and time.monotonic() < deadline:
        await asyncio.sleep(0.01)


async def drain(timeout: float = 5.0) -> None:
    """Flush the events buffered so far, then wait for in-flight batches.

    Never requires a flusher to exist, never raises, and returns within about
    ``timeout`` even under a continuous producer (leftovers stay with the
    flusher / the atexit hook).
    """
    try:
        await _drain_until(time.monotonic() + timeout)
    except Exception as exc:
        logger.debug("capture drain failed (%s)", exc)


async def shutdown(timeout: float = 5.0) -> None:
    """Drain, stop every flusher, then deliver anything a cancelled flush put back.

    For process teardown (FastAPI lifespan, tests), not the request path.
    """
    deadline = time.monotonic() + timeout
    await drain(timeout)
    current_loop = _get_running_loop()
    same_loop_tasks: list[asyncio.Task] = []
    for flusher in list(_flushers.values()):
        if flusher.loop is current_loop and not flusher.task.done():
            same_loop_tasks.append(flusher.task)
        _cancel_flusher(flusher, wait=False)
    _flushers.clear()
    if same_loop_tasks:
        await asyncio.gather(*same_loop_tasks, return_exceptions=True)
    # A flusher cancelled mid-batch re-buffered it; no flusher is left to pick
    # it up, so deliver it here rather than leaving it to the atexit hook.
    if _buffer:
        try:
            await _drain_until(max(deadline, time.monotonic() + min(timeout, 1.0)))
        except Exception as exc:
            logger.debug("capture drain failed (%s)", exc)


async def _await_cancelled(task: asyncio.Task) -> None:
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass


def _cancel_flusher(flusher: _Flusher, *, wait: bool) -> None:
    loop = flusher.loop
    if loop.is_closed() or flusher.task.done():
        return
    if loop is _get_running_loop():
        flusher.task.cancel()
    elif loop.is_running():
        try:
            loop.call_soon_threadsafe(flusher.task.cancel)
        except RuntimeError:
            pass
    else:
        flusher.task.cancel()
        if wait:
            # Deliver the cancellation so the task is not destroyed while pending.
            try:
                loop.run_until_complete(_await_cancelled(flusher.task))
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def should_capture(kind: str) -> bool:
    """Per-run sampling gate for retrieval kinds; every other kind is captured.

    Inside a ``run_scope`` the decision was made ONCE at scope entry (no
    per-emit RNG); without a scope fall back to ``SAMPLE_RATE``.
    """
    if not kind.startswith(RETRIEVAL_KIND_PREFIX):
        return True
    scope = _current_scope.get()
    if scope is not None:
        return scope.sampled
    return random.random() < SAMPLE_RATE


# ---------------------------------------------------------------------------
# Test hooks
# ---------------------------------------------------------------------------


def _configure(
    *,
    queue_size: int | None = None,
    batch_size: int | None = None,
    flush_interval_s: float | None = None,
    sample_rate: float | None = None,
    sink_timeout_s: float | None = None,
) -> None:
    """Override the cached knobs without going through the environment (tests)."""
    global QUEUE_SIZE, BATCH_SIZE, FLUSH_INTERVAL_S, SAMPLE_RATE, SINK_TIMEOUT_S
    if queue_size is not None:
        QUEUE_SIZE = queue_size
    if batch_size is not None:
        BATCH_SIZE = batch_size
    if flush_interval_s is not None:
        FLUSH_INTERVAL_S = flush_interval_s
    if sample_rate is not None:
        SAMPLE_RATE = sample_rate
    if sink_timeout_s is not None:
        SINK_TIMEOUT_S = sink_timeout_s


def _reset_for_tests() -> None:
    """Return the module to its pristine state (sync).

    Cancels flusher tasks whose loop is still open — and, when that loop is not
    running, runs it until the cancellation is delivered — so pytest-asyncio
    0.21.x (which closes loops without cancelling tasks) never destroys a
    pending flusher.
    """
    global _sink, _initialized, _dropped, _in_flight
    for flusher in list(_flushers.values()):
        _cancel_flusher(flusher, wait=True)
    _flushers.clear()
    _buffer.clear()
    _dropped = 0
    _in_flight = 0
    _sink = None
    _initialized = False
    _configure(
        queue_size=_FIELDS["cognee_capture_queue_size"].default,
        batch_size=_FIELDS["cognee_capture_batch_size"].default,
        flush_interval_s=_FIELDS["cognee_capture_flush_interval_s"].default,
        sample_rate=_FIELDS["cognee_capture_sample_rate"].default,
        sink_timeout_s=_FIELDS["cognee_capture_sink_timeout_s"].default,
    )
    get_capture_config.cache_clear()
