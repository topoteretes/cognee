"""Non-blocking eval capture hook (SDK-529).

Captures what the pipeline produces and chooses (per-chunk graphs, merge
decisions, summaries, retrieval candidate pools, run manifests) for offline
evals, at ZERO cost when disabled and near-zero on the hot path when enabled.
Capture never breaks, slows (beyond the snapshot cost), or blocks the
operation it observes.

State
-----
Everything the hook owns lives on one ``CaptureRuntime`` instance,
``_runtime``: the knobs, the registered sink, the buffer, the per-loop
flushers and in-flight batches, the drop counter. The module functions —
``emit()``, ``drain()`` and the rest, the public API — read it. Grouping the
state makes its ownership explicit and lets ``_reset_for_tests()`` swap in a
fresh instance instead of clearing a dozen globals, so no test inherits
another's sink, buffer or flushers. It does not remove the singleton: capture
is process-wide by design (one buffer shared by every loop and thread), so
there is exactly one runtime per process.

Hot-path contract
-----------------
* ``emit()`` starts with ``sink = _runtime.sink; if sink is None: return`` —
  the OFF cost is one global read and one attribute read, no allocation. Only
  after that is the ``CaptureEvent`` built and appended to the deque. On a
  full buffer the newest event is dropped and ``runtime.dropped`` incremented.
  No serialization, no I/O, no await, no logging, no config read, no exception
  path — ever.
* ``is_active()`` is the same two reads in steady state (a flag-guarded
  one-time ``_ensure_initialized()`` on first call). Callers hoist it out of
  loops.
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
``len() >= queue_size`` check is a soft bound across threads (harmless). Run
manifests get headroom past the bound (``_emit_manifest``, up to
``2 * queue_size`` buffered events in total): the manifest is the record that
carries ``events_dropped``, so drop-newest must not eat it while there is room,
but the buffer stays finite under a wedged sink.

Flushing
--------
One flusher task per event loop, started by the first on-loop emit or eagerly
at ``run_scope`` entry (so worker-thread emits during a run are picked up by
the interval tick even when nothing emits on-loop), and woken either by the
``flush_interval_s`` tick or by ``emit()`` when the buffer reaches
``batch_size`` (``call_soon_threadsafe`` — the only loop primitive safe from
any thread/loop; a worker-thread emit targets a flusher whose loop is RUNNING,
since a wake scheduled on a stopped loop would never fire). Serialization runs
in a worker thread because it is pure CPU (~15 ms per 64-graph batch) and would
otherwise freeze concurrent recall coroutines; inside atexit handlers the
executor is gone and it runs inline. Each sink write is bounded by
``sink_timeout_s`` so a wedged sink cannot pin the flusher forever.

Every timed wait on the flusher and drain paths goes through ``_wait_bounded``,
never ``asyncio.wait_for``: on CPython < 3.12 ``wait_for`` swallows a
cancellation that lands in the same loop iteration as the inner awaitable's
completion (bpo-42130), and a flusher that loses its cancel never leaves its
``while True`` — ``asyncio.run()``'s teardown and ``shutdown()`` then wait for
it forever, and a caller cancelled mid-``drain()`` runs on past its own
cancellation.

A flusher cancelled from outside — ``asyncio.run`` / uvicorn tearing the loop
down, ``shutdown()`` — stays in ``runtime.flushers`` as a tombstone: its loop
is going away, so no replacement is started on it. A late emit there (a
``run_tasks`` generator finalized by ``shutdown_asyncgens()`` after the runner
cancelled all tasks, a ``record_operation`` ``finally`` on a cancelled main
task) would otherwise start a fresh flusher that the closing loop destroys
mid-batch; the event stays in the deque for the next ``drain()`` or the atexit
hook instead.

``runtime.in_flight`` holds the batches popped but not yet delivered, per
loop, so ``drain()`` can wait for a batch the flusher is still serializing —
but only on loops that are running: a batch on a loop that stopped can never
complete while it is stopped and must not turn every later ``drain()`` into a
full timeout. A batch stranded on a loop that was CLOSED without cancelling its
tasks (pytest-asyncio 0.21, a ``run_until_complete`` driver that never resumed)
goes back to the head of the buffer when the loop is pruned, so it reaches the
next drain / the atexit hook instead of vanishing with the dead task.
``drain(timeout)`` is authoritative inside a batch too: sink writes are bounded
by what is left of the budget and whatever could not be delivered goes back to
the head of the buffer for the flusher / the atexit hook.

Delivery is at-least-once: a cancelled flush (``asyncio.run`` teardown, uvicorn
shutdown), a drain whose budget ran out mid-write, or a batch recovered from a
dead loop re-buffers the group that was being written along with the rest, so
a consumer may see the same record twice — under the same ``event_id``, which
is assigned at emit and is what consumers dedupe on. Blob names are
collision-free, so nothing is overwritten.

Accounting
----------
Every event is, at any moment, in exactly one of: the buffer, a tracked
in-flight batch, the sink, or dropped. Drops are counted twice over: on the
runtime's ``dropped`` (diagnostics) and on the run the event belongs to
(``RunScope.dropped``, via ``event.scope``) — the latter from whichever thread
drops it, the emit path or the flusher — so a manifest's ``events_dropped``
is that run's own count, not a delta of a shared counter that concurrent runs
pollute. ``RunScope.delivered`` is credited on the sink's first
acknowledgement of each event.
"""

from __future__ import annotations

import asyncio
import atexit
import collections
import itertools
import random
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Iterable, TypeVar
from uuid import UUID, uuid4

from cognee.shared.logging_utils import get_logger

from .config import CaptureConfig, get_capture_config
from .events import CAPTURE_SCHEMA_VERSION, RETRIEVAL_KIND_PREFIX, CaptureEvent
from .sinks import CaptureSink, StorageSink, run_off_loop

if TYPE_CHECKING:
    from .manifest import RunScope

logger = get_logger("eval_capture")

T = TypeVar("T")

_FIELDS = CaptureConfig.model_fields
# Floor for the interval tick: a non-positive interval would spin the flusher.
_MIN_FLUSH_INTERVAL_S: float = 0.01
# How long shutdown() / _reset_for_tests() wait for a cancelled flusher to
# finish: a regression that loses the cancel must fail a test, not hang it.
_CANCEL_GRACE_S: float = 1.0


@dataclass(slots=True)
class _Flusher:
    # The runtime this flusher drains — bound at creation rather than read
    # from the module on every tick, so a flusher that outlives a
    # ``_reset_for_tests()`` swap (one on a foreign loop, cancelled
    # asynchronously) keeps draining the state it was started for.
    runtime: CaptureRuntime
    loop: asyncio.AbstractEventLoop
    wake: asyncio.Event
    # True from the moment a wake.set() is scheduled until the flusher has
    # consumed it, so a synchronous burst past batch_size schedules ONE loop
    # callback instead of one per emit.
    wake_pending: bool = False
    # Bound right after creation: the flush task needs the flusher, the flusher the task.
    # A task that ended cancelled turns this entry into a tombstone for its
    # loop (see _ensure_flusher).
    task: asyncio.Task = field(init=False)


@dataclass(eq=False)
class CaptureRuntime:
    """The process-wide capture state, in one place (see the module docstring).

    Deliberately NOT here, because each must survive a runtime swap:
    the atexit registration flag (one handler per process, or every reset
    would register another), the ``_current_scope`` contextvar (contextvars
    live at module level), and the event-id prefix and sequence (an id must
    stay unique across a reset).
    """

    # Knobs. Defaults mirror CaptureConfig; _load_knobs() overwrites them from
    # the environment once per process, _configure() from tests.
    queue_size: int = _FIELDS["cognee_capture_queue_size"].default
    batch_size: int = _FIELDS["cognee_capture_batch_size"].default
    flush_interval_s: float = _FIELDS["cognee_capture_flush_interval_s"].default
    retrieval_sample_rate: float = _FIELDS["cognee_capture_retrieval_sample_rate"].default
    sink_timeout_s: float = _FIELDS["cognee_capture_sink_timeout_s"].default
    drain_timeout_s: float = _FIELDS["cognee_capture_drain_timeout_s"].default

    # Registration.
    sink: CaptureSink | None = None
    initialized: bool = False
    # Serializes the one-time initialization only. is_active() reads
    # ``initialized`` BEFORE taking it, so the warm path stays plain reads with
    # no locking. Reentrant + ``initializing``: initialization imports the
    # storage chain, and a re-entrant is_active() from inside it must read
    # "off", never deadlock.
    initializing: bool = False
    init_lock: threading.RLock = field(default_factory=threading.RLock)

    # Buffering.
    buffer: collections.deque[CaptureEvent] = field(default_factory=collections.deque)
    # Process-global, monotonic, approximate (lock-free on the emit path).
    # Diagnostics only — the per-run counts live on each RunScope.
    dropped: int = 0

    # Flushing. Batches popped from the buffer but not yet delivered
    # (serializing or awaiting the sink), keyed by the loop doing the work.
    # Each entry is the batch's slot list — an event until its group is
    # acknowledged, None afterwards — so the undelivered remainder is always
    # reachable from here, not only from the frame of the task doing the
    # flush. A loop runs in one thread at a time, so each key is only ever
    # mutated by its owner thread; other threads only read, or pop the key of
    # a CLOSED loop (which no thread runs any more). drain() waits on the
    # RUNNING loops' counts only (see _in_flight_live()).
    in_flight: dict[asyncio.AbstractEventLoop, list[list[CaptureEvent | None]]] = field(
        default_factory=dict
    )
    flushers: dict[asyncio.AbstractEventLoop, _Flusher] = field(default_factory=dict)


_runtime: CaptureRuntime = CaptureRuntime()
_atexit_registered: bool = False

# The active manifest scope (owned here so manifest.py can import it without a
# runtime cycle; manifest.py owns RunScope and run_scope()).
_current_scope: ContextVar[RunScope | None] = ContextVar("cognee_capture_scope", default=None)

# Event ids: "<process>-<seq>". The prefix is unique per process (and restart),
# the sequence is an itertools.count — next() is atomic under the GIL and
# ~40 ns, so a worker-thread emit needs no lock. Formatted at serialization,
# never on the emit path.
_PROCESS_ID: str = uuid4().hex
_event_seq = itertools.count(1)


def _running_loop_fallback() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


# asyncio._get_running_loop returns None instead of raising: no exception path per emit.
_get_running_loop = getattr(asyncio, "_get_running_loop", _running_loop_fallback)


# ---------------------------------------------------------------------------
# Knobs
# ---------------------------------------------------------------------------


def _configure(
    *,
    queue_size: int | None = None,
    batch_size: int | None = None,
    flush_interval_s: float | None = None,
    retrieval_sample_rate: float | None = None,
    sink_timeout_s: float | None = None,
    drain_timeout_s: float | None = None,
) -> None:
    """Set the runtime's knobs (from the config once per process, or from tests).

    Clamped as defence in depth on top of ``CaptureConfig`` validation: a batch
    size of 0 makes ``_flush_one_batch`` pop nothing, and a consumer that pops
    nothing without ever suspending would monopolise its event loop; a
    non-positive interval would spin the flusher.
    """
    runtime = _runtime
    if queue_size is not None:
        runtime.queue_size = max(1, queue_size)
    if batch_size is not None:
        runtime.batch_size = max(1, batch_size)
    if flush_interval_s is not None:
        runtime.flush_interval_s = max(_MIN_FLUSH_INTERVAL_S, flush_interval_s)
    if retrieval_sample_rate is not None:
        runtime.retrieval_sample_rate = retrieval_sample_rate
    if sink_timeout_s is not None:
        runtime.sink_timeout_s = sink_timeout_s
    if drain_timeout_s is not None:
        runtime.drain_timeout_s = drain_timeout_s


def _load_knobs(config: CaptureConfig) -> None:
    _configure(
        queue_size=config.cognee_capture_queue_size,
        batch_size=config.cognee_capture_batch_size,
        flush_interval_s=config.cognee_capture_flush_interval_s,
        retrieval_sample_rate=config.cognee_capture_retrieval_sample_rate,
        sink_timeout_s=config.cognee_capture_sink_timeout_s,
        drain_timeout_s=config.cognee_capture_drain_timeout_s,
    )


# ---------------------------------------------------------------------------
# Initialization and registration
# ---------------------------------------------------------------------------


def _register_atexit() -> None:
    """Flush leftovers at interpreter exit (mirrors ``_register_telemetry_session_atexit``).

    Covers the CLI (``asyncio.run`` per command cancels the flusher, the
    cancelled batch is pushed back, atexit flushes it) and uvicorn (loop closed
    before interpreter exit). Once per process, not per runtime: the handler
    reads whatever ``_runtime`` is at exit.
    """
    global _atexit_registered
    if _atexit_registered:
        return
    _atexit_registered = True
    atexit.register(_flush_at_exit)


def _flush_at_exit() -> None:
    runtime = _runtime
    if runtime.sink is None:
        return
    # Batches stranded on loops that were closed without cancelling their tasks
    # go back to the buffer first, so they are part of what gets persisted.
    _prune_closed_loops(runtime)
    # Only batches on a loop that is still running (a daemon thread's) can
    # complete here; anything on a stopped loop cannot, and waiting for it
    # would just add a full timeout to every interpreter exit.
    if not (runtime.buffer or _in_flight_live(runtime)):
        return
    if _get_running_loop() is not None:
        return
    try:
        # threading._shutdown() has already run here, so the default executor
        # refuses new work; run_off_loop() falls back to inline serialization.
        asyncio.run(drain(1.0))
    except Exception:
        pass


def _ensure_initialized() -> None:
    """Read the config once per process and auto-register the storage sink.

    Guarded by ``initialized`` (NOT by ``sink``). This is the ONLY place
    ``get_capture_config()`` is consulted at runtime, and it is reached from
    ``is_active()`` — never from ``emit()``. A failing initialization (bad
    env, unreachable storage) logs once and leaves capture off: capture must
    never break the operation it observes.

    Runs under ``init_lock`` and publishes ``initialized`` only once ``sink``
    is installed. cognee reaches ``is_active()`` from ``to_thread`` workers and
    the dataset-queue thread, and this body imports the storage chain (hundreds
    of ms cold); publishing the flag first would make every concurrent caller
    in that window see capture as OFF and silently skip a run's early events.
    """
    runtime = _runtime
    with runtime.init_lock:
        # Re-check under the lock: a thread that waited here while another did
        # the work must not redo it. ``initializing`` catches the re-entrant
        # case, which the RLock would otherwise let recurse.
        if runtime.initialized or runtime.initializing:
            return
        runtime.initializing = True
        try:
            config = get_capture_config()
            _load_knobs(config)
            if config.cognee_capture_enabled and runtime.sink is None:
                # Lazy: keeps this module free of the storage/base_config import chain.
                from cognee.infrastructure.files.storage import get_file_storage

                runtime.sink = StorageSink(get_file_storage(config.cognee_capture_dir))
                _register_atexit()
        except Exception as exc:
            logger.warning("eval capture disabled: initialization failed (%s)", exc)
        finally:
            # Set LAST, and inside the lock: a concurrent is_active() must never
            # see ``initialized`` True while ``sink`` is still None, or it would
            # skip capture for the whole (import-chain-long) initialization window.
            runtime.initialized = True
            runtime.initializing = False


def is_active() -> bool:
    """True when a sink is registered. Hoist out of per-node loops."""
    runtime = _runtime
    if not runtime.initialized:
        _ensure_initialized()
    return runtime.sink is not None


def register_capture_sink(sink: CaptureSink | None) -> None:
    """Process-wide sink registration; last wins, ``None`` clears.

    An explicit registration overrides env auto-registration of the SINK; the
    tuning knobs (``COGNEE_CAPTURE_QUEUE_SIZE`` etc.) still come from the
    environment.
    Registering a sink re-arms flushing on loops ``shutdown()`` stopped.

    Runs under ``init_lock``: a registration racing ``_ensure_initialized()``
    (a startup thread calling ``is_active()`` while another registers) must not
    interleave with the env auto-registration, or the explicit sink could be
    overwritten by the ``StorageSink`` — last-wins only holds when the two
    writers are serialized.
    """
    runtime = _runtime
    with runtime.init_lock:
        if not runtime.initialized:
            runtime.initialized = True
            try:
                _load_knobs(get_capture_config())
            except Exception as exc:
                logger.debug("capture config unavailable, keeping default knobs (%s)", exc)
        runtime.sink = sink
    if sink is not None:
        _forget_stopped_flushers(runtime)
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
    runtime = _runtime
    sink = runtime.sink
    if sink is None:
        return

    scope = _current_scope.get()
    if len(runtime.buffer) >= runtime.queue_size:
        # Drop-NEWEST: the buffer is a bounded observation window, not a queue
        # the pipeline waits on. Counted on the run too (see Accounting).
        runtime.dropped += 1
        if scope is not None:
            scope.dropped += 1
        return

    _enqueue(
        runtime,
        CaptureEvent(
            kind=kind,
            payload=payload,
            payload_kind=payload_kind,
            run_id=run_id,
            dataset_id=dataset_id,
            scope=scope,
            stage=stage,
            ts=time.time(),
            seq=next(_event_seq),
        ),
    )
    if scope is not None:
        scope.emitted += 1


def has_room() -> bool:
    """True when the next ``emit()`` would be buffered rather than dropped.

    For emit points whose payload is expensive to build (a graph snapshot): a
    burst past ``queue_size`` would otherwise pay the full build cost for
    events ``emit()`` discards on arrival. Two reads plus a ``len()``; a soft
    check (another thread may fill the buffer in between), never a
    reservation.
    """
    runtime = _runtime
    return runtime.sink is not None and len(runtime.buffer) < runtime.queue_size


def emit_lazy(
    kind: str,
    build_payload: Callable[[], Any],
    *,
    payload_kind: str = "json",
    stage: str | None = None,
    run_id: UUID | str | None = None,
    dataset_id: UUID | str | None = None,
) -> None:
    """``emit()`` for an expensive payload: build it only if it will be buffered.

    ``build_payload`` is the emit point's sanctioned snapshot cost (a
    ``model_dump(mode="json")``, a hash) and runs on the caller's thread, but
    only after the same sink/room checks ``emit()`` makes — so a full buffer
    costs the drop counter and nothing else, instead of a snapshot per dropped
    event. Unlike ``emit()`` this has an exception path: a builder that raises
    costs its own event (logged at debug), never the caller.
    """
    runtime = _runtime
    if runtime.sink is None:
        return
    scope = _current_scope.get()
    if len(runtime.buffer) >= runtime.queue_size:
        runtime.dropped += 1
        if scope is not None:
            scope.dropped += 1
        return
    try:
        payload = build_payload()
    except Exception as exc:
        logger.debug("capture payload build failed (%s)", exc)
        return
    _enqueue(
        runtime,
        CaptureEvent(
            kind=kind,
            payload=payload,
            payload_kind=payload_kind,
            run_id=run_id,
            dataset_id=dataset_id,
            scope=scope,
            stage=stage,
            ts=time.time(),
            seq=next(_event_seq),
        ),
    )
    if scope is not None:
        scope.emitted += 1


def _emit_manifest(
    kind: str,
    payload: Any,
    *,
    payload_kind: str,
    run_id: UUID | str | None,
    dataset_id: UUID | str | None,
) -> None:
    """``emit()`` with headroom past ``queue_size`` — for run manifests only.

    A run that overflowed the buffer must still land its manifest: it is the
    record carrying ``events_dropped`` and the authoritative dataset
    attribution, so plain drop-newest would make overflow unobservable
    offline. The headroom is a second ``queue_size`` — the buffer never holds
    more than ``2 * queue_size`` events. Manifests accumulate as fast as runs
    complete (not as fast as runs are concurrent: a wedged sink, or a sync
    caller with no loop to flush on, leaves every completed run's manifest
    buffered), so past the cap they are dropped and counted like any event.
    """
    runtime = _runtime
    if runtime.sink is None:
        return
    if len(runtime.buffer) >= 2 * runtime.queue_size:
        runtime.dropped += 1
        return
    # Not counted on the run: its manifest is the record carrying the counts,
    # taken before this call.
    _enqueue(
        runtime,
        CaptureEvent(
            kind=kind,
            payload=payload,
            payload_kind=payload_kind,
            run_id=run_id,
            dataset_id=dataset_id,
            scope=_current_scope.get(),
            stage=None,
            ts=time.time(),
            seq=next(_event_seq),
        ),
    )


def _enqueue(runtime: CaptureRuntime, event: CaptureEvent) -> None:
    """Append, make sure this loop has a flusher, fire the batch_size wake."""
    buffer = runtime.buffer
    buffer.append(event)

    loop = _get_running_loop()
    flusher = _ensure_flusher(runtime, loop) if loop is not None else None
    # else: a worker-thread emit; the deque is picked up by whichever flusher runs next.

    if len(buffer) >= runtime.batch_size:
        if flusher is None:
            flusher = _any_live_flusher(runtime)
        if flusher is not None and not flusher.wake_pending and not flusher.loop.is_closed():
            # wake_pending: one callback per batch, even in a synchronous burst
            # where the scheduled wake.set() has not run yet.
            flusher.wake_pending = True
            try:
                flusher.loop.call_soon_threadsafe(flusher.wake.set)
            except RuntimeError:
                flusher.wake_pending = False  # loop closing


def _ensure_flusher(runtime: CaptureRuntime, loop: asyncio.AbstractEventLoop) -> _Flusher | None:
    """Return this loop's live flusher, starting one if absent or crashed.

    ``None`` for a loop whose flusher was CANCELLED: that is the runner tearing
    the loop down (``asyncio.run``, uvicorn) or ``shutdown()``, and a fresh
    flusher started on a closing loop is destroyed with its popped batch. The
    cancelled entry stays as a tombstone until the loop is closed and pruned,
    or a sink is (re)registered. A flusher that CRASHED is replaced.
    """
    flusher = runtime.flushers.get(loop)
    if flusher is not None:
        task = flusher.task
        if not task.done():
            return flusher
        if task.cancelled():
            return None
    return _start_flusher(runtime, loop)


def ensure_flusher() -> None:
    """Start the running loop's flusher eagerly (called at ``run_scope`` entry).

    Emits from plain threads cannot start a flusher (no running loop in the
    emitting thread), so a run whose emit points all execute in worker threads
    would otherwise accumulate events until ``queue_size`` and drop the rest.
    Both wirings open their scope on-loop at run start, so this gives every run
    a live flusher before its first worker-thread emit. Off the hot path.
    """
    runtime = _runtime
    if runtime.sink is None:
        return
    loop = _get_running_loop()
    if loop is not None:
        _ensure_flusher(runtime, loop)


def _start_flusher(runtime: CaptureRuntime, loop: asyncio.AbstractEventLoop) -> _Flusher | None:
    _prune_closed_loops(runtime)

    flusher = _Flusher(runtime=runtime, loop=loop, wake=asyncio.Event())
    flush_coroutine = _flush_loop(flusher)
    try:
        flusher.task = loop.create_task(flush_coroutine)
    except RuntimeError:
        # Loop closing — same posture as _TELEMETRY_TASKS in cognee/shared/utils.py.
        flush_coroutine.close()
        return None

    runtime.flushers[loop] = flusher

    def _on_done(task: asyncio.Task) -> None:
        if runtime.flushers.get(loop) is not flusher:
            return
        if task.cancelled():
            # Keep the entry as a tombstone (see _ensure_flusher): the loop
            # is going away, no replacement is started on it.
            return
        runtime.flushers.pop(loop, None)
        # A crash (a BaseException out of a sink) ended the task; retrieving
        # the exception here keeps asyncio's "Task exception was never
        # retrieved" ERROR out of the logs. The next emit starts a replacement.
        exc = task.exception()
        if exc is not None:
            logger.debug("capture flusher stopped (%r)", exc)

    flusher.task.add_done_callback(_on_done)
    return flusher


def _prune_closed_loops(runtime: CaptureRuntime) -> None:
    """Forget flushers on closed loops; re-buffer the batches stranded there.

    Explicit because ``add_done_callback`` never fires for a task parked on a
    closed loop. A batch such a task had popped can never be acknowledged (the
    task died with its loop), so its undelivered slots go back to the head of
    the buffer for the next drain / the atexit hook — at-least-once: a group
    whose sink write completed but whose acknowledgement never ran is written
    again. A loop that merely stopped keeps its entries: it may run again (a
    driver calling ``run_until_complete`` repeatedly) and its task is still
    cancellable by ``shutdown()`` / ``_reset_for_tests()``;
    ``_any_live_flusher()`` and ``_in_flight_live()`` skip it while it is not
    running.
    """
    # Snapshot the keys — another thread may be mutating the dicts. Only a
    # closed loop's entries are touched, and no thread runs a closed loop.
    for stale_loop in list(runtime.flushers):
        if stale_loop.is_closed():
            runtime.flushers.pop(stale_loop, None)
    for stale_loop in list(runtime.in_flight):
        if stale_loop.is_closed():
            for pending in runtime.in_flight.pop(stale_loop, ()):
                _requeue(runtime, pending)


def _forget_stopped_flushers(runtime: CaptureRuntime) -> None:
    """Drop tombstones so flushing can resume on their loops (sink re-registered)."""
    for loop, flusher in list(runtime.flushers.items()):
        if flusher.task.done():
            runtime.flushers.pop(loop, None)


def _any_live_flusher(runtime: CaptureRuntime | None = None) -> _Flusher | None:
    """A flusher whose loop is RUNNING — the wake target for worker-thread emits.

    A stopped-but-open loop's flusher must not be chosen: ``call_soon_threadsafe``
    on it succeeds but the callback never runs, ``wake_pending`` sticks, and the
    live loop's flusher is never woken by thread emits.
    """
    if runtime is None:
        runtime = _runtime
    for flusher in list(runtime.flushers.values()):
        if flusher.loop.is_running() and not flusher.task.done():
            return flusher
    return None


# ---------------------------------------------------------------------------
# In-flight accounting
# ---------------------------------------------------------------------------


def _track_in_flight(
    runtime: CaptureRuntime,
    loop: asyncio.AbstractEventLoop,
    pending: list[CaptureEvent | None],
) -> None:
    # Only ever called from ``loop``'s own thread (see CaptureRuntime.in_flight).
    runtime.in_flight.setdefault(loop, []).append(pending)


def _untrack_in_flight(
    runtime: CaptureRuntime,
    loop: asyncio.AbstractEventLoop,
    pending: list[CaptureEvent | None],
) -> None:
    batches = runtime.in_flight.get(loop)
    if batches is None:
        return  # already recovered by _prune_closed_loops (loop closed under us)
    for index, candidate in enumerate(batches):
        if candidate is pending:
            del batches[index]
            break
    if not batches:
        runtime.in_flight.pop(loop, None)


def _count_pending(batches: list[list[CaptureEvent | None]]) -> int:
    return sum(1 for pending in list(batches) for event in pending if event is not None)


def _in_flight_total(runtime: CaptureRuntime | None = None) -> int:
    """Popped-but-undelivered events on every loop (tests, diagnostics)."""
    if runtime is None:
        runtime = _runtime
    return sum(_count_pending(batches) for batches in list(runtime.in_flight.values()))


def _in_flight_live(runtime: CaptureRuntime) -> int:
    """Popped-but-undelivered events on loops that can still deliver them."""
    return sum(
        _count_pending(batches)
        for loop, batches in list(runtime.in_flight.items())
        if loop.is_running()
    )


# ---------------------------------------------------------------------------
# Flusher
# ---------------------------------------------------------------------------


async def _wait_bounded(awaitable: Awaitable[T], timeout: float | None) -> T:
    """``asyncio.wait_for`` that cannot swallow the caller's cancellation.

    On CPython < 3.12 ``wait_for`` returns the inner result instead of raising
    when the cancel lands in the same loop iteration as the inner completion
    (bpo-42130): ``Task.cancel()`` consumed the request and it is lost. Here
    the caller parks on ``asyncio.wait``, whose waiter future carries the
    cancellation straight through. Like ``wait_for``, a timeout cancels the
    inner awaitable and waits for that cancellation to land before raising
    ``asyncio.TimeoutError``, so the inner is never left running.

    Unlike ``wait_for``, that post-cancel wait is itself bounded by
    ``_CANCEL_GRACE_S``. A sink that swallows ``CancelledError`` (out of
    contract, but ``register_capture_sink`` is public API) would otherwise pin
    the flusher forever and blow every ``drain()`` budget without limit — and
    because the same wait sits on the ``CancelledError`` branch, no outer
    ``wait_for`` could rescue the caller either. Giving up on the cancel leaves
    the inner running, so the caller must treat the batch as undelivered
    (``_flush_one_batch`` re-buffers it).
    """
    future = asyncio.ensure_future(awaitable)
    if timeout is None:
        # A direct await propagates a racing cancel correctly (it is wait_for's
        # `if fut.done(): return fut.result()` that does not).
        return await future
    try:
        done, _pending = await asyncio.wait({future}, timeout=timeout)
    except asyncio.CancelledError:
        # The caller was cancelled: take the inner down with it, and — like
        # wait_for — do not return while it may still be running.
        if not future.done():
            future.cancel()
            await asyncio.wait({future}, timeout=_CANCEL_GRACE_S)
        raise
    if not done:
        future.cancel()
        await asyncio.wait({future}, timeout=_CANCEL_GRACE_S)
        raise asyncio.TimeoutError
    return future.result()


async def _flush_loop(flusher: _Flusher) -> None:
    runtime = flusher.runtime
    wake = flusher.wake
    while True:
        try:
            try:
                await _wait_bounded(wake.wait(), runtime.flush_interval_s)
            except asyncio.TimeoutError:
                pass  # interval tick
            wake.clear()
            # Re-arm BEFORE flushing so an emit that lands mid-flush can wake us again.
            flusher.wake_pending = False
            # Burst absorption: after a full batch keep flushing until the buffer
            # is below the threshold; sleep only when quiet.
            while runtime.buffer:
                if await _flush_one_batch(runtime) == 0:
                    # Nothing poppable (a drain took it): never spin on an
                    # await that does not suspend.
                    break
                if len(runtime.buffer) < runtime.batch_size:
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
        # A scope without a dataset of its own (an operation recorded inside a
        # pipeline task) is attributed to the enclosing run's dataset.
        dataset_id = event.dataset_id or (
            scope.resolved_dataset_id() if scope is not None else None
        )
        records.append(
            {
                "schema_version": CAPTURE_SCHEMA_VERSION,
                "event_id": f"{_PROCESS_ID}-{event.seq}",
                "kind": event.kind,
                "run_id": None if run_id is None else str(run_id),
                "dataset_id": None if dataset_id is None else str(dataset_id),
                "stage": event.stage,
                "ts": event.ts,
                "payload": payload,
            }
        )
    return records


def _account_dropped(events: Iterable[CaptureEvent | None]) -> int:
    """Count events as lost — on their run's ``dropped`` — and return how many."""
    lost = 0
    for event in events:
        if event is None:
            continue
        lost += 1
        scope = event.scope
        if scope is not None:
            scope.dropped += 1
    return lost


def _account_delivered(events: Iterable[CaptureEvent | None]) -> None:
    """Credit each event's run once, even when at-least-once delivers it twice."""
    for event in events:
        if event is None or event.delivered:
            continue
        event.delivered = True
        scope = event.scope
        if scope is not None:
            scope.delivered += 1


def _requeue(runtime: CaptureRuntime, pending: list[CaptureEvent | None]) -> None:
    """Put the not-yet-delivered events back at the head of the buffer, in order.

    The slots are blanked so the batch counts as nothing in flight afterwards:
    an event is always in exactly one of the buffer, a tracked batch, the sink,
    or dropped.
    """
    leftovers = [event for event in pending if event is not None]
    if leftovers:
        runtime.buffer.extendleft(reversed(leftovers))
        for index in range(len(pending)):
            pending[index] = None


def _remaining(deadline: float | None) -> float | None:
    return None if deadline is None else deadline - time.monotonic()


async def _flush_one_batch(runtime: CaptureRuntime, deadline: float | None = None) -> int:
    """Pop up to batch_size events, serialize off-loop, hand groups to the sink.

    Returns the number of events popped; 0 means there was nothing to pop, and
    callers stop on it so a batch that pops nothing can never spin.

    ``deadline`` (``time.monotonic()``) is ``drain()``'s budget. It bounds the
    serialization wait and every sink write (each is also bounded by
    ``sink_timeout_s``); once it is spent, whatever is not yet delivered —
    including the group that was being written — goes back to the head of the
    buffer for the flusher / the atexit hook (at-least-once, see the module
    docstring). The flusher passes no deadline.

    The popped events are tracked in ``runtime.in_flight`` for this loop from
    pop to delivery so ``drain()`` can wait for a batch another consumer is
    still serializing, and so ``_prune_closed_loops`` can recover the batch if
    the loop dies under it.
    """
    buffer = runtime.buffer
    batch: list[CaptureEvent] = []
    try:
        while buffer and len(batch) < runtime.batch_size:
            batch.append(buffer.popleft())
    except IndexError:
        pass  # raced with another consumer (drain + flusher split the work)
    if not batch:
        return 0

    sink = runtime.sink
    if sink is None:
        # Sink cleared after these were queued: nothing to deliver to. Account
        # for the loss instead of hiding it.
        runtime.dropped += _account_dropped(batch)
        logger.debug("capture sink cleared, %d buffered event(s) dropped", len(batch))
        return len(batch)

    loop = asyncio.get_running_loop()
    # One slot per event, blanked as its group is acknowledged; on cancellation
    # or a spent deadline the rest goes back to the buffer.
    pending: list[CaptureEvent | None] = list(batch)
    _track_in_flight(runtime, loop, pending)
    try:
        records: list[dict] | None = None
        remaining = _remaining(deadline)
        if remaining is None or remaining > 0:
            try:
                records = await _wait_bounded(run_off_loop(_serialize_batch, batch), remaining)
            except asyncio.TimeoutError:
                pass  # budget spent while serializing: the whole batch goes back

        if records is not None:
            groups: dict[tuple[str | None, str], list[int]] = {}
            for index, record in enumerate(records):
                groups.setdefault((record["run_id"], record["kind"]), []).append(index)

            for indexes in groups.values():
                # Bounded: a wedged sink (S3 under partition) must not pin the
                # flusher — nor a drain() past its own budget.
                bound = runtime.sink_timeout_s
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    bound = min(bound, remaining)
                group = [records[index] for index in indexes]
                slots = [pending[index] for index in indexes]
                delivered = False
                try:
                    await _wait_bounded(sink(group), bound)
                    delivered = True
                except asyncio.TimeoutError as exc:
                    if deadline is not None and time.monotonic() >= deadline:
                        break  # the caller's budget ran out, not the sink's: retry later
                    # The sink blew sink_timeout_s; its write was cancelled and
                    # the slots are blanked below, so these events are gone.
                    # Account for the loss instead of hiding it: a run whose sink
                    # is wedged must not report events_dropped = 0.
                    runtime.dropped += _account_dropped(slots)
                    logger.debug(
                        "capture sink timed out, %d event(s) dropped (%s)", len(indexes), exc
                    )
                except Exception as exc:
                    runtime.dropped += _account_dropped(slots)
                    logger.debug("capture sink failed, %d event(s) dropped (%s)", len(indexes), exc)
                if delivered:
                    _account_delivered(slots)
                for index in indexes:
                    pending[index] = None
                # Let other coroutines interleave between sink writes.
                await asyncio.sleep(0)

        # Deadline leftovers (never any for the flusher, which has no deadline).
        _requeue(runtime, pending)
    except Exception as exc:
        # Not re-queued: a deterministic failure would pin the head of the
        # buffer forever. Account for the loss instead of hiding it.
        lost = _account_dropped(pending)
        runtime.dropped += lost
        logger.debug("capture flush failed, %d event(s) dropped (%s)", lost, exc)
    except BaseException:
        # Cancellation (asyncio.run teardown, uvicorn shutdown, a cancelled
        # drain() caller) or a custom BaseException escaping a sink: do not lose
        # the popped batch. KeyboardInterrupt/SystemExit from a sink do NOT
        # reach here - asyncio re-raises those out of the loop itself.
        _requeue(runtime, pending)
        raise
    finally:
        # Whatever was not delivered is either back in the buffer or dropped:
        # in no case is it still in flight.
        _untrack_in_flight(runtime, loop, pending)
    return len(batch)


# ---------------------------------------------------------------------------
# Drain / shutdown
# ---------------------------------------------------------------------------


async def _drain_until(runtime: CaptureRuntime, deadline: float) -> bool:
    # Batches stranded on loops closed since the last pass go back to the
    # buffer first, so this drain covers them too.
    _prune_closed_loops(runtime)
    # (a) Inline on the calling loop: flush what is buffered NOW. The deque is
    # the coordination point — popleft is atomic, so a concurrently running
    # flusher and this drain simply split the work (blob names are
    # collision-free). Budgeted by the entry count so a producer faster than
    # the sink cannot pin the caller (the flusher owns whatever arrives later),
    # and by the deadline, which _flush_one_batch enforces inside the batch.
    buffer = runtime.buffer
    budget = len(buffer)
    while budget > 0 and buffer and time.monotonic() < deadline:
        popped = await _flush_one_batch(runtime, deadline)
        if popped == 0:
            break  # the flusher took the rest; never spin
        budget -= popped
    # (b) Wait for batches other consumers hold (popped, serializing or
    # awaiting their sink): plain counters, no foreign-loop futures — and only
    # on loops that are running. A batch on a stopped loop cannot complete
    # while it is stopped; waiting for it would turn every later drain() in
    # the process into a full timeout.
    while _in_flight_live(runtime) > 0 and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    return not buffer and _in_flight_live(runtime) == 0


async def drain(timeout: float | None = None) -> bool:
    """Flush the events buffered so far, then wait for in-flight batches.

    ``timeout=None`` (the default) means the configured budget:
    ``COGNEE_CAPTURE_DRAIN_TIMEOUT_S``, 5 s unless overridden — this is the
    wait the pipeline wiring adds at every run completion while capture is on.

    Never requires a flusher to exist, never raises (a cancellation of the
    caller propagates, as it must), and returns within about ``timeout`` — the
    budget is enforced inside a batch too, per sink write — even under a
    continuous producer or a wedged sink (leftovers stay with the flusher /
    the atexit hook). "About" is ``timeout`` plus at most one
    ``_CANCEL_GRACE_S``: a write cut off by the budget is cancelled, and a sink
    that does not let that cancel land is abandoned after the grace rather than
    waited on forever.

    Returns True when nothing is left buffered or in flight on this process —
    every event was delivered or accounted as dropped — and False when the
    budget ran out or a concurrent producer kept the buffer non-empty. The
    pipeline wiring uses it to decide whether its manifest can be drained
    inline too.
    """
    runtime = _runtime
    if timeout is None:
        timeout = runtime.drain_timeout_s
    try:
        return await _drain_until(runtime, time.monotonic() + timeout)
    except Exception as exc:
        logger.debug("capture drain failed (%s)", exc)
        return False


async def shutdown(timeout: float = 5.0) -> None:
    """Drain, stop every flusher, then deliver anything a cancelled flush put back.

    For process teardown (FastAPI lifespan, tests), not the request path. The
    stopped flushers stay registered as tombstones, so an emit that arrives
    after shutdown (one more request during a lifespan shutdown) does not
    start a new flusher: it stays in the deque for the atexit hook.
    """
    runtime = _runtime
    deadline = time.monotonic() + timeout
    grace = min(timeout, _CANCEL_GRACE_S)
    await drain(timeout)
    current_loop = _get_running_loop()
    same_loop_tasks: list[asyncio.Task] = []
    for flusher in list(runtime.flushers.values()):
        if flusher.loop is current_loop and not flusher.task.done():
            same_loop_tasks.append(flusher.task)
        _cancel_flusher(flusher, wait=False)
    if same_loop_tasks:
        # Bounded: a flusher that failed to honour its cancel must not hang teardown.
        await asyncio.wait(same_loop_tasks, timeout=grace)
    # A flusher cancelled mid-batch re-buffered it; no flusher is left to pick
    # it up, so deliver it here rather than leaving it to the atexit hook.
    if runtime.buffer:
        try:
            await _drain_until(runtime, max(deadline, time.monotonic() + grace))
        except Exception as exc:
            logger.debug("capture drain failed (%s)", exc)


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
            # Deliver the cancellation so the task is not destroyed while
            # pending. Bounded: a regression that loses the cancel must fail a
            # test, not hang the suite.
            try:
                loop.run_until_complete(asyncio.wait({flusher.task}, timeout=_CANCEL_GRACE_S))
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def should_capture(kind: str) -> bool:
    """Per-run sampling gate for retrieval kinds; every other kind is captured.

    Inside a ``run_scope`` the decision was made ONCE at scope entry (no
    per-emit RNG); without a scope fall back to ``retrieval_sample_rate``.
    """
    if not kind.startswith(RETRIEVAL_KIND_PREFIX):
        return True
    scope = _current_scope.get()
    if scope is not None:
        return scope.sampled
    return random.random() < _runtime.retrieval_sample_rate


# ---------------------------------------------------------------------------
# Test hooks
# ---------------------------------------------------------------------------


def _reset_for_tests() -> None:
    """Swap in a pristine runtime (sync).

    Cancels the current runtime's flusher tasks whose loop is still open —
    and, when that loop is not running, runs it until the cancellation is
    delivered — so pytest-asyncio 0.21.x (which closes loops without
    cancelling tasks) never destroys a pending flusher. The old runtime is
    then simply dropped: a flusher on a foreign running loop, cancelled
    asynchronously, keeps draining the old instance it was bound to and can
    never touch the new one.
    """
    global _runtime
    for flusher in list(_runtime.flushers.values()):
        _cancel_flusher(flusher, wait=True)
    _runtime = CaptureRuntime()
    get_capture_config.cache_clear()
