"""Worker-per-task pipeline executor with bounded queues.

The executor builds one ``asyncio.Queue`` between every pair of adjacent
tasks and runs a pool of worker coroutines per task that pull from the
upstream queue, invoke ``task.execute``, and push to the downstream queue.

Public API (configured per ``Task``):

- ``workers=<WorkerStrategy>`` -- sizing policy for the task's worker pool.
  Defaults to ``FixedWorkers(1)`` when omitted. Two strategies ship:

  - ``FixedWorkers(num_workers)`` -- static pool of ``num_workers`` workers.
  - ``AdaptiveWorkers(initial_workers, max_workers, step, tick_seconds,
    throughput_improvement_ratio)`` -- pool that grows/shrinks at
    ``tick_seconds`` intervals based on observed throughput, bounded by
    ``max_workers`` and stepping by ``step`` workers at a time.

- ``timeout=<seconds>`` -- per-call timeout for ``task.execute``. A
  per-call timeout surfaces as a throttling event for the adaptive
  controller (so it shrinks the pool) AND fails the item via the standard
  ``_ErroredItem`` envelope path -- there is no automatic retry.

Order preservation: ``FixedWorkers(1)`` preserves input order end-to-end.
Any other strategy (including ``AdaptiveWorkers`` and ``FixedWorkers(n>1)``)
may reorder items. Per-item failures are wrapped in ``_ErroredItem``
envelopes and flow through the pipeline so one bad item does not abort its
siblings.

Telemetry: queue-depth samples are collected in-memory and emitted as one
``Task Queue Metrics`` event per task at the end of the run. For adaptive
pools the event additionally carries ``adaptive_final_target`` plus
throttle/grow/shrink counts and peak/mean throughput. Adaptive pools also
resume from the previous run's converged target via an in-process registry,
so repeated runs of the same task skip the warm-up ramp.
"""

import asyncio
import dataclasses
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterable, Optional, Union

from cognee.infrastructure.engine import DataPoint
from cognee.modules.observability import (
    COGNEE_PIPELINE_TASK_NAME,
    COGNEE_RESULT_COUNT,
    COGNEE_RESULT_SUMMARY,
    OtelStatusCode as StatusCode,
    new_span,
)
from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.shared.utils import send_telemetry
from cognee import __version__ as cognee_version

from ..tasks.task import Task


logger = get_logger("worker_pipeline")

_QUEUE_SAMPLE_RING_SIZE = 4096


# ---------------------------------------------------------------------------
# Envelopes and sentinels
# ---------------------------------------------------------------------------


class _Sentinel:
    """End-of-stream marker. Use the singleton ``_SENTINEL``."""

    __slots__ = ()


_SENTINEL = _Sentinel()


class _NoData:
    """Marker used by single-input callers (run_tasks_single) when the first
    task should be invoked with no positional arguments (i.e. caller passed
    ``data=None``)."""

    __slots__ = ()


_NO_DATA = _NoData()


@dataclass
class _ErroredItem:
    """Wraps an exception so it can flow through downstream queues without
    aborting the pipeline."""

    exception: BaseException


@dataclass
class _ItemEnvelope:
    """Queue payload: a value in flight + the original Data item it descended
    from (so ``ctx.data_item`` can be set correctly at every stage) +
    monotonic sequence number (for telemetry / debugging)."""

    value: Any
    origin: Any
    seq: int


# ---------------------------------------------------------------------------
# Instrumented queue
# ---------------------------------------------------------------------------


class InstrumentedQueue(asyncio.Queue):
    """``asyncio.Queue`` that records timestamp + depth on every put/get.

    Samples are kept in a bounded ring buffer (FIFO eviction) to cap memory
    on long runs.
    """

    def __init__(self, maxsize: int = 0, name: str = "queue"):
        super().__init__(maxsize=maxsize)
        self.name = name
        self._samples: deque = deque(maxlen=_QUEUE_SAMPLE_RING_SIZE)

    @property
    def samples(self) -> deque:
        return self._samples

    def _record(self, op: str) -> None:
        loop = asyncio.get_running_loop()
        self._samples.append((loop.time(), op, self.qsize()))

    async def put(self, item):
        await super().put(item)
        self._record("push")

    async def get(self):
        item = await super().get()
        self._record("pop")
        return item


def _compute_queue_stats(samples: deque, maxsize: int) -> dict:
    """Aggregate ring-buffer samples into a telemetry dict.

    ``pct_time_full`` / ``pct_time_empty`` integrate the time-weighted step
    function defined by the (time, depth) samples. When ``maxsize == 0`` the
    queue is unbounded and ``pct_time_full`` is always 0.
    """
    if not samples:
        return {
            "max_depth": 0,
            "mean_depth": 0.0,
            "pct_time_full": 0.0,
            "pct_time_empty": 0.0,
            "pushes": 0,
            "pops": 0,
        }

    pushes = sum(1 for s in samples if s[1] == "push")
    pops = sum(1 for s in samples if s[1] == "pop")
    max_depth = max(s[2] for s in samples)

    total_time = 0.0
    weighted_depth = 0.0
    time_full = 0.0
    time_empty = 0.0
    prev_t, _, prev_depth = samples[0]
    for t, _, depth in list(samples)[1:]:
        dt = t - prev_t
        if dt > 0:
            total_time += dt
            weighted_depth += prev_depth * dt
            if maxsize > 0 and prev_depth >= maxsize:
                time_full += dt
            if prev_depth == 0:
                time_empty += dt
        prev_t, prev_depth = t, depth

    mean_depth = (weighted_depth / total_time) if total_time > 0 else 0.0
    pct_full = (100.0 * time_full / total_time) if total_time > 0 else 0.0
    pct_empty = (100.0 * time_empty / total_time) if total_time > 0 else 0.0

    return {
        "max_depth": max_depth,
        "mean_depth": round(mean_depth, 3),
        "pct_time_full": round(pct_full, 2),
        "pct_time_empty": round(pct_empty, 2),
        "pushes": pushes,
        "pops": pops,
    }


# ---------------------------------------------------------------------------
# Provenance helpers (moved verbatim from the previous run_tasks_single.py)
# ---------------------------------------------------------------------------


def _stamp_provenance(
    data, pipeline_name, task_name, visited=None, node_set=None, user_label=None, content_hash=None
):
    """Recursively stamp DataPoints with provenance. Only sets if currently None."""
    if visited is None:
        visited = set()

    if isinstance(data, DataPoint):
        obj_id = id(data)
        if obj_id in visited:
            return
        visited.add(obj_id)

        if data.source_pipeline is None:
            data.source_pipeline = pipeline_name
        if data.source_task is None:
            data.source_task = task_name
        if data.source_user is None and user_label is not None:
            data.source_user = user_label

        current_node_set = node_set
        if data.source_node_set is not None:
            current_node_set = data.source_node_set
        elif current_node_set is not None and data.source_node_set is None:
            data.source_node_set = current_node_set

        current_hash = content_hash
        if data.source_content_hash is not None:
            current_hash = data.source_content_hash
        elif current_hash is not None and data.source_content_hash is None:
            data.source_content_hash = current_hash

        for field_name in data.model_fields:
            field_value = getattr(data, field_name, None)
            if field_value is not None:
                _stamp_provenance(
                    field_value,
                    pipeline_name,
                    task_name,
                    visited,
                    current_node_set,
                    user_label,
                    current_hash,
                )

    elif isinstance(data, (list, tuple)):
        for item in data:
            _stamp_provenance(
                item, pipeline_name, task_name, visited, node_set, user_label, content_hash
            )


def _extract_node_set(args):
    for arg in args:
        if isinstance(arg, DataPoint) and arg.source_node_set is not None:
            return arg.source_node_set
        if isinstance(arg, (list, tuple)):
            for item in arg:
                if isinstance(item, DataPoint) and item.source_node_set is not None:
                    return item.source_node_set
    return None


def _extract_content_hash(args):
    from cognee.modules.data.models.Data import Data

    for arg in args:
        if isinstance(arg, Data) and arg.content_hash is not None:
            return arg.content_hash
        if isinstance(arg, DataPoint) and arg.source_content_hash is not None:
            return arg.source_content_hash
        if isinstance(arg, (list, tuple)):
            for item in arg:
                if isinstance(item, Data) and item.content_hash is not None:
                    return item.content_hash
                if isinstance(item, DataPoint) and item.source_content_hash is not None:
                    return item.source_content_hash
    return None


def _build_result_summary(executable, task_name: str, count: int) -> str:
    template = getattr(executable, "__task_summary__", None)
    if template:
        return template.format(n=count)
    return f"{task_name} produced {count} result(s)"


# ---------------------------------------------------------------------------
# Adaptive worker pool (AIAD: additive ±step on throttling signal)
# ---------------------------------------------------------------------------


# Substrings that mark an error as throttling-induced rather than a data error.
_THROTTLING_SIGNATURES = (
    "429",
    "rate_limit",
    "rate limit",
    "too many requests",
    "connection error",
    "service unavailable",
    "read timed out",
    "request timed out",
)
_THROTTLING_TYPE_NAMES = {
    "RateLimitError",
    "TimeoutError",
    "ConnectionError",
    "ConnectError",
    "ReadTimeout",
}


def _is_throttling_error(exc: BaseException) -> bool:
    """Conservative classifier: returns True only for errors that strongly
    suggest external resource pressure (so a data-validation error doesn't
    trigger an unnecessary shrink).

    CAUTION: This classifier is intentionally string-based (matching on the
    exception's class name and the lowercased ``str(exc)``) rather than an
    enforced interface contract. That keeps the adaptive pool useful across
    arbitrary third-party libraries — but it also means false positives can
    occur whenever a user-thrown exception happens to contain a throttling-
    like substring (e.g. a ``ValueError`` whose message includes "429",
    "timeout", or "connection error") for unrelated reasons. The consequence
    is at worst an unnecessary pool shrink: the shrink is conservative and
    recoverable, and the pool will grow back via the throughput hill-climber
    once conditions normalize. Authors writing new tasks should nonetheless
    avoid raising exceptions whose ``str()`` matches the substrings in
    ``_THROTTLING_SIGNATURES`` unless the underlying cause really is
    throttling, to keep the adaptive signal clean."""
    type_name = type(exc).__name__
    if type_name in _THROTTLING_TYPE_NAMES:
        return True
    msg = str(exc).lower()
    return any(sig in msg for sig in _THROTTLING_SIGNATURES)


# ---------------------------------------------------------------------------
# Cross-run adaptive-target registry
# ---------------------------------------------------------------------------
# Module-level cache of the last converged target per task. Lets adaptive
# pools resume from the previously-discovered concurrency level instead of
# paying the full discovery cost on every pipeline run. Lives for the
# lifetime of the process (typically a long-running uvicorn server, in which
# case every cognify call benefits after the first).
#
# We only cache the *target* — not the whole pool — because per-run counters
# (throughput history, streak counters, last_action) shouldn't survive across
# runs with potentially different workloads, and ``asyncio.Semaphore`` is
# tied to the event loop it was created on.

_ADAPTIVE_TARGET_REGISTRY: dict[str, int] = {}


def get_last_adaptive_target(task_name: str) -> Optional[int]:
    """Return the last converged target for ``task_name``, or None if no
    adaptive pool has finished a run for that task yet."""
    return _ADAPTIVE_TARGET_REGISTRY.get(task_name)


def reset_adaptive_target_registry() -> None:
    """Drop all cached adaptive targets. Useful for tests."""
    _ADAPTIVE_TARGET_REGISTRY.clear()


class _AdaptivePool:
    """Adaptive worker pool with AIAD control on throttling exceptions.

    Workers always exist at ``max_workers`` count and gate on this pool's
    internal semaphore before processing each item. The semaphore's available
    permit count == ``target`` (the current desired concurrency).

    The control loop runs every ``tick_seconds`` and applies these criteria
    in priority order:

    1. **No completions in window** → no change (insufficient signal).
    2. **Throttling exception/timeout observed** → shrink by ``step``.
    3. **Average busy-worker utilization < 50%** → shrink by ``step``
       (pool is over-provisioned; workers spent most of the tick idle).
    4. **Throughput hill-climbing**: compare completions/s to previous tick.
       If the last action (grow or shrink) improved throughput, continue
       in the same direction; otherwise reverse.
    """

    # Sub-tick interval (seconds) at which the ticker samples how many
    # workers are currently busy. Used to compute avg utilization per tick.
    _BUSY_SAMPLE_INTERVAL_S = 1.0

    def __init__(
        self,
        task_name: str,
        initial: int,
        max_workers: int,
        min_workers: int = 1,
        step: int = 5,
        tick_seconds: float = 10.0,
        throughput_improvement_ratio: float = 1.05,
        underutilization_threshold: float = 0.5,
    ):
        self.task_name = task_name
        self.target = max(min_workers, min(initial, max_workers))
        self.max_workers = max_workers
        self.min_workers = min_workers
        self.step = step
        self.tick_seconds = tick_seconds
        self._sem = asyncio.Semaphore(self.target)

        # Throttling-exception signal
        self._throttle_events_in_window = 0
        self.total_throttle_events = 0

        # Busy / utilization tracking
        self._busy_workers = 0  # currently executing
        self._underutilization_threshold = underutilization_threshold
        # Count of consecutive ticks where util<threshold AND queue had work.
        # We require 2-in-a-row to avoid noise (single low-util ticks can come
        # from transient upstream lag).
        self._consecutive_underutil_ticks = 0
        self._underutil_required_streak = 2
        # Count of consecutive ticks where util was low (regardless of queue).
        # After freeze_after_low_util_ticks ticks like this we conclude we are
        # not the bottleneck — upstream is — and stop hill-climbing. The pool
        # holds position until util rises again (resetting the counter).
        self._consecutive_low_util_ticks = 0
        self._freeze_after_low_util_ticks = 3

        # Throughput hill-climbing state
        self._completed_in_window = 0
        self._last_throughput: Optional[float] = None
        self._last_action: Optional[str] = None  # "grow" | "shrink" | None
        self._throughput_improvement_ratio = throughput_improvement_ratio
        self._throughput_history: list = []  # for telemetry

        self.total_grow_events = 0
        self.total_shrink_events = 0
        self._target_history: list = [(0.0, self.target)]
        self._stop = asyncio.Event()
        self._tick_task: Optional[asyncio.Task] = None
        self._start_time: Optional[float] = None

    @asynccontextmanager
    async def permit(self):
        async with self._sem:
            self._busy_workers += 1
            try:
                yield
            finally:
                self._busy_workers -= 1

    def report_throttling(self) -> None:
        self._throttle_events_in_window += 1
        self.total_throttle_events += 1

    def report_success(self) -> None:
        """Record a successful call. Used to compute per-tick throughput."""
        self._completed_in_window += 1

    async def _grow(self, delta: int) -> None:
        new_target = min(self.target + delta, self.max_workers)
        added = new_target - self.target
        if added <= 0:
            return
        self.target = new_target
        for _ in range(added):
            self._sem.release()
        self.total_grow_events += 1

    async def _shrink(self, delta: int) -> None:
        new_target = max(self.target - delta, self.min_workers)
        removed = self.target - new_target
        if removed <= 0:
            return
        self.target = new_target
        # Burn `removed` permits so the worker pool collapses to the new target.
        # Each acquire blocks until a worker releases its permit (i.e. finishes
        # its current item), so shrink is paced by natural completions.
        for _ in range(removed):
            await self._sem.acquire()
        self.total_shrink_events += 1

    def start_ticker(self, in_queue: "InstrumentedQueue") -> None:
        async def _ticker():
            loop = asyncio.get_running_loop()
            self._start_time = loop.time()
            try:
                while True:
                    # Sample busy-worker count every _BUSY_SAMPLE_INTERVAL_S
                    # across the tick window so we can compute avg utilization.
                    busy_samples: list[int] = []
                    deadline = loop.time() + self.tick_seconds
                    while True:
                        remaining = deadline - loop.time()
                        if remaining <= 0:
                            break
                        step_time = min(self._BUSY_SAMPLE_INTERVAL_S, remaining)
                        try:
                            await asyncio.wait_for(self._stop.wait(), timeout=step_time)
                            return  # stop was set
                        except asyncio.TimeoutError:
                            busy_samples.append(self._busy_workers)

                    # End of tick — read & reset counters.
                    throttled = self._throttle_events_in_window
                    completions = self._completed_in_window
                    self._throttle_events_in_window = 0
                    self._completed_in_window = 0
                    throughput = completions / self.tick_seconds
                    self._throughput_history.append(round(throughput, 3))
                    avg_busy = sum(busy_samples) / len(busy_samples) if busy_samples else 0.0
                    utilization = avg_busy / self.target if self.target > 0 else 0.0
                    prev = self.target
                    queue_has_work = in_queue.qsize() > 0

                    def _log_change(direction: str, reason: str) -> None:
                        logger.info(
                            f"[adaptive {self.task_name}] target {prev} → {self.target} "
                            f"({direction}: {reason}; "
                            f"throughput={throughput:.2f}/s, "
                            f"util={utilization:.0%}, queue={in_queue.qsize()})"
                        )

                    def _log_no_change(reason: str) -> None:
                        logger.info(
                            f"[adaptive {self.task_name}] target {prev} unchanged "
                            f"({reason}; throughput={throughput:.2f}/s, "
                            f"util={utilization:.0%}, queue={in_queue.qsize()})"
                        )

                    # Criterion 1: no completions in window → skip (no signal).
                    if completions == 0:
                        _log_no_change("no completions in window")
                        continue

                    # Criterion 2: throttling exceptions / timeouts → shrink.
                    if throttled > 0:
                        if self.target > self.min_workers:
                            await self._shrink(self.step)
                            self._last_action = "shrink"
                            self._last_throughput = throughput
                            _log_change("shrink", f"{throttled} timeout/throttle event(s)")
                        else:
                            _log_no_change(f"{throttled} throttle event(s) but at min_workers")
                        now = loop.time() - self._start_time
                        self._target_history.append((now, self.target))
                        continue

                    # Criterion 3: under-utilized → shrink (pool is over-provisioned).
                    # Gated on (a) there is real work in the queue — otherwise
                    # low util is just upstream-starvation, not over-provisioning —
                    # and (b) the signal has persisted for N consecutive ticks
                    # to avoid one-off noise.
                    underutil = utilization < self._underutilization_threshold
                    if underutil and queue_has_work:
                        self._consecutive_underutil_ticks += 1
                    else:
                        self._consecutive_underutil_ticks = 0

                    if self._consecutive_underutil_ticks >= self._underutil_required_streak:
                        if self.target > self.min_workers:
                            await self._shrink(self.step)
                            self._last_action = "shrink"
                            self._last_throughput = throughput
                            _log_change(
                                "shrink",
                                f"util {utilization:.0%} < "
                                f"{self._underutilization_threshold:.0%} "
                                f"for {self._consecutive_underutil_ticks} consecutive tick(s)",
                            )
                            self._consecutive_underutil_ticks = 0
                        else:
                            _log_no_change("under-utilized but at min_workers")
                        now = loop.time() - self._start_time
                        self._target_history.append((now, self.target))
                        continue

                    # Track sustained low utilization to detect "we are not the
                    # bottleneck" — upstream is. When this holds for K ticks
                    # in a row, freeze hill-climbing (don't oscillate around a
                    # workload our pool size can't influence).
                    if underutil:
                        self._consecutive_low_util_ticks += 1
                    else:
                        self._consecutive_low_util_ticks = 0

                    if self._consecutive_low_util_ticks >= self._freeze_after_low_util_ticks:
                        _log_no_change(
                            f"frozen (util < {self._underutilization_threshold:.0%} for "
                            f"{self._consecutive_low_util_ticks} ticks — upstream-bound)"
                        )
                        self._last_throughput = throughput
                        now = loop.time() - self._start_time
                        self._target_history.append((now, self.target))
                        continue

                    # Criterion 4: hill-climb on throughput.
                    # Always take an exploratory grow when last_action is None
                    # (we have throughput data but haven't probed grow vs
                    # shrink yet). Subsequent ticks reverse direction when an
                    # action fails to improve throughput.
                    if self._last_action is None:
                        if self.target < self.max_workers:
                            await self._grow(self.step)
                            self._last_action = "grow"
                            _log_change(
                                "grow",
                                f"exploratory probe (throughput={throughput:.2f}/s)",
                            )
                        else:
                            _log_no_change("at max_workers, can't explore higher")
                    else:
                        improved = (
                            throughput >= self._last_throughput * self._throughput_improvement_ratio
                        )
                        if self._last_action == "grow":
                            next_action = "grow" if improved else "shrink"
                        else:  # last_action == "shrink"
                            next_action = "shrink" if improved else "grow"

                        if next_action == "grow" and self.target < self.max_workers:
                            await self._grow(self.step)
                            self._last_action = "grow"
                            _log_change(
                                "grow",
                                f"throughput {self._last_throughput:.2f}→{throughput:.2f}/s "
                                f"({'helped' if improved else 'did not help'})",
                            )
                        elif next_action == "shrink" and self.target > self.min_workers:
                            await self._shrink(self.step)
                            self._last_action = "shrink"
                            _log_change(
                                "shrink",
                                f"throughput {self._last_throughput:.2f}→{throughput:.2f}/s "
                                f"({'helped' if improved else 'did not help'})",
                            )
                        else:
                            _log_no_change("hill-climb suggests no change (at bound)")

                    self._last_throughput = throughput
                    now = asyncio.get_running_loop().time() - self._start_time
                    self._target_history.append((now, self.target))
            except asyncio.CancelledError:
                return

        self._tick_task = asyncio.create_task(_ticker())

    async def stop_ticker(self) -> None:
        self._stop.set()
        if self._tick_task is not None and not self._tick_task.done():
            try:
                await self._tick_task
            except asyncio.CancelledError:
                pass
        # Persist the converged target so the next pipeline run for the same
        # task can resume here instead of paying the discovery cost again.
        _ADAPTIVE_TARGET_REGISTRY[self.task_name] = self.target

    def stats(self) -> dict:
        targets = [t for _, t in self._target_history] or [self.target]
        ths = self._throughput_history
        return {
            "adaptive_initial_target": self._target_history[0][1]
            if self._target_history
            else self.target,
            "adaptive_final_target": self.target,
            "adaptive_min_target": min(targets),
            "adaptive_max_target": max(targets),
            "adaptive_mean_target": round(sum(targets) / len(targets), 2),
            "adaptive_throttle_events": self.total_throttle_events,
            "adaptive_grow_events": self.total_grow_events,
            "adaptive_shrink_events": self.total_shrink_events,
            "adaptive_peak_throughput": max(ths) if ths else 0.0,
            "adaptive_mean_throughput": round(sum(ths) / len(ths), 3) if ths else 0.0,
        }


# ---------------------------------------------------------------------------
# Worker strategy (public API)
# ---------------------------------------------------------------------------


class WorkerStrategy:
    """How many worker coroutines a task gets and whether the count adapts
    at runtime. Set via ``task_config={"workers": ...}``."""


@dataclass
class FixedWorkers(WorkerStrategy):
    """Run a fixed number of worker coroutines for this task.

    ``num_workers=1`` preserves input order. Larger values run items in
    parallel; output order is no longer guaranteed.
    """

    num_workers: int = 1


@dataclass
class AdaptiveWorkers(WorkerStrategy):
    """Run a worker pool whose live count adapts at runtime via throughput
    hill-climbing, with throttling-exception and low-utilization safety
    overrides.

    Starts at ``initial_workers``; the pool can grow up to ``max_workers``
    (defaults to ``data_per_batch`` when None) and shrink to ``min_workers``
    (defaults to ``max(1, initial_workers // 4)`` when None — a quarter of
    the starting concurrency, so a transient throttle storm cannot collapse
    the pool to a single worker that then takes many ticks to climb back).
    Adjustment cadence is one tick every ``tick_seconds``; each adjustment
    is ±``step`` workers.

    Output order is not preserved.
    """

    initial_workers: int = 40
    max_workers: Optional[int] = None  # None → data_per_batch
    min_workers: Optional[int] = None  # None → max(1, initial_workers // 4)
    step: int = 10
    tick_seconds: float = 20.0
    throughput_improvement_ratio: float = 1.05


# Framework default: single worker, preserves input order. Callers that want
# parallelism opt in explicitly via FixedWorkers(N) or AdaptiveWorkers().
_DEFAULT_STRATEGY = FixedWorkers(num_workers=1)


# ---------------------------------------------------------------------------
# Config resolution (internal)
# ---------------------------------------------------------------------------


# Adaptive-related defaults come from ``AdaptiveWorkers``; do not edit here.
# ``_StageConfig`` intentionally omits defaults for adaptive fields so the two
# classes can't drift — ``_resolve_stage_configs`` always populates them from
# either the user-supplied strategy or ``AdaptiveWorkers()`` defaults.
@dataclass
class _StageConfig:
    queue_maxsize: int
    num_workers: int
    next_batch_size: int
    adaptive: bool
    initial_workers: int
    min_workers: int
    step: int
    tick_seconds: float
    throughput_improvement_ratio: float
    per_call_timeout: Optional[float] = None


def _resolve_stage_configs(tasks: list[Task], data_per_batch: int) -> list[_StageConfig]:
    configs: list[_StageConfig] = []
    # Single source of truth for adaptive defaults — used both when the user
    # supplies an ``AdaptiveWorkers`` (via the strategy instance) and when the
    # stage is non-adaptive (we still need to populate the fields with valid
    # values, even though they won't be consulted).
    _adaptive_defaults = AdaptiveWorkers()
    for i, task in enumerate(tasks):
        cfg = task.task_config or {}
        strategy = cfg.get("workers", _DEFAULT_STRATEGY)
        if not isinstance(strategy, WorkerStrategy):
            raise TypeError(
                f"Task '{task.executable.__name__}': task_config['workers'] must be a "
                f"WorkerStrategy instance, got {type(strategy).__name__}"
            )

        if isinstance(strategy, AdaptiveWorkers):
            adaptive = True
            num_workers = strategy.max_workers or data_per_batch
            initial_workers = strategy.initial_workers
            # Resolve min_workers floor: when unspecified, default to a quarter
            # of initial_workers (floor 1) so a throttle storm can't collapse
            # the pool down to a single worker.
            if strategy.min_workers is None:
                min_workers = max(1, strategy.initial_workers // 4)
            else:
                min_workers = strategy.min_workers
            step = strategy.step
            tick_seconds = strategy.tick_seconds
            throughput_improvement_ratio = strategy.throughput_improvement_ratio
        elif isinstance(strategy, FixedWorkers):
            adaptive = False
            num_workers = strategy.num_workers
            initial_workers = strategy.num_workers
            # Adaptive fields are unused for FixedWorkers stages, but populate
            # them from ``AdaptiveWorkers()`` defaults so ``_StageConfig`` is
            # always valid and the defaults live in exactly one place.
            min_workers = 1
            step = _adaptive_defaults.step
            tick_seconds = _adaptive_defaults.tick_seconds
            throughput_improvement_ratio = _adaptive_defaults.throughput_improvement_ratio
        else:
            raise TypeError(
                f"Task '{task.executable.__name__}': unknown WorkerStrategy subclass "
                f"{type(strategy).__name__}"
            )

        # Default queue_maxsize tracks data_per_batch so the buffer can absorb
        # the whole input batch and never stalls the producer when num_workers
        # is the same size. Floor at 4 so tiny pipelines still have a small
        # buffer.
        queue_maxsize = int(cfg.get("queue_maxsize", max(data_per_batch, 4)))
        next_batch_size = (
            int(tasks[i + 1].task_config.get("batch_size", 1)) if i + 1 < len(tasks) else 1
        )
        per_call_timeout_raw = cfg.get("timeout")
        per_call_timeout = float(per_call_timeout_raw) if per_call_timeout_raw is not None else None
        configs.append(
            _StageConfig(
                queue_maxsize=queue_maxsize,
                num_workers=int(num_workers),
                next_batch_size=next_batch_size,
                adaptive=adaptive,
                initial_workers=int(initial_workers),
                min_workers=int(min_workers),
                step=int(step),
                tick_seconds=float(tick_seconds),
                throughput_improvement_ratio=float(throughput_improvement_ratio),
                per_call_timeout=per_call_timeout,
            )
        )
    return configs


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def _telemetry_props(task_name: str, user: Optional[User]) -> dict:
    return {
        "task_name": task_name,
        "cognee_version": cognee_version,
        "tenant_id": str(user.tenant_id) if user and user.tenant_id else "Single User Tenant",
    }


def _user_id(user: Optional[User]):
    return user.id if user is not None else None


async def _run_worker(
    task: Task,
    task_name: str,
    task_type: str,
    in_queue: InstrumentedQueue,
    out_queue: Optional[InstrumentedQueue],
    next_batch_size: int,
    shared_ctx: Optional[PipelineContext],
    user: User,
    user_label: Optional[str],
    pipe_name: Optional[str],
    provenance_visited: Optional[set],
    pool: Optional[_AdaptivePool] = None,
    per_call_timeout: Optional[float] = None,
):
    while True:
        envelope = await in_queue.get()
        if envelope is _SENTINEL:
            return

        # Pass-through error envelopes so per-item failure short-circuits to output.
        if isinstance(envelope.value, _ErroredItem):
            if out_queue is not None:
                await out_queue.put(envelope)
            continue

        # When adaptive, gate on the pool's permit semaphore so the live
        # concurrency stays at pool.target. Non-adaptive: nullcontext.
        async with pool.permit() if pool is not None else _nullcontext():
            # Build a per-item ctx by cloning the shared one; _provenance_visited
            # is a Set passed by reference so all clones share the same instance.
            kwargs = {}
            if shared_ctx is not None and task.accepts_ctx:
                item_ctx = dataclasses.replace(shared_ctx, data_item=envelope.origin)
                kwargs["ctx"] = item_ctx

            logger.info(f"{task_type} task started: `{task_name}`")
            send_telemetry(
                f"{task_type} Task Started",
                _user_id(user),
                additional_properties=_telemetry_props(task_name, user),
            )

            with new_span(f"cognee.pipeline.task.{task_name}") as span:
                span.set_attribute(COGNEE_PIPELINE_TASK_NAME, task_name)
                try:
                    result_count = 0
                    args_for_execute = (
                        [] if isinstance(envelope.value, _NoData) else [envelope.value]
                    )
                    input_node_set = _extract_node_set(args_for_execute)
                    input_content_hash = _extract_content_hash(args_for_execute)

                    async def _consume_execute():
                        nonlocal result_count
                        async for result_data in task.execute(
                            args_for_execute, kwargs, next_batch_size
                        ):
                            if isinstance(result_data, list):
                                result_count += len(result_data)
                            else:
                                result_count += 1
                            _stamp_provenance(
                                result_data,
                                pipe_name,
                                task_name,
                                visited=provenance_visited,
                                node_set=input_node_set,
                                user_label=user_label,
                                content_hash=input_content_hash,
                            )
                            if out_queue is not None:
                                await out_queue.put(
                                    _ItemEnvelope(
                                        value=result_data,
                                        origin=envelope.origin,
                                        seq=envelope.seq,
                                    )
                                )

                    if per_call_timeout is not None:
                        await asyncio.wait_for(_consume_execute(), timeout=per_call_timeout)
                    else:
                        await _consume_execute()

                    if pool is not None:
                        pool.report_success()

                    span.set_attribute(COGNEE_RESULT_COUNT, result_count)
                    span.set_attribute(
                        COGNEE_RESULT_SUMMARY,
                        _build_result_summary(task.executable, task_name, result_count),
                    )

                    logger.info(f"{task_type} task completed: `{task_name}`")
                    send_telemetry(
                        f"{task_type} Task Completed",
                        _user_id(user),
                        additional_properties=_telemetry_props(task_name, user),
                    )
                except asyncio.TimeoutError as timeout_exc:
                    # A single hung call (LLM, DB, etc.) exceeded
                    # per_call_timeout. Treat as a throttling signal so the
                    # adaptive pool shrinks, and forward an _ErroredItem so
                    # downstream + run_tasks can record the per-item failure.
                    span.set_status(StatusCode.ERROR, str(timeout_exc))
                    span.record_exception(timeout_exc)
                    logger.error(
                        f"{task_type} task timed out after {per_call_timeout}s: `{task_name}`",
                        exc_info=True,
                    )
                    send_telemetry(
                        f"{task_type} Task Errored",
                        _user_id(user),
                        additional_properties=_telemetry_props(task_name, user),
                    )
                    if pool is not None:
                        pool.report_throttling()
                    if out_queue is not None:
                        await out_queue.put(
                            _ItemEnvelope(
                                value=_ErroredItem(timeout_exc),
                                origin=envelope.origin,
                                seq=envelope.seq,
                            )
                        )
                    # do not re-raise — error is reported through the envelope channel
                except Exception as error:
                    span.set_status(StatusCode.ERROR, str(error))
                    span.record_exception(error)
                    logger.error(
                        f"{task_type} task errored: `{task_name}`\n{str(error)}\n",
                        exc_info=True,
                    )
                    send_telemetry(
                        f"{task_type} Task Errored",
                        _user_id(user),
                        additional_properties=_telemetry_props(task_name, user),
                    )
                    if pool is not None and _is_throttling_error(error):
                        pool.report_throttling()
                    if out_queue is not None:
                        await out_queue.put(
                            _ItemEnvelope(
                                value=_ErroredItem(error), origin=envelope.origin, seq=envelope.seq
                            )
                        )
                    # do not re-raise — error is reported through the envelope channel


@asynccontextmanager
async def _nullcontext():
    yield


async def _run_stage(
    task: Task,
    in_queue: InstrumentedQueue,
    out_queue: Optional[InstrumentedQueue],
    num_workers: int,
    next_workers_count: int,
    next_batch_size: int,
    shared_ctx: Optional[PipelineContext],
    user: User,
    user_label: Optional[str],
    pipe_name: Optional[str],
    provenance_visited: Optional[set],
    pool: Optional[_AdaptivePool] = None,
    per_call_timeout: Optional[float] = None,
):
    task_name = task.executable.__name__
    task_type = task.task_type
    if pool is not None:
        pool.start_ticker(in_queue)
    workers = [
        asyncio.create_task(
            _run_worker(
                task=task,
                task_name=task_name,
                task_type=task_type,
                in_queue=in_queue,
                out_queue=out_queue,
                next_batch_size=next_batch_size,
                shared_ctx=shared_ctx,
                user=user,
                user_label=user_label,
                pipe_name=pipe_name,
                provenance_visited=provenance_visited,
                pool=pool,
                per_call_timeout=per_call_timeout,
            )
        )
        for _ in range(num_workers)
    ]
    try:
        await asyncio.gather(*workers)
    finally:
        if pool is not None:
            await pool.stop_ticker()
        if out_queue is not None:
            for _ in range(next_workers_count):
                await out_queue.put(_SENTINEL)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def _as_async_iterable(data_iterable) -> AsyncIterator:
    if hasattr(data_iterable, "__aiter__"):
        async for item in data_iterable:
            yield item
    else:
        for item in data_iterable:
            yield item


async def run_worker_pipeline(
    tasks: list[Task],
    data_iterable: Union[Iterable[Any], AsyncIterator[Any]],
    user: User,
    ctx: Optional[PipelineContext] = None,
    data_per_batch: int = 1,
    pipeline_name: Optional[str] = None,
) -> AsyncIterator:
    """Run ``tasks`` as a worker-per-task pipeline. Yields each result that
    leaves the final stage in the order it lands in the output queue
    (single-worker stages preserve input order).

    Items that fail in any stage flow to the consumer as
    ``_ItemEnvelope(value=_ErroredItem(exc), ...)`` so the caller can yield
    per-item errors without aborting the rest of the run.

    Queue-depth telemetry is emitted as ``Task Queue Metrics`` events at end of run.
    """
    if not tasks:
        async for item in _as_async_iterable(data_iterable):
            yield _ItemEnvelope(value=item, origin=item, seq=0)
        return

    stage_configs = _resolve_stage_configs(tasks, data_per_batch=data_per_batch)

    # Queues: one per stage's input + one output queue at the tail. N+1 total.
    queues: list[InstrumentedQueue] = []
    for i, cfg in enumerate(stage_configs):
        task_name = tasks[i].executable.__name__
        queues.append(InstrumentedQueue(maxsize=cfg.queue_maxsize, name=f"q_in_{task_name}"))
    queues.append(InstrumentedQueue(maxsize=0, name="q_out"))

    user_label = getattr(user, "email", None) or (str(user.id) if user else None)
    pipe_name = (ctx.pipeline_name if ctx else None) or pipeline_name
    provenance_visited = ctx._provenance_visited if ctx else None

    # Build per-stage adaptive pools (or None for static stages).
    # If the cross-run registry has a target from a previous run for the same
    # task, use it as the starting point instead of cfg.initial_workers —
    # this skips the discovery climb on repeat runs in long-lived processes.
    pools: list[Optional[_AdaptivePool]] = []
    for i, cfg in enumerate(stage_configs):
        if cfg.adaptive:
            task_name = tasks[i].executable.__name__
            cached = _ADAPTIVE_TARGET_REGISTRY.get(task_name)
            if cached is not None:
                # Clamp cached target to current max in case data_per_batch
                # (and thus max_workers) shrunk between runs.
                resume_initial = min(cached, cfg.num_workers)
                logger.info(
                    f"[adaptive {task_name}] resuming from previous run's "
                    f"target={cached} (clamped to {resume_initial})"
                )
            else:
                resume_initial = cfg.initial_workers
            pools.append(
                _AdaptivePool(
                    task_name=task_name,
                    initial=resume_initial,
                    max_workers=cfg.num_workers,
                    min_workers=cfg.min_workers,
                    step=cfg.step,
                    tick_seconds=cfg.tick_seconds,
                    throughput_improvement_ratio=cfg.throughput_improvement_ratio,
                )
            )
        else:
            pools.append(None)

    # Spawn one stage coordinator per task.
    stage_tasks: list[asyncio.Task] = []
    for i, task in enumerate(tasks):
        cfg = stage_configs[i]
        in_q = queues[i]
        out_q = queues[i + 1]
        next_workers = stage_configs[i + 1].num_workers if i + 1 < len(stage_configs) else 1
        stage_tasks.append(
            asyncio.create_task(
                _run_stage(
                    task=task,
                    in_queue=in_q,
                    out_queue=out_q,
                    num_workers=cfg.num_workers,
                    next_workers_count=next_workers,
                    next_batch_size=cfg.next_batch_size,
                    shared_ctx=ctx,
                    user=user,
                    user_label=user_label,
                    pipe_name=pipe_name,
                    provenance_visited=provenance_visited,
                    pool=pools[i],
                    per_call_timeout=cfg.per_call_timeout,
                )
            )
        )

    # Producer: push every data item then one sentinel per first-stage worker.
    # Each yielded entry from ``data_iterable`` may be a (value, origin) tuple
    # — when the value pushed into the first task should differ from the
    # ``origin`` propagated to ctx.data_item (e.g. multi-item dataset runs
    # where the first task expects a list and origin is the raw item).
    async def _produce():
        head_q = queues[0]
        seq = 0
        async for item in _as_async_iterable(data_iterable):
            if isinstance(item, tuple) and len(item) == 2:
                value, origin = item
            else:
                value, origin = item, item
            await head_q.put(_ItemEnvelope(value=value, origin=origin, seq=seq))
            seq += 1
        for _ in range(stage_configs[0].num_workers):
            await head_q.put(_SENTINEL)

    producer_task = asyncio.create_task(_produce())

    # Consumer: drain the output queue until we've seen the EOF sentinel
    # pushed by the final stage's coordinator after its workers have all
    # exited. (Intermediate stages push N sentinels to the next stage's
    # queue so each of its workers can exit; the final stage pushes a single
    # marker to the output channel for this consumer.)
    output_q = queues[-1]
    sentinels_remaining = 1

    consumer_error: Optional[BaseException] = None
    try:
        while sentinels_remaining > 0:
            envelope = await output_q.get()
            if envelope is _SENTINEL:
                sentinels_remaining -= 1
                continue
            yield envelope
    except BaseException as e:
        consumer_error = e

    # Either way, await producer + stages so we surface their exceptions.
    try:
        await producer_task
    except BaseException as e:
        if consumer_error is None:
            consumer_error = e

    try:
        await asyncio.gather(*stage_tasks)
    except BaseException as e:
        if consumer_error is None:
            consumer_error = e

    # Emit one Task Queue Metrics event per task (using the upstream-of-queue
    # task as the label so a "full queue X" event indicates the downstream
    # task X cannot keep up).
    for i, task in enumerate(tasks):
        in_q = queues[i]
        stats = _compute_queue_stats(in_q.samples, in_q.maxsize)
        adaptive_stats = pools[i].stats() if pools[i] is not None else {}
        send_telemetry(
            "Task Queue Metrics",
            _user_id(user),
            additional_properties={
                "pipeline_name": str(pipe_name) if pipe_name else None,
                "task_name": task.executable.__name__,
                "queue_maxsize": in_q.maxsize,
                "num_workers": stage_configs[i].num_workers,
                "adaptive": stage_configs[i].adaptive,
                **stats,
                **adaptive_stats,
                "cognee_version": cognee_version,
                "tenant_id": str(user.tenant_id)
                if user and user.tenant_id
                else "Single User Tenant",
            },
        )

    if consumer_error is not None:
        raise consumer_error
