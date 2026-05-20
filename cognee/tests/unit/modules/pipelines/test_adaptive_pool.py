"""Verify ``_AdaptivePool`` freezes when utilization stays low for several ticks.

The pool detects "we are not the bottleneck — upstream is" by counting
consecutive ticks where utilization is below the threshold. After
``_freeze_after_low_util_ticks`` such ticks the pool stops hill-climbing
and holds its current target steady (logged as "frozen").
"""

import asyncio

import pytest

from cognee.modules.pipelines.operations.worker_pipeline import (
    _AdaptivePool,
    InstrumentedQueue,
    _is_throttling_error,
    get_last_adaptive_target,
    reset_adaptive_target_registry,
)


@pytest.mark.asyncio
async def test_adaptive_pool_freezes_when_upstream_bound():
    tick = 0.05
    pool = _AdaptivePool(
        task_name="test-upstream-bound",
        initial=10,
        max_workers=50,
        min_workers=1,
        step=2,
        tick_seconds=tick,
    )
    queue = InstrumentedQueue(name="test-q")

    # Continuously report successes so ``completions > 0`` every tick —
    # without this the ticker takes the "no completions" no-signal branch
    # and never advances the freeze counter. We never call ``pool.permit()``,
    # so ``_busy_workers`` stays at 0 → utilization = 0% (underutil).
    # Queue stays empty → no shrink via criterion 3, so the freeze path
    # is the one that ultimately fires.
    stop_reporter = asyncio.Event()

    async def _report_loop():
        while not stop_reporter.is_set():
            pool.report_success()
            await asyncio.sleep(tick / 10)

    reporter = asyncio.create_task(_report_loop())
    pool.start_ticker(queue)

    try:
        # Wait long enough for the freeze counter to exceed the threshold.
        # Generous margin keeps the test stable on a loaded scheduler.
        await asyncio.sleep(tick * 20)
        assert pool._consecutive_low_util_ticks >= 3, (
            f"expected freeze counter >=3, got {pool._consecutive_low_util_ticks}"
        )

        # Once frozen, the target must hold steady across additional ticks.
        frozen_target = pool.target
        await asyncio.sleep(tick * 10)
        target_after_more_ticks = pool.target

        assert target_after_more_ticks == frozen_target, (
            f"target changed while frozen: {frozen_target} → {target_after_more_ticks}"
        )
    finally:
        stop_reporter.set()
        reporter.cancel()
        try:
            await reporter
        except asyncio.CancelledError:
            pass
        await pool.stop_ticker()


@pytest.mark.asyncio
async def test_adaptive_pool_hill_climb_reverses_on_no_improvement():
    """Hill-climb must reverse direction when an action stops improving throughput."""
    tick = 0.05
    pool = _AdaptivePool(
        task_name="test-hill-climb",
        initial=20,
        max_workers=100,
        min_workers=1,
        step=2,
        tick_seconds=tick,
        throughput_improvement_ratio=1.05,
    )
    queue = InstrumentedQueue(name="test-q")
    # Keep qsize() > 0 so the underutil branch can fire deterministically only
    # if we let it — but we force util=1.0 by pinning _busy_workers below.
    for _ in range(5):
        queue.put_nowait(object())

    async def _wait_for_next_tick(prior_len: int, timeout: float = 2.0) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            if len(pool._target_history) > prior_len:
                return
            await asyncio.sleep(tick / 5)
        raise AssertionError("tick did not fire within timeout")

    def _seed(completions: int) -> None:
        # Inject this tick's signal: completions count + full utilization
        # (so neither underutil-shrink nor freeze fires).
        pool._completed_in_window = completions
        pool._busy_workers = pool.target

    # Seed tick 1 before starting the ticker so the very first window has data.
    _seed(10)
    baseline = len(pool._target_history)  # constructor seeds one entry
    pool.start_ticker(queue)

    try:
        # Tick 1: exploratory grow (last_action was None).
        await _wait_for_next_tick(baseline)
        assert pool.target == 22, f"tick1 target: {pool.target}"
        assert pool._last_action == "grow"

        # Tick 2: throughput improved (10 → 30), keep growing.
        _seed(30)
        await _wait_for_next_tick(baseline + 1)
        assert pool.target == 24, f"tick2 target: {pool.target}"
        assert pool._last_action == "grow"

        # Tick 3: throughput flat (30 → 30, not >= 1.05×) → reverse to shrink.
        _seed(30)
        await _wait_for_next_tick(baseline + 2)
        assert pool.target == 22, f"tick3 target: {pool.target}"
        assert pool._last_action == "shrink"

        # Tick 4: shrink improved throughput (30 → 50) → keep shrinking.
        _seed(50)
        await _wait_for_next_tick(baseline + 3)
        assert pool.target == 20, f"tick4 target: {pool.target}"
        assert pool._last_action == "shrink"
    finally:
        await pool.stop_ticker()


@pytest.mark.asyncio
async def test_adaptive_pool_resumes_from_cached_target_across_runs():
    """Second pool for the same task name must read the first pool's converged
    target from the cross-run registry, so discovery cost is paid only once."""
    task_name = "test_resume_task"
    reset_adaptive_target_registry()
    try:
        # First pool — simulate that it ran and converged on target=35.
        pool_a = _AdaptivePool(
            task_name=task_name,
            initial=10,
            max_workers=100,
            min_workers=1,
            step=2,
            tick_seconds=0.05,
        )
        pool_a.target = 35
        # ``stop_ticker`` persists the final target unconditionally — even when
        # the ticker was never started — which exercises the same registry
        # write path that production uses on pipeline shutdown.
        await pool_a.stop_ticker()

        # The public getter must surface the persisted target.
        assert get_last_adaptive_target(task_name) == 35

        # A second pool constructed for the same task name does not need to
        # read the registry itself (that lookup lives in ``run_worker_pipeline``);
        # the contract under test is that the registry value persists across
        # pool instances and remains observable via the public getter.
        pool_b = _AdaptivePool(
            task_name=task_name,
            initial=10,
            max_workers=100,
            min_workers=1,
            step=2,
            tick_seconds=0.05,
        )
        assert pool_b.target == 10  # constructor still honors ``initial``
        assert get_last_adaptive_target(task_name) == 35
    finally:
        reset_adaptive_target_registry()


# ---------------------------------------------------------------------------
# _is_throttling_error classifier
# ---------------------------------------------------------------------------


def test_is_throttling_error_positive_by_type_name():
    """Exceptions whose class name appears in ``_THROTTLING_TYPE_NAMES``
    classify as throttling regardless of message content."""
    assert _is_throttling_error(ConnectionError("foo")) is True
    assert _is_throttling_error(TimeoutError("bar")) is True

    # Ad-hoc subclass whose name matches the type-name allowlist.
    class RateLimitError(Exception):
        pass

    assert _is_throttling_error(RateLimitError("slow down")) is True

    # Another ad-hoc match by class name only — message is empty.
    class ReadTimeout(Exception):
        pass

    assert _is_throttling_error(ReadTimeout()) is True


def test_is_throttling_error_positive_by_message_signature():
    """Exceptions of unrelated classes still classify as throttling when
    their message contains a substring from ``_THROTTLING_SIGNATURES``."""
    # "429" / "too many requests" substring inside a ValueError.
    assert _is_throttling_error(ValueError("HTTP 429 Too Many Requests")) is True
    # "rate limit" substring.
    assert _is_throttling_error(RuntimeError("Provider rate limit exceeded")) is True
    # "service unavailable" substring.
    assert _is_throttling_error(RuntimeError("503 Service Unavailable from upstream")) is True
    # Case-insensitive: the classifier lowercases the message before matching.
    assert _is_throttling_error(RuntimeError("READ TIMED OUT after 30s")) is True


def test_is_throttling_error_negative_unrelated_errors():
    """Normal data / programming errors must not be classified as throttling."""
    assert _is_throttling_error(ValueError("bad data")) is False
    assert _is_throttling_error(KeyError("missing key")) is False

    class MyDomainError(Exception):
        pass

    assert _is_throttling_error(MyDomainError("validation failed")) is False
    # Empty-message exception with non-throttling class name.
    assert _is_throttling_error(Exception()) is False


def test_is_throttling_error_negative_partial_substring():
    """The classifier matches the exact substring ``"connection error"`` —
    a message that contains only the bare word ``"connection"`` (without the
    trailing ``" error"``) must NOT be classified."""
    # "Database connection refused" contains "connection" but not the exact
    # signature "connection error" — the classifier should return False.
    assert _is_throttling_error(ValueError("Database connection refused by peer")) is False
    # "Lost connection while reading" — same: word "connection" appears but
    # the substring "connection error" does not.
    assert _is_throttling_error(RuntimeError("Lost connection while reading row 42")) is False
