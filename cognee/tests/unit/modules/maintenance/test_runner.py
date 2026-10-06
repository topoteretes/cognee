"""``run_maintenance``: which jobs run, and the rules every job gets for free.

Fake jobs only -- no database. The runner must select by pipeline, honour the
operator switch and each gate, skip a job whose gate raises, keep going after
a job fails, never raise, and on cancellation let the running job finish and
start no other.
"""

import asyncio

import pytest

from cognee.modules.maintenance import (
    REASON_DISABLED_BY_CONFIG,
    REASON_GATE_ERRORED,
    BaseMaintenanceJob,
    JobResult,
    get_maintenance_config,
    run_maintenance,
)


class FakeJob(BaseMaintenanceJob):
    def __init__(self, name, pipelines=("cognify_pipeline",), gate=None, run=None):
        self.name = name
        self.pipelines = frozenset(pipelines)
        self._gate = gate
        self._run = run
        self.calls = []

    def gate(self, ctx):
        return self._gate(ctx) if self._gate else None

    async def run(self, ctx):
        self.calls.append(ctx)
        if self._run:
            return await self._run(ctx)
        return JobResult.completed(self.name, things=1)


@pytest.fixture(autouse=True)
def no_disabled_jobs(monkeypatch):
    monkeypatch.delenv("MAINTENANCE_JOBS_DISABLED", raising=False)
    get_maintenance_config.cache_clear()
    yield
    get_maintenance_config.cache_clear()


async def _run(jobs, pipeline_name="cognify_pipeline", **kwargs):
    return await run_maintenance(pipeline_name=pipeline_name, jobs=jobs, **kwargs)


@pytest.mark.asyncio
async def test_runs_the_jobs_for_the_pipeline_in_order_with_its_context():
    first, other, second = (
        FakeJob("first"),
        FakeJob("other", ("memify_pipeline",)),
        FakeJob("second"),
    )

    results = await _run(
        [first, other, second], pipeline_run_id="run-1", dataset="ds", user="u", extra="ignored"
    )

    assert [(r.job, r.status) for r in results] == [("first", "completed"), ("second", "completed")]
    assert other.calls == []
    (ctx,) = first.calls
    assert (ctx.pipeline_name, ctx.pipeline_run_id, ctx.dataset, ctx.user) == (
        "cognify_pipeline",
        "run-1",
        "ds",
        "u",
    )
    assert all(r.duration_ms >= 0 for r in results)


@pytest.mark.asyncio
async def test_no_triggered_jobs_is_a_no_op():
    assert await _run([FakeJob("x", ("memify_pipeline",))]) == []


@pytest.mark.asyncio
async def test_a_job_disabled_by_config_is_skipped(monkeypatch):
    # Real job names are validated against the registry, so disable the real one.
    monkeypatch.setenv("MAINTENANCE_JOBS_DISABLED", "vector_compaction")
    get_maintenance_config.cache_clear()
    job = FakeJob("vector_compaction")

    (result,) = await _run([job])

    assert (result.status, result.reason) == ("skipped", REASON_DISABLED_BY_CONFIG)
    assert job.calls == []


@pytest.mark.asyncio
async def test_a_gate_reason_skips_the_job():
    job = FakeJob("gated", gate=lambda ctx: "backend_unsupported")

    (result,) = await _run([job])

    assert (result.status, result.reason) == ("skipped", "backend_unsupported")
    assert job.calls == []


@pytest.mark.asyncio
async def test_a_gate_that_raises_skips_the_job():
    """When a job cannot tell whether it should run, it does not."""

    def broken_gate(ctx):
        raise RuntimeError("config unreadable")

    job, after = FakeJob("broken", gate=broken_gate), FakeJob("after")

    results = await _run([job, after])

    assert (results[0].status, results[0].reason) == ("skipped", REASON_GATE_ERRORED)
    assert job.calls == []
    assert results[1].status == "completed"


@pytest.mark.asyncio
async def test_unreadable_configuration_runs_nothing(monkeypatch):
    def broken_config():
        raise ValueError("bad MAINTENANCE_JOBS_DISABLED")

    monkeypatch.setattr("cognee.modules.maintenance.runner.get_maintenance_config", broken_config)
    job = FakeJob("any")

    (result,) = await _run([job])

    assert (result.status, result.reason) == ("skipped", REASON_GATE_ERRORED)
    assert job.calls == []


@pytest.mark.asyncio
async def test_a_failing_job_is_errored_and_the_next_still_runs():
    async def explode(ctx):
        raise RuntimeError("disk on fire")

    failing, after = FakeJob("failing", run=explode), FakeJob("after")

    results = await _run([failing, after])

    assert results[0].status == "errored"
    assert "disk on fire" in results[0].error
    assert results[1].status == "completed"


@pytest.mark.asyncio
async def test_cancellation_lets_the_running_job_finish_and_starts_no_other():
    """A job's database work does not stop with its coroutine; the dataset
    context must not be torn down under it. A repeated cancel included."""
    started, finished = asyncio.Event(), []

    async def slow(ctx):
        started.set()
        await asyncio.sleep(0.3)
        finished.append("slow")
        return JobResult.completed("slow")

    slow_job, never = FakeJob("slow", run=slow), FakeJob("never")
    task = asyncio.ensure_future(_run([slow_job, never]))
    await started.wait()
    task.cancel()
    await asyncio.sleep(0.05)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert finished == ["slow"]
    assert never.calls == []


@pytest.mark.asyncio
async def test_an_errored_jobs_text_is_scrubbed_like_a_pipeline_error():
    """The error is logged and kept on the result: no credentials in it."""

    async def leak(ctx):
        raise RuntimeError(
            "connect failed with Bearer abcdef0123456789 key sk-live_0123456789abcdef "
            "for /Users/alice/.cognee"
        )

    (result,) = await _run([FakeJob("leaky", run=leak)])

    assert result.status == "errored"
    assert "RuntimeError" in result.error
    for secret in ("abcdef0123456789", "sk-live_0123456789abcdef", "alice"):
        assert secret not in result.error, result.error


def _shared_stores(monkeypatch, shared: bool):
    monkeypatch.setattr("cognee.modules.maintenance.runner._stores_are_shared", lambda: shared)


class StoreJob(FakeJob):
    scope = "store"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "shared, last, store_job_runs",
    [
        (True, False, False),  # multi-user off, not the last dataset: wait for the last
        (True, True, True),  # multi-user off, last dataset: the one pass
        (False, False, True),  # multi-user on: every dataset has its own store
        (False, True, True),
    ],
)
async def test_store_scoped_jobs_run_once_on_a_shared_store(
    monkeypatch, shared, last, store_job_runs
):
    _shared_stores(monkeypatch, shared)
    store_job, dataset_job = StoreJob("store_job"), FakeJob("dataset_job")

    results = await _run([store_job, dataset_job], last_in_invocation=last)

    by_job = {result.job: result for result in results}
    if store_job_runs:
        assert by_job["store_job"].status == "completed"
    else:
        assert (by_job["store_job"].status, by_job["store_job"].reason) == (
            "skipped",
            "shared_store_runs_after_last_dataset",
        )
        assert store_job.calls == []
    # A dataset-scoped job maintains each dataset's own data: always runs.
    assert by_job["dataset_job"].status == "completed"


@pytest.mark.asyncio
async def test_the_shared_store_rule_follows_the_access_control_setting(monkeypatch):
    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "false")
    store_job = StoreJob("store_job")

    (result,) = await _run([store_job], last_in_invocation=False)

    assert result.reason == "shared_store_runs_after_last_dataset"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome, level",
    [
        (JobResult.completed("job", things=1), "info"),
        (JobResult.already_completed("job"), "debug"),
        (JobResult.completed("job", collection_errors=1), "warning"),
        (JobResult(job="job", status="errored", error="all failed"), "warning"),
    ],
)
async def test_the_summary_is_logged_at_the_level_the_outcome_deserves(monkeypatch, outcome, level):
    """Failures -- even partial ones -- are warnings, never buried at debug."""
    import cognee.modules.maintenance.runner as runner_module

    logged = []
    for name in ("debug", "info", "warning"):
        monkeypatch.setattr(
            runner_module.logger, name, lambda *args, _name=name, **kwargs: logged.append(_name)
        )

    async def returns_outcome(ctx):
        return outcome

    await _run([FakeJob("job", run=returns_outcome)])

    assert logged == [level]
