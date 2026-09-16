import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.modules.cognify import recovery as recovery_module
from cognee.modules.pipelines.models import PipelineRunStatus


class _FakeSession:
    def __init__(self, dataset=None):
        self._dataset = dataset

    async def get(self, _model, _dataset_id):
        return self._dataset

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False


class _FakeEngine:
    def __init__(self, sessions):
        self._sessions = list(sessions)

    def get_async_session(self):
        return self._sessions.pop(0)


@asynccontextmanager
async def _no_op_context(*_args, **_kwargs):
    yield


@pytest.mark.asyncio
async def test_recover_stale_cognify_runs_executes_rollback_for_latest_candidate(monkeypatch):
    dataset_id = uuid4()
    owner_id = uuid4()
    pipeline_run_id = uuid4()

    pipeline_id = uuid4()
    stale_run = SimpleNamespace(
        pipeline_name="cognify_pipeline",
        dataset_id=dataset_id,
        pipeline_run_id=pipeline_run_id,
        pipeline_id=pipeline_id,
        status=PipelineRunStatus.DATASET_PROCESSING_STARTED,
        origin="api",
        user_id=None,
        created_at=datetime.now(timezone.utc) - timedelta(hours=2),
        started_at=datetime.now(timezone.utc) - timedelta(hours=2),
        run_info={"data": "summarized-payload"},
    )
    dataset = SimpleNamespace(id=dataset_id, owner_id=owner_id)

    dataset_session = _FakeSession(dataset=dataset)
    engine = _FakeEngine([dataset_session])

    rollback_calls = []
    close_calls = []

    async def _rollback_handler(**kwargs):
        rollback_calls.append(kwargs)

    async def _log_error(**kwargs):
        close_calls.append(kwargs)

    async def _fake_latest_runs(_dataset_ids, _pipeline_name):
        return {dataset_id: stale_run}

    monkeypatch.setattr(recovery_module, "get_relational_engine", lambda: engine)
    monkeypatch.setattr(recovery_module, "get_latest_pipeline_runs_by_datasets", _fake_latest_runs)
    monkeypatch.setattr(recovery_module, "set_database_global_context_variables", _no_op_context)

    monkeypatch.setattr(recovery_module, "cognify_rollback_handler", _rollback_handler)
    monkeypatch.setattr(recovery_module, "log_pipeline_run_error", _log_error)

    await recovery_module.recover_stale_cognify_runs_on_startup()

    assert len(rollback_calls) == 1
    assert rollback_calls[0]["pipeline_run_id"] == pipeline_run_id
    assert rollback_calls[0]["dataset"] == dataset
    # The run is closed as ERRORED, keeping its identity, and the error class
    # is what distinguishes a killed run from one whose work actually failed.
    assert len(close_calls) == 1
    assert close_calls[0]["dataset_id"] == dataset_id
    assert close_calls[0]["pipeline_run_id"] == pipeline_run_id
    assert close_calls[0]["pipeline_id"] == pipeline_id
    assert type(close_calls[0]["e"]).__name__ == "AbandonedPipelineRunError"
    # The STARTED row's already-summarized payload is passed through rather
    # than re-summarized.
    assert close_calls[0]["data_info"] == "summarized-payload"


@pytest.mark.asyncio
async def test_recover_stale_cognify_runs_skips_missing_dataset(monkeypatch):
    dataset_id = uuid4()
    stale_run = SimpleNamespace(
        pipeline_name="cognify_pipeline",
        dataset_id=dataset_id,
        pipeline_run_id=uuid4(),
        status=PipelineRunStatus.DATASET_PROCESSING_STARTED,
        origin="api",
        user_id=None,
        created_at=datetime.now(timezone.utc) - timedelta(hours=2),
    )

    dataset_session = _FakeSession(dataset=None)
    engine = _FakeEngine([dataset_session])

    rollback_calls = []

    async def _rollback_handler(**kwargs):
        rollback_calls.append(kwargs)

    async def _fake_latest_runs(_dataset_ids, _pipeline_name):
        return {dataset_id: stale_run}

    monkeypatch.setattr(recovery_module, "get_relational_engine", lambda: engine)
    monkeypatch.setattr(recovery_module, "get_latest_pipeline_runs_by_datasets", _fake_latest_runs)
    monkeypatch.setattr(recovery_module, "set_database_global_context_variables", _no_op_context)

    monkeypatch.setattr(recovery_module, "cognify_rollback_handler", _rollback_handler)

    await recovery_module.recover_stale_cognify_runs_on_startup()

    assert rollback_calls == []


@pytest.mark.asyncio
async def test_a_recent_run_is_left_alone(monkeypatch):
    """Origin alone cannot tell a dead process's row from a live sibling's: a
    rolling deploy boots a new instance of the same surface while the old one
    is still finishing, and both stamp the same origin. A run that only just
    started is far more likely to be that live sibling than a genuinely dead
    process, so the age floor holds it back until enough time has passed that
    "still running" stops being the likely explanation."""
    dataset_id = uuid4()
    owner_id = uuid4()
    recent_run = SimpleNamespace(
        pipeline_name="cognify_pipeline",
        dataset_id=dataset_id,
        pipeline_run_id=uuid4(),
        pipeline_id=uuid4(),
        status=PipelineRunStatus.DATASET_PROCESSING_STARTED,
        origin="api",
        user_id=None,
        created_at=datetime.now(timezone.utc),
        started_at=datetime.now(timezone.utc),
        run_info={},
    )
    dataset = SimpleNamespace(id=dataset_id, owner_id=owner_id)

    engine = _FakeEngine([_FakeSession(dataset=dataset)])

    rollback_calls = []
    close_calls = []

    async def _rollback_handler(**kwargs):
        rollback_calls.append(kwargs)

    async def _log_error(**kwargs):
        close_calls.append(kwargs)

    async def _fake_latest_runs(_dataset_ids, _pipeline_name):
        return {dataset_id: recent_run}

    monkeypatch.setattr(recovery_module, "get_relational_engine", lambda: engine)
    monkeypatch.setattr(recovery_module, "get_latest_pipeline_runs_by_datasets", _fake_latest_runs)
    monkeypatch.setattr(recovery_module, "set_database_global_context_variables", _no_op_context)

    monkeypatch.setattr(recovery_module, "cognify_rollback_handler", _rollback_handler)
    monkeypatch.setattr(recovery_module, "log_pipeline_run_error", _log_error)

    await recovery_module.recover_stale_cognify_runs_on_startup()

    assert rollback_calls == []
    assert close_calls == []


@pytest.mark.asyncio
async def test_a_run_past_the_age_floor_is_recovered(monkeypatch):
    """The other half of the same coin: once a STARTED row is old enough that
    a boot overlap no longer explains it, origin match is sufficient and the
    run is closed exactly as it always was."""
    dataset_id = uuid4()
    owner_id = uuid4()
    old_run = SimpleNamespace(
        pipeline_name="cognify_pipeline",
        dataset_id=dataset_id,
        pipeline_run_id=uuid4(),
        pipeline_id=uuid4(),
        status=PipelineRunStatus.DATASET_PROCESSING_STARTED,
        origin="api",
        user_id=None,
        created_at=datetime.now(timezone.utc) - timedelta(hours=2),
        started_at=datetime.now(timezone.utc) - timedelta(hours=2),
        run_info={},
    )
    dataset = SimpleNamespace(id=dataset_id, owner_id=owner_id)

    engine = _FakeEngine([_FakeSession(dataset=dataset)])

    rollback_calls = []
    close_calls = []

    async def _rollback_handler(**kwargs):
        rollback_calls.append(kwargs)

    async def _log_error(**kwargs):
        close_calls.append(kwargs)

    async def _fake_latest_runs(_dataset_ids, _pipeline_name):
        return {dataset_id: old_run}

    monkeypatch.setattr(recovery_module, "get_relational_engine", lambda: engine)
    monkeypatch.setattr(recovery_module, "get_latest_pipeline_runs_by_datasets", _fake_latest_runs)
    monkeypatch.setattr(recovery_module, "set_database_global_context_variables", _no_op_context)

    monkeypatch.setattr(recovery_module, "cognify_rollback_handler", _rollback_handler)
    monkeypatch.setattr(recovery_module, "log_pipeline_run_error", _log_error)

    await recovery_module.recover_stale_cognify_runs_on_startup()

    assert len(rollback_calls) == 1
    assert len(close_calls) == 1


@pytest.mark.asyncio
async def test_a_failing_rollback_leaves_the_run_open_for_the_next_boot(monkeypatch):
    """The STARTED row is the retry token. Closing the run before the rollback
    finished would mark it terminal over a half-deleted graph that nothing
    would ever revisit, so a raising rollback must leave it open instead."""
    dataset_id = uuid4()
    owner_id = uuid4()
    stale_run = SimpleNamespace(
        pipeline_name="cognify_pipeline",
        dataset_id=dataset_id,
        pipeline_run_id=uuid4(),
        pipeline_id=uuid4(),
        status=PipelineRunStatus.DATASET_PROCESSING_STARTED,
        origin="api",
        user_id=None,
        created_at=datetime.now(timezone.utc) - timedelta(hours=2),
        started_at=datetime.now(timezone.utc) - timedelta(hours=2),
        run_info={},
    )
    dataset = SimpleNamespace(id=dataset_id, owner_id=owner_id)

    engine = _FakeEngine([_FakeSession(dataset=dataset)])

    close_calls = []

    async def _rollback_handler(**kwargs):
        raise RuntimeError("graph store unreachable")

    async def _log_error(**kwargs):
        close_calls.append(kwargs)

    async def _fake_latest_runs(_dataset_ids, _pipeline_name):
        return {dataset_id: stale_run}

    monkeypatch.setattr(recovery_module, "get_relational_engine", lambda: engine)
    monkeypatch.setattr(recovery_module, "get_latest_pipeline_runs_by_datasets", _fake_latest_runs)
    monkeypatch.setattr(recovery_module, "set_database_global_context_variables", _no_op_context)

    monkeypatch.setattr(recovery_module, "cognify_rollback_handler", _rollback_handler)
    monkeypatch.setattr(recovery_module, "log_pipeline_run_error", _log_error)

    # Startup must survive it: one dataset failing to recover cannot stop the
    # server from coming up.
    await recovery_module.recover_stale_cognify_runs_on_startup()

    assert close_calls == []


def _started_run(dataset_id, *, origin, user_id=None, tenant_id=None, status=None):
    return SimpleNamespace(
        pipeline_name="cognify_pipeline",
        dataset_id=dataset_id,
        pipeline_run_id=uuid4(),
        pipeline_id=uuid4(),
        status=status or PipelineRunStatus.DATASET_PROCESSING_STARTED,
        origin=origin,
        user_id=user_id,
        tenant_id=tenant_id,
        created_at=datetime.now(timezone.utc) - timedelta(hours=2),
        started_at=datetime.now(timezone.utc) - timedelta(hours=2),
        run_info={},
    )


def _drive(monkeypatch, run, dataset):
    """Run recovery over one candidate, returning (rollbacks, closes)."""
    rollback_calls = []
    close_calls = []

    async def _rollback_handler(**kwargs):
        rollback_calls.append(kwargs)

    async def _log_error(**kwargs):
        close_calls.append(kwargs)

    async def _fake_latest_runs(_dataset_ids, _pipeline_name):
        return {run.dataset_id: run}

    monkeypatch.setattr(
        recovery_module,
        "get_relational_engine",
        lambda: _FakeEngine([_FakeSession(dataset=dataset)]),
    )
    monkeypatch.setattr(recovery_module, "get_latest_pipeline_runs_by_datasets", _fake_latest_runs)
    monkeypatch.setattr(recovery_module, "set_database_global_context_variables", _no_op_context)
    monkeypatch.setattr(recovery_module, "cognify_rollback_handler", _rollback_handler)
    monkeypatch.setattr(recovery_module, "log_pipeline_run_error", _log_error)

    return rollback_calls, close_calls


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", ["sdk", "cli", "mcp", "background", None])
async def test_a_run_this_server_did_not_start_is_left_alone(monkeypatch, origin):
    """A shared relational database is the normal case, not an exotic one:
    docker-compose runs the API and the MCP server against one, and the CLI
    without --api-url executes in its own process against the same SQLite
    file. A booting server cannot observe whether such a run is still going,
    and guessing wrong deletes that run's graph, so it does not touch it.
    "background" is in this list too: it is stamped by whatever surface spawned
    the work, so an SDK script's background bridge carries it as well. Server
    background work does not need it, because its STARTED row is written in the
    request context and already reads "api".

    NULL covers rows written before the origin stamp existed. Those cannot be
    attributed either way, so they are skipped unless an operator opts in."""
    dataset_id = uuid4()
    dataset = SimpleNamespace(id=dataset_id, owner_id=uuid4())
    run = _started_run(dataset_id, origin=origin)

    rollback_calls, close_calls = _drive(monkeypatch, run, dataset)

    await recovery_module.recover_stale_cognify_runs_on_startup()

    assert rollback_calls == []
    assert close_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", ["api"])
async def test_a_run_this_kind_of_process_started_is_closed(monkeypatch, origin):
    """The mirror of the test above: a run a server started is one a server
    may close, because its process is the one that just restarted. This is the
    only origin that says that, which is why the set has exactly one member."""
    dataset_id = uuid4()
    dataset = SimpleNamespace(id=dataset_id, owner_id=uuid4())
    run = _started_run(dataset_id, origin=origin)

    rollback_calls, close_calls = _drive(monkeypatch, run, dataset)

    await recovery_module.recover_stale_cognify_runs_on_startup()

    assert len(rollback_calls) == 1
    assert len(close_calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal_status",
    [
        PipelineRunStatus.DATASET_PROCESSING_COMPLETED,
        PipelineRunStatus.DATASET_PROCESSING_ERRORED,
    ],
)
async def test_an_already_closed_run_is_not_rolled_back_again(monkeypatch, terminal_status):
    """Without this filter a closed run stays a candidate, so its rollback
    repeats on every single boot, deleting nothing new but doing the whole
    destructive pass again each time."""
    dataset_id = uuid4()
    dataset = SimpleNamespace(id=dataset_id, owner_id=uuid4())
    run = _started_run(dataset_id, origin="api", status=terminal_status)

    rollback_calls, close_calls = _drive(monkeypatch, run, dataset)

    await recovery_module.recover_stale_cognify_runs_on_startup()

    assert rollback_calls == []
    assert close_calls == []


@pytest.mark.asyncio
async def test_the_closing_row_carries_the_runs_own_attribution(monkeypatch):
    """The activity feed filters on user_id, so a closing row with no user is
    invisible to the very person whose run it closes. The values are copied
    off the STARTED row rather than looked up: a lookup is one more thing that
    can fail after the rollback has already deleted the graph."""
    dataset_id = uuid4()
    starter_id = uuid4()
    tenant_id = uuid4()
    # Deliberately not the dataset owner: a collaborator's run is theirs.
    dataset = SimpleNamespace(id=dataset_id, owner_id=uuid4())
    run = _started_run(dataset_id, origin="api", user_id=starter_id, tenant_id=tenant_id)

    _, close_calls = _drive(monkeypatch, run, dataset)

    await recovery_module.recover_stale_cognify_runs_on_startup()

    assert len(close_calls) == 1
    assert close_calls[0]["user"].id == starter_id
    assert close_calls[0]["user"].tenant_id == tenant_id


@pytest.mark.asyncio
async def test_a_run_with_no_recorded_starter_closes_without_one(monkeypatch):
    """Legacy rows carry no user. Inventing one, by falling back to the dataset
    owner, would attribute the close to someone who did not start the run and
    leave the feed with a start and a close under two different users."""
    dataset_id = uuid4()
    dataset = SimpleNamespace(id=dataset_id, owner_id=uuid4())
    run = _started_run(dataset_id, origin="api", user_id=None)

    _, close_calls = _drive(monkeypatch, run, dataset)

    await recovery_module.recover_stale_cognify_runs_on_startup()

    assert len(close_calls) == 1
    assert close_calls[0]["user"] is None


@pytest.mark.asyncio
async def test_unattributed_runs_are_skipped_unless_an_operator_opts_in(monkeypatch):
    """Rows written before the origin stamp existed cannot be attributed, so
    closing them would mean guessing again. Skipping them leaves a deployment
    upgrading with its stuck runs still stuck, which is a real cost, so an
    operator who knows only one process reaches the database can say so."""
    dataset_id = uuid4()
    dataset = SimpleNamespace(id=dataset_id, owner_id=uuid4())
    run = _started_run(dataset_id, origin=None)

    rollback_calls, close_calls = _drive(monkeypatch, run, dataset)
    monkeypatch.setattr(recovery_module, "_RECOVER_UNATTRIBUTED", False)

    await recovery_module.recover_stale_cognify_runs_on_startup()

    assert rollback_calls == []
    assert close_calls == []

    monkeypatch.setattr(recovery_module, "_RECOVER_UNATTRIBUTED", True)

    await recovery_module.recover_stale_cognify_runs_on_startup()

    assert len(rollback_calls) == 1
    assert len(close_calls) == 1


@pytest.mark.asyncio
async def test_each_surface_closes_only_its_own_runs(monkeypatch):
    """The API and the MCP server share a relational database in this repo's
    own compose file, and neither can observe whether the other's run is still
    going. So each sweeps the origin it stamps and leaves the rest alone: MCP
    closing an api-stamped run would be the same mistake as a server closing
    an SDK script's."""
    dataset_id = uuid4()
    dataset = SimpleNamespace(id=dataset_id, owner_id=uuid4())
    run = _started_run(dataset_id, origin="mcp")

    rollback_calls, close_calls = _drive(monkeypatch, run, dataset)

    # The API's default set does not own it.
    await recovery_module.recover_stale_cognify_runs_on_startup()

    assert rollback_calls == []
    assert close_calls == []

    # The MCP server's does.
    await recovery_module.recover_stale_cognify_runs_on_startup(owned_origins=frozenset({"mcp"}))

    assert len(rollback_calls) == 1
    assert len(close_calls) == 1


@pytest.mark.asyncio
async def test_a_surface_does_not_close_another_surfaces_runs(monkeypatch):
    """The mirror: MCP must not reach for the API's rows either."""
    dataset_id = uuid4()
    dataset = SimpleNamespace(id=dataset_id, owner_id=uuid4())
    run = _started_run(dataset_id, origin="api")

    rollback_calls, close_calls = _drive(monkeypatch, run, dataset)

    await recovery_module.recover_stale_cognify_runs_on_startup(owned_origins=frozenset({"mcp"}))

    assert rollback_calls == []
    assert close_calls == []


@pytest.mark.asyncio
async def test_a_run_just_inside_the_age_floor_is_left_alone(monkeypatch):
    """One second younger than the threshold must not be recovered — the
    filter is a strict "at least this old", not "roughly this old"."""
    dataset_id = uuid4()
    dataset = SimpleNamespace(id=dataset_id, owner_id=uuid4())
    run = _started_run(dataset_id, origin="api")
    run.created_at = datetime.now(timezone.utc) - timedelta(
        seconds=recovery_module._STALE_RUN_MIN_AGE_SECONDS - 1
    )
    run.started_at = run.created_at

    rollback_calls, close_calls = _drive(monkeypatch, run, dataset)

    await recovery_module.recover_stale_cognify_runs_on_startup()

    assert rollback_calls == []
    assert close_calls == []


def test_is_older_than_threshold_at_exactly_the_floor_is_true():
    """The one case the "just younger" test above cannot pin: a run exactly
    as old as the floor. _is_older_than_threshold uses >=, so this must be
    True — a test that used `threshold - 1` for "left alone" would still
    pass if the implementation used `>` instead of `>=`, since neither value
    tells the two apart. This one does."""
    run = SimpleNamespace(
        started_at=datetime.now(timezone.utc)
        - timedelta(seconds=recovery_module._STALE_RUN_MIN_AGE_SECONDS),
        created_at=None,
    )
    assert recovery_module._is_older_than_threshold(run) is True


@pytest.mark.asyncio
async def test_a_run_with_no_timestamp_is_left_alone(monkeypatch):
    """Neither started_at nor created_at present is not evidence of age
    either way, so it is skipped rather than treated as old enough."""
    dataset_id = uuid4()
    dataset = SimpleNamespace(id=dataset_id, owner_id=uuid4())
    run = _started_run(dataset_id, origin="api")
    run.created_at = None
    run.started_at = None

    rollback_calls, close_calls = _drive(monkeypatch, run, dataset)

    await recovery_module.recover_stale_cognify_runs_on_startup()

    assert rollback_calls == []
    assert close_calls == []


def test_is_older_than_threshold_falls_back_to_created_at():
    """started_at can be NULL (an INITIATED row that never ticked past it in
    some path); created_at is the fallback rather than treating that as
    ageless."""
    run = SimpleNamespace(
        started_at=None,
        created_at=datetime.now(timezone.utc) - timedelta(hours=2),
    )
    assert recovery_module._is_older_than_threshold(run) is True


def test_is_older_than_threshold_handles_naive_datetimes():
    """A row written by a backend that stores naive UTC must not raise or be
    silently misjudged when compared against an aware `now()`."""
    naive_two_hours_ago = (datetime.now(timezone.utc) - timedelta(hours=2)).replace(tzinfo=None)
    run = SimpleNamespace(
        started_at=naive_two_hours_ago,
        created_at=None,
    )
    assert recovery_module._is_older_than_threshold(run) is True


@pytest.mark.asyncio
async def test_a_second_stdio_sessions_fresh_run_is_left_alone(monkeypatch):
    """The exact scenario from the MCP review comment
    (github.com/topoteretes/cognee/pull/4983#discussion_r4004955294): stdio is
    one process per client, so every IDE window or agent session against the
    same local database stamps "mcp" too. A second session booting while the
    first is still mid-ingest must not roll that ingest back just because the
    origin matches — the age floor is what stops it, the same mechanism that
    protects a rolling API deploy, with no MCP-specific carve-out needed."""
    dataset_id = uuid4()
    dataset = SimpleNamespace(id=dataset_id, owner_id=uuid4())
    run = _started_run(dataset_id, origin="mcp")
    run.created_at = datetime.now(timezone.utc)
    run.started_at = run.created_at

    rollback_calls, close_calls = _drive(monkeypatch, run, dataset)

    await recovery_module.recover_stale_cognify_runs_on_startup(owned_origins=frozenset({"mcp"}))

    assert rollback_calls == []
    assert close_calls == []


@pytest.mark.asyncio
async def test_a_stdio_sessions_run_past_the_age_floor_is_recovered(monkeypatch):
    """The other half: once that same "mcp"-origin row is old enough that a
    sibling session no longer explains it, it is closed exactly as any other
    surface's stale run would be."""
    dataset_id = uuid4()
    dataset = SimpleNamespace(id=dataset_id, owner_id=uuid4())
    run = _started_run(dataset_id, origin="mcp")

    rollback_calls, close_calls = _drive(monkeypatch, run, dataset)

    await recovery_module.recover_stale_cognify_runs_on_startup(owned_origins=frozenset({"mcp"}))

    assert len(rollback_calls) == 1
    assert len(close_calls) == 1


def test_parse_non_negative_int_falls_back_on_garbage(monkeypatch, caplog):
    """A typo in an env var this module's own docs tell operators to set
    must degrade to the default, not crash the process that imports this
    module at boot (my-python-reviewer finding, SDK-591)."""
    monkeypatch.setenv("TEST_COGNEE_INT_VAR", "not-a-number")
    with caplog.at_level("WARNING"):
        result = recovery_module._parse_non_negative_int("TEST_COGNEE_INT_VAR", 42)
    assert result == 42
    assert "not-a-number" in caplog.text


def test_parse_non_negative_int_falls_back_on_negative(monkeypatch, caplog):
    """A negative value would silently disable the floor by making every run
    look old enough; treat it the same as garbage."""
    monkeypatch.setenv("TEST_COGNEE_INT_VAR", "-5")
    with caplog.at_level("WARNING"):
        result = recovery_module._parse_non_negative_int("TEST_COGNEE_INT_VAR", 42)
    assert result == 42
    assert "-5" in caplog.text


def test_parse_non_negative_int_accepts_zero(monkeypatch):
    """Zero is a legitimate value (disables the periodic sweep entirely) and
    must not be treated as falsy-garbage."""
    monkeypatch.setenv("TEST_COGNEE_INT_VAR", "0")
    assert recovery_module._parse_non_negative_int("TEST_COGNEE_INT_VAR", 42) == 0


def test_parse_non_negative_int_uses_default_when_unset(monkeypatch):
    monkeypatch.delenv("TEST_COGNEE_INT_VAR", raising=False)
    assert recovery_module._parse_non_negative_int("TEST_COGNEE_INT_VAR", 42) == 42


@pytest.mark.asyncio
async def test_periodic_sweep_calls_recovery_on_each_tick(monkeypatch):
    """The loop must actually re-invoke the sweep on the timer, not just sleep
    forever — the whole point is to give a row a second chance once it clears
    the age floor."""
    calls = []

    async def _fake_recover(owned_origins=None):
        calls.append(owned_origins)

    monkeypatch.setattr(recovery_module, "recover_stale_cognify_runs_on_startup", _fake_recover)

    sleep_calls = []

    async def _fake_sleep(_seconds):
        sleep_calls.append(_seconds)
        if len(sleep_calls) >= 3:
            raise asyncio.CancelledError

    monkeypatch.setattr(recovery_module.asyncio, "sleep", _fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        await recovery_module._periodic_recovery_loop(frozenset({"api"}), 5)

    # sleep, recover, sleep, recover, sleep (raises here) — the 3rd sleep
    # ends the loop before a 3rd recover call happens.
    assert calls == [frozenset({"api"})] * 2
    assert sleep_calls == [5, 5, 5]


@pytest.mark.asyncio
async def test_periodic_sweep_survives_a_failing_iteration(monkeypatch):
    """One failed sweep must not kill the loop — the next interval should
    still get its turn."""
    calls = []

    async def _flaky_recover(owned_origins=None):
        calls.append(owned_origins)
        if len(calls) == 1:
            raise RuntimeError("relational database unreachable")

    monkeypatch.setattr(recovery_module, "recover_stale_cognify_runs_on_startup", _flaky_recover)

    async def _fake_sleep(_seconds):
        if len(calls) >= 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(recovery_module.asyncio, "sleep", _fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        await recovery_module._periodic_recovery_loop(frozenset({"api"}), 5)

    assert len(calls) == 2


def test_start_periodic_recovery_sweep_disabled_returns_none(monkeypatch):
    """interval_seconds=0 must not schedule a task at all — a caller that
    checks ``is None`` should never see a task it then has to clean up."""
    task = recovery_module.start_periodic_recovery_sweep(interval_seconds=0)
    assert task is None


@pytest.mark.asyncio
async def test_start_and_stop_periodic_recovery_sweep_round_trip(monkeypatch):
    """The task actually runs on the event loop and stop_ actually ends it —
    not two functions that merely look like they cooperate."""
    calls = []

    async def _fake_recover(owned_origins=None):
        calls.append(owned_origins)

    monkeypatch.setattr(recovery_module, "recover_stale_cognify_runs_on_startup", _fake_recover)

    task = recovery_module.start_periodic_recovery_sweep(
        owned_origins=frozenset({"api"}), interval_seconds=0.01
    )
    assert task is not None

    # Let it tick at least once before stopping.
    await asyncio.sleep(0.03)
    await recovery_module.stop_periodic_recovery_sweep(task)

    assert task.cancelled() or task.done()
    assert len(calls) >= 1


@pytest.mark.asyncio
async def test_stop_periodic_recovery_sweep_is_a_no_op_for_none():
    """The disabled case from start_ (returns None) must be safe to pass
    straight into stop_ without a None-check at every call site."""
    await recovery_module.stop_periodic_recovery_sweep(None)
