"""SQL-level tests for recall's graph build-status probe (graph_warmup).

The unit suite stubs ``get_graph_build_status`` wholesale, so the probe's
actual ``pipeline_runs`` queries are pinned here against a real (tmp sqlite)
relational engine. The critical property: errored ``add_pipeline`` (staging)
rows never flag ``build_failed`` — otherwise one errored ``add()`` of a
corrupt file would make every recall() on a populated graph return zero
results, and datasets built by custom pipelines (which log nothing beyond
staging) would lose their fail-safe warm fall-through.
"""

import pathlib
import types
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import select

import cognee
from cognee.infrastructure.databases.relational import create_db_and_tables, get_relational_engine
from cognee.modules.operations import record_operation
from cognee.modules.pipelines.models import PipelineRun
from cognee.modules.pipelines.models.PipelineRun import PipelineRunStatus
from cognee.modules.recall.methods.graph_warmup import (
    STATE_BUILD_FAILED,
    STATE_NEVER_BUILT,
    STATE_WARM,
    get_graph_build_status,
)


@pytest_asyncio.fixture
async def clean_test_environment(request, tmp_path, monkeypatch):
    """Scope every write to a per-test tmp root; never touch ambient data."""
    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "false")

    root = pathlib.Path(tmp_path) / request.node.name

    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )

    create_relational_engine.cache_clear()

    cognee.config.set_relational_db_config({"db_provider": "sqlite"})
    cognee.config.system_root_directory(str(root / "system"))
    cognee.config.data_root_directory(str(root / "data"))

    await create_db_and_tables()

    yield

    create_relational_engine.cache_clear()


def _permit(monkeypatch, dataset_ids):
    """Stub permission resolution; the probe intersects with these ids."""
    import cognee.modules.users.permissions.methods as permission_methods

    async def fake_get_permitted_dataset_ids(user_id):
        return list(dataset_ids)

    monkeypatch.setattr(
        permission_methods, "get_permitted_dataset_ids", fake_get_permitted_dataset_ids
    )


async def _insert_run(
    dataset_id,
    pipeline_name,
    status,
    error_class=None,
    error_message=None,
):
    async with get_relational_engine().get_async_session() as session:
        session.add(
            PipelineRun(
                pipeline_run_id=uuid4(),
                pipeline_id=uuid4(),
                pipeline_name=pipeline_name,
                dataset_id=dataset_id,
                status=status,
                run_info={},
                error_class=error_class,
                error_message=error_message,
            )
        )
        await session.commit()


_USER = types.SimpleNamespace(id=uuid4(), tenant_id=None)


@pytest.mark.asyncio
async def test_errored_staging_run_keeps_failsafe_warm(clean_test_environment, monkeypatch):
    """An errored add() must not read as build_failed: staging rows are
    excluded from the errored query, so the staging-only fail-safe (warm)
    still applies — e.g. a graph populated by a custom pipeline."""
    dataset_id = uuid4()
    _permit(monkeypatch, [dataset_id])

    await _insert_run(
        dataset_id,
        "add_pipeline",
        PipelineRunStatus.DATASET_PROCESSING_ERRORED,
        error_class="UnicodeDecodeError",
        error_message="corrupt file",
    )

    probe = await get_graph_build_status(_USER, [dataset_id])
    assert probe.state == STATE_WARM


@pytest.mark.asyncio
async def test_operation_record_does_not_make_unbuilt_dataset_warm(
    clean_test_environment, monkeypatch
):
    """A session-only remember record is not evidence of a built graph."""
    dataset_id = uuid4()
    _permit(monkeypatch, [dataset_id])

    # This is the row produced by remember(..., session_id=...): it carries
    # the dataset for attribution, but has no pipeline_name or status.
    async with record_operation("remember", dataset_id=dataset_id, session_id="session-1"):
        pass

    async with get_relational_engine().get_async_session() as session:
        rows = (
            (await session.execute(select(PipelineRun).where(PipelineRun.dataset_id == dataset_id)))
            .scalars()
            .all()
        )

    assert len(rows) == 1
    operation_row = rows[0]
    assert operation_row.dataset_id == dataset_id
    assert operation_row.operation_name == "remember"
    assert operation_row.pipeline_name is None
    assert operation_row.status is None

    probe = await get_graph_build_status(_USER, [dataset_id])
    assert probe.state == STATE_NEVER_BUILT


@pytest.mark.asyncio
async def test_errored_graph_write_flags_build_failed_until_a_build_completes(
    clean_test_environment, monkeypatch
):
    dataset_id = uuid4()
    _permit(monkeypatch, [dataset_id])

    # No runs at all: never_built.
    probe = await get_graph_build_status(_USER, [dataset_id])
    assert probe.state == STATE_NEVER_BUILT

    # A staged add() plus an ERRORED graph-writing attempt: build_failed,
    # carrying the classified root cause.
    await _insert_run(dataset_id, "add_pipeline", PipelineRunStatus.DATASET_PROCESSING_COMPLETED)
    await _insert_run(
        dataset_id,
        "cognify_pipeline",
        PipelineRunStatus.DATASET_PROCESSING_ERRORED,
        error_class="AuthenticationError",
        error_message="invalid api key",
    )

    probe = await get_graph_build_status(_USER, [dataset_id])
    assert probe.state == STATE_BUILD_FAILED
    assert probe.error_class == "AuthenticationError"
    assert probe.error_message == "invalid api key"

    # Once any graph-writing run completes, the dataset is warm — the stale
    # errored row no longer matters.
    await _insert_run(
        dataset_id, "cognify_pipeline", PipelineRunStatus.DATASET_PROCESSING_COMPLETED
    )

    probe = await get_graph_build_status(_USER, [dataset_id])
    assert probe.state == STATE_WARM
