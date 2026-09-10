"""Custom store configs route update() through the full rebuild flow.

The incremental engine resolves its engines from global/dataset context, so
it cannot honor per-call ``vector_db_config``/``graph_db_config`` — running
it anyway would silently read and write the DEFAULT stores while the
caller's stores never see the edit. update() must therefore skip the
incremental attempt whenever either config is provided and take the full
flow, whose pipelines apply the configs. Without configs the incremental
path stays first choice.
"""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

import cognee.api.v1.update.update  # bind the real submodule
from cognee.api.v1.update.incremental import RefusalReason
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunCompleted

update_module = sys.modules["cognee.api.v1.update.update"]
data_methods_module = sys.modules["cognee.modules.data.methods"]

pytestmark = pytest.mark.asyncio

FULL_RUN_ID = uuid4()


def _full_result(dataset_id):
    """What cognify() hands back for the rebuild: one completed run per dataset."""
    return {
        dataset_id: PipelineRunCompleted(
            pipeline_run_id=FULL_RUN_ID, dataset_id=dataset_id, dataset_name="ds"
        )
    }


def _engine_summary(status="incremental"):
    """The chunk-level engine's raw summary, as incremental_update() returns it."""
    return {
        "status": status,
        "regions": 1,
        "deleted_chunks": 1,
        "added_chunks": 2,
        "reused_chunks": 0,
        "kept_chunks": 5,
        "reindexed_chunks": 0,
        "pipeline_run_id": uuid4(),
    }


def _relational_engine_stub(row):
    session = MagicMock()
    session.get = AsyncMock(return_value=row)
    session.commit = AsyncMock()
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=session)
    context.__aexit__ = AsyncMock(return_value=False)
    engine = MagicMock()
    engine.get_async_session = MagicMock(return_value=context)
    return engine


def _patches(data_id, incremental, full_result, row=None):
    row = row if row is not None else SimpleNamespace(id=data_id, legacy_id=None, owner_id=uuid4())
    relational_module = sys.modules["cognee.infrastructure.databases.relational"]
    return (
        patch.object(data_methods_module, "resolve_data_id", AsyncMock(return_value=data_id)),
        patch.object(
            relational_module,
            "get_relational_engine",
            MagicMock(return_value=_relational_engine_stub(row)),
        ),
        patch.object(update_module, "incremental_update", incremental),
        patch.object(update_module, "datasets", SimpleNamespace(delete_data=AsyncMock())),
        patch.object(update_module, "add", AsyncMock()),
        patch.object(update_module, "cognify", AsyncMock(return_value=full_result)),
    )


async def test_custom_configs_skip_the_incremental_path():
    data_id, dataset_id = uuid4(), uuid4()
    incremental = AsyncMock()
    full_result = _full_result(dataset_id)

    for config_kwargs in (
        {"vector_db_config": {"vector_db_provider": "custom"}},
        {"graph_db_config": {"graph_database_provider": "custom"}},
        {
            "vector_db_config": {"vector_db_provider": "custom"},
            "graph_db_config": {"graph_database_provider": "custom"},
        },
    ):
        incremental.reset_mock()
        p1, p2, p3, p4, p5, p6 = _patches(data_id, incremental, full_result)
        with p1, p2, p3, p4, p5, p6:
            result = await update_module.update(
                data_id=data_id,
                data="new content",
                dataset_id=dataset_id,
                user=SimpleNamespace(id=uuid4()),
                **config_kwargs,
            )
        incremental.assert_not_called()
        assert result.mode == "full_rebuild", "custom-config updates must run the full flow"
        assert result.status == "updated"
        assert result.fallback_reason is RefusalReason.PER_CALL_DB_CONFIG
        assert result.pipeline_run_id == FULL_RUN_ID


async def test_node_set_change_skips_the_incremental_path():
    data_id, dataset_id = uuid4(), uuid4()
    incremental = AsyncMock()
    full_result = _full_result(dataset_id)

    p1, p2, p3, p4, p5, p6 = _patches(data_id, incremental, full_result)
    with p1, p2, p3, p4, p5, p6:
        result = await update_module.update(
            data_id=data_id,
            data="new content",
            dataset_id=dataset_id,
            user=SimpleNamespace(id=uuid4()),
            node_set=["updated-group"],
        )

    incremental.assert_not_called()
    assert result.mode == "full_rebuild"
    assert result.fallback_reason is RefusalReason.UNSUPPORTED_METADATA


@pytest.mark.parametrize(
    "config_kwargs",
    (
        {"graph_model": type("CustomGraph", (), {})},
        {"custom_prompt": "Extract only explicitly stated facts."},
    ),
)
async def test_custom_extraction_config_skips_the_incremental_path(config_kwargs):
    data_id, dataset_id = uuid4(), uuid4()
    incremental = AsyncMock()
    full_result = _full_result(dataset_id)

    p1, p2, p3, p4, p5, p6 = _patches(data_id, incremental, full_result)
    with p1, p2, p3, p4, p5, p6:
        result = await update_module.update(
            data_id=data_id,
            data="new content",
            dataset_id=dataset_id,
            user=SimpleNamespace(id=uuid4()),
            **config_kwargs,
        )

    incremental.assert_not_called()
    assert result.mode == "full_rebuild"
    assert result.fallback_reason is RefusalReason.CUSTOM_EXTRACTION_CONFIG


async def test_multi_item_input_is_rejected_not_multiplied():
    """update() is the ONLY place that rejects a multi-item list.

    The incremental engine used to re-check this and raise
    IncrementalUpdateNotPossible — which means "fall back", so a two-item
    update would have been routed into a full flow that had already unwrapped
    it. That copy is gone, which makes this check the only thing standing
    between a caller and a silently multiplied update.
    """
    from cognee.modules.ingestion.exceptions import IngestionError

    data_id, dataset_id = uuid4(), uuid4()
    incremental = AsyncMock()

    p1, p2, p3, p4, p5, p6 = _patches(data_id, incremental, _full_result(dataset_id))
    with p1, p2, p3, p4, p5, p6, pytest.raises(IngestionError):
        await update_module.update(
            data_id=data_id,
            data=["first document", "second document"],
            dataset_id=dataset_id,
            user=SimpleNamespace(id=uuid4()),
        )

    incremental.assert_not_called()


async def test_single_item_list_is_unwrapped():
    """The permissive shape the HTTP router sends is accepted, not refused."""
    data_id, dataset_id = uuid4(), uuid4()
    incremental = AsyncMock(return_value=_engine_summary())

    p1, p2, p3, p4, p5, p6 = _patches(data_id, incremental, _full_result(dataset_id))
    with p1, p2, p3, p4, p5, p6:
        await update_module.update(
            data_id=data_id,
            data=["only document"],
            dataset_id=dataset_id,
            user=SimpleNamespace(id=uuid4()),
        )

    assert incremental.await_args.kwargs["data"] == "only document"


async def test_the_full_fallback_keeps_the_original_row_owner():
    """Re-ingestion must not hand the document to whoever updated it.

    The fallback deletes the row and calls add(), which mints a fresh Data
    with owner_id=user.id. A collaborator authorized by the dataset ACL would
    otherwise take ownership of a document they merely edited — a permission
    change nobody asked for, and one the incremental branch never makes.
    """
    data_id, dataset_id, original_owner = uuid4(), uuid4(), uuid4()
    row = SimpleNamespace(id=data_id, legacy_id=None, owner_id=original_owner)
    collaborator = SimpleNamespace(id=uuid4())

    p1, p2, p3, p4, p5, p6 = _patches(data_id, AsyncMock(), _full_result(dataset_id), row=row)
    with p1, p2, p3, p4, p5, p6:
        await update_module.update(
            data_id=data_id,
            data="new content",
            dataset_id=dataset_id,
            user=collaborator,
            chunk_level_diff=False,
        )

    assert row.owner_id == original_owner


async def test_no_configs_take_the_incremental_path():
    data_id, dataset_id = uuid4(), uuid4()
    summary = _engine_summary()
    incremental = AsyncMock(return_value=summary)

    p1, p2, p3, p4, p5, p6 = _patches(data_id, incremental, _full_result(dataset_id))
    with p1, p2, p3, p4, p5, p6:
        result = await update_module.update(
            data_id=data_id,
            data="new content",
            dataset_id=dataset_id,
            user=SimpleNamespace(id=uuid4()),
        )
    incremental.assert_awaited_once()
    assert (result.mode, result.status) == ("incremental", "updated")
    assert result.fallback_reason is None
    assert (result.chunks.regions, result.chunks.deleted, result.chunks.added) == (1, 1, 2)
    assert (result.chunks.kept, result.chunks.reindexed) == (5, 0)
    assert result.pipeline_run_id == summary["pipeline_run_id"]
