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
        "total_chunks": 7,
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
        patch.object(update_module, "forget", AsyncMock()),
        patch.object(data_methods_module, "reset_data_pipeline_status", AsyncMock()),
        patch.object(update_module, "add", AsyncMock()),
        patch.object(update_module, "cognify", AsyncMock(return_value=full_result)),
        patch.object(update_module, "recorded_chunk_budget", AsyncMock(return_value=None)),
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
        p1, p2, p3, p4, p5, p6, p7, p8 = _patches(data_id, incremental, full_result)
        with p1, p2, p3, p4, p5, p6, p7, p8:
            result = await update_module.update(
                data_id=data_id,
                data="new content",
                dataset_id=dataset_id,
                user=SimpleNamespace(id=uuid4()),
                **config_kwargs,
            )
        incremental.assert_not_called()
        assert result["status"] == "full_rebuild", "custom-config updates must run the full flow"
        assert result["fallback"]["reason"] is RefusalReason.PER_CALL_DB_CONFIG
        assert result["pipeline_run_id"] == FULL_RUN_ID


async def test_node_set_change_skips_the_incremental_path():
    data_id, dataset_id = uuid4(), uuid4()
    incremental = AsyncMock()
    full_result = _full_result(dataset_id)

    p1, p2, p3, p4, p5, p6, p7, p8 = _patches(data_id, incremental, full_result)
    with p1, p2, p3, p4, p5, p6, p7, p8:
        result = await update_module.update(
            data_id=data_id,
            data="new content",
            dataset_id=dataset_id,
            user=SimpleNamespace(id=uuid4()),
            node_set=["updated-group"],
        )

    incremental.assert_not_called()
    assert result["status"] == "full_rebuild"
    assert result["fallback"]["reason"] is RefusalReason.UNSUPPORTED_METADATA


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

    p1, p2, p3, p4, p5, p6, p7, p8 = _patches(data_id, incremental, full_result)
    with p1, p2, p3, p4, p5, p6, p7, p8:
        result = await update_module.update(
            data_id=data_id,
            data="new content",
            dataset_id=dataset_id,
            user=SimpleNamespace(id=uuid4()),
            **config_kwargs,
        )

    incremental.assert_not_called()
    assert result["status"] == "full_rebuild"
    assert result["fallback"]["reason"] is RefusalReason.CUSTOM_EXTRACTION_CONFIG


async def test_multi_item_input_is_rejected_not_multiplied():
    """One data_id with several inputs is refused before any work.

    A list is a batch, one document per input, and a single id cannot name
    them all — applying it to each would silently multiply the update. The
    incremental engine used to re-check this and raise
    IncrementalUpdateNotPossible, which means "fall back"; that copy is gone,
    so this check in update() is the only thing standing in the way.
    """
    from cognee.modules.ingestion.exceptions import IngestionError

    data_id, dataset_id = uuid4(), uuid4()
    incremental = AsyncMock()

    p1, p2, p3, p4, p5, p6, p7, p8 = _patches(data_id, incremental, _full_result(dataset_id))
    with p1, p2, p3, p4, p5, p6, p7, p8, pytest.raises(IngestionError):
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

    p1, p2, p3, p4, p5, p6, p7, p8 = _patches(data_id, incremental, _full_result(dataset_id))
    with p1, p2, p3, p4, p5, p6, p7, p8:
        await update_module.update(
            data_id=data_id,
            data=["only document"],
            dataset_id=dataset_id,
            user=SimpleNamespace(id=uuid4()),
        )

    assert incremental.await_args.kwargs["data"] == "only document"


async def test_the_full_fallback_keeps_the_original_row_owner():
    """Re-ingestion must not hand the document to whoever updated it.

    The rebuild never deletes the row: it drops the document's memory and
    refreshes the same row with a pinned add(), so a collaborator authorized
    by the dataset ACL keeps editing a document that stays owned by whoever
    ingested it — the incremental branch and the rebuild agree on this.
    """
    data_id, dataset_id, original_owner = uuid4(), uuid4(), uuid4()
    row = SimpleNamespace(id=data_id, legacy_id=None, owner_id=original_owner)
    collaborator = SimpleNamespace(id=uuid4())

    p1, p2, p3, p4, p5, p6, p7, p8 = _patches(
        data_id, AsyncMock(), _full_result(dataset_id), row=row
    )
    with p1, p2, p3, p4, p5, p6, p7, p8:
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

    p1, p2, p3, p4, p5, p6, p7, p8 = _patches(data_id, incremental, _full_result(dataset_id))
    with p1, p2, p3, p4, p5, p6, p7, p8:
        result = await update_module.update(
            data_id=data_id,
            data="new content",
            dataset_id=dataset_id,
            user=SimpleNamespace(id=uuid4()),
        )
    incremental.assert_awaited_once()
    assert result["status"] == "incremental"
    assert result["fallback"] is None
    assert (result["regions"], result["deleted_chunks"], result["added_chunks"]) == (1, 1, 2)
    assert (result["kept_chunks"], result["reindexed_chunks"], result["total_chunks"]) == (5, 0, 7)
    assert result["pipeline_run_id"] == summary["pipeline_run_id"]
