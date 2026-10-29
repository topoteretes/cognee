"""Rollback resets the stamp slot of the pipeline whose run is rolled back (SDK-397)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from cognee.modules.cognify import rollback
from cognee.modules.pipelines.models.DataItemStatus import DataItemStatus

DATASET_ID = uuid4()
DONE = DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED


def _session(records):
    result = MagicMock()
    result.scalars.return_value.all.return_value = records
    return SimpleNamespace(execute=AsyncMock(return_value=result))


@pytest.mark.asyncio
async def test_reset_clears_only_the_named_pipelines_slot():
    record = SimpleNamespace(
        id=uuid4(),
        pipeline_status={
            "add_pipeline": {str(DATASET_ID): DONE},
            "indexing_pipeline": {str(DATASET_ID): DONE},
            "cognify_pipeline": {str(DATASET_ID): DONE},
        },
    )
    with patch.object(rollback.orm_attributes, "flag_modified"):
        await rollback._reset_pipeline_status(
            _session([record]), {record.id}, DATASET_ID, "indexing_pipeline"
        )

    assert str(DATASET_ID) not in record.pipeline_status["indexing_pipeline"]
    assert str(DATASET_ID) in record.pipeline_status["add_pipeline"]
    assert str(DATASET_ID) in record.pipeline_status["cognify_pipeline"]


@pytest.mark.asyncio
async def test_reset_defaults_to_the_cognify_slot():
    record = SimpleNamespace(
        id=uuid4(),
        pipeline_status={
            "indexing_pipeline": {str(DATASET_ID): DONE},
            "cognify_pipeline": {str(DATASET_ID): DONE},
        },
    )
    with patch.object(rollback.orm_attributes, "flag_modified"):
        await rollback._reset_pipeline_status(_session([record]), {record.id}, DATASET_ID)

    assert str(DATASET_ID) not in record.pipeline_status["cognify_pipeline"]
    assert str(DATASET_ID) in record.pipeline_status["indexing_pipeline"]


@pytest.mark.asyncio
async def test_completed_ids_read_the_named_pipelines_slot():
    done = SimpleNamespace(
        id=uuid4(), pipeline_status={"indexing_pipeline": {str(DATASET_ID): DONE}}
    )
    pending = SimpleNamespace(
        id=uuid4(), pipeline_status={"cognify_pipeline": {str(DATASET_ID): DONE}}
    )

    kept = await rollback._completed_data_ids(
        _session([done, pending]), {done.id, pending.id}, DATASET_ID, "indexing_pipeline"
    )

    assert kept == {done.id}


@pytest.mark.asyncio
async def test_the_handler_passes_the_pipeline_name_through_the_graph_provenance_path():
    unified = SimpleNamespace(
        supports_graph_provenance_delete=lambda: True,
        graph=SimpleNamespace(
            find_node_source_refs_by_pipeline_run=AsyncMock(return_value={}),
            find_edge_source_refs_by_pipeline_run=AsyncMock(return_value={}),
        ),
        rollback_by_pipeline_run_id=AsyncMock(),
    )
    reset = AsyncMock()
    session = SimpleNamespace(commit=AsyncMock())

    class _Session:
        async def __aenter__(self):
            return session

        async def __aexit__(self, *args):
            return False

    engine = SimpleNamespace(get_async_session=lambda: _Session())
    run_id = uuid4()
    with (
        patch.object(rollback, "get_unified_engine", AsyncMock(return_value=unified)),
        patch.object(rollback, "stores_provenance_in_graph", AsyncMock(return_value=True)),
        patch.object(rollback, "get_relational_engine", return_value=engine),
        patch.object(rollback, "_reset_pipeline_status", reset),
    ):
        await rollback.cognify_rollback_handler(
            run_id, SimpleNamespace(id=DATASET_ID), pipeline_name="indexing_pipeline"
        )

    unified.rollback_by_pipeline_run_id.assert_awaited_once_with(str(run_id), keep_data_ids=None)
    assert reset.await_args.args[2:] == (DATASET_ID, "indexing_pipeline")
