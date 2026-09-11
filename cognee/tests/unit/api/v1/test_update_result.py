"""update() answers in one shape on every path, and a full rebuild says why.

The chunk-level engine and the full rebuild used to return different values
(a summary dict versus a pipeline-run mapping), and the reason for a rebuild
lived only in the server log. Every path now returns one dict, a superset of
the old summary, and every rebuild carries its ``fallback`` reason (SDK-587).
"""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

import cognee.api.v1.update.update  # bind the real submodule
from cognee.api.v1.update import UpdateResult
from cognee.api.v1.update.incremental import IncrementalUpdateNotPossible, RefusalReason
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunCompleted,
    PipelineRunErrored,
)

update_module = sys.modules["cognee.api.v1.update.update"]
data_methods_module = sys.modules["cognee.modules.data.methods"]

pytestmark = pytest.mark.asyncio

COUNTER_KEYS = (
    "regions",
    "deleted_chunks",
    "added_chunks",
    "reused_chunks",
    "kept_chunks",
    "reindexed_chunks",
    "total_chunks",
)


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


def _run(dataset_id, errored=False):
    if errored:
        return PipelineRunErrored(
            pipeline_run_id=uuid4(),
            dataset_id=dataset_id,
            dataset_name="ds",
            error_class="LLMRateLimitError",
            error_message="rate limited",
        )
    return PipelineRunCompleted(pipeline_run_id=uuid4(), dataset_id=dataset_id, dataset_name="ds")


class _Stack:
    """update() with its collaborators stubbed; records what the rebuild called."""

    def __init__(self, data_id, dataset_id, incremental, cognify_run):
        row = SimpleNamespace(id=data_id, legacy_id=None, owner_id=uuid4())
        relational_module = sys.modules["cognee.infrastructure.databases.relational"]
        self.forget = AsyncMock()
        self.reset_status = AsyncMock()
        self.add = AsyncMock()
        self.cognify = AsyncMock(return_value={dataset_id: cognify_run})
        self.recorded_budget = AsyncMock(return_value=None)
        self.patches = (
            patch.object(data_methods_module, "resolve_data_id", AsyncMock(return_value=data_id)),
            patch.object(
                relational_module,
                "get_relational_engine",
                MagicMock(return_value=_relational_engine_stub(row)),
            ),
            patch.object(update_module, "incremental_update", incremental),
            patch.object(update_module, "forget", self.forget),
            patch.object(data_methods_module, "reset_data_pipeline_status", self.reset_status),
            patch.object(update_module, "add", self.add),
            patch.object(update_module, "cognify", self.cognify),
            patch.object(update_module, "recorded_chunk_budget", self.recorded_budget),
        )

    def __enter__(self):
        for p in self.patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self.patches:
            p.stop()
        return False


async def _update(stack, data_id, dataset_id, **kwargs):
    with stack:
        return await update_module.update(
            data_id=data_id,
            data="new content",
            dataset_id=dataset_id,
            user=SimpleNamespace(id=uuid4()),
            **kwargs,
        )


async def test_engine_refusal_becomes_a_full_rebuild_with_its_reason():
    data_id, dataset_id = uuid4(), uuid4()
    refusal = IncrementalUpdateNotPossible(
        "no stored processed text for this data item", RefusalReason.NO_BASELINE
    )
    run = _run(dataset_id)
    stack = _Stack(data_id, dataset_id, AsyncMock(side_effect=refusal), run)

    result = await _update(stack, data_id, dataset_id)

    assert isinstance(result, dict)
    assert result["status"] == "full_rebuild"
    assert result["fallback"] == {
        "reason": RefusalReason.NO_BASELINE,
        "detail": "no stored processed text for this data item",
    }
    assert result["error"] is None
    assert all(result[key] is None for key in COUNTER_KEYS), "a rebuild has no chunk diff"
    assert result["pipeline_run_id"] == run.pipeline_run_id
    assert (result["data_id"], result["dataset_id"]) == (data_id, dataset_id)
    assert result["duration_seconds"] >= 0
    stack.forget.assert_awaited_once()
    stack.reset_status.assert_awaited_once()
    stack.add.assert_awaited_once()


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"chunk_level_diff": False}, RefusalReason.DISABLED),
        ({"node_set": ["group"]}, RefusalReason.UNSUPPORTED_METADATA),
        ({"custom_prompt": "be brief"}, RefusalReason.CUSTOM_EXTRACTION_CONFIG),
        ({"graph_model": type("Custom", (), {})}, RefusalReason.CUSTOM_EXTRACTION_CONFIG),
        ({"vector_db_config": {"vector_db_provider": "x"}}, RefusalReason.PER_CALL_DB_CONFIG),
        ({"graph_db_config": {"graph_database_provider": "x"}}, RefusalReason.PER_CALL_DB_CONFIG),
    ],
)
async def test_every_pre_check_downgrade_names_its_reason(kwargs, reason):
    data_id, dataset_id = uuid4(), uuid4()
    incremental = AsyncMock()
    stack = _Stack(data_id, dataset_id, incremental, _run(dataset_id))

    result = await _update(stack, data_id, dataset_id, **kwargs)

    incremental.assert_not_called()
    assert result["status"] == "full_rebuild"
    assert result["fallback"]["reason"] is reason
    assert result["fallback"]["detail"], "the reason comes with a sentence for the caller"


async def test_incremental_result_keeps_the_old_summary_keys_and_adds_the_new_ones():
    """The chunk-level summary that shipped before is a strict subset: same
    keys, same values, so a caller reading result["kept_chunks"] or comparing
    result["status"] == "incremental" is unaffected."""
    data_id, dataset_id = uuid4(), uuid4()
    run_id = uuid4()
    summary = {
        "status": "incremental",
        "regions": 2,
        "deleted_chunks": 3,
        "added_chunks": 4,
        "reused_chunks": 1,
        "kept_chunks": 7,
        "reindexed_chunks": 2,
        "total_chunks": 11,
        "pipeline_run_id": run_id,
    }
    stack = _Stack(data_id, dataset_id, AsyncMock(return_value=summary), _run(dataset_id))

    result = await _update(stack, data_id, dataset_id)

    old_summary = {key: value for key, value in summary.items() if key != "pipeline_run_id"}
    assert {key: result[key] for key in old_summary} == old_summary
    assert result["fallback"] is None and result["error"] is None
    assert result["pipeline_run_id"] == run_id
    assert (result["data_id"], result["dataset_id"]) == (data_id, dataset_id)
    stack.forget.assert_not_awaited()
    stack.cognify.assert_not_awaited()


async def test_unchanged_content_is_reported_as_unchanged():
    data_id, dataset_id = uuid4(), uuid4()
    summary = {
        "status": "unchanged",
        "regions": 0,
        "deleted_chunks": 0,
        "added_chunks": 0,
        "reused_chunks": 0,
        "kept_chunks": 9,
        "reindexed_chunks": 0,
        "total_chunks": 9,
        "pipeline_run_id": None,
    }
    stack = _Stack(data_id, dataset_id, AsyncMock(return_value=summary), _run(dataset_id))

    result = await _update(stack, data_id, dataset_id)

    assert result["status"] == "unchanged"
    assert (result["kept_chunks"], result["total_chunks"]) == (9, 9)
    assert result["pipeline_run_id"] is None


async def test_errored_rebuild_is_a_failed_result_not_an_exception():
    data_id, dataset_id = uuid4(), uuid4()
    errored = _run(dataset_id, errored=True)
    stack = _Stack(data_id, dataset_id, AsyncMock(), errored)

    result = await _update(stack, data_id, dataset_id, chunk_level_diff=False)

    assert result["status"] == "failed"
    assert result["fallback"]["reason"] is RefusalReason.DISABLED
    assert result["error"] == {"error_class": "LLMRateLimitError", "message": "rate limited"}
    assert result["pipeline_run_id"] == errored.pipeline_run_id
    assert (result["data_id"], result["dataset_id"]) == (data_id, dataset_id), (
        "a failed result keeps the ids the caller needs to retry"
    )


def test_result_schema_round_trips_through_json():
    result = UpdateResult(
        status="full_rebuild",
        data_id=uuid4(),
        dataset_id=uuid4(),
        duration_seconds=1.5,
        pipeline_run_id=uuid4(),
        fallback={"reason": RefusalReason.UNSUPPORTED_CHUNKER, "detail": "chunked by x, not y"},
    )
    body = result.model_dump(mode="json")
    assert body["fallback"] == {"reason": "unsupported_chunker", "detail": "chunked by x, not y"}
    assert body["kept_chunks"] is None and body["error"] is None
    assert UpdateResult.model_validate(body).model_dump() == result.model_dump()


async def test_dlt_replacement_under_another_name_is_refused_before_the_delete():
    """The rebuild deletes first, so a replacement the re-add would refuse
    must be refused while the manifest still exists."""
    dlt = pytest.importorskip("dlt")
    from cognee.exceptions import CogneeValidationError

    resolve_module = sys.modules["cognee.tasks.ingestion.resolve_dlt_sources"]
    data_id, dataset_id = uuid4(), uuid4()
    stack = _Stack(data_id, dataset_id, AsyncMock(), _run(dataset_id))

    with (
        stack,
        patch.object(
            data_methods_module,
            "get_authorized_dataset",
            AsyncMock(return_value=SimpleNamespace(id=dataset_id, name="ds")),
        ),
        patch.object(resolve_module, "get_unique_data_id", AsyncMock(return_value=uuid4())),
        pytest.raises(CogneeValidationError, match="same source name"),
    ):
        await update_module.update(
            data_id=data_id,
            data=dlt.resource([{"id": 1}], name="renamed", primary_key="id"),
            dataset_id=dataset_id,
            user=SimpleNamespace(id=uuid4()),
            chunk_level_diff=False,
        )

    stack.forget.assert_not_awaited()
    stack.add.assert_not_awaited()
