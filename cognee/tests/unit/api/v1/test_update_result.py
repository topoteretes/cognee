"""update() answers in one shape on every path, and a full rebuild says why.

The chunk-level engine and the full rebuild used to return different values
(a summary dict versus a pipeline-run mapping), and the reason for a rebuild
lived only in the server log. Every path now returns an ``UpdateResult`` with
the same fields, and every rebuild carries its ``fallback_reason`` (SDK-587).
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
        self.delete_data = AsyncMock()
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
            patch.object(update_module, "datasets", SimpleNamespace(delete_data=self.delete_data)),
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

    assert isinstance(result, UpdateResult)
    assert (result.status, result.mode) == ("updated", "full_rebuild")
    assert result.fallback_reason is RefusalReason.NO_BASELINE
    assert result.fallback_detail == "no stored processed text for this data item"
    assert result.chunks is None, "a rebuild has no chunk diff to report"
    assert result.pipeline_run_id == run.pipeline_run_id
    assert (result.data_id, result.dataset_id) == (data_id, dataset_id)
    stack.delete_data.assert_awaited_once()
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
    assert result.mode == "full_rebuild"
    assert result.fallback_reason is reason
    assert result.fallback_detail, "the reason comes with a sentence for the caller"


async def test_incremental_result_carries_the_chunk_counters_and_no_reason():
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
        "pipeline_run_id": run_id,
    }
    stack = _Stack(data_id, dataset_id, AsyncMock(return_value=summary), _run(dataset_id))

    result = await _update(stack, data_id, dataset_id)

    assert (result.status, result.mode) == ("updated", "incremental")
    assert result.chunks.model_dump() == {
        "regions": 2,
        "deleted": 3,
        "added": 4,
        "reused": 1,
        "kept": 7,
        "reindexed": 2,
    }
    assert result.fallback_reason is None and result.fallback_detail is None
    assert result.pipeline_run_id == run_id
    stack.delete_data.assert_not_awaited()
    stack.cognify.assert_not_awaited()


async def test_unchanged_content_is_reported_as_unchanged():
    data_id, dataset_id = uuid4(), uuid4()
    summary = {
        "status": "unchanged",
        "regions": 0,
        "deleted_chunks": 0,
        "added_chunks": 0,
        "reused_chunks": 0,
        "kept_chunks": 0,
        "reindexed_chunks": 0,
        "pipeline_run_id": None,
    }
    stack = _Stack(data_id, dataset_id, AsyncMock(return_value=summary), _run(dataset_id))

    result = await _update(stack, data_id, dataset_id)

    assert (result.status, result.mode) == ("unchanged", "incremental")
    assert result.chunks.model_dump() == dict.fromkeys(
        ("regions", "deleted", "added", "reused", "kept", "reindexed"), 0
    )
    assert result.pipeline_run_id is None


async def test_errored_rebuild_is_a_failed_result_not_an_exception():
    data_id, dataset_id = uuid4(), uuid4()
    errored = _run(dataset_id, errored=True)
    stack = _Stack(data_id, dataset_id, AsyncMock(), errored)

    result = await _update(stack, data_id, dataset_id, chunk_level_diff=False)

    assert (result.status, result.mode) == ("failed", "full_rebuild")
    assert result.fallback_reason is RefusalReason.DISABLED
    assert (result.error_class, result.error_message) == ("LLMRateLimitError", "rate limited")
    assert result.pipeline_run_id == errored.pipeline_run_id
    assert (result.data_id, result.dataset_id) == (data_id, dataset_id), (
        "a failed result keeps the ids the caller needs to retry"
    )


def test_result_serializes_to_plain_json():
    result = UpdateResult(
        data_id=uuid4(),
        dataset_id=uuid4(),
        status="updated",
        mode="full_rebuild",
        fallback_reason=RefusalReason.UNSUPPORTED_CHUNKER,
        fallback_detail="document was chunked by x, not y",
        pipeline_run_id=uuid4(),
    )
    body = result.model_dump(mode="json")
    assert body["fallback_reason"] == "unsupported_chunker"
    assert body["chunks"] is None
    assert UpdateResult.model_validate(body) == result
