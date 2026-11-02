"""Which entities each conflict review covers, and what it stamps for the next one to retry."""

import importlib
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.memify_pipelines.review_conflicts_pipeline import ConflictReviewResult
from cognee.modules.improve.watermarks import StageWatermark
from cognee.tasks.memify.review_conflicts.read_facts import StoredConflicts

module = importlib.import_module("cognee.modules.improve.review_conflicts")


@pytest.fixture
def selection(monkeypatch):
    user = SimpleNamespace(id=uuid4())
    dataset = SimpleNamespace(id=uuid4(), owner_id=user.id)
    touched = AsyncMock(return_value=set())
    to_repair = AsyncMock(return_value=StoredConflicts({}, [], []))
    pipeline = AsyncMock(return_value=ConflictReviewResult(object(), []))
    monkeypatch.setattr(module, "get_touched_entity_ids", touched)
    monkeypatch.setattr(module, "read_stored_conflicts", to_repair)
    dataset_node_ids = {"a node of this dataset"}
    monkeypatch.setattr(module, "read_dataset_node_ids", AsyncMock(return_value=dataset_node_ids))
    monkeypatch.setattr(module, "review_conflicts_pipeline", pipeline)
    graph = object()
    monkeypatch.setattr(module, "get_graph_engine", AsyncMock(return_value=graph))
    events = []

    @asynccontextmanager
    async def lock(dataset_id):
        assert dataset_id == dataset.id
        events.append("lock")
        yield
        events.append("unlock")

    @asynccontextmanager
    async def context(dataset_id, owner_id):
        assert (dataset_id, owner_id) == (dataset.id, user.id)
        assert events[-1] == "lock"
        events.append("context")
        yield
        events.append("close context")

    monkeypatch.setattr(module, "dataset_lock", lock)
    monkeypatch.setattr(module, "set_database_global_context_variables", context)

    async def review(watermark):
        return await module.review_conflicts(watermark, dataset=dataset, user=user)

    return SimpleNamespace(
        review=review,
        dataset=dataset,
        user=user,
        touched=touched,
        to_repair=to_repair,
        dataset_node_ids=dataset_node_ids,
        graph=graph,
        pipeline=pipeline,
        events=events,
    )


@pytest.mark.asyncio
async def test_first_review_reads_all_entities_without_precheck(selection):
    await selection.review(None)
    selection.pipeline.assert_awaited_once_with(selection.dataset.id, selection.user)
    selection.touched.assert_not_awaited()
    selection.to_repair.assert_not_awaited()


@pytest.mark.asyncio
async def test_touched_entities_skip_graph_precheck(selection):
    since = datetime(2025, 1, 1, tzinfo=timezone.utc)
    selection.touched.return_value = {"b", "a"}
    await selection.review(StageWatermark(uuid4(), since, {}))
    selection.pipeline.assert_awaited_once_with(
        selection.dataset.id, selection.user, entity_ids=["a", "b"], since=since
    )
    selection.touched.assert_awaited_once_with(selection.dataset.id, since)
    selection.to_repair.assert_not_awaited()
    assert selection.events == []


@pytest.mark.parametrize(
    "pending", [([], []), (["unfinished-write"], []), ([], ["missing-subject-conflict"])]
)
@pytest.mark.asyncio
async def test_empty_touched_set_checks_retry_work_before_pipeline(selection, pending):
    selection.to_repair.return_value = StoredConflicts({}, *pending)

    async def pipeline(*args, **kwargs):
        assert selection.events == ["lock", "context", "close context", "unlock"]
        assert kwargs["entity_ids"] == []
        return ConflictReviewResult(object(), [])

    selection.pipeline.side_effect = pipeline
    review = await selection.review(
        StageWatermark(uuid4(), datetime(2025, 1, 1, tzinfo=timezone.utc), {})
    )
    selection.to_repair.assert_awaited_once_with(
        selection.graph, selection.dataset.id, selection.dataset_node_ids
    )
    if any(pending):
        assert review.run_result is not None
        selection.pipeline.assert_awaited_once()
    else:
        assert review is None
        assert module.build_stamp_fields(review) == {}
        selection.pipeline.assert_not_awaited()


@pytest.mark.asyncio
async def test_previous_failures_are_reviewed_again_without_new_writes(selection):
    since = datetime(2025, 1, 1, tzinfo=timezone.utc)
    await selection.review(StageWatermark(uuid4(), since, {"retry_entity_ids": ["alice"]}))
    selection.pipeline.assert_awaited_once_with(
        selection.dataset.id,
        selection.user,
        entity_ids=["alice"],
        since=None,  # alice's older facts were never judged, so nothing is narrowed
    )
    selection.to_repair.assert_not_awaited()


@pytest.mark.asyncio
async def test_malformed_retry_ids_are_ignored(selection):
    await selection.review(
        StageWatermark(
            uuid4(), datetime(2025, 1, 1, tzinfo=timezone.utc), {"retry_entity_ids": "alice"}
        )
    )
    selection.to_repair.assert_awaited_once()


def test_mass_failure_withholds_the_stamp(monkeypatch):
    monkeypatch.setattr(module, "REVIEW_RETRY_ENTITY_LIMIT", 1)
    review = ConflictReviewResult(object(), ["alice", "bob"])
    assert module.build_stamp_fields(review) is None
