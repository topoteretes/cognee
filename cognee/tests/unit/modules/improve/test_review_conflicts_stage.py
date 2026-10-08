"""Conflict review gates work before memify and retain retries after failures."""

import importlib
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.modules.improve import (
    GraphCapabilities,
    ImproveConfig,
    ImproveRunInputs,
    graph_changes,
    stages,
)
from cognee.modules.improve.stage import execute_stage
from cognee.modules.pipelines.models import PipelineRunCompleted, PipelineRunErrored


@pytest.fixture
def review(monkeypatch):
    user = SimpleNamespace(id=uuid4())
    dataset = SimpleNamespace(id=uuid4(), owner_id=user.id)
    inputs = ImproveRunInputs(
        user=user,
        dataset_id=dataset.id,
        dataset=dataset,
        session_ids=(),
        config=ImproveConfig(),
        capabilities=GraphCapabilities.assume_supported(),
        review_conflicts=True,
        improve_operation_id=uuid4(),
    )
    watermark = AsyncMock(return_value=None)
    touched = AsyncMock(return_value=set())
    lost = AsyncMock(return_value=([], []))
    pipeline = AsyncMock(
        return_value=PipelineRunCompleted(
            pipeline_run_id=uuid4(), dataset_id=dataset.id, dataset_name="docs", status="completed"
        )
    )
    monkeypatch.setattr(graph_changes, "last_improve_watermark", watermark)
    evidence = importlib.import_module("cognee.modules.provenance.edge_evidence.lookup")
    reader = importlib.import_module("cognee.tasks.memify.review_conflicts.read_facts")
    wrapper = importlib.import_module("cognee.memify_pipelines.review_conflicts")
    monkeypatch.setattr(evidence, "get_touched_entity_ids", touched)
    monkeypatch.setattr(reader, "find_conflicts_with_lost_citations", lost)
    monkeypatch.setattr(wrapper, "review_conflicts_pipeline", pipeline)
    monkeypatch.setattr(stages, "llm_available", lambda: True)
    config = importlib.import_module("cognee.modules.provenance.edge_evidence.config")
    monkeypatch.setattr(
        config, "get_provenance_config", lambda: SimpleNamespace(edge_evidence_enabled=True)
    )
    graph = object()
    graph_module = importlib.import_module("cognee.infrastructure.databases.graph")
    monkeypatch.setattr(graph_module, "get_graph_engine", AsyncMock(return_value=graph))
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

    monkeypatch.setattr(
        importlib.import_module("cognee.infrastructure.locks.dataset_lock"), "dataset_lock", lock
    )
    monkeypatch.setattr(
        importlib.import_module("cognee.context_global_variables"),
        "set_database_global_context_variables",
        context,
    )
    return SimpleNamespace(
        inputs=inputs,
        watermark=watermark,
        touched=touched,
        lost=lost,
        pipeline=pipeline,
        events=events,
        graph=graph,
        evidence_config=config,
    )


@pytest.mark.parametrize(
    "reason",
    ["opt_in_disabled", "no_llm_configured", "edge_evidence_disabled", "disabled_by_config"],
)
@pytest.mark.asyncio
async def test_gates_do_no_pipeline_work(review, monkeypatch, reason):
    from dataclasses import replace

    inputs = review.inputs
    if reason == "opt_in_disabled":
        inputs = replace(inputs, review_conflicts=False)
    elif reason == "no_llm_configured":
        monkeypatch.setattr(stages, "llm_available", lambda: False)
    elif reason == "edge_evidence_disabled":
        monkeypatch.setattr(
            review.evidence_config,
            "get_provenance_config",
            lambda: SimpleNamespace(edge_evidence_enabled=False),
        )
    else:
        inputs = replace(inputs, config=ImproveConfig(stages_disabled=["review_conflicts"]))
    result = await execute_stage(stages.ReviewConflictsStage(), inputs)
    assert (result.status, result.reason) == ("skipped", reason)
    review.watermark.assert_not_awaited()
    review.pipeline.assert_not_awaited()


@pytest.mark.asyncio
async def test_first_review_reads_all_entities_without_precheck(review):
    result = await execute_stage(stages.ReviewConflictsStage(), review.inputs)
    assert result.status == "completed"
    assert result.counts == {"entities_failed": 0}
    assert result.run_info_stamp["review_conflicts"]["status"] == "completed"
    review.pipeline.assert_awaited_once_with(
        dataset=review.inputs.dataset_id, user=review.inputs.user, entity_ids=None, since=None
    )
    review.watermark.assert_awaited_once_with(
        review.inputs.dataset_id,
        "review_conflicts",
        exclude_operation_id=review.inputs.improve_operation_id,
    )
    review.touched.assert_not_awaited()
    review.lost.assert_not_awaited()


@pytest.mark.asyncio
async def test_touched_entities_skip_graph_precheck(review):
    since = datetime(2025, 1, 1, tzinfo=timezone.utc)
    review.watermark.return_value = graph_changes.ImproveWatermark(uuid4(), since, {})
    review.touched.return_value = {"b", "a"}
    await execute_stage(stages.ReviewConflictsStage(), review.inputs)
    review.pipeline.assert_awaited_once_with(
        dataset=review.inputs.dataset_id,
        user=review.inputs.user,
        entity_ids=["a", "b"],
        since=since,
    )
    review.lost.assert_not_awaited()
    assert review.events == []


@pytest.mark.parametrize(
    "pending", [([], []), ([{"review_pending": True}], []), ([], ["missing-subject-conflict"])]
)
@pytest.mark.asyncio
async def test_empty_touched_set_checks_retry_work_before_pipeline(review, pending):
    review.watermark.return_value = graph_changes.ImproveWatermark(
        uuid4(), datetime(2025, 1, 1, tzinfo=timezone.utc), {}
    )
    review.lost.return_value = pending

    async def pipeline(**kwargs):
        assert review.events == ["lock", "context", "close context", "unlock"]
        assert kwargs["entity_ids"] == []
        return PipelineRunCompleted(
            pipeline_run_id=uuid4(),
            dataset_id=review.inputs.dataset_id,
            dataset_name="docs",
            status="completed",
        )

    review.pipeline.side_effect = pipeline
    result = await execute_stage(stages.ReviewConflictsStage(), review.inputs)
    assert result.run_info_stamp is not None
    if any(pending):
        assert result.status == "completed"
        review.pipeline.assert_awaited_once()
    else:
        assert (result.status, result.reason) == ("already_completed", "no_new_facts")
        review.pipeline.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_llm_entities_are_reported_and_completed_run_stamps(review, caplog):
    review.pipeline.return_value.payload = {"unreviewed_entity_ids": ["alice"]}
    result = await execute_stage(stages.ReviewConflictsStage(), review.inputs)
    assert result.status == "completed"
    assert result.counts == {"entities_failed": 1}
    assert result.run_info_stamp["review_conflicts"]["retry_entity_ids"] == ["alice"]
    assert "alice" in caplog.text


@pytest.mark.asyncio
async def test_previous_failures_are_reviewed_again_without_new_writes(review):
    since = datetime(2025, 1, 1, tzinfo=timezone.utc)
    review.watermark.return_value = graph_changes.ImproveWatermark(
        uuid4(), since, {"retry_entity_ids": ["alice"]}
    )
    review.touched.return_value = set()
    await execute_stage(stages.ReviewConflictsStage(), review.inputs)
    review.pipeline.assert_awaited_once_with(
        dataset=review.inputs.dataset_id,
        user=review.inputs.user,
        entity_ids=["alice"],
        since=since,
    )
    review.lost.assert_not_awaited()


@pytest.mark.asyncio
async def test_malformed_retry_ids_are_ignored(review):
    review.watermark.return_value = graph_changes.ImproveWatermark(
        uuid4(), datetime(2025, 1, 1, tzinfo=timezone.utc), {"retry_entity_ids": "alice"}
    )
    review.touched.return_value = set()
    await execute_stage(stages.ReviewConflictsStage(), review.inputs)
    review.lost.assert_awaited_once()


@pytest.mark.asyncio
async def test_mass_failure_withholds_the_stamp(review, monkeypatch):
    monkeypatch.setattr(stages, "REVIEW_RETRY_ENTITY_LIMIT", 1)
    review.pipeline.return_value.payload = {"unreviewed_entity_ids": ["alice", "bob"]}
    result = await execute_stage(stages.ReviewConflictsStage(), review.inputs)
    assert result.status == "completed"
    assert result.run_info_stamp is None


@pytest.mark.parametrize("budget", [False, True])
@pytest.mark.asyncio
async def test_pipeline_errors_never_stamp_and_keep_budget_classification(review, budget):
    review.pipeline.return_value = PipelineRunErrored(
        pipeline_run_id=uuid4(),
        dataset_id=review.inputs.dataset_id,
        dataset_name="docs",
        status="errored",
        error_class="LLMPaymentRequiredError" if budget else "RuntimeError",
        error_message="write failed",
    )
    result = await execute_stage(stages.ReviewConflictsStage(), review.inputs)
    assert result.status == "errored"
    assert result.run_info_stamp is None
    assert result.budget_exhausted is budget
