"""The conflict review stage gates work before memify and stamps only a completed review."""

import importlib
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.memify_pipelines.review_conflicts_pipeline import ConflictReviewResult
from cognee.modules.improve import (
    GraphCapabilities,
    ImproveConfig,
    ImproveRunInputs,
    stages,
    watermarks,
)
from cognee.modules.improve.stage import execute_stage
from cognee.modules.pipelines.models import PipelineRunCompleted, PipelineRunErrored

wrapper = importlib.import_module("cognee.modules.improve.review_conflicts")


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
    completed = PipelineRunCompleted(
        pipeline_run_id=uuid4(), dataset_id=dataset.id, dataset_name="docs", status="completed"
    )
    watermark = AsyncMock(return_value=None)
    review_conflicts = AsyncMock(return_value=ConflictReviewResult(completed, []))
    monkeypatch.setattr(watermarks, "last_improve_watermark", watermark)
    monkeypatch.setattr(wrapper, "review_conflicts", review_conflicts)
    monkeypatch.setattr(stages, "llm_available", lambda: True)
    config = importlib.import_module("cognee.modules.provenance.edge_evidence.config")
    monkeypatch.setattr(
        config, "get_provenance_config", lambda: SimpleNamespace(edge_evidence_enabled=True)
    )
    return SimpleNamespace(
        inputs=inputs,
        completed=completed,
        watermark=watermark,
        review_conflicts=review_conflicts,
        evidence_config=config,
    )


@pytest.mark.parametrize(
    "reason",
    ["turned_off", "no_llm_configured", "edge_evidence_disabled", "disabled_by_config"],
)
@pytest.mark.asyncio
async def test_gates_do_no_pipeline_work(review, monkeypatch, reason):
    from dataclasses import replace

    inputs = review.inputs
    if reason == "turned_off":
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
    review.review_conflicts.assert_not_awaited()


@pytest.mark.asyncio
async def test_completed_review_reads_and_stamps_the_watermark(review):
    watermark = watermarks.StageWatermark(uuid4(), datetime(2025, 1, 1, tzinfo=timezone.utc), {})
    review.watermark.return_value = watermark
    result = await execute_stage(stages.ReviewConflictsStage(), review.inputs)
    assert result.status == "completed"
    assert result.counts == {"entities_failed": 0, "entities_skipped": 0}
    assert result.run_info_stamp["review_conflicts"]["status"] == "completed"
    review.watermark.assert_awaited_once_with(
        review.inputs.dataset_id,
        "review_conflicts",
        exclude_operation_id=review.inputs.improve_operation_id,
    )
    review.review_conflicts.assert_awaited_once_with(
        watermark, dataset=review.inputs.dataset, user=review.inputs.user
    )


@pytest.mark.asyncio
async def test_nothing_to_review_is_already_completed_and_stamps(review):
    review.review_conflicts.return_value = None
    result = await execute_stage(stages.ReviewConflictsStage(), review.inputs)
    assert (result.status, result.reason) == ("already_completed", "no_new_facts")
    stamp = result.run_info_stamp["review_conflicts"]
    assert stamp["status"] == "already_completed"
    assert "retry_entity_ids" not in stamp


@pytest.mark.asyncio
async def test_failed_entities_are_counted_and_stamped_for_retry(review):
    review.review_conflicts.return_value = ConflictReviewResult(review.completed, ["alice"])
    result = await execute_stage(stages.ReviewConflictsStage(), review.inputs)
    assert result.status == "completed"
    assert result.counts == {"entities_failed": 1, "entities_skipped": 0}
    assert result.run_info_stamp["review_conflicts"]["retry_entity_ids"] == ["alice"]


@pytest.mark.asyncio
async def test_skipped_entities_are_counted_but_not_stamped_for_retry(review):
    """A subject too large for one call would only be skipped again."""
    review.review_conflicts.return_value = ConflictReviewResult(review.completed, [], ["alice"])
    result = await execute_stage(stages.ReviewConflictsStage(), review.inputs)
    assert result.status == "completed"
    assert result.counts == {"entities_failed": 0, "entities_skipped": 1}
    assert result.run_info_stamp["review_conflicts"]["retry_entity_ids"] == []


@pytest.mark.asyncio
async def test_withheld_stamp_is_not_written(review, monkeypatch):
    monkeypatch.setattr(wrapper, "REVIEW_RETRY_ENTITY_LIMIT", 1)
    review.review_conflicts.return_value = ConflictReviewResult(review.completed, ["alice", "bob"])
    result = await execute_stage(stages.ReviewConflictsStage(), review.inputs)
    assert result.status == "completed"
    assert result.counts == {"entities_failed": 2, "entities_skipped": 0}
    assert result.run_info_stamp is None


@pytest.mark.parametrize("budget", [False, True])
@pytest.mark.asyncio
async def test_pipeline_errors_never_stamp_and_keep_budget_classification(review, budget):
    errored = PipelineRunErrored(
        pipeline_run_id=uuid4(),
        dataset_id=review.inputs.dataset_id,
        dataset_name="docs",
        status="errored",
        error_class="LLMPaymentRequiredError" if budget else "RuntimeError",
        error_message="write failed",
    )
    review.review_conflicts.return_value = ConflictReviewResult(errored, [])
    result = await execute_stage(stages.ReviewConflictsStage(), review.inputs)
    assert result.status == "errored"
    assert result.run_info_stamp is None
    assert result.budget_exhausted is budget
