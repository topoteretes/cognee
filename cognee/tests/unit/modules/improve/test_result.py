"""StageResult / ImproveResult: statuses come from PipelineRunInfo, skipped needs a reason."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from cognee.modules.improve import REASON_LOCK_HELD, ImproveResult, StageResult
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunAlreadyCompleted,
    PipelineRunCompleted,
    PipelineRunErrored,
    PipelineRunStarted,
)


def _info(cls, **extra):
    return cls(pipeline_run_id=uuid4(), dataset_id=uuid4(), dataset_name="d", **extra)


def test_skipped_requires_a_reason():
    with pytest.raises(ValidationError):
        StageResult(stage="x", status="skipped")
    assert StageResult.skipped("x", "why").reason == "why"


def test_status_vocabulary_is_closed():
    with pytest.raises(ValidationError):
        StageResult(stage="x", status="running")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "cls,expected",
    [
        (PipelineRunCompleted, "completed"),
        (PipelineRunAlreadyCompleted, "already_completed"),
        (PipelineRunErrored, "errored"),
        (PipelineRunStarted, "completed"),
    ],
)
def test_from_pipeline_run_maps_status_from_run_info(cls, expected):
    info = _info(cls)
    result = StageResult.from_pipeline_run("s", {info.dataset_id: info})
    assert result.status == expected
    assert result.run is info
    assert result.raw_run == {info.dataset_id: info}


def test_from_pipeline_run_carries_error_message():
    info = _info(PipelineRunErrored, error_class="Boom", error_message="it broke")
    result = StageResult.from_pipeline_run("s", info)
    assert result.status == "errored"
    assert result.error == "it broke"


def test_from_pipeline_run_without_run_info_is_completed():
    result = StageResult.from_pipeline_run("s", {})
    assert result.status == "completed"
    assert result.run is None


def test_improve_result_status_summary():
    empty = ImproveResult()
    assert empty.status == "completed"

    mixed = ImproveResult(
        stages=[
            StageResult.completed("a"),
            StageResult.skipped("b", "x"),
            StageResult.errored("c", "e"),
        ]
    )
    assert mixed.status == "errored"
    assert [s.stage for s in mixed.ok] == ["a"]
    assert mixed.stage("b").reason == "x"
    assert mixed.stage("zzz") is None
    assert mixed.stage_summary() == "a=completed,b=skipped,c=errored"

    running = ImproveResult(background=True, finished=False)
    assert running.status == "running"

    all_skipped = ImproveResult.all_skipped(["a", "b"], REASON_LOCK_HELD, session_ids=["s"])
    assert all_skipped.status == "skipped"
    assert all_skipped.lock_held is True
    assert all_skipped.session_ids == ["s"]
    assert all_skipped.to_legacy_dict() == {}


def test_model_dump_includes_status_and_serializes_run_info():
    info = _info(PipelineRunCompleted)
    result = ImproveResult(
        dataset_id=uuid4(),
        stages=[StageResult.from_pipeline_run("triplet_enrichment", {info.dataset_id: info})],
        memify_run={},
    )
    dumped = result.model_dump(mode="json")
    assert dumped["status"] == "completed"
    assert dumped["stages"][0]["run"]["status"] == "PipelineRunCompleted"
    assert "raw_run" not in dumped["stages"][0]


@pytest.mark.asyncio
async def test_wait_is_a_noop_for_foreground_results():
    result = ImproveResult()
    assert await result.wait() is result
