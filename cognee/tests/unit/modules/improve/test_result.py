"""StageResult / ImproveResult: statuses come from PipelineRunInfo, skipped needs a reason."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from cognee.infrastructure.llm.exceptions import LLMPaymentRequiredError
from cognee.modules.improve import (
    REASON_BUDGET_EXHAUSTED,
    ImproveResult,
    StageResult,
)
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


BUDGET_SENTENCE = "Budget has been exceeded! Current cost: 20.0, Max budget: 20.0"


def test_budget_exhausted_reads_the_exception_of_a_stage_that_raised():
    raised = StageResult.errored("s", LLMPaymentRequiredError())
    assert raised.exception is not None
    assert raised.budget_exhausted is True

    # The provider's own error, not yet converted, and one level down a chain.
    wrapper = RuntimeError("stage failed")
    wrapper.__cause__ = Exception(f"litellm.RateLimitError: {BUDGET_SENTENCE}")
    assert StageResult.errored("s", wrapper).budget_exhausted is True

    # A classification for the orchestrator, not a field of the reported result.
    assert "budget_exhausted" not in raised.model_dump(mode="json")


def test_budget_exhausted_reads_the_run_info_of_a_stage_that_did_not_raise():
    """The wrapped pipeline reported PipelineRunErrored: there is no exception,
    only the run's error class and the error text taken from its message."""
    converted = _info(
        PipelineRunErrored,
        error_class="LLMPaymentRequiredError",
        error_message="LLMPaymentRequiredError: LLM provider requires payment or token budget "
        "is exhausted. (Status code: 402)",
    )
    by_class = StageResult.from_pipeline_run("s", {converted.dataset_id: converted})
    assert by_class.exception is None
    assert by_class.budget_exhausted is True

    provider_error = _info(
        PipelineRunErrored,
        error_class="RateLimitError",
        error_message=f"litellm.RateLimitError: {BUDGET_SENTENCE}",
    )
    by_message = StageResult.from_pipeline_run("s", provider_error)
    assert by_message.exception is None
    assert by_message.budget_exhausted is True

    # A stage that reports a failure as plain text has only the text.
    assert StageResult.errored("s", f"LLM budget exhausted: {BUDGET_SENTENCE}").budget_exhausted


def test_budget_exhausted_is_false_for_any_other_outcome():
    assert StageResult.errored("s", ValueError("nope")).budget_exhausted is False
    assert StageResult.errored("s", "embedding backend down").budget_exhausted is False
    unrelated = _info(
        PipelineRunErrored, error_class="AuthenticationError", error_message="invalid api key"
    )
    assert StageResult.from_pipeline_run("s", unrelated).budget_exhausted is False
    # Only an errored stage can be the one that hit the budget: the stages
    # skipped after it carry the reason but did not run.
    assert StageResult.completed("s").budget_exhausted is False
    assert StageResult.skipped("s", REASON_BUDGET_EXHAUSTED).budget_exhausted is False


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
    assert mixed.stage("b").reason == "x"
    assert mixed.stage("zzz") is None
    assert mixed.stage_summary() == "a=completed,b=skipped,c=errored"

    running = ImproveResult(background=True, finished=False)
    assert running.status == "running"

    all_skipped = ImproveResult(
        stages=[StageResult.skipped("a", "x"), StageResult.skipped("b", "y")]
    )
    assert all_skipped.status == "skipped"


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
