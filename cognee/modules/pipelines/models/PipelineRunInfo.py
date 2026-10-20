from typing import Any
from uuid import UUID

from pydantic import BaseModel

from cognee.modules.data.models.Data import Data


class PipelineRunInfo(BaseModel):
    status: str
    pipeline_run_id: UUID
    dataset_id: UUID
    dataset_name: str
    # Data must be mentioned in typing to allow custom encoders for Data to be activated
    payload: Any | list[Data] | None = None
    data_ingestion_info: list | None = None
    # The dataset's memory state once this event has happened, set on the
    # milestones of cognify()'s chained runs (see
    # cognee.modules.cognify.memory_pipelines): "cached", "caching_failed",
    # "indexed", "indexing_failed", "ready" or "extraction_failed". None on
    # every other event.
    memory_state: str | None = None
    # Set by a chained execution (cognify()'s caching, indexing and cognify
    # runs) on the events of its later runs: the id of the run the chain
    # started with, which is the id a caller was handed. The background
    # executor delivers such an event to that run's subscribers as well.
    chain_run_id: UUID | None = None

    model_config = {
        "arbitrary_types_allowed": True,
        "from_attributes": True,
        # Add custom encoding handler for Data ORM model
        "json_encoders": {Data: lambda d: d.to_json()},
    }


class PipelineRunStarted(PipelineRunInfo):
    status: str = "PipelineRunStarted"


class PipelineRunYield(PipelineRunInfo):
    status: str = "PipelineRunYield"


class PipelineRunCompleted(PipelineRunInfo):
    status: str = "PipelineRunCompleted"


class PipelineRunAlreadyCompleted(PipelineRunInfo):
    status: str = "PipelineRunAlreadyCompleted"


class PipelineRunStageCompleted(PipelineRunInfo):
    """A preparation run of cognify() completed; the dataset's memory moved up a state.

    cognify() builds memory in three runs (caching, indexing, cognify). Its
    chained runner yields this in place of the caching and indexing runs'
    PipelineRunCompleted, so a consumer that stops at "completed" (the
    /cognify/subscribe websocket) keeps following into the run that comes
    next. ``pipeline_name`` names the run that completed, ``memory_state`` the
    state reached ("cached", "indexed"); ``data_ingestion_info`` is that run's.
    """

    status: str = "PipelineRunStageCompleted"
    pipeline_name: str | None = None


class PipelineRunErrored(PipelineRunInfo):
    status: str = "PipelineRunErrored"

    # Failure detail so callers (remember(), cognify(raise_on_error=True), the
    # recall warm-up marker, MCP cognify_status) can say WHAT failed instead of
    # just "errored". error_message is PII-scrubbed; payload keeps the legacy
    # repr for backward compatibility.
    error_class: str | None = None
    error_message: str | None = None


class PipelineRunProgress(PipelineRunInfo):
    status: str = "PipelineRunProgress"
    completed_items: int | None = None
    total_items: int | None = None
    current_stage: str | None = None
    stage_index: int | None = None
    stage_total: int | None = None


def get_errored_run_info(result) -> PipelineRunErrored | None:
    """First ``PipelineRunErrored`` in a cognify()/run_pipeline result, or None.

    Blocking pipeline executors return ``{dataset_id: PipelineRunInfo}`` (or a
    bare run info); callers that pass ``raise_on_error=False`` use this to tell
    a failed build apart from a completed one.
    """
    if isinstance(result, PipelineRunErrored):
        return result
    if isinstance(result, dict):
        for run_info in result.values():
            if isinstance(run_info, PipelineRunErrored):
                return run_info
    return None
