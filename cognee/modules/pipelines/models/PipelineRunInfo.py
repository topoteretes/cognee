from typing import Any
from uuid import UUID

from pydantic import BaseModel, model_serializer

from cognee.modules.data.models.Data import Data


class PipelineRunInfo(BaseModel):
    status: str
    pipeline_run_id: UUID
    dataset_id: UUID
    dataset_name: str
    # Data must be mentioned in typing to allow custom encoders for Data to be activated
    payload: Any | list[Data] | None = None
    # Per-item results: one {"run_info": PipelineRunInfo, "data_id": UUID}
    # entry per data item the run handled. Incremental runs also add
    # "data_name", "data_location" (the caller's source; None for raw text)
    # and "data_label", so ids can be mapped back to inputs. For an add() run,
    # read ``added_data_ids`` for just the ids.
    data_ingestion_info: list | None = None
    # Ids of the data items an add() run stored, or found already holding the
    # same content. Despite the name, those existing data items are included:
    # nothing new was added for them, but the id still names where the caller's
    # content lives (their data_ingestion_info entry says
    # PipelineRunAlreadyCompleted). Only add() fills it; None on every other
    # pipeline's run infos and on an add() that has no per-item results yet
    # (background run).
    added_data_ids: list[UUID] | None = None

    model_config = {
        "arbitrary_types_allowed": True,
        "from_attributes": True,
        # Add custom encoding handler for Data ORM model
        "json_encoders": {Data: lambda d: d.to_json()},
    }

    # Leave added_data_ids out unless add() filled it, so the run infos nested in
    # data_ingestion_info, progress ticks and other pipelines' results don't
    # each carry "added_data_ids": null. No return annotation on purpose: with one,
    # pydantic replaces the model's serialization schema (OpenAPI) with a dict.
    @model_serializer(mode="wrap")
    def _omit_unset_added_data_ids(self, handler):
        serialized = handler(self)
        if self.added_data_ids is None and isinstance(serialized, dict):
            serialized.pop("added_data_ids", None)
        return serialized


class PipelineRunStarted(PipelineRunInfo):
    status: str = "PipelineRunStarted"


class PipelineRunYield(PipelineRunInfo):
    status: str = "PipelineRunYield"


class PipelineRunCompleted(PipelineRunInfo):
    status: str = "PipelineRunCompleted"


class PipelineRunAlreadyCompleted(PipelineRunInfo):
    status: str = "PipelineRunAlreadyCompleted"


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
