from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, computed_field

from cognee.modules.data.models.Data import Data
from cognee.modules.pipelines.utils import iter_ingestion_entries


def extract_data_ids(data_ingestion_info: Any) -> list[UUID]:
    """Ids of the data items a run processed, read from its per-item results.

    ``run_tasks`` reports one ``{"run_info": ..., "data_id": ...}`` entry per
    data item. Entries whose item errored are skipped (they identify nothing
    that was stored); entries reporting ``PipelineRunAlreadyCompleted`` are
    kept — the row exists and holds the content, which is exactly what a
    caller re-adding known content wants back. Order follows the run's
    results; duplicates and ids that are not UUIDs are dropped.
    """
    data_ids: list[UUID] = []
    seen: set[UUID] = set()
    for status, entry_data_ids in iter_ingestion_entries(data_ingestion_info):
        if status == "PipelineRunErrored":
            continue
        for data_id in entry_data_ids:
            if data_id not in seen:
                seen.add(data_id)
                data_ids.append(data_id)
    return data_ids


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
    # and "data_label", so ids can be mapped back to inputs. Read ``data_ids``
    # for just the ids.
    data_ingestion_info: list | None = None

    model_config = {
        "arbitrary_types_allowed": True,
        "from_attributes": True,
        # Add custom encoding handler for Data ORM model
        "json_encoders": {Data: lambda d: d.to_json()},
    }

    @computed_field(return_type=list[UUID] | None, exclude_if=lambda value: value is None)
    @property
    def data_ids(self) -> list[UUID] | None:
        """Ids of the ``Data`` rows this run processed, in result order.

        For ``add()`` this is the id of every item that was stored (or already
        existed with the same content — dedup returns the existing row), so a
        caller can key later ``update()`` / ``delete_data()`` / ``find_data``
        calls on cognee's own ids instead of re-deriving them. Items that
        errored are left out. ``None`` for run infos that carry no per-item
        results (``PipelineRunStarted``, progress ticks, the run infos nested
        in ``data_ingestion_info``), which serialize without the key.
        Serialized with the model, so ``POST /api/v1/add`` responses carry it
        too.
        """
        if self.data_ingestion_info is None:
            return None
        return extract_data_ids(self.data_ingestion_info)


class PipelineRunStarted(PipelineRunInfo):
    status: str = "PipelineRunStarted"


class PipelineRunYield(PipelineRunInfo):
    status: str = "PipelineRunYield"


class PipelineRunCompleted(PipelineRunInfo):
    status: str = "PipelineRunCompleted"

    # Set when the step after the completed run failed (``AfterRunHookError``:
    # the document structure pass or the vector store compaction); the run's data
    # is stored. PII-scrubbed. Left out of the serialized run when unset.
    after_run_error: str | None = Field(default=None, exclude_if=lambda value: value is None)


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


def get_after_run_failures(result) -> list[tuple[UUID, str, str]]:
    """``(dataset_id, dataset_name, after_run_error)`` of each completed run in a
    cognify()/run_pipeline result whose step after the run failed.

    What ``cognify(raise_on_error=False)`` returns instead of raising
    ``AfterRunHookError``; read like ``get_errored_run_info``.
    """
    runs = result.values() if isinstance(result, dict) else [result]
    return [
        (run.dataset_id, run.dataset_name, run.after_run_error)
        for run in runs
        if isinstance(run, PipelineRunCompleted) and run.after_run_error
    ]
