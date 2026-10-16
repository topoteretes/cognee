from typing import Any
from uuid import UUID

from pydantic import BaseModel, computed_field

from cognee.modules.data.models.Data import Data


def extract_data_ids(data_ingestion_info: Any) -> list[UUID]:
    """Ids of the data items a run processed, read from its per-item results.

    ``run_tasks`` reports one ``{"run_info": ..., "data_id": ...}`` entry per
    data item. Entries whose item errored are skipped (they identify nothing
    that was stored); entries reporting ``PipelineRunAlreadyCompleted`` are
    kept — the row exists and holds the content, which is exactly what a
    caller re-adding known content wants back. Order follows the run's
    results; duplicates and ids that are not UUIDs are dropped.
    """
    if not isinstance(data_ingestion_info, list):
        return []

    data_ids: list[UUID] = []
    seen: set[UUID] = set()
    for entry in data_ingestion_info:
        if not isinstance(entry, dict):
            continue
        status = getattr(entry.get("run_info"), "status", "") or ""
        if "Errored" in status:
            continue
        raw_id = entry.get("data_id")
        if raw_id is None:
            continue
        try:
            data_id = raw_id if isinstance(raw_id, UUID) else UUID(str(raw_id))
        except (ValueError, TypeError, AttributeError):
            continue
        if data_id in seen:
            continue
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
    # entry per data item the run handled. Read ``data_ids`` instead of
    # walking this.
    data_ingestion_info: list | None = None

    model_config = {
        "arbitrary_types_allowed": True,
        "from_attributes": True,
        # Add custom encoding handler for Data ORM model
        "json_encoders": {Data: lambda d: d.to_json()},
    }

    @computed_field(return_type=list[UUID])
    @property
    def data_ids(self) -> list[UUID]:
        """Ids of the ``Data`` rows this run processed, in result order.

        For ``add()`` this is the id of every item that was stored (or already
        existed with the same content — dedup returns the existing row), so a
        caller can key later ``update()`` / ``delete_data()`` / ``find_data``
        calls on cognee's own ids instead of re-deriving them. Empty for run
        infos that carry no per-item results (``PipelineRunStarted``,
        progress ticks) and for items that errored. Serialized with the model,
        so ``POST /api/v1/add`` responses carry it too.
        """
        return extract_data_ids(self.data_ingestion_info)


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
