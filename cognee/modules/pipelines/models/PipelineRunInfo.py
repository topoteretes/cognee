from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from cognee.modules.data.models.Data import Data


def _run_info_status(run_info: Any) -> str:
    """Status of a per-item ``run_info``, whether it is a run-info object or the
    dict a JSON round-trip (``model_validate(json.loads(...))``) leaves behind."""
    if isinstance(run_info, dict):
        return str(run_info.get("status") or "")
    return str(getattr(run_info, "status", "") or "")


def _to_uuid(value: Any) -> UUID | None:
    if isinstance(value, UUID):
        return value
    if value is None:
        return None
    try:
        return UUID(str(value))
    except (ValueError, TypeError, AttributeError):
        return None


def extract_data_ids(
    data_ingestion_info: Any, *, include_already_completed: bool = True
) -> list[UUID]:
    """Ids of the ``Data`` rows a run's per-item results name.

    ``run_tasks`` collects one ``{"run_info": ..., "data_id": ...}`` entry per
    work item. The ids come back in the order the entries were collected,
    deduplicated, and without the items that errored (nothing was stored for
    them). Entries reporting ``PipelineRunAlreadyCompleted`` are kept by
    default: for ``add()`` that is a dedup hit, the row that already holds the
    content. Pass ``include_already_completed=False`` to keep only the items
    this run processed itself (what ``cognify`` counts as its own work; see
    ``cognify/rollback.py``).

    Entries are read in their live form or after a dict/JSON round-trip, and
    ids that are not UUIDs are dropped.
    """
    if not isinstance(data_ingestion_info, list):
        return []

    data_ids: list[UUID] = []
    seen: set[UUID] = set()
    for entry in data_ingestion_info:
        if not isinstance(entry, dict):
            continue
        status = _run_info_status(entry.get("run_info"))
        if "Errored" in status:
            continue
        if not include_already_completed and "AlreadyCompleted" in status:
            continue
        raw_ids = entry.get("data_ids")
        if not isinstance(raw_ids, list):
            raw_ids = [entry.get("data_id")]
        for raw_id in raw_ids:
            data_id = _to_uuid(raw_id)
            if data_id is None or data_id in seen:
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
    # entry per work item the run handled.
    data_ingestion_info: list | None = None
    # Ids of the Data rows the run's per-item results name. add() fills this
    # on the run info it returns (its docstring states the exact semantics);
    # every other run info leaves it None, and a None is left out of the
    # serialized form, so the nested per-item run infos and progress ticks
    # never carry the key.
    data_ids: list[UUID] | None = Field(default=None, exclude_if=lambda value: value is None)

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
