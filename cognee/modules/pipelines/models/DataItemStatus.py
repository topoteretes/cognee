import enum
from typing import Any


class DataItemStatus(str, enum.Enum):
    DATA_ITEM_PROCESSING_COMPLETED = "DATA_ITEM_PROCESSING_COMPLETED"


def is_data_item_completed(status_value: Any) -> bool:
    """Whether a per-item ``pipeline_status`` slot records a completed run.

    The slot (``Data.pipeline_status[pipeline_name][str(dataset_id)]``) has two
    accepted representations: the legacy bare string/enum value, and a dict
    carrying the status under its ``"status"`` key (the shape planned for the
    hash-aware rollout, so it can grow fields like ``content_hash`` without
    breaking readers). Anything else — a missing slot, ``None``, an unrelated
    string, or a dict without a completed status — is not completed.
    """
    if isinstance(status_value, dict):
        status_value = status_value.get("status")
    return status_value == DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED


def is_data_item_completed_in(pipeline_status: Any, pipeline_name: str, dataset_id: Any) -> bool:
    """Whether a row's ``pipeline_status`` records ``pipeline_name`` completed for the dataset."""
    status_for_pipeline = (pipeline_status or {}).get(pipeline_name) or {}
    return is_data_item_completed(status_for_pipeline.get(str(dataset_id)))
