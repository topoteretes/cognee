from uuid import UUID
from typing import Union

from cognee.modules.pipelines.models.DataItemStatus import DataItemStatus


def is_data_item_cognified(data, pipeline_name: str, dataset_id: Union[UUID, str]) -> bool:
    """True if ``data`` has been fully processed by ``pipeline_name`` for ``dataset_id``.

    Reads the per-item ``pipeline_status`` marker (written only on full-item success,
    so it is genuinely all-or-nothing): ``pipeline_status[pipeline_name][dataset_id]
    == DATA_ITEM_PROCESSING_COMPLETED``. Shared by the incremental skip/progress logic
    and the dataset-data endpoint so the two can't drift.
    """
    return (getattr(data, "pipeline_status", None) or {}).get(pipeline_name, {}).get(
        str(dataset_id)
    ) == DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED
