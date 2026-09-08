"""Item-level completion counts for a dataset's incremental processing.

The dataset-level status (``pipeline_runs``) says whether a run is going or
done; this answers the operator's other question — how many of the dataset's
items carry the per-item completion stamp for a pipeline, and how many are
still pending — without reaching into the database by hand.
"""

from uuid import UUID

from .get_dataset_data import get_dataset_data
from .publish_updated_data import COGNIFY_PIPELINE_NAME


async def get_dataset_processing_status(
    dataset_id: UUID, pipeline_name: str = COGNIFY_PIPELINE_NAME
) -> dict:
    """Count a dataset's items by per-item completion for ``pipeline_name``.

    Returns ``{"total": int, "completed": int, "pending": int}`` where
    ``pending = total - completed``. An item is completed when its
    ``pipeline_status[pipeline_name][str(dataset_id)]`` slot satisfies
    :func:`is_data_item_completed` (legacy string or dict representation).
    Rows with a ``NULL`` ``pipeline_status`` count as pending. Per-item
    errored state is not persisted today, so it is not reported.
    """
    # Imported on use: the pipeline layer imports this package at module
    # scope, so a top-level import here would close a cycle (see
    # publish_updated_data._completed_status for the full story).
    from cognee.modules.pipelines.models.DataItemStatus import is_data_item_completed

    data_items = await get_dataset_data(dataset_id)
    dataset_key = str(dataset_id)

    total = len(data_items)
    completed = 0
    for item in data_items:
        status_for_pipeline = (item.pipeline_status or {}).get(pipeline_name) or {}
        if is_data_item_completed(status_for_pipeline.get(dataset_key)):
            completed += 1

    return {"total": total, "completed": completed, "pending": total - completed}
