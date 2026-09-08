"""Item-level completion counts for a dataset's incremental processing.

The dataset-level status (``pipeline_runs``) says whether a run is going or
done; this answers the operator's other question — how many of the dataset's
items carry the per-item completion stamp for a pipeline, and how many are
still pending — without reaching into the database by hand. On request it
also names the items, so "which ones?" needs no second query.
"""

from uuid import UUID

from .get_dataset_data import get_dataset_data
from .publish_updated_data import COGNIFY_PIPELINE_NAME


async def get_dataset_processing_status(
    dataset_id: UUID,
    pipeline_name: str = COGNIFY_PIPELINE_NAME,
    include_items: bool = False,
) -> dict:
    """Count a dataset's items by per-item completion for ``pipeline_name``.

    Returns ``{"total": int, "completed": int, "pending": int}`` where
    ``pending = total - completed``. An item is completed when its
    ``pipeline_status[pipeline_name][str(dataset_id)]`` slot satisfies
    :func:`is_data_item_completed` (legacy string or dict representation).
    Rows with a ``NULL`` ``pipeline_status`` count as pending. Per-item
    errored state is not persisted today, so it is not reported.

    With ``include_items=True`` the result also carries
    ``"items": [{"id": UUID, "name": str, "completed": bool}, ...]`` in the
    same order ``get_dataset_data`` returns them (the order
    ``GET /datasets/{id}/data`` shows), one entry per row counted above.
    """
    # Imported on use: the pipeline layer imports this package at module
    # scope, so a top-level import here would close a cycle (see
    # publish_updated_data._completed_status for the full story).
    from cognee.modules.pipelines.models.DataItemStatus import is_data_item_completed

    data_items = await get_dataset_data(dataset_id)
    dataset_key = str(dataset_id)

    items = []
    for item in data_items:
        status_for_pipeline = (item.pipeline_status or {}).get(pipeline_name) or {}
        items.append(
            {
                "id": item.id,
                "name": item.name,
                "completed": is_data_item_completed(status_for_pipeline.get(dataset_key)),
            }
        )

    total = len(items)
    completed = sum(1 for item in items if item["completed"])
    result = {"total": total, "completed": completed, "pending": total - completed}
    if include_items:
        result["items"] = items
    return result
