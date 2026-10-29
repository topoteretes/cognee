"""Choose what each conflict review covers, from the last review's watermark.

``review_conflicts`` is what the improve stage runs. It reviews the entities a completed
write touched since the last review started, and the entities that review could not
review; with neither, it runs only when a stored conflict needs repair. With no watermark
yet, it reviews every entity. ``build_stamp_fields`` says what the next watermark records.
"""

from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.locks.dataset_lock import dataset_lock
from cognee.memify_pipelines.review_conflicts_pipeline import (
    ConflictReviewResult,
    review_conflicts_pipeline,
)
from cognee.modules.data.models import Dataset
from cognee.modules.provenance.edge_evidence.lookup import get_touched_entity_ids
from cognee.modules.users.models import User
from cognee.tasks.memify.review_conflicts.read_facts import (
    read_dataset_node_ids,
    read_stored_conflicts,
)

from .watermarks import StageWatermark

# The watermark stamp field that lists the entities the next review must review again.
RETRY_ENTITY_IDS = "retry_entity_ids"

# Entity ids a review carries on its watermark stamp for the next run. Beyond this the
# stamp is withheld instead, so the next run redoes the whole window rather than carry
# a large id list in the operation row.
REVIEW_RETRY_ENTITY_LIMIT = 500


def _parse_retry_entity_ids(watermark: StageWatermark) -> list[str]:
    """Entity ids the last review stamped for retry; a malformed value counts as none."""
    ids = (watermark.stamp or {}).get(RETRY_ENTITY_IDS)
    if not isinstance(ids, list):
        return []
    return [entity_id for entity_id in ids if isinstance(entity_id, str)]


def build_stamp_fields(review: ConflictReviewResult | None) -> dict | None:
    """The fields a review adds to its watermark stamp; None withholds the stamp.

    A review that started no run (None) has nothing to retry, so it stamps no list.
    """
    if review is None:
        return {}
    if len(review.failed_entity_ids) > REVIEW_RETRY_ENTITY_LIMIT:
        return None
    return {RETRY_ENTITY_IDS: review.failed_entity_ids}


async def _has_conflicts_to_repair(dataset: Dataset) -> bool:
    """Does a stored conflict need repair: a lost citation, an unfinished write, or a subject
    that is gone or outside the dataset?"""
    # Match memify's lock/context order and release both before its wrapper
    # runs, so the dataset queue slot is never held waiting.
    async with (
        dataset_lock(dataset.id),
        set_database_global_context_variables(dataset.id, dataset.owner_id),
    ):
        graph_engine = await get_graph_engine()
        dataset_node_ids = await read_dataset_node_ids(graph_engine, str(dataset.id))
        stored_conflicts = await read_stored_conflicts(graph_engine, dataset.id, dataset_node_ids)
    return bool(stored_conflicts.ids_to_review or stored_conflicts.ids_to_drop)


async def review_conflicts(
    watermark: StageWatermark | None, *, dataset: Dataset, user: User
) -> ConflictReviewResult | None:
    """Review the entities changed since the last review, those it left to retry, and repairs.

    Returns None when nothing needed a review, so no run started.
    """
    if watermark is None:
        # Nothing has been reviewed yet, so review every entity.
        return await review_conflicts_pipeline(dataset.id, user)
    retry_ids = _parse_retry_entity_ids(watermark)
    # An entity the last review could not review is touched by no later write, so
    # only its stamped id selects it again.
    touched_ids = await get_touched_entity_ids(dataset.id, watermark.started_at)
    entity_ids = sorted(touched_ids | set(retry_ids))
    if not entity_ids and not await _has_conflicts_to_repair(dataset):
        return None
    # An entity to retry missed its last review, so its older facts were never
    # judged: a run that retries one shows every subject whole.
    since = None if retry_ids else watermark.started_at
    return await review_conflicts_pipeline(dataset.id, user, entity_ids=entity_ids, since=since)
