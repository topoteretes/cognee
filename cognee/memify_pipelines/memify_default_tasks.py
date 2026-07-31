from cognee.shared.logging_utils import get_logger
from cognee.modules.pipelines.tasks.task import Task
from cognee.tasks.memify.get_triplet_datapoints import get_triplet_datapoints
from cognee.tasks.memify.extract_user_sessions import extract_user_sessions
from cognee.tasks.memify.cognify_session import cognify_session
from cognee.tasks.storage.index_data_points import index_data_points

logger = get_logger("memify_default_tasks")


def get_default_memify_extraction_tasks():
    from cognee.modules.cognify.config import get_cognify_config

    if not get_cognify_config().triplet_embedding:
        return []

    # This walks and re-embeds every triplet in the graph, so it bills per
    # existing triplet rather than per newly stored one. Say so: the automatic
    # path (remember -> improve -> memify) reaches it without the caller ever
    # naming it, and the cost only shows up on the embedding provider's bill.
    logger.info(
        "memify: TRIPLET_EMBEDDING is enabled, so enrichment re-embeds every triplet in "
        "the graph — cost grows with total graph size, not with what was just stored. "
        "Pass self_improvement=False to remember() to skip this after a write."
    )
    return [Task(get_triplet_datapoints, triplets_batch_size=100)]


def get_default_memify_enrichment_tasks():
    return [Task(index_data_points, task_config={"batch_size": 100})]


def get_session_memify_tasks():
    """Return (extraction_tasks, enrichment_tasks) for session cognification."""
    return (
        [Task(extract_user_sessions)],
        [Task(cognify_session)],
    )
