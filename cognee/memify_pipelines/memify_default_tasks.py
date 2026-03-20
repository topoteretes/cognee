from cognee.modules.pipelines.tasks.task import Task
from cognee.tasks.memify.get_triplet_datapoints import get_triplet_datapoints
from cognee.tasks.memify.extract_user_sessions import extract_user_sessions
from cognee.tasks.memify.cognify_session import cognify_session
from cognee.tasks.storage.index_data_points import index_data_points


def get_default_memify_extraction_tasks():
    return [Task(get_triplet_datapoints, triplets_batch_size=100)]


def get_default_memify_enrichment_tasks():
    return [Task(index_data_points, task_config={"batch_size": 100})]


def get_session_memify_tasks():
    """Return (extraction_tasks, enrichment_tasks) for session cognification."""
    return (
        [Task(extract_user_sessions)],
        [Task(cognify_session)],
    )
