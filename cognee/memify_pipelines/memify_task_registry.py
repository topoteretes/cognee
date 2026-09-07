"""Registry of memify tasks that can be selected by name.

The memify HTTP endpoint (and ``cognee.memify()``) accept task names as plain
strings. This module maps those names onto ready-to-run ``Task`` instances so
the string-based contract advertised by the API actually works.

Only tasks that are runnable without required keyword arguments are exposed
here. Tasks that need call-specific parameters (e.g. ``extract_feedback_qas``
requires ``session_ids``) stay SDK-only, where callers construct the ``Task``
themselves.
"""

from typing import Callable, Dict, List, Optional, Sequence, Union

from cognee.exceptions import CogneeValidationError
from cognee.modules.pipelines.tasks.task import Task

# Name -> zero-argument factory producing a fresh Task instance.
# Defaults mirror how the dedicated memify pipelines construct these tasks.
_MEMIFY_TASK_FACTORIES: Dict[str, Callable[[], Task]] = {}


def _build_task_factories() -> Dict[str, Callable[[], Task]]:
    from cognee.tasks.memify.extract_subgraph import extract_subgraph
    from cognee.tasks.memify.extract_subgraph_chunks import extract_subgraph_chunks
    from cognee.tasks.memify.get_triplet_datapoints import get_triplet_datapoints
    from cognee.tasks.memify.extract_user_sessions import extract_user_sessions
    from cognee.tasks.memify.cognify_session import cognify_session
    from cognee.tasks.memify.extract_agent_trace_feedbacks import extract_agent_trace_feedbacks
    from cognee.tasks.memify.cognify_agent_trace_feedback import cognify_agent_trace_feedback
    from cognee.tasks.memify.apply_feedback_weights import apply_feedback_weights
    from cognee.tasks.memify.consolidate_entities import (
        detect_entity_duplicates,
        merge_entity_duplicates,
    )
    from cognee.tasks.storage.index_data_points import index_data_points

    return {
        "extract_subgraph": lambda: Task(extract_subgraph),
        "extract_subgraph_chunks": lambda: Task(extract_subgraph_chunks),
        "get_triplet_datapoints": lambda: Task(get_triplet_datapoints, triplets_batch_size=100),
        "extract_user_sessions": lambda: Task(extract_user_sessions),
        "cognify_session": lambda: Task(cognify_session),
        "extract_agent_trace_feedbacks": lambda: Task(extract_agent_trace_feedbacks),
        "cognify_agent_trace_feedback": lambda: Task(cognify_agent_trace_feedback),
        "apply_feedback_weights": lambda: Task(
            apply_feedback_weights, task_config={"batch_size": 100}
        ),
        "detect_entity_duplicates": lambda: Task(detect_entity_duplicates),
        "merge_entity_duplicates": lambda: Task(merge_entity_duplicates),
        "index_data_points": lambda: Task(index_data_points, task_config={"batch_size": 100}),
    }


def _get_task_factories() -> Dict[str, Callable[[], Task]]:
    global _MEMIFY_TASK_FACTORIES
    if not _MEMIFY_TASK_FACTORIES:
        _MEMIFY_TASK_FACTORIES = _build_task_factories()
    return _MEMIFY_TASK_FACTORIES


def supported_memify_task_names() -> List[str]:
    """Return the sorted names of memify tasks selectable by name."""
    return sorted(_get_task_factories())


def resolve_memify_tasks(
    tasks: Optional[Sequence[Union[Task, str]]],
) -> Optional[List[Task]]:
    """Resolve a mixed list of Task instances and task names into Task instances.

    Passes ``None``/empty input through unchanged so callers keep their
    default-task fallback. Raises ``CogneeValidationError`` (HTTP 422) when a
    name is unknown or an entry is neither a Task nor a string.
    """
    if not tasks:
        return None

    factories = _get_task_factories()
    resolved_tasks = []

    for task in tasks:
        if isinstance(task, Task):
            resolved_tasks.append(task)
        elif isinstance(task, str):
            factory = factories.get(task)
            if factory is None:
                raise CogneeValidationError(
                    message=(
                        f"Unknown memify task name: '{task}'. "
                        f"Supported task names: {', '.join(supported_memify_task_names())}. "
                        "Tasks requiring parameters must be passed as Task instances "
                        "through the Python SDK."
                    ),
                    log=False,
                )
            resolved_tasks.append(factory())
        else:
            raise CogneeValidationError(
                message=(
                    "Memify tasks must be Task instances or supported task names, "
                    f"got {type(task).__name__}."
                ),
                log=False,
            )

    return resolved_tasks
