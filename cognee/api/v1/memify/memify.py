"""Deprecated public ``memify()`` (SDK-1115), kept for one release.

Triplet embeddings were removed, and with them the default memify tasks that
built them. ``cognee.memify()`` still runs the tasks a caller names, but warns
that it goes away in the next release: ``cognee.improve()`` enriches a dataset
and ``cognee.run_custom_pipeline()`` runs custom tasks. The internal runner the
improve stages use is ``cognee.modules.memify.run_memify_tasks``; it stays.
"""

import warnings
from collections.abc import Callable, Sequence
from typing import Any
from uuid import UUID

from cognee.exceptions import CogneeValidationError
from cognee.modules.data.constants import DEFAULT_DATASET_NAME
from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger

logger = get_logger("memify")

MEMIFY_DEPRECATION_MESSAGE = (
    "cognee.memify() is deprecated and will be removed in the next release: "
    "cognee 1.6.4 was the last release in which memify() ran its default tasks "
    "(triplet embeddings); since then, a call without tasks runs nothing. "
    "Use cognee.improve() to enrich a dataset, or cognee.run_custom_pipeline() "
    "to run your own tasks."
)

# Task names that built triplet embeddings. They no longer exist.
_REMOVED_TASK_NAMES = frozenset({"get_triplet_datapoints"})

# Extraction tasks that read the dataset's graph through the graph engine
# themselves and ignore what they are seeded with. run_legacy_memify hands
# them the ``[{}]`` placeholder the SDK pipelines pass instead of projecting
# the whole graph into a memory fragment they would discard.
_GRAPH_READING_EXTRACTION_NAMES = frozenset({"extract_code_docs"})


def _reads_the_graph_itself(extraction_tasks: list[Task]) -> bool:
    return bool(extraction_tasks) and all(
        getattr(task.executable, "__name__", None) in _GRAPH_READING_EXTRACTION_NAMES
        for task in extraction_tasks
    )


def _task_factories() -> dict[str, Callable[[], Task]]:
    """Name -> zero-argument factory, for the names the memify API accepted."""
    from cognee.memify_pipelines.cognify_code_docs import (
        cognify_chunks_per_batch,
        cognify_code_doc_chunks,
    )
    from cognee.shared.data_models import KnowledgeGraph
    from cognee.tasks.code_docs.add_code_doc_edges import add_code_doc_edges
    from cognee.tasks.code_docs.extract_code_docs import extract_code_docs
    from cognee.tasks.memify.apply_feedback_weights import apply_feedback_weights
    from cognee.tasks.memify.cognify_agent_trace_feedback import cognify_agent_trace_feedback
    from cognee.tasks.memify.cognify_session import cognify_session
    from cognee.tasks.memify.consolidate_entities import (
        detect_entity_duplicates,
        merge_entity_duplicates,
    )
    from cognee.tasks.memify.extract_agent_trace_feedbacks import extract_agent_trace_feedbacks
    from cognee.tasks.memify.extract_subgraph import extract_subgraph
    from cognee.tasks.memify.extract_subgraph_chunks import extract_subgraph_chunks
    from cognee.tasks.memify.extract_user_sessions import extract_user_sessions
    from cognee.tasks.storage.add_data_points import add_data_points
    from cognee.tasks.storage.index_data_points import index_data_points

    return {
        "extract_subgraph": lambda: Task(extract_subgraph),
        "extract_subgraph_chunks": lambda: Task(extract_subgraph_chunks),
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
        # The code docs pipeline (cognee/memify_pipelines/cognify_code_docs.py) by
        # name: extraction ["extract_code_docs"]; enrichment ["add_data_points",
        # "add_code_doc_edges"] (embed-only) or ["cognify_code_doc_chunks",
        # "add_data_points", "add_code_doc_edges"] (cognify's entity extraction
        # and summaries over the chunks, at cognify's defaults, one cognify
        # batch at a time).
        "extract_code_docs": lambda: Task(extract_code_docs, needs_llm=False),
        "cognify_code_doc_chunks": lambda: Task(
            cognify_code_doc_chunks,
            graph_model=KnowledgeGraph,
            chunks_per_batch=cognify_chunks_per_batch(),
        ),
        "add_data_points": lambda: Task(
            add_data_points,
            task_config={"batch_size": cognify_chunks_per_batch()},
            needs_llm=False,
        ),
        "add_code_doc_edges": lambda: Task(add_code_doc_edges, needs_llm=False),
    }


def resolve_memify_tasks(tasks: Sequence[Task | str] | None) -> list[Task]:
    """Resolve Task instances and built-in task names into Task instances.

    Raises ``CogneeValidationError`` (HTTP 422) for an unknown or removed name,
    or an entry that is neither a Task nor a string.
    """
    if not tasks:
        return []

    factories = _task_factories()
    resolved = []
    for task in tasks:
        if isinstance(task, Task):
            resolved.append(task)
        elif isinstance(task, str):
            if task in _REMOVED_TASK_NAMES:
                raise CogneeValidationError(
                    message=f"Memify task '{task}' was removed together with triplet embeddings.",
                    log=False,
                )
            factory = factories.get(task)
            if factory is None:
                raise CogneeValidationError(
                    message=(
                        f"Unknown memify task name: '{task}'. "
                        f"Supported task names: {', '.join(sorted(factories))}. "
                        "Tasks requiring parameters must be passed as Task instances "
                        "through the Python SDK."
                    ),
                    log=False,
                )
            resolved.append(factory())
        else:
            raise CogneeValidationError(
                message=(
                    "Memify tasks must be Task instances or supported task names, "
                    f"got {type(task).__name__}."
                ),
                log=False,
            )
    return resolved


async def memify(
    extraction_tasks: Sequence[Task | str] | None = None,
    enrichment_tasks: Sequence[Task | str] | None = None,
    data: Any | None = None,
    dataset: str | UUID = DEFAULT_DATASET_NAME,
    user: User = None,
    node_type: type | None = NodeSet,
    node_name: list[str] | None = None,
    vector_db_config: dict | None = None,
    graph_db_config: dict | None = None,
    run_in_background: bool = False,
):
    """Deprecated: run enrichment tasks over a dataset. Removed in the next release.

    Use ``cognee.improve()`` to enrich a dataset, or
    ``cognee.run_custom_pipeline()`` to run your own tasks.

    The named or given tasks run as before. With no tasks, nothing runs and
    ``{}`` is returned: the default tasks built triplet embeddings, which were
    removed. Without ``data``, the dataset's graph (filtered by ``node_type`` /
    ``node_name``) is passed to the first task, as before.
    """
    warnings.warn(MEMIFY_DEPRECATION_MESSAGE, FutureWarning, stacklevel=2)
    return await run_legacy_memify(
        extraction_tasks=extraction_tasks,
        enrichment_tasks=enrichment_tasks,
        data=data,
        dataset=dataset,
        user=user,
        node_type=node_type,
        node_name=node_name,
        vector_db_config=vector_db_config,
        graph_db_config=graph_db_config,
        run_in_background=run_in_background,
    )


async def run_legacy_memify(
    extraction_tasks: Sequence[Task | str] | None = None,
    enrichment_tasks: Sequence[Task | str] | None = None,
    data: Any | None = None,
    dataset: str | UUID = DEFAULT_DATASET_NAME,
    user: User = None,
    node_type: type | None = NodeSet,
    node_name: list[str] | None = None,
    vector_db_config: dict | None = None,
    graph_db_config: dict | None = None,
    run_in_background: bool = False,
):
    """``memify()`` without its warning, for the deprecated callers that already warned.

    ``improve(extraction_tasks=..., enrichment_tasks=...)`` runs its custom
    tasks through this, as the ``custom_tasks`` stage.
    """
    resolved_extraction_tasks = resolve_memify_tasks(extraction_tasks)
    resolved_enrichment_tasks = resolve_memify_tasks(enrichment_tasks)

    from cognee.context_global_variables import set_database_global_context_variables
    from cognee.modules.engine.operations.setup import setup
    from cognee.modules.migrations.startup import run_migrations_and_block
    from cognee.modules.pipelines.layers.resolve_authorized_user_datasets import (
        resolve_authorized_user_datasets,
    )
    from cognee.modules.retrieval.utils.brute_force_triplet_search import get_memory_fragment

    # Same order as before SDK-1115: the migration gate comes before dataset
    # resolution (which can write the Dataset row and its ACL) and before the
    # graph read, and an unknown or unauthorized dataset still raises when no
    # tasks are given.
    await setup()
    await run_migrations_and_block(dataset, user)
    user, authorized_datasets = await resolve_authorized_user_datasets(dataset, user)
    authorized_dataset = authorized_datasets[0]

    if not resolved_extraction_tasks and not resolved_enrichment_tasks:
        logger.warning(
            "memify() was called without tasks; nothing ran. Its default tasks built "
            "triplet embeddings, which were removed."
        )
        return {}

    if not data and _reads_the_graph_itself(resolved_extraction_tasks):
        data = [{}]
    elif not data:
        async with set_database_global_context_variables(
            authorized_dataset.id, authorized_dataset.owner_id
        ):
            memory_fragment = await get_memory_fragment(node_type=node_type, node_name=node_name)
        # One element: the whole subgraph is one data item.
        data = [memory_fragment]

    from cognee.modules.memify import run_memify_tasks

    return await run_memify_tasks(
        resolved_extraction_tasks,
        resolved_enrichment_tasks,
        data=data,
        dataset=authorized_dataset.id,
        user=user,
        vector_db_config=vector_db_config,
        graph_db_config=graph_db_config,
        run_in_background=run_in_background,
    )
