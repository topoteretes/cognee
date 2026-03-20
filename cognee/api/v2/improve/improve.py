from uuid import UUID
from typing import Union, Optional, List, Type, Any

from cognee.modules.users.models import User
from cognee.modules.pipelines.tasks.task import Task


async def improve(
    extraction_tasks: Union[List[Task], List[str]] = None,
    enrichment_tasks: Union[List[Task], List[str]] = None,
    data: Optional[Any] = None,
    dataset: Union[str, UUID] = "main_dataset",
    user: User = None,
    node_type: Optional[Type] = None,
    node_name: Optional[List[str]] = None,
    vector_db_config: Optional[dict] = None,
    graph_db_config: Optional[dict] = None,
    run_in_background: bool = False,
):
    """Enrich an existing knowledge graph with additional context and rules.

    This is a memory-oriented alias for ``cognee.memify()``.  All arguments
    are forwarded unchanged.

    Args:
        extraction_tasks: Tasks for graph/data extraction.
        enrichment_tasks: Tasks for graph enrichment.
        data: Custom input data. Uses the existing graph when *None*.
        dataset: Dataset name or UUID to process.
        user: User context for permissions.
        node_type: Filter graph to specific entity types.
        node_name: Filter graph to specific named entities.
        vector_db_config: Custom vector DB config.
        graph_db_config: Custom graph DB config.
        run_in_background: Run processing asynchronously.

    Returns:
        Pipeline run info (same as ``cognee.memify()``).
    """
    from cognee.modules.memify import memify

    # Resolve default node_type here to avoid import at module level
    if node_type is None:
        from cognee.modules.engine.models.node_set import NodeSet

        node_type = NodeSet

    return await memify(
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
