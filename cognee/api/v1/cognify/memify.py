from typing import Union, Optional, List, Type, Any
from dataclasses import field
from uuid import UUID

from cognee.shared.logging_utils import get_logger

from cognee.modules.retrieval.utils.brute_force_triplet_search import get_memory_fragment

from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.pipelines import run_pipeline
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.models import User
from cognee.modules.pipelines.layers.resolve_authorized_user_datasets import (
    resolve_authorized_user_datasets,
)
from cognee.modules.pipelines.layers.reset_dataset_pipeline_run_status import (
    reset_dataset_pipeline_run_status,
)
from cognee.modules.engine.operations.setup import setup
from cognee.modules.pipelines.layers.pipeline_execution_mode import get_pipeline_executor

logger = get_logger("memify")


async def memify(
    data_streaming_tasks: List[Task],
    data_processing_tasks: List[Task] = [],
    data_persistence_tasks: List[Task] = [],
    data: Optional[Any] = None,
    datasets: Union[str, list[str], list[UUID]] = None,
    user: User = None,
    node_type: Optional[Type] = NodeSet,
    node_name: Optional[List[str]] = None,
    cypher_query: Optional[str] = None,
    vector_db_config: dict = None,
    graph_db_config: dict = None,
    run_in_background: bool = False,
):
    """
    Prerequisites:
        - **LLM_API_KEY**: Must be configured (required for entity extraction and graph generation)
        - **Data Added**: Must have data previously added via `cognee.add()` and `cognee.cognify()`
        - **Vector Database**: Must be accessible for embeddings storage
        - **Graph Database**: Must be accessible for relationship storage

    Args:
        datasets: Dataset name(s) or dataset uuid to process. Processes all available data if None.
            - Single dataset: "my_dataset"
            - Multiple datasets: ["docs", "research", "reports"]
            - None: Process all datasets for the user
        user: User context for authentication and data access. Uses default if None.
        vector_db_config: Custom vector database configuration for embeddings storage.
        graph_db_config: Custom graph database configuration for relationship storage.
        run_in_background: If True, starts processing asynchronously and returns immediately.
                          If False, waits for completion before returning.
                          Background mode recommended for large datasets (>100MB).
                          Use pipeline_run_id from return value to monitor progress.
    """

    if not data:
        if cypher_query:
            pass
        else:
            memory_fragment = await get_memory_fragment(node_type=node_type, node_name=node_name)
            # Subgraphs should be a single element in the list to represent one data item
            data = [memory_fragment]

    memify_tasks = [
        *data_streaming_tasks,  # Unpack tasks provided to memify pipeline
        *data_processing_tasks,
        *data_persistence_tasks,
    ]

    await setup()

    user, authorized_datasets = await resolve_authorized_user_datasets(datasets, user)

    for dataset in authorized_datasets:
        await reset_dataset_pipeline_run_status(
            dataset.id, user, pipeline_names=["memify_pipeline"]
        )

    # By calling get pipeline executor we get a function that will have the run_pipeline run in the background or a function that we will need to wait for
    pipeline_executor_func = get_pipeline_executor(run_in_background=run_in_background)

    # Run the run_pipeline in the background or blocking based on executor
    return await pipeline_executor_func(
        pipeline=run_pipeline,
        tasks=memify_tasks,
        user=user,
        data=data,
        datasets=datasets,
        vector_db_config=vector_db_config,
        graph_db_config=graph_db_config,
        incremental_loading=False,
        pipeline_name="memify_pipeline",
    )
