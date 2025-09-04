from typing import Union, Optional, List, Type, Any
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
from cognee.tasks.memify.extract_subgraph_chunks import extract_subgraph_chunks
from cognee.tasks.codingagents.coding_rule_associations import (
    add_rule_associations,
)

logger = get_logger("memify")


async def memify(
    extraction_tasks: Union[List[Task], List[str]] = [Task(extract_subgraph_chunks)],
    enrichment_tasks: Union[List[Task], List[str]] = [Task(add_rule_associations)],
    data: Optional[Any] = None,
    datasets: Union[str, list[str], list[UUID]] = None,
    user: User = None,
    node_type: Optional[Type] = NodeSet,
    node_name: Optional[List[str]] = None,
    vector_db_config: Optional[dict] = None,
    graph_db_config: Optional[dict] = None,
    run_in_background: bool = False,
):
    """
    Enrichment pipeline in Cognee, can work with already built graphs. If no data is provided existing knowledge graph will be used as data,
    custom data can also be provided instead which can be processed with provided extraction and enrichment tasks.

    Provided tasks and data will be arranged to run the Cognee pipeline and execute graph enrichment/creation.

    This is the core processing step in Cognee that converts raw text and documents
    into an intelligent knowledge graph. It analyzes content, extracts entities and
    relationships, and creates semantic connections for enhanced search and reasoning.

    Args:
        extraction_tasks: List of Cognee Tasks to execute for graph/data extraction.
        enrichment_tasks: List of Cognee Tasks to handle enrichment of provided graph/data from extraction tasks.
        data: The data to ingest. Can be anything when custom extraction and enrichment tasks are used.
              Data provided here will be forwarded to the first extraction task in the pipeline as input.
              If no data is provided the whole graph (or subgraph if node_name/node_type is specified) will be forwarded
        datasets: Dataset name(s) or dataset uuid to process. Processes all available datasets if None.
            - Single dataset: "my_dataset"
            - Multiple datasets: ["docs", "research", "reports"]
            - None: Process all datasets for the user
        user: User context for authentication and data access. Uses default if None.
        node_type: Filter graph to specific entity types (for advanced filtering). Used when no data is provided.
        node_name: Filter graph to specific named entities (for targeted search). Used when no data is provided.
        vector_db_config: Custom vector database configuration for embeddings storage.
        graph_db_config: Custom graph database configuration for relationship storage.
        run_in_background: If True, starts processing asynchronously and returns immediately.
                          If False, waits for completion before returning.
                          Background mode recommended for large datasets (>100MB).
                          Use pipeline_run_id from return value to monitor progress.
    """

    if not data:
        memory_fragment = await get_memory_fragment(node_type=node_type, node_name=node_name)
        # Subgraphs should be a single element in the list to represent one data item
        data = [memory_fragment]

    memify_tasks = [
        *extraction_tasks,  # Unpack tasks provided to memify pipeline
        *enrichment_tasks,
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
