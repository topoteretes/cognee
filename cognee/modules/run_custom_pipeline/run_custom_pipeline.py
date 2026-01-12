from typing import Union, Optional, List, Type, Any
from uuid import UUID

from cognee.shared.logging_utils import get_logger

from cognee.modules.pipelines import run_pipeline
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.models import User
from cognee.modules.pipelines.layers.pipeline_execution_mode import get_pipeline_executor

logger = get_logger()


async def run_custom_pipeline(
    tasks: Union[List[Task], List[str]] = None,
    data: Any = None,
    dataset: Union[str, UUID] = "main_dataset",
    user: User = None,
    vector_db_config: Optional[dict] = None,
    graph_db_config: Optional[dict] = None,
    use_pipeline_cache: bool = False,
    incremental_loading: bool = False,
    data_per_batch: int = 20,
    run_in_background: bool = False,
    pipeline_name: str = "custom_pipeline",
):
    """
    Custom pipeline in Cognee, can work with already built graphs. Data needs to be provided which can be processed
    with provided tasks.

    Provided tasks and data will be arranged to run the Cognee pipeline and execute graph enrichment/creation.

    This is the core processing step in Cognee that converts raw text and documents
    into an intelligent knowledge graph. It analyzes content, extracts entities and
    relationships, and creates semantic connections for enhanced search and reasoning.

    Args:
        tasks: List of Cognee Tasks to execute.
        data: The data to ingest. Can be anything when custom extraction and enrichment tasks are used.
              Data provided here will be forwarded to the first extraction task in the pipeline as input.
        dataset: Dataset name or dataset uuid to process.
        user: User context for authentication and data access. Uses default if None.
        vector_db_config: Custom vector database configuration for embeddings storage.
        graph_db_config: Custom graph database configuration for relationship storage.
        use_pipeline_cache: If True, pipelines with the same ID that are currently executing and pipelines with the same ID that were completed won't process data again.
                        Pipelines ID is created based on the generate_pipeline_id function. Pipeline status can be manually reset with the reset_dataset_pipeline_run_status function.
        incremental_loading: If True, only new or modified data will be processed to avoid duplication. (Only works if data is used with the Cognee python Data model).
                            The incremental system stores and compares hashes of processed data in the Data model and skips data with the same content hash.
        data_per_batch: Number of data items to be processed in parallel.
        run_in_background: If True, starts processing asynchronously and returns immediately.
                          If False, waits for completion before returning.
                          Background mode recommended for large datasets (>100MB).
                          Use pipeline_run_id from return value to monitor progress.
    """

    custom_tasks = [
        *tasks,
    ]

    # By calling get pipeline executor we get a function that will have the run_pipeline run in the background or a function that we will need to wait for
    pipeline_executor_func = get_pipeline_executor(run_in_background=run_in_background)

    # Run the run_pipeline in the background or blocking based on executor
    return await pipeline_executor_func(
        pipeline=run_pipeline,
        tasks=custom_tasks,
        user=user,
        data=data,
        datasets=dataset,
        vector_db_config=vector_db_config,
        graph_db_config=graph_db_config,
        use_pipeline_cache=use_pipeline_cache,
        incremental_loading=incremental_loading,
        data_per_batch=data_per_batch,
        pipeline_name=pipeline_name,
    )
