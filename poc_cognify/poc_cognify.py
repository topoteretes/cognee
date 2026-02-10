import asyncio
from pydantic import BaseModel
from typing import Union, Optional
from uuid import UUID

from cognee import run_custom_pipeline
from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.ontology.ontology_env_config import get_ontology_env_config
from cognee.shared.logging_utils import get_logger
from cognee.shared.data_models import KnowledgeGraph
from cognee.infrastructure.llm import get_max_chunk_tokens

from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.ontology.ontology_config import Config
from cognee.modules.ontology.get_default_ontology_resolver import (
    get_default_ontology_resolver,
    get_ontology_resolver_from_env,
)
from cognee.modules.users.models import User

from cognee.tasks.documents import (
    classify_documents,
    extract_chunks_from_documents,
)
from cognee.tasks.storage import add_data_points
from cognee.tasks.summarization import summarize_text
from poc_extract_graph_from_data import poc_extract_graph_from_data

logger = get_logger("cognify")

update_status_lock = asyncio.Lock()


async def cognify_single_add_datapoints(
    datasets: Union[str, list[str], list[UUID]] = None,
    user: User = None,
    graph_model: BaseModel = KnowledgeGraph,
    chunker=TextChunker,
    chunk_size: int = None,
    chunks_per_batch: int = None,
    config: Config = None,
    vector_db_config: dict = None,
    graph_db_config: dict = None,
    incremental_loading: bool = True,
    custom_prompt: Optional[str] = None,
    data_per_batch: int = 20,
    **kwargs,
):
    if config is None:
        ontology_config = get_ontology_env_config()
        if (
            ontology_config.ontology_file_path
            and ontology_config.ontology_resolver
            and ontology_config.matching_strategy
        ):
            config: Config = {
                "ontology_config": {
                    "ontology_resolver": get_ontology_resolver_from_env(**ontology_config.to_dict())
                }
            }
        else:
            config: Config = {
                "ontology_config": {"ontology_resolver": get_default_ontology_resolver()}
            }
    cognify_config = get_cognify_config()
    embed_triplets = cognify_config.triplet_embedding

    if chunks_per_batch is None:
        chunks_per_batch = (
            cognify_config.chunks_per_batch if cognify_config.chunks_per_batch is not None else 100
        )

    tasks = [
        Task(classify_documents),
        Task(
            extract_chunks_from_documents,
            max_chunk_size=chunk_size or get_max_chunk_tokens(),
            chunker=chunker,
        ),  # Extract text chunks based on the document type.
        Task(
            poc_extract_graph_from_data,
            graph_model=graph_model,
            config=config,
            custom_prompt=custom_prompt,
            task_config={"batch_size": chunks_per_batch},
            **kwargs,
        ),  # Generate knowledge graphs from the document chunks.
        Task(
            summarize_text,
            task_config={"batch_size": chunks_per_batch},
        ),
        Task(
            add_data_points,
            embed_triplets=embed_triplets,
            task_config={"batch_size": chunks_per_batch},
        ),
    ]

    # Run the run_pipeline in the background or blocking based on executor
    return await run_custom_pipeline(
        tasks=tasks,
        user=user,
        dataset=datasets,
        vector_db_config=vector_db_config,
        graph_db_config=graph_db_config,
        incremental_loading=incremental_loading,
        use_pipeline_cache=True,
        pipeline_name="poc_cognify_pipeline",
        data_per_batch=data_per_batch,
    )
