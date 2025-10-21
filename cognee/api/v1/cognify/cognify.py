import asyncio
from pydantic import BaseModel
from typing import Union, Optional
from uuid import UUID

from cognee.modules.ontology.ontology_env_config import get_ontology_env_config
from cognee.shared.logging_utils import get_logger
from cognee.shared.data_models import KnowledgeGraph
from cognee.infrastructure.llm import get_max_chunk_tokens

from cognee.modules.pipelines import run_pipeline
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.ontology.ontology_config import Config
from cognee.modules.ontology.get_default_ontology_resolver import (
    get_default_ontology_resolver,
    get_ontology_resolver_from_env,
)
from cognee.modules.users.models import User

from cognee.tasks.documents import (
    check_permissions_on_dataset,
    classify_documents,
    extract_chunks_from_documents,
)
from cognee.tasks.graph import extract_graph_from_data
from cognee.tasks.storage import add_data_points
from cognee.tasks.summarization import summarize_text
from cognee.modules.pipelines.layers.pipeline_execution_mode import get_pipeline_executor
from cognee.tasks.temporal_graph.extract_events_and_entities import extract_events_and_timestamps
from cognee.tasks.temporal_graph.extract_knowledge_graph_from_events import (
    extract_knowledge_graph_from_events,
)


logger = get_logger("cognify")

update_status_lock = asyncio.Lock()


async def cognify(
    datasets: Union[str, list[str], list[UUID]] = None,
    user: User = None,
    graph_model: BaseModel = KnowledgeGraph,
    chunker=TextChunker,
    chunk_size: int = None,
    chunks_per_batch: int = None,
    config: Config = None,
    vector_db_config: dict = None,
    graph_db_config: dict = None,
    run_in_background: bool = False,
    incremental_loading: bool = True,
    custom_prompt: Optional[str] = None,
    temporal_cognify: bool = False,
    data_per_batch: int = 20,
):
    """
    Transform ingested data into a structured knowledge graph.

    This is the core processing step in Cognee that converts raw text and documents
    into an intelligent knowledge graph. It analyzes content, extracts entities and
    relationships, and creates semantic connections for enhanced search and reasoning.

    Prerequisites:
        - **LLM_API_KEY**: Must be configured (required for entity extraction and graph generation)
        - **Data Added**: Must have data previously added via `cognee.add()`
        - **Vector Database**: Must be accessible for embeddings storage
        - **Graph Database**: Must be accessible for relationship storage

    Input Requirements:
        - **Datasets**: Must contain data previously added via `cognee.add()`
        - **Content Types**: Works with any text-extractable content including:
            * Natural language documents
            * Structured data (CSV, JSON)
            * Code repositories
            * Academic papers and technical documentation
            * Mixed multimedia content (with text extraction)

    Processing Pipeline:
        1. **Document Classification**: Identifies document types and structures
        2. **Permission Validation**: Ensures user has processing rights
        3. **Text Chunking**: Breaks content into semantically meaningful segments
        4. **Entity Extraction**: Identifies key concepts, people, places, organizations
        5. **Relationship Detection**: Discovers connections between entities
        6. **Graph Construction**: Builds semantic knowledge graph with embeddings
        7. **Content Summarization**: Creates hierarchical summaries for navigation

    Graph Model Customization:
        The `graph_model` parameter allows custom knowledge structures:
        - **Default**: General-purpose KnowledgeGraph for any domain
        - **Custom Models**: Domain-specific schemas (e.g., scientific papers, code analysis)
        - **Ontology Integration**: Use `ontology_file_path` for predefined vocabularies

    Args:
        datasets: Dataset name(s) or dataset uuid to process. Processes all available data if None.
            - Single dataset: "my_dataset"
            - Multiple datasets: ["docs", "research", "reports"]
            - None: Process all datasets for the user
        user: User context for authentication and data access. Uses default if None.
        graph_model: Pydantic model defining the knowledge graph structure.
                    Defaults to KnowledgeGraph for general-purpose processing.
        chunker: Text chunking strategy (TextChunker, LangchainChunker).
                - TextChunker: Paragraph-based chunking (default, most reliable)
                - LangchainChunker: Recursive character splitting with overlap
                Determines how documents are segmented for processing.
        chunk_size: Maximum tokens per chunk. Auto-calculated based on LLM if None.
                   Formula: min(embedding_max_completion_tokens, llm_max_completion_tokens // 2)
                   Default limits: ~512-8192 tokens depending on models.
                   Smaller chunks = more granular but potentially fragmented knowledge.
        chunks_per_batch: Number of chunks to be processed in a single batch in Cognify tasks.
        vector_db_config: Custom vector database configuration for embeddings storage.
        graph_db_config: Custom graph database configuration for relationship storage.
        run_in_background: If True, starts processing asynchronously and returns immediately.
                          If False, waits for completion before returning.
                          Background mode recommended for large datasets (>100MB).
                          Use pipeline_run_id from return value to monitor progress.
        custom_prompt: Optional custom prompt string to use for entity extraction and graph generation.
                      If provided, this prompt will be used instead of the default prompts for
                      knowledge graph extraction. The prompt should guide the LLM on how to
                      extract entities and relationships from the text content.

    Returns:
        Union[dict, list[PipelineRunInfo]]:
            - **Blocking mode**: Dictionary mapping dataset_id -> PipelineRunInfo with:
                * Processing status (completed/failed/in_progress)
                * Extracted entity and relationship counts
                * Processing duration and resource usage
                * Error details if any failures occurred
            - **Background mode**: List of PipelineRunInfo objects for tracking progress
                * Use pipeline_run_id to monitor status
                * Check completion via pipeline monitoring APIs

    Next Steps:
        After successful cognify processing, use search functions to query the knowledge:

        ```python
        import cognee
        from cognee import SearchType

        # Process your data into knowledge graph
        await cognee.cognify()

        # Query for insights using different search types:

        # 1. Natural language completion with graph context
        insights = await cognee.search(
            "What are the main themes?",
            query_type=SearchType.GRAPH_COMPLETION
        )

        # 2. Get entity relationships and connections
        relationships = await cognee.search(
            "connections between concepts",
            query_type=SearchType.GRAPH_COMPLETION
        )

        # 3. Find relevant document chunks
        chunks = await cognee.search(
            "specific topic",
            query_type=SearchType.CHUNKS
        )
        ```

    Advanced Usage:
        ```python
        # Custom domain model for scientific papers
        class ScientificPaper(DataPoint):
            title: str
            authors: List[str]
            methodology: str
            findings: List[str]

        await cognee.cognify(
            datasets=["research_papers"],
            graph_model=ScientificPaper,
            ontology_file_path="scientific_ontology.owl"
        )

        # Background processing for large datasets
        run_info = await cognee.cognify(
            datasets=["large_corpus"],
            run_in_background=True
        )
        # Check status later with run_info.pipeline_run_id
        ```


    Environment Variables:
        Required:
        - LLM_API_KEY: API key for your LLM provider

        Optional (same as add function):
        - LLM_PROVIDER, LLM_MODEL, VECTOR_DB_PROVIDER, GRAPH_DATABASE_PROVIDER
        - LLM_RATE_LIMIT_ENABLED: Enable rate limiting (default: False)
        - LLM_RATE_LIMIT_REQUESTS: Max requests per interval (default: 60)
    """
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

    if temporal_cognify:
        tasks = await get_temporal_tasks(
            user=user, chunker=chunker, chunk_size=chunk_size, chunks_per_batch=chunks_per_batch
        )
    else:
        tasks = await get_default_tasks(
            user=user,
            graph_model=graph_model,
            chunker=chunker,
            chunk_size=chunk_size,
            config=config,
            custom_prompt=custom_prompt,
            chunks_per_batch=chunks_per_batch,
        )

    # By calling get pipeline executor we get a function that will have the run_pipeline run in the background or a function that we will need to wait for
    pipeline_executor_func = get_pipeline_executor(run_in_background=run_in_background)

    # Run the run_pipeline in the background or blocking based on executor
    return await pipeline_executor_func(
        pipeline=run_pipeline,
        tasks=tasks,
        user=user,
        datasets=datasets,
        vector_db_config=vector_db_config,
        graph_db_config=graph_db_config,
        incremental_loading=incremental_loading,
        pipeline_name="cognify_pipeline",
        data_per_batch=data_per_batch,
    )


async def get_default_tasks(  # TODO: Find out a better way to do this (Boris's comment)
    user: User = None,
    graph_model: BaseModel = KnowledgeGraph,
    chunker=TextChunker,
    chunk_size: int = None,
    config: Config = None,
    custom_prompt: Optional[str] = None,
    chunks_per_batch: int = 100,
) -> list[Task]:
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

    if chunks_per_batch is None:
        chunks_per_batch = 100

    default_tasks = [
        Task(classify_documents),
        Task(check_permissions_on_dataset, user=user, permissions=["write"]),
        Task(
            extract_chunks_from_documents,
            max_chunk_size=chunk_size or get_max_chunk_tokens(),
            chunker=chunker,
        ),  # Extract text chunks based on the document type.
        Task(
            extract_graph_from_data,
            graph_model=graph_model,
            config=config,
            custom_prompt=custom_prompt,
            task_config={"batch_size": chunks_per_batch},
        ),  # Generate knowledge graphs from the document chunks.
        Task(
            summarize_text,
            task_config={"batch_size": chunks_per_batch},
        ),
        Task(add_data_points, task_config={"batch_size": chunks_per_batch}),
    ]

    return default_tasks


async def get_temporal_tasks(
    user: User = None, chunker=TextChunker, chunk_size: int = None, chunks_per_batch: int = 10
) -> list[Task]:
    """
    Builds and returns a list of temporal processing tasks to be executed in sequence.

    The pipeline includes:
    1. Document classification.
    2. Dataset permission checks (requires "write" access).
    3. Document chunking with a specified or default chunk size.
    4. Event and timestamp extraction from chunks.
    5. Knowledge graph extraction from events.
    6. Batched insertion of data points.

    Args:
        user (User, optional): The user requesting task execution, used for permission checks.
        chunker (Callable, optional): A text chunking function/class to split documents. Defaults to TextChunker.
        chunk_size (int, optional): Maximum token size per chunk. If not provided, uses system default.
        chunks_per_batch (int, optional): Number of chunks to process in a single batch in Cognify

    Returns:
        list[Task]: A list of Task objects representing the temporal processing pipeline.
    """
    if chunks_per_batch is None:
        chunks_per_batch = 10

    temporal_tasks = [
        Task(classify_documents),
        Task(check_permissions_on_dataset, user=user, permissions=["write"]),
        Task(
            extract_chunks_from_documents,
            max_chunk_size=chunk_size or get_max_chunk_tokens(),
            chunker=chunker,
        ),
        Task(extract_events_and_timestamps, task_config={"batch_size": chunks_per_batch}),
        Task(extract_knowledge_graph_from_events),
        Task(add_data_points, task_config={"batch_size": chunks_per_batch}),
    ]

    return temporal_tasks
