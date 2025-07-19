import asyncio
from pydantic import BaseModel
from typing import Union, Optional
from uuid import UUID

from cognee.shared.logging_utils import get_logger
from cognee.shared.data_models import KnowledgeGraph
from cognee.infrastructure.llm import get_max_chunk_tokens

from cognee.modules.pipelines import cognee_pipeline
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.ontology.rdf_xml.OntologyResolver import OntologyResolver
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunCompleted, PipelineRunErrored
from cognee.modules.pipelines.queues.pipeline_run_info_queues import push_to_queue
from cognee.modules.users.models import User

from cognee.tasks.documents import (
    check_permissions_on_dataset,
    classify_documents,
    extract_chunks_from_documents,
)
from cognee.tasks.graph import extract_graph_from_data
from cognee.tasks.storage import add_data_points
from cognee.tasks.summarization import summarize_text

logger = get_logger("cognify")

update_status_lock = asyncio.Lock()


async def cognify(
    datasets: Union[str, list[str], list[UUID]] = None,
    user: User = None,
    graph_model: BaseModel = KnowledgeGraph,
    chunker=TextChunker,
    chunk_size: int = None,
    ontology_file_path: Optional[str] = None,
    vector_db_config: dict = None,
    graph_db_config: dict = None,
    run_in_background: bool = False,
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
                   Formula: min(embedding_max_tokens, llm_max_tokens // 2)
                   Default limits: ~512-8192 tokens depending on models.
                   Smaller chunks = more granular but potentially fragmented knowledge.
        ontology_file_path: Path to RDF/OWL ontology file for domain-specific entity types.
                          Useful for specialized fields like medical or legal documents.
        vector_db_config: Custom vector database configuration for embeddings storage.
        graph_db_config: Custom graph database configuration for relationship storage.
        run_in_background: If True, starts processing asynchronously and returns immediately.
                          If False, waits for completion before returning.
                          Background mode recommended for large datasets (>100MB).
                          Use pipeline_run_id from return value to monitor progress.

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
            query_type=SearchType.INSIGHTS
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

    Raises:
        DatasetNotFoundError: If specified datasets don't exist
        PermissionError: If user lacks processing rights
        InvalidValueError: If LLM_API_KEY is not set
        OntologyParsingError: If ontology file is malformed
        ValueError: If chunks exceed max token limits (reduce chunk_size)
        DatabaseNotCreatedError: If databases are not properly initialized
    """
    tasks = await get_default_tasks(user, graph_model, chunker, chunk_size, ontology_file_path)

    if run_in_background:
        return await run_cognify_as_background_process(
            tasks=tasks,
            user=user,
            datasets=datasets,
            vector_db_config=vector_db_config,
            graph_db_config=graph_db_config,
        )
    else:
        return await run_cognify_blocking(
            tasks=tasks,
            user=user,
            datasets=datasets,
            vector_db_config=vector_db_config,
            graph_db_config=graph_db_config,
        )


async def run_cognify_blocking(
    tasks,
    user,
    datasets,
    graph_db_config: dict = None,
    vector_db_config: dict = False,
):
    total_run_info = {}

    async for run_info in cognee_pipeline(
        tasks=tasks,
        datasets=datasets,
        user=user,
        pipeline_name="cognify_pipeline",
        graph_db_config=graph_db_config,
        vector_db_config=vector_db_config,
    ):
        if run_info.dataset_id:
            total_run_info[run_info.dataset_id] = run_info
        else:
            total_run_info = run_info

    return total_run_info


async def run_cognify_as_background_process(
    tasks,
    user,
    datasets,
    graph_db_config: dict = None,
    vector_db_config: dict = False,
):
    # Convert dataset to list if it's a string
    if isinstance(datasets, str):
        datasets = [datasets]

    # Store pipeline status for all pipelines
    pipeline_run_started_info = {}

    async def handle_rest_of_the_run(pipeline_list):
        # Execute all provided pipelines one by one to avoid database write conflicts
        for pipeline in pipeline_list:
            while True:
                try:
                    pipeline_run_info = await anext(pipeline)

                    push_to_queue(pipeline_run_info.pipeline_run_id, pipeline_run_info)

                    if isinstance(pipeline_run_info, PipelineRunCompleted) or isinstance(
                        pipeline_run_info, PipelineRunErrored
                    ):
                        break
                except StopAsyncIteration:
                    break

    # Start all pipelines to get started status
    pipeline_list = []
    for dataset in datasets:
        pipeline_run = cognee_pipeline(
            tasks=tasks,
            user=user,
            datasets=dataset,
            pipeline_name="cognify_pipeline",
            graph_db_config=graph_db_config,
            vector_db_config=vector_db_config,
        )

        # Save dataset Pipeline run started info
        run_info = await anext(pipeline_run)
        pipeline_run_started_info[run_info.dataset_id] = run_info

        if pipeline_run_started_info[run_info.dataset_id].payload:
            # Remove payload info to avoid serialization
            # TODO: Handle payload serialization
            pipeline_run_started_info[run_info.dataset_id].payload = []

        pipeline_list.append(pipeline_run)

    # Send all started pipelines to execute one by one in background
    asyncio.create_task(handle_rest_of_the_run(pipeline_list=pipeline_list))

    return pipeline_run_started_info


async def get_default_tasks(  # TODO: Find out a better way to do this (Boris's comment)
    user: User = None,
    graph_model: BaseModel = KnowledgeGraph,
    chunker=TextChunker,
    chunk_size: int = None,
    ontology_file_path: Optional[str] = None,
) -> list[Task]:
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
            ontology_adapter=OntologyResolver(ontology_file=ontology_file_path),
            task_config={"batch_size": 10},
        ),  # Generate knowledge graphs from the document chunks.
        Task(
            summarize_text,
            task_config={"batch_size": 10},
        ),
        Task(add_data_points, task_config={"batch_size": 10}),
    ]

    return default_tasks
