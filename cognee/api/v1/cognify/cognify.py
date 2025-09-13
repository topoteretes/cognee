from pydantic import BaseModel
from typing import Union, Optional, Type
from uuid import UUID
import os


from cognee.shared.data_models import KnowledgeGraph
from cognee.infrastructure.llm.utils import get_max_chunk_tokens
from cognee.shared.logging_utils import get_logger

from cognee.modules.pipelines.operations.pipeline import run_pipeline
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.ontology.rdf_xml.OntologyResolver import OntologyResolver
from cognee.modules.users.models import User

logger = get_logger()

from cognee.tasks.documents import (
    check_permissions_on_dataset,
    classify_documents,
    extract_chunks_from_documents,
)
from cognee.tasks.graph import extract_graph_from_data
from cognee.tasks.storage import add_data_points
from cognee.tasks.summarization import summarize_text
from cognee.tasks.translation import translate_content, get_available_providers, validate_provider
from cognee.modules.pipelines.layers.pipeline_execution_mode import get_pipeline_executor


class TranslationProviderError(ValueError):
    """Error related to translation provider initialization."""
    pass

class UnknownTranslationProviderError(TranslationProviderError):
    """Unknown translation provider name."""

class ProviderInitializationError(TranslationProviderError):
    """Provider failed to initialize (likely missing dependency or bad config)."""


_WARNED_ENV_VARS: set[str] = set()

def _parse_batch_env(var: str, default: int = 10) -> int:
<<<<<<< HEAD
=======
    """
    Parse an environment variable as a positive integer (minimum 1), falling back to a default.
    
    If the environment variable named `var` is unset, the provided `default` is returned.
    If the variable is set but cannot be parsed as an integer, `default` is returned and a
    one-time warning is logged for that variable (the variable name is recorded in
    `_WARNED_ENV_VARS` to avoid repeated warnings).
    
    Parameters:
        var: Name of the environment variable to read.
        default: Fallback integer value returned when the variable is missing or invalid.
    
    Returns:
        An integer >= 1 representing the parsed value or the fallback `default`.
    """
>>>>>>> 9f6b2dca51a936a9de482fc9f3c64934502240b6
    raw = os.getenv(var)
    if raw is None:
        return default
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        if var not in _WARNED_ENV_VARS:
            logger.warning("Invalid int for %s=%r; using default=%d", var, raw, default)
            _WARNED_ENV_VARS.add(var)
        return default

# Constants for batch processing
DEFAULT_BATCH_SIZE = _parse_batch_env("COGNEE_DEFAULT_BATCH_SIZE", 10)

async def cognify(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    datasets: Optional[Union[str, UUID, list[str], list[UUID]]] = None,
    user: Optional[User] = None,
    graph_model: Type[BaseModel] = KnowledgeGraph,
    chunker=TextChunker,
    chunk_size: Optional[int] = None,
    ontology_file_path: Optional[str] = None,
    vector_db_config: Optional[dict] = None,
    graph_db_config: Optional[dict] = None,
    run_in_background: bool = False,
    incremental_loading: bool = True,
    custom_prompt: Optional[str] = None,
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

        Note: To include a Translation step after chunking, use
        `get_default_tasks_with_translation(...)`.

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
        ontology_file_path: Path to RDF/OWL ontology file for domain-specific entity types.
                          Useful for specialized fields like medical or legal documents.
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
        from cognee.api.v1.search import SearchType

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

        New in this version:
        - COGNEE_DEFAULT_BATCH_SIZE: Default batch size for processing (default: 10)
    """
    tasks = get_default_tasks(
        user, graph_model, chunker, chunk_size, ontology_file_path, custom_prompt
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
    )


def get_default_tasks(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    user: Optional[User] = None,
    graph_model: Type[BaseModel] = KnowledgeGraph,
    chunker=TextChunker,
    chunk_size: Optional[int] = None,
    ontology_file_path: Optional[str] = None,
    custom_prompt: Optional[str] = None,
) -> list[Task]:
    """
    Return the standard, non-translation Task list used by the cognify pipeline.
    
    This builds the default processing pipeline (no automatic translation) and returns
    a list of Task objects in execution order:
    1. classify_documents
    2. check_permissions_on_dataset (enforces write permission for `user`)
    3. extract_chunks_from_documents (uses `chunker` and `chunk_size`)
    4. extract_graph_from_data (uses `graph_model`, optional `ontology_file_path`, and `custom_prompt`)
    5. summarize_text
    6. add_data_points
    
    Notes:
    - Batch sizes for downstream tasks use the module-level DEFAULT_BATCH_SIZE.
    - If `chunk_size` is not provided, the token limit from get_max_chunk_tokens() is used.
    
    Parameters:
        user: Optional user context used for the permission check.
        graph_model: Model class used to construct knowledge graph instances.
        chunker: Chunking strategy or class used to split documents into chunks.
        chunk_size: Optional max tokens per chunk; if omitted, defaults to get_max_chunk_tokens().
        ontology_file_path: Optional path to an ontology file passed to the extractor.
        custom_prompt: Optional custom prompt applied during graph extraction.
    
    Returns:
        List[Task]: Ordered list of Task objects for the cognify pipeline (no translation).
    """
    # Precompute max_chunk_size for stability
    max_chunk = chunk_size or get_max_chunk_tokens()
    default_tasks = [
        Task(classify_documents),
        Task(check_permissions_on_dataset, user=user, permissions=["write"]),
        Task(
            extract_chunks_from_documents,
            max_chunk_size=max_chunk,
            chunker=chunker,
        ),  # Extract text chunks based on the document type.
        Task(
            extract_graph_from_data,
            graph_model=graph_model,
            ontology_adapter=OntologyResolver(ontology_file=ontology_file_path),
            custom_prompt=custom_prompt,
            task_config={"batch_size": DEFAULT_BATCH_SIZE},
        ),  # Generate knowledge graphs from the document chunks.
        Task(
            summarize_text,
            task_config={"batch_size": DEFAULT_BATCH_SIZE},
        ),
        Task(add_data_points, task_config={"batch_size": DEFAULT_BATCH_SIZE}),
    ]

    return default_tasks


def get_default_tasks_with_translation(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    user: Optional[User] = None,
    graph_model: Type[BaseModel] = KnowledgeGraph,
    chunker=TextChunker,
    chunk_size: Optional[int] = None,
    ontology_file_path: Optional[str] = None,
    custom_prompt: Optional[str] = None,
    translation_provider: str = "noop",
) -> list[Task]:
    """
    Return the default Cognify pipeline task list with an added translation step.
    
    Constructs the standard processing pipeline (classify -> permission check -> chunk extraction -> translate -> graph extraction -> summarize -> add data points),
    validates and initializes the named translation provider, and applies module DEFAULT_BATCH_SIZE to downstream batchable tasks.
    
    Parameters:
        translation_provider (str): Name of a registered translation provider (case-insensitive). Defaults to `"noop"` which is a no-op provider.
    
    Returns:
        list[Task]: Ordered Task objects ready to be executed by the pipeline executor.
    
    Raises:
        UnknownTranslationProviderError: If the given provider name is not in get_available_providers().
        ProviderInitializationError: If the provider fails to initialize or validate via validate_provider().
    """
    # Fail fast on unknown providers (keeps errors close to the API surface)
    translation_provider = (translation_provider or "noop").strip().lower()
    # Validate provider using public API
    if translation_provider not in get_available_providers():
        available = ", ".join(get_available_providers())
        logger.error("Unknown provider '%s'. Available: %s", translation_provider, available)
        raise UnknownTranslationProviderError(f"Unknown provider '{translation_provider}'")
    # Instantiate to validate dependencies; include provider-specific config errors
    try:
        validate_provider(translation_provider)
    except Exception as e:  # we want to convert provider init errors
        available = ", ".join(get_available_providers())
        logger.error(
            "Provider '%s' failed to initialize (available: %s).",
            translation_provider,
            available,
            exc_info=True,
        )
        raise ProviderInitializationError() from e
    
    # Precompute max_chunk_size for stability
    max_chunk = chunk_size or get_max_chunk_tokens()
    
    default_tasks = [
        Task(classify_documents),
        Task(check_permissions_on_dataset, user=user, permissions=["write"]),
        Task(
            extract_chunks_from_documents,
            max_chunk_size=max_chunk,
            chunker=chunker,
        ),  # Extract text chunks based on the document type.
        Task(
            translate_content,
            target_language="en",
            translation_provider=translation_provider,
            task_config={"batch_size": DEFAULT_BATCH_SIZE},
        ),  # Auto-translate non-English content and attach metadata
        Task(
            extract_graph_from_data,
            graph_model=graph_model,
            ontology_adapter=OntologyResolver(ontology_file=ontology_file_path),
            custom_prompt=custom_prompt,
            task_config={"batch_size": DEFAULT_BATCH_SIZE},
        ),  # Generate knowledge graphs from the document chunks.
        Task(
            summarize_text,
            task_config={"batch_size": DEFAULT_BATCH_SIZE},
        ),
        Task(add_data_points, task_config={"batch_size": DEFAULT_BATCH_SIZE}),
    ]

    return default_tasks
