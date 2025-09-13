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
    Orchestrate processing of datasets into a knowledge graph.
    
    Builds the default Cognify task sequence (classification, permission check, chunking,
    graph extraction, summarization, indexing) and executes it via the pipeline
    executor. Use get_default_tasks_with_translation(...) to include an automatic
    translation step before graph extraction.
    
    Parameters:
        datasets: Optional dataset id or list of ids to process. If None, processes all
            datasets available to the user.
        user: Optional user context used for permission checks; defaults to the current
            runtime user if omitted.
        graph_model: Pydantic model type that defines the structure of produced graph
            DataPoints (default: KnowledgeGraph).
        chunker: Chunking strategy/class used to split documents (default: TextChunker).
        chunk_size: Optional max tokens per chunk; when None a sensible default is used.
        ontology_file_path: Optional path to an ontology (RDF/OWL) used by the extractor.
        vector_db_config: Optional mapping of vector DB configuration (overrides defaults).
        graph_db_config: Optional mapping of graph DB configuration (overrides defaults).
        run_in_background: If True, starts the pipeline asynchronously and returns
            background run info; if False, waits for completion and returns results.
        incremental_loading: If True, performs incremental loading to avoid reprocessing
            unchanged content.
        custom_prompt: Optional prompt to override the default prompt used for graph
            extraction.
    
    Returns:
        The pipeline executor result. In blocking mode this is the pipeline run result
        (per-dataset run info and status). In background mode this returns information
        required to track the background run (e.g., pipeline_run_id and submission status).
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
