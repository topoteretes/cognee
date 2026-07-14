import asyncio
from pydantic import BaseModel
from typing import Union, Optional
from uuid import UUID

from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.cognify.rollback import cognify_rollback_handler
from cognee.modules.ontology.ontology_env_config import get_ontology_env_config
from cognee.shared.logging_utils import get_logger
from cognee.shared.data_models import KnowledgeGraph
from cognee.infrastructure.llm import get_max_chunk_tokens

from cognee.modules.pipelines import run_pipeline
from cognee.modules.pipelines.tasks.task import Task
from cognee.infrastructure.databases.vector.embeddings.config import EmbeddingConfig
from cognee.infrastructure.llm.config import LLMConfig
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
from cognee.tasks.graph.extract_graph_and_summarize import extract_graph_and_summarize
from cognee.tasks.storage import add_data_points
from cognee.tasks.ingestion.extract_dlt_fk_edges import extract_dlt_fk_edges
from cognee.modules.pipelines.layers.pipeline_execution_mode import get_pipeline_executor
from cognee.tasks.temporal_graph.extract_events_and_entities import extract_events_and_timestamps
from cognee.tasks.temporal_graph.extract_knowledge_graph_from_events import (
    extract_knowledge_graph_from_events,
)
from cognee.modules.observability import new_span, COGNEE_PIPELINE_NAME, COGNEE_RESULT_SUMMARY


logger = get_logger("cognify")


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
    llm_config: Optional[LLMConfig] = None,
    embedding_config: Optional[EmbeddingConfig] = None,
    data_cache: bool = True,
    **kwargs,
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
        2. **Text Chunking**: Breaks content into semantically meaningful segments
        3. **Entity Extraction**: Identifies key concepts, people, places, organizations
        4. **Relationship Detection**: Discovers connections between entities
        5. **Graph Construction**: Builds semantic knowledge graph with embeddings
        6. **Content Summarization**: Creates text summaries for navigation

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
    # Route to remote instance if connected via serve()
    from cognee.api.v1.serve.state import get_remote_client

    client = get_remote_client()
    if client is not None:
        return await client.cognify(
            datasets,
            chunk_size=chunk_size,
            chunks_per_batch=chunks_per_batch,
            custom_prompt=custom_prompt,
            run_in_background=run_in_background,
        )

    with new_span("cognee.api.cognify") as span:
        span.set_attribute(COGNEE_PIPELINE_NAME, "cognify")
        if datasets is not None:
            span.set_attribute("cognee.cognify.datasets", str(datasets))

        from cognee.modules.migrations.startup import run_migrations_and_block

        await run_migrations_and_block(datasets, user)

        if config is None:
            ontology_config = get_ontology_env_config()
            if (
                ontology_config.ontology_file_path
                and ontology_config.ontology_resolver
                and ontology_config.matching_strategy
            ):
                config: Config = {
                    "ontology_config": {
                        "ontology_resolver": get_ontology_resolver_from_env(
                            **ontology_config.to_dict()
                        )
                    }
                }
            else:
                config: Config = {
                    "ontology_config": {"ontology_resolver": get_default_ontology_resolver()}
                }

        if temporal_cognify:
            tasks = await get_temporal_tasks(
                user=user,
                chunker=chunker,
                chunk_size=chunk_size,
                chunks_per_batch=chunks_per_batch,
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
                **kwargs,
            )

        # By calling get pipeline executor we get a function that will have the run_pipeline run in the background or a function that we will need to wait for
        pipeline_executor_func = get_pipeline_executor(run_in_background=run_in_background)

        # DLT-source manifest items run the deterministic DLT pipeline; all
        # other data items keep the standard flow. The split is per data item,
        # so mixed datasets get both pipelines, each with its own item subset.
        runs = await _plan_cognify_runs(datasets, user)

        shared_run_kwargs = {
            "pipeline": run_pipeline,
            "user": user,
            "vector_db_config": vector_db_config,
            "graph_db_config": graph_db_config,
            "incremental_loading": incremental_loading,
            "use_pipeline_cache": False,
            "data_per_batch": data_per_batch,
            "rollback_handler": cognify_rollback_handler,
            "llm_config": llm_config,
            "embedding_config": embedding_config,
            "data_cache": data_cache,
        }

        if not runs:
            # Run the run_pipeline in the background or blocking based on executor
            result = await pipeline_executor_func(
                tasks=tasks,
                datasets=datasets,
                pipeline_name="cognify_pipeline",
                **shared_run_kwargs,
            )
        else:
            tasks_for = {
                "dlt_cognify_pipeline": await get_dlt_tasks(
                    chunk_size=chunk_size,
                    chunks_per_batch=chunks_per_batch,
                ),
                "cognify_pipeline": tasks,
            }
            partial_results = []
            for pipeline_name, dataset_ids, items in runs:
                run_kwargs = {"data": items} if items is not None else {}
                partial_results.append(
                    await pipeline_executor_func(
                        tasks=tasks_for[pipeline_name],
                        datasets=dataset_ids,
                        pipeline_name=pipeline_name,
                        **run_kwargs,
                        **shared_run_kwargs,
                    )
                )
            result = _merge_pipeline_results(partial_results)

        dataset_desc = str(datasets) if datasets else "all datasets"
        span.set_attribute(
            COGNEE_RESULT_SUMMARY,
            f"Cognify completed for {dataset_desc}",
        )

        return result


async def get_default_tasks(  # TODO: Find out a better way to do this (Boris's comment)
    user: User = None,
    graph_model: BaseModel = KnowledgeGraph,
    chunker=TextChunker,
    chunk_size: int = None,
    config: Config = None,
    custom_prompt: Optional[str] = None,
    chunks_per_batch: int = None,
    **kwargs,
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

    cognify_config = get_cognify_config()
    embed_triplets = cognify_config.triplet_embedding

    if chunks_per_batch is None:
        chunks_per_batch = (
            cognify_config.chunks_per_batch if cognify_config.chunks_per_batch is not None else 100
        )

    default_tasks = [
        # EXTRACT: classify raw Data items into typed Document objects
        Task(classify_documents),
        # EXTRACT: split Documents into semantic text chunks
        Task(
            extract_chunks_from_documents,
            max_chunk_size=chunk_size or await get_max_chunk_tokens(),
            chunker=chunker,
        ),
        # COGNIFY: LLM-extract entities and relationships into a knowledge graph
        # COGNIFY: LLM-summarize each chunk for retrieval
        Task(
            extract_graph_and_summarize,
            graph_model=graph_model,
            config=config,
            custom_prompt=custom_prompt,
            task_config={"batch_size": chunks_per_batch},
            **kwargs,
        ),
        # LOAD: persist nodes, edges, and embeddings to graph/vector DBs
        Task(
            add_data_points,
            embed_triplets=embed_triplets,
            task_config={"batch_size": chunks_per_batch},
        ),
        Task(extract_dlt_fk_edges),
    ]

    return default_tasks


async def _plan_cognify_runs(datasets, user) -> list[tuple[str, list, Optional[list]]]:
    """Plan pipeline runs for cognify routing.

    DLT-source manifests (external_metadata.source == "dlt_source") run the
    DLT pipeline; all other items run the standard cognify pipeline. The split
    is per data item, so mixed datasets get both pipelines, each with its own
    item subset.

    Returns [(pipeline_name, dataset_ids, data_or_None)]:
    - ("dlt_cognify_pipeline", [ds_id], manifest_items) per manifest dataset
    - ("cognify_pipeline", [ds_id], regular_items) for manifest datasets that
      also contain non-manifest items
    - one ("cognify_pipeline", [ids...], None) entry for all datasets without
      manifests, items loaded by the pipeline itself (unchanged behavior)

    Returns [] when no dataset contains a DLT manifest, in which case the
    caller keeps the original single-call path untouched.
    """
    from sqlalchemy import select

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.models import Data, DatasetData
    from cognee.modules.data.methods import get_authorized_existing_datasets, get_dataset_data
    from cognee.modules.users.methods import get_default_user
    from cognee.tasks.ingestion.dlt_utils import is_dlt_source_manifest

    if user is None:
        user = await get_default_user()

    dataset_list = datasets if isinstance(datasets, list) or datasets is None else [datasets]
    authorized_datasets = await get_authorized_existing_datasets(
        datasets=dataset_list, permission_type="write", user=user
    )
    if not authorized_datasets:
        return []

    # One filtered query to find which requested datasets contain a manifest,
    # instead of loading every dataset's data items.
    authorized_ids = [dataset.id for dataset in authorized_datasets]
    async with get_relational_engine().get_async_session() as session:
        manifest_dataset_ids = set(
            (
                await session.execute(
                    select(DatasetData.dataset_id)
                    .join(Data, Data.id == DatasetData.data_id)
                    .where(
                        DatasetData.dataset_id.in_(authorized_ids),
                        Data.external_metadata["source"].as_string() == "dlt_source",
                    )
                    .distinct()
                )
            )
            .scalars()
            .all()
        )

    if not manifest_dataset_ids:
        return []

    runs: list[tuple[str, list, Optional[list]]] = []
    regular_ids = []
    for dataset in authorized_datasets:
        if dataset.id not in manifest_dataset_ids:
            regular_ids.append(dataset.id)
            continue

        manifest_items, regular_items = [], []
        for item in await get_dataset_data(dataset.id):
            (manifest_items if is_dlt_source_manifest(item) else regular_items).append(item)

        runs.append(("dlt_cognify_pipeline", [dataset.id], manifest_items))
        if regular_items:
            runs.append(("cognify_pipeline", [dataset.id], regular_items))

    if regular_ids:
        runs.append(("cognify_pipeline", regular_ids, None))

    return runs


def _merge_pipeline_results(partial_results: list) -> dict:
    """Merge results from multiple pipeline executor calls.

    Both executor modes return dicts keyed by dataset_id. For a dataset with
    two runs (DLT + regular) the later (regular) run info wins, as it runs
    after the DLT pipeline.
    """
    merged = {}
    for partial in partial_results:
        merged.update(partial)
    return merged


async def get_dlt_tasks(chunk_size: int = None, chunks_per_batch: int = None) -> list[Task]:
    """Deterministic pipeline for DLT-source manifest datasets.

    No LLM tasks: each manifest row becomes one DocumentChunk (vector-indexed
    by add_data_points) and the graph structure comes from the relational
    schema via extract_dlt_source_edges.
    """
    from cognee.tasks.ingestion.extract_dlt_source_edges import extract_dlt_source_edges

    cognify_config = get_cognify_config()
    if chunks_per_batch is None:
        chunks_per_batch = (
            cognify_config.chunks_per_batch if cognify_config.chunks_per_batch is not None else 100
        )

    return [
        # EXTRACT: classify manifest Data items into DltSourceDocument objects
        Task(classify_documents),
        # EXTRACT: one DocumentChunk per manifest row (no text chunking)
        Task(
            extract_chunks_from_documents,
            max_chunk_size=chunk_size or await get_max_chunk_tokens(),
            chunker=TextChunker,
        ),
        # LOAD: persist row chunks and embeddings to graph/vector DBs
        Task(
            add_data_points,
            task_config={"batch_size": chunks_per_batch},
        ),
        # LOAD: schema nodes and deterministic FK edges from the manifest.
        # emitted_schema_docs is shared across batches of this pipeline run so
        # schema nodes are only emitted (and embedded) for the first batch.
        Task(extract_dlt_source_edges, emitted_schema_docs=set()),
    ]


async def get_temporal_tasks(
    user: User = None, chunker=TextChunker, chunk_size: int = None, chunks_per_batch: int = None
) -> list[Task]:
    """
    Builds and returns a list of temporal processing tasks to be executed in sequence.

    The pipeline includes:
    1. Document classification.
    2. Document chunking with a specified or default chunk size.
    3. Event and timestamp extraction from chunks.
    4. Knowledge graph extraction from events.
    5. Batched insertion of data points.

    Args:
        user (User, optional): The user requesting task execution.
        chunker (Callable, optional): A text chunking function/class to split documents. Defaults to TextChunker.
        chunk_size (int, optional): Maximum token size per chunk. If not provided, uses system default.
        chunks_per_batch (int, optional): Number of chunks to process in a single batch in Cognify

    Returns:
        list[Task]: A list of Task objects representing the temporal processing pipeline.
    """
    if chunks_per_batch is None:
        from cognee.modules.cognify.config import get_cognify_config

        configured = get_cognify_config().chunks_per_batch
        chunks_per_batch = configured if configured is not None else 10

    temporal_tasks = [
        # EXTRACT: classify raw Data items into typed Document objects
        Task(classify_documents),
        # EXTRACT: split Documents into semantic text chunks
        Task(
            extract_chunks_from_documents,
            max_chunk_size=chunk_size or await get_max_chunk_tokens(),
            chunker=chunker,
        ),
        # COGNIFY: extract temporal events and timestamps from chunks
        Task(extract_events_and_timestamps, task_config={"batch_size": chunks_per_batch}),
        # COGNIFY: build knowledge graph from extracted events
        Task(extract_knowledge_graph_from_events),
        # LOAD: persist nodes, edges, and embeddings to graph/vector DBs
        Task(add_data_points, task_config={"batch_size": chunks_per_batch}),
    ]

    return temporal_tasks
