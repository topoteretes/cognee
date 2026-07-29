import asyncio
from pydantic import BaseModel
from typing import Collection, Union, Optional
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
from cognee.tasks.graph import detect_contradictions
from cognee.tasks.graph.resolve_temporal_contradictions import resolve_temporal_contradictions
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
    functional_relationships: Optional[Collection[str]] = None,
    data_per_batch: int = 20,
    llm_config: Optional[LLMConfig] = None,
    embedding_config: Optional[EmbeddingConfig] = None,
    data_cache: bool = True,
    dry_run: bool = False,
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
        functional_relationships: Relationship names that hold a single target per source
                      (e.g. {"ceo_of"}). Once the graph is written, conflicting assertions
                      of those relationships are resolved by recency: the most recent one
                      stays current and the older ones are tagged as superseded — nothing
                      is deleted. Off by default, because most cognee relationships are
                      legitimately many-valued and collapsing them would corrupt the graph.
        dry_run: If True, return a stage-level estimate of LLM token usage and rough cost
                 without making LLM calls or writing graph results. The estimate covers all
                 data in the selected dataset(s); an incremental run may process fewer items.

    Returns:
        Union[dict, list[PipelineRunInfo], DryRunEstimate]:
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
        if dry_run:
            raise ValueError(
                "dry_run is not supported while connected to a remote Cognee instance. "
                "Call cognee.disconnect() to estimate against local data."
            )
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

        if dry_run:
            if temporal_cognify:
                raise ValueError("dry_run is supported for the default cognify pipeline only.")
            from cognee.modules.cognify.estimator import estimate_cognify_dry_run

            return await estimate_cognify_dry_run(
                datasets,
                user=user,
                graph_model=graph_model,
                chunker=chunker,
                chunk_size=chunk_size or await get_max_chunk_tokens(),
                custom_prompt=custom_prompt,
            )

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
                functional_relationships=functional_relationships,
                **kwargs,
            )

        # By calling get pipeline executor we get a function that will have the run_pipeline run in the background or a function that we will need to wait for
        pipeline_executor_func = get_pipeline_executor(run_in_background=run_in_background)

        # DLT-source manifest items run the deterministic DLT pipeline; all
        # other data items keep the standard flow (see _plan_cognify_runs).
        runs = await _plan_cognify_runs(datasets, user)

        result = await _execute_cognify_runs(
            runs,
            executor=pipeline_executor_func,
            cognify_tasks=tasks,
            chunk_size=chunk_size,
            chunks_per_batch=chunks_per_batch,
            user=user,
            vector_db_config=vector_db_config,
            graph_db_config=graph_db_config,
            incremental_loading=incremental_loading,
            data_per_batch=data_per_batch,
            llm_config=llm_config,
            embedding_config=embedding_config,
            data_cache=data_cache,
        )

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
    functional_relationships: Optional[Collection[str]] = None,
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
    check_contradictions = cognify_config.contradiction_detection

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
        # COGNIFY (opt-in): flag facts in this ingestion that contradict facts
        # already in the graph. Runs last so both new and existing facts are
        # persisted and comparable. Default OFF — when the flag is off this spread
        # is empty and the task list is identical to the pre-detection pipeline.
        *(
            [Task(detect_contradictions, task_config={"batch_size": chunks_per_batch})]
            if check_contradictions
            else []
        ),
    ]

    # OPTIONAL: for the relationships declared single-valued, tag the assertions a
    # more recent one replaced. Runs last so the new facts are already stored and
    # comparable with the ones they supersede; disabled by default.
    if functional_relationships:
        default_tasks.append(
            Task(
                resolve_temporal_contradictions,
                functional_relationships=functional_relationships,
                task_config={"batch_size": chunks_per_batch},
            )
        )

    return default_tasks


async def _execute_cognify_runs(
    runs,
    executor,
    cognify_tasks,
    chunk_size,
    chunks_per_batch,
    user,
    vector_db_config,
    graph_db_config,
    incremental_loading,
    data_per_batch,
    llm_config,
    embedding_config,
    data_cache,
):
    """Execute the planned pipeline runs and merge their results.

    Each run is (pipeline_name, datasets, items): items is an explicit data
    subset for mixed datasets, or None to let the pipeline load the dataset
    itself. The DLT task list is built lazily, only when a run needs it.
    """
    tasks_for = {"cognify_pipeline": cognify_tasks}
    partial_results = []
    for pipeline_name, run_datasets, items in runs:
        if pipeline_name not in tasks_for:
            tasks_for[pipeline_name] = await get_dlt_tasks(
                chunk_size=chunk_size, chunks_per_batch=chunks_per_batch
            )
        data_kwargs = {"data": items} if items is not None else {}
        partial_results.append(
            await executor(
                pipeline=run_pipeline,
                tasks=tasks_for[pipeline_name],
                datasets=run_datasets,
                pipeline_name=pipeline_name,
                user=user,
                vector_db_config=vector_db_config,
                graph_db_config=graph_db_config,
                incremental_loading=incremental_loading,
                use_pipeline_cache=False,
                data_per_batch=data_per_batch,
                rollback_handler=cognify_rollback_handler,
                llm_config=llm_config,
                embedding_config=embedding_config,
                data_cache=data_cache,
                **data_kwargs,
            )
        )
    return _merge_pipeline_results(partial_results)


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

    When no dataset contains a manifest, the plan is a single standard run
    that passes the original ``datasets`` argument through unchanged (items
    are loaded by the pipeline itself).

    Routing never guesses: the standard plan is only chosen when the probe
    *proves* there is nothing to route (no datasets, or no manifests in
    them). A probe failure raises — silently falling back could send
    manifest data through the LLM pipeline, producing a wrong graph at LLM
    cost with no error anywhere.
    """
    from cognee.exceptions import CogneeSystemError

    try:
        return await _probe_cognify_runs(datasets, user)
    except Exception as error:
        raise CogneeSystemError(
            f"DLT cognify routing probe failed for datasets {datasets!r}; "
            "cannot determine which pipeline the data requires. "
            f"Underlying error: {error!r}"
        ) from error


async def _probe_cognify_runs(datasets, user) -> list[tuple[str, list, Optional[list]]]:
    """The fallible half of ``_plan_cognify_runs``: resolve datasets and split
    manifest items from regular ones."""
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
        return [("cognify_pipeline", datasets, None)]

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
        return [("cognify_pipeline", datasets, None)]

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
        # emitted_schema_docs and emitted_value_node_ids are shared across
        # batches of this pipeline run so schema nodes and column value nodes
        # are only emitted (and embedded) once per run, not once per batch.
        Task(
            extract_dlt_source_edges,
            emitted_schema_docs=set(),
            emitted_value_node_ids=set(),
        ),
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
