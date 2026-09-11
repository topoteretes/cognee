import asyncio
from pydantic import BaseModel
from typing import Collection, Literal, Union, Optional
from uuid import UUID

from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.cognify.rollback import cognify_rollback_handler
from cognee.modules.cognify.routing import CognifyRoute, cognify_route_for
from cognee.modules.ontology.ontology_env_config import get_ontology_env_config
from cognee.shared.logging_utils import get_logger
from cognee.shared.data_models import KnowledgeGraph
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.llm import get_max_chunk_tokens

from cognee.modules.pipelines import run_pipeline
from cognee.modules.pipelines.tasks.task import Task
from cognee.infrastructure.databases.vector.embeddings.config import EmbeddingConfig
from cognee.infrastructure.llm.config import LLMConfig
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.ontology.ontology_config import Config
from cognee.modules.ontology.get_default_ontology_resolver import (
    get_configured_ontology_mode,
    get_configured_ontology_resolver,
)
from cognee.modules.users.models import User

from cognee.tasks.documents import (
    classify_documents,
    extract_chunks_from_documents,
)
from cognee.tasks.code_graph.code_files import get_code_file_tasks
from cognee.tasks.code_graph.code_repo import get_code_repo_tasks
from cognee.tasks.graph.extract_graph_and_summarize import extract_graph_and_summarize
from cognee.tasks.graph import detect_contradictions
from cognee.tasks.provenance import record_provenance
from cognee.tasks.graph.resolve_temporal_contradictions import resolve_temporal_contradictions
from cognee.tasks.storage import add_data_points
from cognee.modules.pipelines.layers.pipeline_execution_mode import get_pipeline_executor
from cognee.tasks.temporal_graph.extract_events_and_entities import extract_events_and_timestamps
from cognee.tasks.temporal_graph.extract_knowledge_graph_from_events import (
    extract_knowledge_graph_from_events,
)
from cognee.modules.observability import (
    new_span,
    COGNEE_PIPELINE_NAME,
    COGNEE_RESULT_SUMMARY,
    MEMORY_SYSTEM,
    MEMORY_OPERATION,
    record_operation_duration,
    increment_graph_edges,
    increment_graph_nodes,
)


logger = get_logger("cognify")


def _wrap_cognify_exception(error: BaseException, datasets) -> "Exception":
    """Wrap a run-level pipeline exception in a typed CognifyFailedError.

    CognifyFailedError instances pass through unchanged so double-wrapping
    can't happen.
    """
    from cognee.modules.operations import scrub_error_message
    from cognee.modules.pipelines.exceptions import CognifyFailedError

    if isinstance(error, CognifyFailedError):
        return error
    return CognifyFailedError(
        dataset_name=str(datasets) if datasets else None,
        error_class=type(error).__name__,
        error_message=scrub_error_message(error),
        # On this path raise_on_error=False re-raises the original exception —
        # there is no errored run info to hand back.
        hint="Pass raise_on_error=False to get the original exception instead.",
    )


def raise_if_cognify_errored(result) -> None:
    """Raise ``CognifyFailedError`` if any dataset's foreground run errored.

    ``result`` is the blocking executor's ``{dataset_id: PipelineRunInfo}``
    mapping (or a bare run info when no dataset id was present). The first
    errored run wins — a day-0 user has exactly one dataset, and for batch
    users the exception names the dataset so the rest can be retried.
    """
    from cognee.modules.pipelines.exceptions import CognifyFailedError
    from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunErrored

    if isinstance(result, PipelineRunErrored):
        run_infos = [result]
    elif isinstance(result, dict):
        run_infos = [info for info in result.values() if isinstance(info, PipelineRunErrored)]
    else:
        return

    for run_info in run_infos:
        raise CognifyFailedError(
            dataset_name=getattr(run_info, "dataset_name", None),
            error_class=getattr(run_info, "error_class", None),
            error_message=getattr(run_info, "error_message", None)
            or str(getattr(run_info, "payload", "") or ""),
        )


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
    raise_on_error: bool = True,
    chunk_attachment: Optional[Literal["direct", "all"]] = None,
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
        chunk_attachment: How widely each chunk links into the graph extracted from it.
                 Accepts "direct", "all", or None; omitting it is the same as "direct".
                 - "direct": the chunk links to the extracted root, or - if that root is a
                   transparent container - to the children that replaced it. Today's behaviour.
                 - "all": the chunk links once to every stored node reachable from that root,
                   so any entity is one hop from its source chunk.
                 Requires a custom DataPoint graph_model; passing it with KnowledgeGraph
                 raises, since that path already attaches every extracted entity to its chunk.
                 Applies to standard-routed items only, exactly like graph_model - DLT-source
                 manifests and code files run their own task lists and ignore both.
                 Orthogonal to metadata["transparent"], which is a property of the model.
                 SDK-only: not exposed over the REST API. Raises with temporal_cognify=True
                 or while connected to a remote instance; permitted with dry_run=True.
                 Cost of "all": index_graph_edges embeds one EdgeType per distinct edge text,
                 and contains edge text is "<chunk label> contains <node label>." - so a model
                 yielding N nodes per chunk means roughly N extra embedded rows per chunk.

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
    if chunk_attachment is not None:
        # cognify() forwards unknown kwargs into the LLM call, which also takes
        # **kwargs, so an unusable value here has to raise rather than vanish.
        if chunk_attachment not in ("direct", "all"):
            raise ValueError(
                f"Invalid chunk_attachment {chunk_attachment!r}; expected 'direct', 'all', or None."
            )
        if not (isinstance(graph_model, type) and issubclass(graph_model, DataPoint)):
            raise ValueError(
                "chunk_attachment requires a custom DataPoint graph_model; "
                f"{getattr(graph_model, '__name__', graph_model)!r} is not a DataPoint subclass."
            )
        if temporal_cognify:
            raise ValueError(
                "chunk_attachment is not supported with temporal_cognify=True; the temporal "
                "pipeline does not attach extracted graphs to chunks."
            )

    # Route to remote instance if connected via serve()
    from cognee.api.v1.serve.state import get_remote_client

    client = get_remote_client()
    if client is not None:
        if chunk_attachment is not None:
            raise ValueError(
                "chunk_attachment is not supported while connected to a remote Cognee "
                "instance. Call cognee.disconnect() to use it locally."
            )
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

    import time as _time

    _cognify_start_ns = _time.monotonic_ns()

    with new_span("memory.process") as span:
        span.set_attribute(MEMORY_SYSTEM, "cognee")
        span.set_attribute(MEMORY_OPERATION, "process")
        span.set_attribute(COGNEE_PIPELINE_NAME, "cognify")
        if datasets is not None:
            span.set_attribute("cognee.cognify.datasets", str(datasets))

        from cognee.modules.migrations.startup import run_migrations_and_block

        await run_migrations_and_block(datasets, user)

        resolved_resolver = get_configured_ontology_resolver(config)
        resolved_ontology_mode = get_configured_ontology_mode(config)
        config = {
            "ontology_config": {
                "ontology_resolver": resolved_resolver,
                "ontology_mode": resolved_ontology_mode,
            }
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
                chunk_attachment=chunk_attachment,
                **kwargs,
            )

        # By calling get pipeline executor we get a function that will have the run_pipeline run in the background or a function that we will need to wait for
        pipeline_executor_func = get_pipeline_executor(run_in_background=run_in_background)

        # Per-item routing: each data item resolves to the task list its kind
        # requires — DLT-source manifests run the deterministic DLT list, code
        # files run the enola code graph list, everything else runs the
        # standard (or temporal) list. The lists are built once up front and
        # the resolver is a sync closure over them (the distributed runner
        # materializes per-item task columns, so it needs concrete lists, not
        # an async factory). One run_pipeline call, one cognify_pipeline run
        # per dataset, mixed datasets included. Every route is wired
        # EXPLICITLY, the standard route included — no implicit default. An
        # unmapped route (a CognifyRoute member added without a task list
        # here) raises KeyError instead of silently running the standard LLM
        # list on data that was routed away from it.
        tasks_by_route = {
            CognifyRoute.STANDARD: tasks,
            CognifyRoute.DLT_SOURCE: await get_dlt_tasks(
                chunk_size=chunk_size, chunks_per_batch=chunks_per_batch
            ),
            CognifyRoute.CODE: get_code_file_tasks(),
            CognifyRoute.CODE_REPO: get_code_repo_tasks(),
        }

        def resolve_cognify_tasks(data_item):
            return tasks_by_route[cognify_route_for(data_item)]

        try:
            result = await pipeline_executor_func(
                pipeline=run_pipeline,
                datasets=datasets,
                tasks=resolve_cognify_tasks,
                pipeline_name="cognify_pipeline",
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
            )
        except Exception as error:
            # Run-level failures (e.g. an AuthenticationError escaping a task)
            # re-raise straight out of the pipeline generator, bypassing the
            # errored-run-info path below. Wrap them in the same typed,
            # classified exception so foreground callers see ONE failure
            # surface either way; raise_on_error=False keeps the raw exception
            # (today's behavior). Already-typed cognee errors — e.g.
            # PermissionDeniedError / DatasetNotFoundError from dataset
            # resolution — pass through unchanged so callers' except clauses
            # keep matching.
            from cognee.exceptions import CogneeApiError

            if raise_on_error and not run_in_background and not isinstance(error, CogneeApiError):
                raise _wrap_cognify_exception(error, datasets) from error
            raise

        # Loud-by-default failure: a silently "errored" run info is invisible to
        # first-time users (57% of first SDK cognify runs errored and 0 of those
        # accounts ever searched — the run object was never inspected). Raise a
        # typed, classified error instead; batch/pipeline users opt out with
        # raise_on_error=False. Background runs can't raise here — their errors
        # land on the run record and the warm-up marker.
        if raise_on_error and not run_in_background:
            raise_if_cognify_errored(result)

        dataset_desc = str(datasets) if datasets else "all datasets"
        span.set_attribute(
            COGNEE_RESULT_SUMMARY,
            f"Cognify completed for {dataset_desc}",
        )

        _duration_ms = (_time.monotonic_ns() - _cognify_start_ns) / 1_000_000
        _attrs = {"memory.system": "cognee", "memory.operation": "process"}
        record_operation_duration(_duration_ms, _attrs)

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
    chunk_attachment: Optional[Literal["direct", "all"]] = None,
    **kwargs,
) -> list[Task]:
    cognify_config = get_cognify_config()
    embed_triplets = cognify_config.triplet_embedding
    check_contradictions = cognify_config.contradiction_detection
    track_provenance = cognify_config.provenance_tracking

    if chunks_per_batch is None:
        chunks_per_batch = (
            cognify_config.chunks_per_batch if cognify_config.chunks_per_batch is not None else 2000
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
            chunk_attachment=chunk_attachment,
            task_config={"batch_size": chunks_per_batch},
            **kwargs,
        ),
        # LOAD: persist nodes, edges, and embeddings to graph/vector DBs
        Task(
            add_data_points,
            embed_triplets=embed_triplets,
            task_config={"batch_size": chunks_per_batch},
        ),
        # COGNIFY (opt-in): append an audit-grade provenance ledger entry for every
        # document/chunk/entity/relationship this ingestion produced. Runs right
        # after add_data_points (node ids persisted and stable) and before the
        # contradiction spread so the ledger never depends on contradiction edges.
        # Default OFF.
        *(
            [Task(record_provenance, task_config={"batch_size": chunks_per_batch})]
            if track_provenance
            else []
        ),
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


async def get_dlt_tasks(chunk_size: int = None, chunks_per_batch: int = None) -> list[Task]:
    """Deterministic pipeline for DLT-source manifest datasets.

    No LLM tasks: each manifest row becomes one DocumentChunk (vector-indexed
    by add_data_points) and the graph structure comes from the relational
    schema via extract_dlt_source_edges. Deliberate omissions relative to
    get_default_tasks: contradiction detection (an LLM pass; DLT rows are
    deterministic relational data) and functional_relationships (a cognify()
    parameter that only applies to LLM-extracted temporal facts).
    """
    from cognee.tasks.ingestion.extract_dlt_source_edges import extract_dlt_source_edges
    from cognee.tasks.ingestion.purge_stale_dlt_source_artifacts import (
        purge_stale_dlt_source_artifacts,
    )

    cognify_config = get_cognify_config()
    if chunks_per_batch is None:
        chunks_per_batch = (
            cognify_config.chunks_per_batch if cognify_config.chunks_per_batch is not None else 100
        )

    return [
        # EXTRACT: classify manifest Data items into DltSourceDocument objects
        Task(classify_documents),
        # PURGE: manifests have stable ids — drop the source's previously
        # derived artifacts so re-emission replaces instead of accreting.
        Task(purge_stale_dlt_source_artifacts),
        # EXTRACT: one DocumentChunk per manifest row (no text chunking)
        Task(
            extract_chunks_from_documents,
            max_chunk_size=chunk_size or await get_max_chunk_tokens(),
            chunker=TextChunker,
        ),
        # LOAD: persist row chunks and embeddings to graph/vector DBs
        Task(
            add_data_points,
            embed_triplets=cognify_config.triplet_embedding,
            task_config={"batch_size": chunks_per_batch},
        ),
        # LOAD: schema nodes and deterministic FK edges from the manifest.
        # Cross-batch dedup state lives in ctx.extras (per data item = per
        # source), so these Task objects are safe to share across datasets.
        Task(extract_dlt_source_edges),
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
