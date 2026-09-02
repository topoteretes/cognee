import asyncio
import inspect
from typing import Type, List, Literal, Optional
from pydantic import BaseModel

from cognee.modules.pipelines.tasks.task import task_summary
from cognee.modules.ontology.ontology_config import Config
from cognee.modules.ontology.get_default_ontology_resolver import (
    get_configured_ontology_mode,
    get_configured_ontology_resolver,
)
from cognee.modules.ontology.base_ontology_resolver import BaseOntologyResolver
from cognee.modules.ontology.construct_data_points_and_edges_with_ontology import (
    construct_data_points_and_edges_with_ontology,
)
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.graph.utils import (
    attach_new_edges_to_data_points,
    collect_stored_data_points,
    construct_data_points_and_edges,
    find_existing_edge_identities,
)
from cognee.shared.data_models import KnowledgeGraph
from cognee.infrastructure.llm.extraction import extract_content_graph
from cognee.infrastructure.llm.pipeline_stage import pipeline_stage
from cognee.infrastructure.engine import DataPoint
from cognee.shared.logging_utils import get_logger
from cognee.tasks.graph.exceptions import (
    InvalidGraphModelError,
    InvalidDataChunksError,
    InvalidChunkGraphInputError,
    InvalidOntologyAdapterError,
)

logger = get_logger("extract_graph_from_data")


def _remove_duplicate_extracted_nodes_by_id(
    data_chunks: List[DocumentChunk],
    extracted_graphs: list[KnowledgeGraph],
) -> None:
    """Keep the first extracted node for each graph-local ID.

    Eval capture (SDK-529): while capture is active, every chunk graph that lost nodes
    emits ONE ``extraction.dropped_duplicates`` event carrying the dropped ids, and the
    run manifest counts them under ``extraction.dropped_duplicate_nodes``. The event
    carries ``chunk_id`` as the join key to its ``extraction.chunk_graph`` record:
    ``chunk_index`` is the position within this batch and restarts at 0 for every
    batch of a run, so it alone cannot identify the chunk. Nothing is collected when
    capture is off.

    The loop runs over ``extracted_graphs``, never over ``zip(...)``: deduplication is
    the caller's contract and must cover every graph even if the two lists ever differ
    in length (reachable through the ``calculate_chunk_graphs`` hook). Only the capture
    event needs a chunk, so a graph without one is deduplicated silently.
    """
    from cognee.modules.observability import capture as eval_capture

    active = eval_capture.is_active()

    for chunk_index, extracted_graph in enumerate(extracted_graphs):
        if not extracted_graph:
            continue

        chunk = data_chunks[chunk_index] if chunk_index < len(data_chunks) else None

        nodes_by_id = {}
        dropped_node_ids = [] if active else None
        for node in extracted_graph.nodes:
            if node.id in nodes_by_id:
                # NOTE: This is a lossy strategy; duplicate IDs may deserve more careful handling.
                logger.warning("Ignoring duplicate extracted node ID: %s", node.id)
                if dropped_node_ids is not None:
                    dropped_node_ids.append(node.id)
                continue

            nodes_by_id[node.id] = node

        if len(nodes_by_id) != len(extracted_graph.nodes):
            extracted_graph.nodes = list(nodes_by_id.values())

        if dropped_node_ids and chunk is not None:
            eval_capture.emit(
                eval_capture.KIND_EXTRACTION_DROPPED_DUPLICATES,
                {
                    "chunk_id": str(chunk.id),
                    "chunk_index": chunk_index,
                    "dropped_node_ids": dropped_node_ids,
                    "count": len(dropped_node_ids),
                },
                payload_kind="json",
                stage="extract_graph_from_data",
            )
            eval_capture.bump("extraction.dropped_duplicate_nodes", len(dropped_node_ids))


def capture_chunk_graphs(data_chunks: List[DocumentChunk], chunk_graphs: list) -> None:
    """Snapshot every raw per-chunk graph for eval capture (SDK-529).

    Runs right after extraction and BEFORE ``_remove_duplicate_extracted_nodes_by_id``
    and the ontology canonicalization, both of which mutate the graphs in place — so
    the payload is a ``model_dump(mode="json")`` taken now, never the live object or a
    ``model_copy``. ``chunk_id`` is the join key shared with the dropped-duplicates and
    fuzzy-match events; ``chunk_index`` is the position in this batch (it restarts at 0
    per batch); ``chunk_size_chars`` records the chunk boundary. A structural no-op (no
    dump, no allocation) when capture is off.

    The whole per-chunk payload build is guarded, because the graph is an object this
    function does not own: the standard route hands it freshly validated LLM output,
    but the ``calculate_chunk_graphs`` extension hook and a custom DataPoint-derived
    ``graph_model`` (``arbitrary_types_allowed``) can supply a graph whose dump raises —
    a reference cycle or a non-JSON-native field value. Capture never breaks the
    extraction it observes, so a graph that will not dump costs its own event and
    nothing else.
    """
    from cognee.modules.observability import capture as eval_capture

    if not eval_capture.is_active():
        return

    for chunk_index, (chunk, chunk_graph) in enumerate(zip(data_chunks, chunk_graphs)):
        try:
            payload = {
                "chunk_id": str(chunk.id),
                "chunk_index": chunk_index,
                "chunk_size_chars": len(chunk.text),
                "graph": (
                    chunk_graph.model_dump(mode="json")
                    if isinstance(chunk_graph, BaseModel)
                    else None
                ),
            }
        except Exception as exc:
            # Capture never breaks the extraction it observes.
            logger.debug("chunk graph capture skipped (%s)", exc)
            continue

        eval_capture.emit(
            eval_capture.KIND_EXTRACTION_CHUNK_GRAPH,
            payload,
            payload_kind="json",
            stage="extract_graph_from_data",
        )


def _note_ontology_config(
    ontology_resolver: Optional[BaseOntologyResolver], ontology_mode: Optional[str]
) -> None:
    """Record the run's effective ontology configuration on the eval-capture manifest.

    SDK-529. ``ontology.mode`` is the mode this integration applies (a ``None``
    per-call mode falls back to ONTOLOGY_MODE, exactly as the constructor does); the
    resolver and its matching strategy are reported by class name, and
    ``ontology.threshold`` is the strategy's ``cutoff`` where it has one
    (``FuzzyMatchingStrategy``). All ``None`` for a run without an ontology. A no-op
    when capture is off.
    """
    from cognee.modules.observability import capture as eval_capture

    if not eval_capture.is_active():
        return

    if ontology_mode is None:
        ontology_mode = get_configured_ontology_mode()
    strategy = getattr(ontology_resolver, "matching_strategy", None)
    eval_capture.note("ontology.mode", ontology_mode)
    eval_capture.note(
        "ontology.resolver",
        None if ontology_resolver is None else type(ontology_resolver).__name__,
    )
    eval_capture.note(
        "ontology.matching_strategy", None if strategy is None else type(strategy).__name__
    )
    eval_capture.note("ontology.threshold", getattr(strategy, "cutoff", None))


def _stamp_provenance_deep(data, pipeline_name, task_name, visited=None):
    """Recursively stamp all reachable DataPoints with provenance info."""
    if visited is None:
        visited = set()

    if isinstance(data, DataPoint):
        obj_id = id(data)
        if obj_id in visited:
            return
        visited.add(obj_id)

        if data.source_pipeline is None:
            data.source_pipeline = pipeline_name
        if data.source_task is None:
            data.source_task = task_name

        for field_name in data.model_fields:
            field_value = getattr(data, field_name, None)
            if field_value is not None:
                _stamp_provenance_deep(field_value, pipeline_name, task_name, visited)

    elif isinstance(data, (list, tuple)):
        for item in data:
            _stamp_provenance_deep(item, pipeline_name, task_name, visited)


async def integrate_chunk_graphs(
    data_chunks: list[DocumentChunk],
    chunk_graphs: list,
    graph_model: Type[BaseModel],
    ontology_resolver: Optional[BaseOntologyResolver],
    chunk_attachment: Optional[Literal["direct", "all"]] = None,
    pipeline_name: str = None,
    task_name: str = None,
    ontology_mode: Optional[str] = None,
    **kwargs,
) -> List[DocumentChunk]:
    """Convert extracted graphs into linked data points for later storage.

    Graphs take the pure construction path when no ontology resolver is provided.
    Otherwise, ontology matches are canonicalized and enriched before construction.
    Candidate edges that already exist in graph storage are not attached again.

    Args:
        data_chunks: List of document chunks containing source data
        chunk_graphs: List of knowledge graphs corresponding to each chunk
        graph_model: Pydantic model class for graph data validation
        ontology_resolver: Optional resolver for ontology canonicalization and enrichment
        chunk_attachment: How widely each chunk links into its extracted graph.
            ``"all"`` links it to every stored node; omitted and ``"direct"``
            keep the single-root linkage. Custom DataPoint models only.
        ontology_mode: Per-call ontology mode ("annotate" or "strict"); None
            falls back to the ONTOLOGY_MODE environment value.

    Returns:
        The input chunks, updated with their extracted entities

    Raises:
        InvalidChunkGraphInputError: If input validation fails
        InvalidGraphModelError: If graph model validation fails
        InvalidOntologyAdapterError: If ontology resolver validation fails
    """

    if not isinstance(data_chunks, list) or not isinstance(chunk_graphs, list):
        raise InvalidChunkGraphInputError("data_chunks and chunk_graphs must be lists.")
    if len(data_chunks) != len(chunk_graphs):
        raise InvalidChunkGraphInputError(
            f"length mismatch: {len(data_chunks)} chunks vs {len(chunk_graphs)} graphs."
        )
    if not isinstance(graph_model, type) or not issubclass(graph_model, BaseModel):
        raise InvalidGraphModelError(graph_model)
    if ontology_resolver is not None and not hasattr(ontology_resolver, "get_subgraph"):
        raise InvalidOntologyAdapterError(type(ontology_resolver).__name__)

    _note_ontology_config(ontology_resolver, ontology_mode)

    if not issubclass(graph_model, KnowledgeGraph):
        for chunk_index, chunk_graph in enumerate(chunk_graphs):
            if chunk_attachment == "all" and isinstance(chunk_graph, DataPoint):
                # The field name supplies the "contains" relationship, and the shared
                # edge-text policy fills the label - no Edge wrapper needed here.
                data_chunks[chunk_index].contains = await collect_stored_data_points(chunk_graph)
            else:
                data_chunks[chunk_index].contains = chunk_graph

        return data_chunks

    _remove_duplicate_extracted_nodes_by_id(data_chunks, chunk_graphs)

    if ontology_resolver is None:
        data_points_by_id, edges_by_identity = construct_data_points_and_edges(
            data_chunks,
            chunk_graphs,
        )
    else:
        data_points_by_id, edges_by_identity = construct_data_points_and_edges_with_ontology(
            data_chunks,
            chunk_graphs,
            ontology_resolver,
            ontology_mode=ontology_mode,
        )

    existing_edge_identities = await find_existing_edge_identities(edges_by_identity.keys())
    attach_new_edges_to_data_points(
        data_points_by_id,
        edges_by_identity,
        existing_edge_identities,
    )
    constructed_data_points = list(data_points_by_id.values())

    if constructed_data_points:
        if pipeline_name or task_name:
            for data_point in constructed_data_points:
                _stamp_provenance_deep(data_point, pipeline_name, task_name)

        cache_entity_embeddings = kwargs.get("cache_entity_embeddings")
        if callable(cache_entity_embeddings):
            callback_result = cache_entity_embeddings(constructed_data_points, **kwargs)
            if inspect.isawaitable(callback_result):
                await callback_result

    return data_chunks


@task_summary("Extracted graph from {n} chunk(s)")
async def extract_graph_from_data(
    data_chunks: List[DocumentChunk],
    graph_model: Type[BaseModel],
    config: Optional[Config] = None,
    custom_prompt: Optional[str] = None,
    ctx=None,
    chunk_attachment: Optional[Literal["direct", "all"]] = None,
    **kwargs,
) -> List[DocumentChunk]:
    """
    Extracts and integrates a knowledge graph from the text content of document chunks using a specified graph model.
    """
    pipeline_name = ctx.pipeline_name if ctx else None

    if not isinstance(data_chunks, list) or not data_chunks:
        raise InvalidDataChunksError("must be a non-empty list of DocumentChunk.")
    if not all(hasattr(c, "text") for c in data_chunks):
        raise InvalidDataChunksError("each chunk must have a 'text' attribute")
    if not isinstance(graph_model, type) or not issubclass(graph_model, BaseModel):
        raise InvalidGraphModelError(graph_model)

    calculate_chunk_graphs = kwargs.get("calculate_chunk_graphs")
    if callable(calculate_chunk_graphs):
        extracted = calculate_chunk_graphs(data_chunks, graph_model, custom_prompt, **kwargs)
        chunk_graphs = await extracted if inspect.isawaitable(extracted) else extracted
    else:
        with pipeline_stage("extraction"):
            chunk_graphs = await asyncio.gather(
                *[
                    extract_content_graph(
                        chunk.text, graph_model, custom_prompt=custom_prompt, **kwargs
                    )
                    for chunk in data_chunks
                ]
            )
    # Eval capture (SDK-529): snapshot the raw graphs before anything mutates them.
    capture_chunk_graphs(data_chunks, chunk_graphs)

    cache_entity_embeddings = kwargs.get("cache_entity_embeddings")
    if callable(cache_entity_embeddings):
        callback_result = cache_entity_embeddings(chunk_graphs, **kwargs)
        if inspect.isawaitable(callback_result):
            await callback_result

    ontology_resolver = get_configured_ontology_resolver(config)
    ontology_mode = get_configured_ontology_mode(config)

    task_name = "extract_graph_from_data"

    integrated = await integrate_chunk_graphs(
        data_chunks,
        chunk_graphs,
        graph_model,
        ontology_resolver,
        chunk_attachment=chunk_attachment,
        pipeline_name=pipeline_name,
        task_name=task_name,
        ontology_mode=ontology_mode,
        **kwargs,
    )

    return integrated
