import asyncio
import inspect
from typing import Type, List, Optional
from pydantic import BaseModel

from cognee.modules.pipelines.tasks.task import task_summary
from cognee.modules.ontology.ontology_env_config import get_ontology_env_config
from cognee.modules.ontology.ontology_config import Config
from cognee.modules.ontology.get_default_ontology_resolver import (
    get_default_ontology_resolver,
    get_ontology_resolver_from_env,
)
from cognee.modules.ontology.base_ontology_resolver import BaseOntologyResolver
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.graph.utils import (
    expand_with_nodes_and_edges,
    retrieve_existing_edges,
)
from cognee.shared.data_models import KnowledgeGraph
from cognee.infrastructure.llm.extraction import extract_content_graph
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.engine.models.Edge import Edge
from cognee.modules.engine.models import Event
from cognee.modules.engine.utils import generate_node_id
from cognee.modules.engine.utils.generate_event_datapoint import generate_event_datapoint
from cognee.tasks.graph.exceptions import (
    InvalidGraphModelError,
    InvalidDataChunksError,
    InvalidChunkGraphInputError,
    InvalidOntologyAdapterError,
)


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


async def _integrate_events(
    data_chunks: List[DocumentChunk],
    chunk_graphs: list,
    entity_nodes: list,
) -> None:
    """Convert EventNode objects from LLM output into Event DataPoints and wire them into chunks.

    For each event in a chunk's graph:
    1. Convert the EventNode (pydantic BaseModel) to an Event DataPoint via generate_event_datapoint.
    2. Resolve participant_node_ids to actual Entity DataPoints already attached to the chunk.
    3. If the event has a `supersedes` description, search for the matching prior event
       in the vector DB and update its status — one targeted search, not a bulk scan.
    4. Append the Event (with participant edges) to chunk.contains.
    """
    if not chunk_graphs:
        return

    # Build a lookup: generate_node_id(original_node_id) → Entity DataPoint
    entity_lookup = {}
    for node in entity_nodes:
        entity_lookup[node.id] = node

    # Collect events that supersede prior events for batch reconciliation
    supersession_queue: list[tuple] = []  # (event_node, event_dp)

    for data_chunk, graph in zip(data_chunks, chunk_graphs):
        if not graph or not hasattr(graph, "events") or not graph.events:
            continue

        for event_node in graph.events:
            event_dp = generate_event_datapoint(event_node)
            event_dp.status = (
                str(event_node.status.value)
                if hasattr(event_node.status, "value")
                else str(event_node.status)
            )

            # Resolve participants and attach as event.attributes = [(Edge, [Entity]), ...]
            event_dp.attributes = []
            for participant_id in event_node.participant_node_ids:
                resolved_id = generate_node_id(participant_id)
                entity = entity_lookup.get(resolved_id)
                if entity is not None:
                    edge = Edge(relationship_type="participated_in")
                    event_dp.attributes.append((edge, [entity]))

            if data_chunk.contains is None:
                data_chunk.contains = []
            data_chunk.contains.append(event_dp)

            if event_node.supersedes:
                supersession_queue.append((event_node, event_dp))

    # Reconcile: for events that supersede prior events, find and update the old ones.
    if supersession_queue:
        await _reconcile_superseded_events(supersession_queue)


async def _reconcile_superseded_events(
    supersession_queue: list[tuple],
) -> None:
    """Search for prior events matching each supersedes description and update their status.

    Only called for events where the LLM flagged a supersession — typically a small
    fraction of all extracted events, so cost is bounded.
    """
    from cognee.infrastructure.databases.vector import get_vector_engine
    from cognee.infrastructure.databases.graph import get_graph_engine

    vector_engine = get_vector_engine()

    # Check if the Event_name collection exists (it won't on first cognify)
    if not await vector_engine.has_collection("Event_name"):
        return

    graph_engine = await get_graph_engine()

    for event_node, event_dp in supersession_queue:
        # One targeted search per superseding event
        results = await vector_engine.search(
            collection_name="Event_name",
            query_text=event_node.supersedes,
            query_vector=None,
            limit=3,
        )

        if not results:
            continue

        # Update the best match — the closest event to the supersedes description.
        # Only update if the new event's status represents a status transition
        # (e.g., planned → cancelled, planned → completed).
        best_match = results[0]
        await graph_engine.update_node_properties(
            {str(best_match.id): {"status": event_dp.status}}
        )


async def integrate_chunk_graphs(
    data_chunks: list[DocumentChunk],
    chunk_graphs: list,
    graph_model: Type[BaseModel],
    ontology_resolver: BaseOntologyResolver,
    pipeline_name: str = None,
    task_name: str = None,
    **kwargs,
) -> List[DocumentChunk]:
    """Integrate chunk graphs with ontology validation and store in databases.

    This function processes document chunks and their associated knowledge graphs,
    validates entities against an ontology resolver, and stores the integrated
    data points and edges in the configured databases.

    Args:
        data_chunks: List of document chunks containing source data
        chunk_graphs: List of knowledge graphs corresponding to each chunk
        graph_model: Pydantic model class for graph data validation
        ontology_resolver: Resolver for validating entities against ontology

    Returns:
        List of updated DocumentChunk objects with integrated data

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
    if ontology_resolver is None or not hasattr(ontology_resolver, "get_subgraph"):
        raise InvalidOntologyAdapterError(
            type(ontology_resolver).__name__ if ontology_resolver else "None"
        )

    if graph_model is not KnowledgeGraph:
        for chunk_index, chunk_graph in enumerate(chunk_graphs):
            data_chunks[chunk_index].contains = chunk_graph

        return data_chunks

    existing_edges_map = await retrieve_existing_edges(
        data_chunks,
        chunk_graphs,
    )

    data_chunks, entity_nodes = expand_with_nodes_and_edges(
        data_chunks, chunk_graphs, ontology_resolver, existing_edges_map
    )

    # Process events extracted inline by the LLM.
    await _integrate_events(data_chunks, chunk_graphs, entity_nodes)

    if entity_nodes:
        if pipeline_name or task_name:
            for node in entity_nodes:
                _stamp_provenance_deep(node, pipeline_name, task_name)

        cache_entity_embeddings = kwargs.get("cache_entity_embeddings")
        if callable(cache_entity_embeddings):
            callback_result = cache_entity_embeddings(entity_nodes, **kwargs)
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

    # Skip LLM extraction for DLT row chunks — their graph is built
    # deterministically by extract_dlt_fk_edges from schema metadata.
    from cognee.modules.data.processing.document_types import DltRowDocument

    dlt_chunks = [
        c for c in data_chunks if isinstance(getattr(c, "is_part_of", None), DltRowDocument)
    ]
    non_dlt_chunks = [c for c in data_chunks if c not in dlt_chunks]

    if not non_dlt_chunks:
        return data_chunks

    calculate_chunk_graphs = kwargs.get("calculate_chunk_graphs")
    if callable(calculate_chunk_graphs):
        extracted = calculate_chunk_graphs(non_dlt_chunks, graph_model, custom_prompt, **kwargs)
        chunk_graphs = await extracted if inspect.isawaitable(extracted) else extracted
    else:
        chunk_graphs = await asyncio.gather(
            *[
                extract_content_graph(
                    chunk.text, graph_model, custom_prompt=custom_prompt, **kwargs
                )
                for chunk in non_dlt_chunks
            ]
        )
    cache_entity_embeddings = kwargs.get("cache_entity_embeddings")
    if callable(cache_entity_embeddings):
        callback_result = cache_entity_embeddings(chunk_graphs, **kwargs)
        if inspect.isawaitable(callback_result):
            await callback_result

    # Note: Filter edges with missing source or target nodes
    if graph_model == KnowledgeGraph:
        for graph in chunk_graphs:
            valid_node_ids = {node.id for node in graph.nodes}
            graph.edges = [
                edge
                for edge in graph.edges
                if edge.source_node_id in valid_node_ids and edge.target_node_id in valid_node_ids
            ]

    # Extract resolver from config if provided, otherwise get default
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

    ontology_resolver = config["ontology_config"]["ontology_resolver"]

    task_name = "extract_graph_from_data"

    integrated = await integrate_chunk_graphs(
        non_dlt_chunks,
        chunk_graphs,
        graph_model,
        ontology_resolver,
        pipeline_name=pipeline_name,
        task_name=task_name,
        **kwargs,
    )

    return integrated + dlt_chunks
