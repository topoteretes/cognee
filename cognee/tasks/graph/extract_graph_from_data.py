import asyncio
import inspect
from typing import Type, List, Optional
from pydantic import BaseModel

from cognee.modules.pipelines.tasks.task import task_summary
from cognee.modules.ontology.ontology_config import Config
from cognee.modules.ontology.get_default_ontology_resolver import get_configured_ontology_resolver
from cognee.modules.ontology.base_ontology_resolver import BaseOntologyResolver
from cognee.modules.ontology.construct_data_points_and_edges_with_ontology import (
    construct_data_points_and_edges_with_ontology,
)
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.graph.utils import (
    attach_new_edges_to_data_points,
    construct_data_points_and_edges,
    find_existing_edge_identities,
)
from cognee.shared.data_models import KnowledgeGraph
from cognee.infrastructure.llm.extraction import extract_content_graph
from cognee.infrastructure.llm.pipeline_stage import pipeline_stage
from cognee.infrastructure.engine import DataPoint
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


async def integrate_chunk_graphs(
    data_chunks: list[DocumentChunk],
    chunk_graphs: list,
    graph_model: Type[BaseModel],
    ontology_resolver: Optional[BaseOntologyResolver],
    pipeline_name: str = None,
    task_name: str = None,
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

    if not issubclass(graph_model, KnowledgeGraph):
        for chunk_index, chunk_graph in enumerate(chunk_graphs):
            data_chunks[chunk_index].contains = chunk_graph

        return data_chunks

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

    # Partition in a single pass: a list-membership check against dlt_chunks
    # rescans the list for every chunk (O(n^2) with Pydantic __eq__ comparisons),
    # which becomes a bottleneck on the extraction hot path for large DLT sources.
    dlt_chunks = []
    non_dlt_chunks = []
    for c in data_chunks:
        if isinstance(getattr(c, "is_part_of", None), DltRowDocument):
            dlt_chunks.append(c)
        else:
            non_dlt_chunks.append(c)

    if not non_dlt_chunks:
        return data_chunks

    calculate_chunk_graphs = kwargs.get("calculate_chunk_graphs")
    if callable(calculate_chunk_graphs):
        extracted = calculate_chunk_graphs(non_dlt_chunks, graph_model, custom_prompt, **kwargs)
        chunk_graphs = await extracted if inspect.isawaitable(extracted) else extracted
    else:
        with pipeline_stage("extraction"):
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

    ontology_resolver = get_configured_ontology_resolver(config)

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
