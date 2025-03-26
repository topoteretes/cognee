import asyncio
from typing import Type, List
from cognee.modules.users.models import User
from pydantic import BaseModel

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.ontology.rdf_xml.OntologyResolver import OntologyResolver
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.data.extraction.knowledge_graph import extract_content_graph
from cognee.modules.graph.utils import (
    expand_with_nodes_and_edges,
    retrieve_existing_edges,
)
from cognee.shared.data_models import KnowledgeGraph
from cognee.tasks.storage import add_data_points


async def integrate_chunk_graphs(
    data_chunks: list[DocumentChunk],
    chunk_graphs: list,
    graph_model: Type[BaseModel],
    ontology_adapter: OntologyResolver,
    user: User = None,
) -> List[DocumentChunk]:
    """Updates DocumentChunk objects, integrates data points and edges into databases."""
    graph_engine = await get_graph_engine()

    if graph_model is not KnowledgeGraph:
        for chunk_index, chunk_graph in enumerate(chunk_graphs):
            data_chunks[chunk_index].contains = chunk_graph

        await add_data_points(chunk_graphs, user=user)
        return data_chunks

    existing_edges_map = await retrieve_existing_edges(
        data_chunks,
        chunk_graphs,
        graph_engine,
    )

    graph_nodes, graph_edges = expand_with_nodes_and_edges(
        data_chunks, chunk_graphs, ontology_adapter, existing_edges_map
    )

    if len(graph_nodes) > 0:
        await add_data_points(graph_nodes, user=user)

    if len(graph_edges) > 0:
        await graph_engine.add_edges(graph_edges, user=user)

    return data_chunks


async def extract_graph_from_data(
    data_chunks: List[DocumentChunk],
    graph_model: Type[BaseModel],
    user: User = None,
    ontology_adapter: OntologyResolver = None,
) -> List[DocumentChunk]:
    """
    Extracts a knowledge graph from the text content of document chunks using a specified graph model.
    """
    chunk_graphs = await asyncio.gather(
        *[extract_content_graph(chunk.text, graph_model) for chunk in data_chunks]
    )

    return await integrate_chunk_graphs(
        data_chunks, chunk_graphs, graph_model, ontology_adapter or OntologyResolver(), user=user
    )
