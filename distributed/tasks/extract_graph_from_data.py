import asyncio
from typing import Type, List

from pydantic import BaseModel

from cognee.modules.graph.utils import (
    expand_with_nodes_and_edges,
    retrieve_existing_edges,
)
from cognee.shared.data_models import KnowledgeGraph
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.ontology.rdf_xml.OntologyResolver import OntologyResolver
from cognee.modules.data.extraction.knowledge_graph import extract_content_graph


async def extract_graph_from_data(
    data_chunks: list[DocumentChunk],
    graph_model: Type[BaseModel],
    ontology_adapter: OntologyResolver = OntologyResolver(),
) -> List[DocumentChunk]:
    """Extracts and integrates a knowledge graph from the text content of document chunks using a specified graph model."""
    chunk_graphs = await asyncio.gather(
        *[extract_content_graph(chunk.text, graph_model) for chunk in data_chunks]
    )

    if graph_model is not KnowledgeGraph:
        for chunk_index, chunk_graph in enumerate(chunk_graphs):
            data_chunks[chunk_index].contains = chunk_graph

        return data_chunks

    existing_edges_map = await retrieve_existing_edges(
        data_chunks,
        chunk_graphs,
    )

    graph_nodes, graph_edges = expand_with_nodes_and_edges(
        data_chunks, chunk_graphs, ontology_adapter, existing_edges_map
    )

    return data_chunks, graph_nodes, graph_edges
