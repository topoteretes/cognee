import asyncio
from typing import Type
from uuid import uuid5
from pydantic import BaseModel
from cognee.modules.graph.utils import get_graph_from_model
from cognee.tasks.summarization.models import TextSummary
from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.data.extraction.extract_summary import extract_summary


async def summarize_text(
    data_points_and_relationships: tuple[list[DocumentChunk], list[DataPoint], list],
    summarization_model: Type[BaseModel],
):
    document_chunks = data_points_and_relationships[0]
    nodes = data_points_and_relationships[1]
    relationships = data_points_and_relationships[2]

    if len(document_chunks) == 0:
        return document_chunks

    chunk_summaries = await asyncio.gather(
        *[extract_summary(chunk.text, summarization_model) for chunk in document_chunks]
    )

    summaries = [
        TextSummary(
            id=uuid5(chunk.id, "TextSummary"),
            made_from=chunk,
            text=chunk_summaries[chunk_index].summary,
        )
        for (chunk_index, chunk) in enumerate(document_chunks)
    ]

    data_points = summaries + nodes

    added_nodes = {}
    added_edges = {}
    visited_properties = {}

    nodes_and_edges: list[tuple] = await asyncio.gather(
        *[
            get_graph_from_model(
                data_point,
                added_nodes=added_nodes,
                added_edges=added_edges,
                visited_properties=visited_properties,
            )
            for data_point in data_points
        ]
    )

    graph_data_deduplication = GraphDataDeduplication()
    deduplicated_nodes_and_edges = [graph_data_deduplication.deduplicate_nodes_and_edges(nodes, edges + relationships) for nodes, edges in nodes_and_edges]

    return deduplicated_nodes_and_edges


class GraphDataDeduplication:
    nodes_and_edges_map: dict

    def __init__(self):
        self.reset()

    def reset(self):
        self.nodes_and_edges_map = {}

    def deduplicate_nodes_and_edges(self, nodes: list, edges: list):
        final_nodes = []
        final_edges = []

        for node in nodes:
            node_key = str(node.id)
            if node_key not in self.nodes_and_edges_map:
                self.nodes_and_edges_map[node_key] = True
                final_nodes.append(node)

        for edge in edges:
            edge_key = str(edge[0]) + str(edge[2]) + str(edge[1])
            if edge_key not in self.nodes_and_edges_map:
                self.nodes_and_edges_map[edge_key] = True
                final_edges.append(edge)

        return final_nodes, final_edges
