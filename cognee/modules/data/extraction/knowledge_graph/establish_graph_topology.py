from typing import Type
from pydantic import BaseModel
from cognee.shared.data_models import KnowledgeGraph
from cognee.infrastructure.databases.graph import get_graph_engine
from ...processing.chunk_types.DocumentChunk import DocumentChunk
from .add_model_class_to_graph import add_model_class_to_graph

async def establish_graph_topology(data_chunks: list[DocumentChunk], topology_model: Type[BaseModel]):
    if topology_model == KnowledgeGraph:
        return data_chunks

    graph_engine = await get_graph_engine()

    await add_model_class_to_graph(topology_model, graph_engine)

    return data_chunks


def generate_node_id(node_id: str) -> str:
    return node_id.upper().replace(" ", "_").replace("'", "")
