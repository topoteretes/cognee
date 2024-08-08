from typing import Type
from pydantic import BaseModel

from cognee.modules.data.processing.chunk_types.DocumentChunk import DocumentChunk
from cognee.shared.data_models import KnowledgeGraph
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.data.extraction.knowledge_graph.add_model_class_to_graph import add_model_class_to_graph


async def chunk_to_graph_decomposition(data_chunks: list[DocumentChunk], topology_model: Type[BaseModel]):
    if topology_model == KnowledgeGraph:
        return data_chunks

    graph_engine = await get_graph_engine()

    await add_model_class_to_graph(topology_model, graph_engine)

    return data_chunks


def generate_node_id(node_id: str) -> str:
    return node_id.upper().replace(" ", "_").replace("'", "")
