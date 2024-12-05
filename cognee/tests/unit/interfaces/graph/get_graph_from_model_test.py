import asyncio
import random
from typing import List
from uuid import uuid5, NAMESPACE_OID

from cognee.infrastructure.engine import DataPoint
from cognee.modules.graph.utils import get_graph_from_model

class Document(DataPoint):
    path: str

class DocumentChunk(DataPoint):
    part_of: Document
    text: str
    contains: List["Entity"] = None

class EntityType(DataPoint):
    name: str

class Entity(DataPoint):
    name: str
    is_type: EntityType

DocumentChunk.model_rebuild()


async def get_graph_from_model_test():
    document = Document(path = "file_path")

    document_chunks = [DocumentChunk(
        id = uuid5(NAMESPACE_OID, f"file{file_index}"),
        text = "some text",
        part_of = document,
        contains = [],
    ) for file_index in range(1)]

    for document_chunk in document_chunks:
        document_chunk.contains.append(Entity(
            name = f"Entity",
            is_type = random.choice([
                EntityType(
                    name = "Type 1",
                ),
                EntityType(
                    name = "Type 2",
                ),
            ]),
        ))

    nodes = []
    edges = []

    added_nodes = {}
    added_edges = {}

    results = await asyncio.gather(*[
        get_graph_from_model(
            document_chunk,
            added_nodes = added_nodes,
            added_edges = added_edges,
        ) for document_chunk in document_chunks
    ])

    for result_nodes, result_edges in results:
        nodes.extend(result_nodes)
        edges.extend(result_edges)

    assert len(nodes) == 4
    assert len(edges) == 3

if __name__ == "__main__":
    asyncio.run(get_graph_from_model_test())
