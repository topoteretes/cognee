import asyncio
from typing import List
from uuid import NAMESPACE_OID, uuid5

from cognee.infrastructure.engine import DataPoint
from cognee.modules.graph.utils import get_graph_from_model


class Document(DataPoint):
    path: str
    metadata: dict = {"index_fields": []}


class DocumentChunk(DataPoint):
    part_of: Document
    text: str
    contains: List["Entity"] = None
    metadata: dict = {"index_fields": ["text"]}


class EntityType(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}


class Entity(DataPoint):
    name: str
    is_type: EntityType
    metadata: dict = {"index_fields": ["name"]}


DocumentChunk.model_rebuild()


async def get_graph_from_model_test():
    document = Document(path="file_path")

    document_chunk = DocumentChunk(
        id=uuid5(NAMESPACE_OID, "file_name"),
        text="some text",
        part_of=document,
        contains=[],
    )

    document_chunk.contains.append(
        Entity(
            name="Entity",
            is_type=EntityType(
                name="Type 1",
            ),
        )
    )

    added_nodes = {}
    added_edges = {}
    visited_properties = {}

    result = await get_graph_from_model(
        document_chunk,
        added_nodes=added_nodes,
        added_edges=added_edges,
        visited_properties=visited_properties,
    )

    nodes = result[0]
    edges = result[1]

    assert len(nodes) == 4
    assert len(edges) == 3

    document_chunk_node = next(filter(lambda node: node.type is "DocumentChunk", nodes))
    assert not hasattr(document_chunk_node, "part_of"), "Expected part_of attribute to be removed"


if __name__ == "__main__":
    asyncio.run(get_graph_from_model_test())
