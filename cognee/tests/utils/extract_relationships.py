from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.engine.models import Entity, EntityType
from cognee.shared.data_models import KnowledgeGraph


def extract_relationships(
    document_chunk: DocumentChunk, graph: KnowledgeGraph, cache: dict | None = None
):
    if cache is None:
        cache = {}
    relationships = []

    for edge in graph.edges:
        edge_id = f"{edge.source_node_id}_{edge.relationship_name}_{edge.target_node_id}"

        if edge_id not in cache:
            relationship = (
                Entity.id_for(edge.source_node_id),
                Entity.id_for(edge.target_node_id),
                edge.relationship_name,
            )
            cache[edge_id] = relationship
        else:
            relationship = cache[edge_id]

        relationships.append(relationship)

    for node in graph.nodes:
        node_id = Entity.id_for(node.id)
        type_node_id = EntityType.id_for(node.type)
        type_edge_id = f"{node_id!s}_is_a_{type_node_id!s}"

        if type_edge_id not in cache:
            relationship = (
                node_id,
                type_node_id,
                "is_a",
            )
            cache[type_edge_id] = relationship
        else:
            relationship = cache[type_edge_id]

        relationships.append(relationship)

        chunk_edge_id = f"{document_chunk.id!s}_contains_{node_id!s}"

        if chunk_edge_id not in cache:
            relationship = (
                document_chunk.id,
                node_id,
                "contains",
            )
            cache[chunk_edge_id] = relationship
        else:
            relationship = cache[chunk_edge_id]

        relationships.append(relationship)

    return relationships
