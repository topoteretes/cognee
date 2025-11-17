from cognee.shared.data_models import KnowledgeGraph
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.engine.utils import generate_edge_id, generate_node_id


def extract_relationships(document_chunk: DocumentChunk, graph: KnowledgeGraph, cache: dict = {}):
    relationships = []

    for edge in graph.edges:
        edge_id = f"{edge.source_node_id}_{edge.relationship_name}_{edge.target_node_id}"

        if edge_id not in cache:
            relationship = (
                generate_edge_id(edge.source_node_id),
                generate_edge_id(edge.target_node_id),
                edge.relationship_name,
            )
            cache[edge_id] = relationship
        else:
            relationship = cache[edge_id]

        relationships.append(relationship)

    for node in graph.nodes:
        node_id = generate_node_id(node.id)
        type_node_id = generate_node_id(node.type)
        type_edge_id = f"{str(node_id)}_is_a_{str(type_node_id)}"

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

        chunk_edge_id = f"{str(document_chunk.id)}_contains_{str(node_id)}"

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
