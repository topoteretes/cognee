from uuid import UUID

from cognee.api.v1.exceptions.exceptions import DocumentSubgraphNotFoundError
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.engine import DataPoint
from cognee.modules.data.models import Data
from cognee.modules.graph.models.EdgeType import EdgeType
from cognee.modules.graph.utils.prepare_edges_for_storage import get_edge_retrieval_text
from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.graph.utils.convert_node_to_data_point import get_all_subclasses


logger = get_logger()


def _is_contains_edge(edge: dict) -> bool:
    relationship_name = str(edge.get("relationship_name", ""))
    return relationship_name == "contains" or "relationship_name: contains;" in relationship_name


def _get_edge_vector_text(edge: dict) -> str:
    return get_edge_retrieval_text(edge.get("edge_text"), edge.get("relationship_name"))


async def legacy_delete(data: Data, mode: str = "soft"):
    """Delete a single document by its content hash."""

    # Delete from graph database
    deleted_node_ids = await delete_document_subgraph(data.id, mode)

    # Delete from vector database
    vector_engine = get_vector_engine()

    # Determine vector collections dynamically
    subclasses = get_all_subclasses(DataPoint)
    vector_collections = []

    for subclass in subclasses:
        index_fields = subclass.model_fields["metadata"].default.get("index_fields", [])
        for field_name in index_fields:
            vector_collections.append(f"{subclass.__name__}_{field_name}")

    # If no collections found, use default collections
    if not vector_collections:
        vector_collections = [
            "DocumentChunk_text",
            "EdgeType_relationship_name",
            "EntityType_name",
            "Entity_name",
            "TextDocument_name",
            "TextSummary_text",
        ]

    # Delete records from each vector collection that exists
    for collection in vector_collections:
        if await vector_engine.has_collection(collection):
            await vector_engine.delete_data_points(
                collection, [str(node_id) for node_id in deleted_node_ids]
            )


async def delete_document_subgraph(document_id: UUID, mode: str = "soft"):
    """Delete a document and all its related nodes in the correct order."""
    graph_db = await get_graph_engine()
    subgraph = await graph_db.get_document_subgraph(str(document_id))
    if not subgraph:
        raise DocumentSubgraphNotFoundError(f"Document not found with id: {document_id}")

    # Delete in the correct order to maintain graph integrity
    deletion_order = [
        ("orphan_entities", "orphaned entities"),
        ("orphan_types", "orphaned entity types"),
        (
            "made_from_nodes",
            "made_from nodes",
        ),  # Move before chunks since summaries are connected to chunks
        ("chunks", "document chunks"),
        ("document", "document"),
    ]

    deleted_node_ids = []
    for key, description in deletion_order:
        nodes = subgraph[key]
        if nodes:
            for node in nodes:
                node_id = node["id"]

                if key == "chunks":
                    chunk_connections = await graph_db.get_connections(node_id)
                    deleted_node_ids.extend(
                        [
                            str(EdgeType.id_for(_get_edge_vector_text(edge)))
                            for (__, edge, __) in chunk_connections
                            if _is_contains_edge(edge) and _get_edge_vector_text(edge)
                        ]
                    )

                await graph_db.delete_node(node_id)
                deleted_node_ids.append(node_id)

    # If hard mode, also delete degree-one nodes
    if mode == "hard":
        # Get and delete degree one entity nodes
        degree_one_entity_nodes = await graph_db.get_degree_one_nodes("Entity")
        for node in degree_one_entity_nodes:
            await graph_db.delete_node(node["id"])
            deleted_node_ids.append(node["id"])

        # Get and delete degree one entity types
        degree_one_entity_types = await graph_db.get_degree_one_nodes("EntityType")
        for node in degree_one_entity_types:
            await graph_db.delete_node(node["id"])
            deleted_node_ids.append(node["id"])

    return deleted_node_ids
