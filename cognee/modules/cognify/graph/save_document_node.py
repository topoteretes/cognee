from uuid import UUID
from cognee.shared.data_models import Document
from cognee.infrastructure.databases.graph import get_graph_engine

async def save_document_node(document: Document, parent_node_id: UUID = None):
    graph_engine = get_graph_engine()

    await graph_engine.add_node(document.id, document.model_dump())

    if parent_node_id:
        await graph_engine.add_edge(
            parent_node_id,
            document.id,
            "has_document",
            dict(relationship_name = "has_document"),
        )

    return document
