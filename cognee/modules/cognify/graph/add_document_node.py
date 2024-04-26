from cognee.shared.data_models import Document
from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface

async def add_document_node(graph_client: GraphDBInterface, parent_node_id, document_metadata):
    document_id = f"DOCUMENT__{document_metadata['id']}"





    document = await graph_client.extract_node(document_id)

    if not document:
        document = Document(
            id = document_id,
            title = document_metadata["name"],
            file_path = document_metadata["file_path"],
        ).model_dump()

    document["entity_type"] = "Document"

    await graph_client.add_node(document_id, document)

    await graph_client.add_edge(parent_node_id, document_id, "has_document", dict(relationship_name = "has_document"))

    return document_id
