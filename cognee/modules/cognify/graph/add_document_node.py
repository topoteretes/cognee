from cognee.shared.data_models import Document
# from cognee.modules.cognify.graph.add_label_nodes import add_label_nodes
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

    document["type"] = "Document"

    await graph_client.add_node(document_id, document)
    print(f"Added document node: {document_id}")

    await graph_client.add_edge(
        parent_node_id,
        document_id,
        "has_document",
        dict(relationship_name = "has_document"),
    )

    #
    # await add_label_nodes(graph_client, document_id, document_metadata["keywords"].split("|"))

    return document_id
