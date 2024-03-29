from cognee.infrastructure.databases.graph.get_graph_client import get_graph_client
from cognee.shared.data_models import GraphDBType, Document, DocumentType, Category, Relationship
from .create import add_node_and_edge

def create_category(category_name: str):
    return Category(
        category_id = category_name.lower(),
        name = category_name,
        default_relationship = Relationship(type = "belongs_to")
    )

async def add_document_node(graph_client, parent_id, document_data):
    # graph_client = get_graph_client(GraphDBType.NETWORKX)
    # await graph_client.load_graph_from_file()

    document_id = f"DOCUMENT:{document_data['id']}"

    document = Document(
        doc_id = document_id,
        title = document_data["name"],
        doc_type = DocumentType(type_id = "PDF", description = "Portable Document Format"),
        categories = list(map(create_category, document_data["categories"])) if "categories" in document_data else [],
    )

    document_dict = document.model_dump()
    relationship = Relationship(type = "has_document").model_dump()

    await add_node_and_edge(graph_client, parent_id, document_id, document_dict, relationship)
