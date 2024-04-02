from cognee.infrastructure import infrastructure_config
from cognee.infrastructure.databases.graph.get_graph_client import get_graph_client
from cognee.shared.data_models import GraphDBType, Document, DocumentType, Category, Relationship
from .create import add_node, add_edge

def create_category(category_name: str):
    return Category(
        category_id = category_name.lower(),
        name = category_name,
        default_relationship = Relationship(type = "belongs_to")
    )

async def add_document_node(graph_client, parent_id, document_data):
    # graph_client = get_graph_client(GraphDBType.NETWORKX)
    # await graph_client.load_graph_from_file()

    document_id = f"DOCUMENT_{document_data['id']}"

    document = Document(
        doc_id = document_id,
        description = document_data["description"],
        title = document_data["name"],
        doc_type = DocumentType(type_id = "PDF", description = "Portable Document Format"),
        categories = list(map(create_category, document_data["categories"])) if "categories" in document_data else [],
    )

    document_dict = document.model_dump()
    relationship = Relationship(type = "has_document").model_dump()

    created_node_ids = await add_node(graph_client, parent_id, document_id, document_dict)

    if infrastructure_config.get_config()["graph_engine"] == GraphDBType.NEO4J:
        print("graph_client: ", graph_client)
        await graph_client.add_edge( parent_id, document_id, relationship['type'])


