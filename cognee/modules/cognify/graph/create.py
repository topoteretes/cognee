""" This module is responsible for creating a semantic graph """
from typing import  Optional, Any
from pydantic import BaseModel
from cognee.infrastructure.databases.graph.get_graph_client import get_graph_client
from cognee.shared.data_models import GraphDBType


async def generate_node_id(instance: BaseModel) -> str:
    for field in ["id", "doc_id", "location_id", "type_id"]:
        if hasattr(instance, field):
            return f"{instance.__class__.__name__}:{getattr(instance, field)}"
    return f"{instance.__class__.__name__}:default"

async def add_node_and_edge(client, parent_id: Optional[str], node_id: str, node_data: dict, relationship_data: dict):
    await client.add_node(node_id, **node_data)  # Add the current node with its data
    if parent_id:
        # Add an edge between the parent node and the current node with the correct relationship data
        await client.add_edge(parent_id, node_id, **relationship_data)


async def process_attribute(graph_client, parent_id: Optional[str], attribute: str, value: Any):
    if isinstance(value, BaseModel):
        node_id = await generate_node_id(value)

        node_data = value.model_dump(exclude = {"default_relationship"})

        # Use the specified default relationship for the edge between the parent node and the current node
        relationship_data = value.default_relationship.model_dump() if hasattr(value, "default_relationship") else {}

        await add_node_and_edge(graph_client, parent_id, node_id, node_data, relationship_data)

        # Recursively process nested attributes to ensure all nodes and relationships are added to the graph
        for sub_attr, sub_val in value.__dict__.items():  # Access attributes and their values directly
            await process_attribute(graph_client, node_id, sub_attr, sub_val)

    elif isinstance(value, list) and all(isinstance(item, BaseModel) for item in value):
        # For lists of BaseModel instances, process each item in the list
        for item in value:
            await process_attribute(graph_client, parent_id, attribute, item)

async def create_dynamic(graph_model) :
    root_id = await generate_node_id(graph_model)

    node_data = graph_model.model_dump(exclude = {"default_relationship", "id"})

    graph_client = get_graph_client(GraphDBType.NETWORKX)

    await graph_client.add_node(root_id, **node_data)

    for attribute_name, attribute_value in graph_model:
        await process_attribute(graph_client, root_id, attribute_name, attribute_value)

    return graph_client


async def create_semantic_graph(graph_model_instance):
    # Dynamic graph creation based on the provided graph model instance
    graph = await create_dynamic(graph_model_instance)

    return graph



# if __name__ == "__main__":
#     import asyncio

#     # Assuming all necessary imports and GraphDBType, get_graph_client, Document, DocumentType, etc. are defined

#     # Initialize the graph client
#     graph_client = get_graph_client(GraphDBType.NETWORKX)

#     # Define a GraphModel instance with example data
#     graph_model_instance = DefaultGraphModel(
#         id="user123",
#         documents=[
#             Document(
#                 doc_id="doc1",
#                 title="Document 1",
#                 summary="Summary of Document 1",
#                 content_id="content_id_for_doc1",
#                 doc_type=DocumentType(type_id="PDF", description="Portable Document Format"),
#                 categories=[
#                     Category(category_id="finance", name="Finance", default_relationship=Relationship(type="belongs_to")),
#                     Category(category_id="tech", name="Technology", default_relationship=Relationship(type="belongs_to"))
#                 ],
#                 default_relationship=Relationship(type="has_document")
#             ),
#             Document(
#                 doc_id="doc2",
#                 title="Document 2",
#                 summary="Summary of Document 2",
#                 content_id="content_id_for_doc2",
#                 doc_type=DocumentType(type_id="TXT", description="Text File"),
#                 categories=[
#                     Category(category_id="health", name="Health", default_relationship=Relationship(type="belongs_to")),
#                     Category(category_id="wellness", name="Wellness", default_relationship=Relationship(type="belongs_to"))
#                 ],
#                 default_relationship=Relationship(type="has_document")
#             )
#         ],
#         user_properties=UserProperties(
#             custom_properties={"age": "30"},
#             location=UserLocation(location_id="ny", description="New York", default_relationship=Relationship(type="located_in"))
#         ),
#         default_fields={
#             "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         }
#     )

#     # Run the graph creation asynchronously
#     G = asyncio.run(create_semantic_graph(graph_model_instance, graph_client))

#     # Optionally, here you can add more nodes, edges, or perform other operations on the graph G

# async def create_semantic_graph(
# ):
#     graph_type = GraphDBType.NETWORKX
#
#     # Call the get_graph_client function with the selected graph type
#     graph_client = get_graph_client(graph_type)
#
#     print(graph_client)
#
#     await graph_client.load_graph_from_file()
#     #
#     #
#     #
#     # b = await graph_client.add_node("23ds",     {
#     #     "username": "exampleUser",
#     #     "email": "user@example.com"
#     # })
#     #
#     # await graph_client.save_graph_to_file(b)
#     graph_model_instance = DefaultGraphModel(
#         id="user123",
#         documents=[
#             Document(
#                 doc_id="doc1",
#                 title="Document 1",
#                 summary="Summary of Document 1",
#                 content_id="content_id_for_doc1",  # Assuming external content storage ID
#                 doc_type=DocumentType(type_id="PDF", description="Portable Document Format"),
#                 categories=[
#                     Category(category_id="finance", name="Finance",
#                              default_relationship=Relationship(type="belongs_to")),
#                     Category(category_id="tech", name="Technology",
#                              default_relationship=Relationship(type="belongs_to"))
#                 ],
#                 default_relationship=Relationship(type='has_document')
#             ),
#             Document(
#                 doc_id="doc2",
#                 title="Document 2",
#                 summary="Summary of Document 2",
#                 content_id="content_id_for_doc2",
#                 doc_type=DocumentType(type_id="TXT", description="Text File"),
#                 categories=[
#                     Category(category_id="health", name="Health", default_relationship=Relationship(type="belongs_to")),
#                     Category(category_id="wellness", name="Wellness",
#                              default_relationship=Relationship(type="belongs_to"))
#                 ],
#                 default_relationship=Relationship(type='has_document')
#             )
#         ],
#         user_properties=UserProperties(
#             custom_properties={"age": "30"},
#             location=UserLocation(location_id="ny", description="New York",
#                                   default_relationship=Relationship(type='located_in'))
#         ),
#         default_fields={
#             'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             'updated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         }
#     )
#
#     G = await create_dynamic(graph_model_instance, graph_client)
#
#     # print("Nodes and their data:")
#     # for node, data in G.graph.nodes(data=True):
#     #     print(node, data)
#     #
#     # # Print edges with their data
#     # print("\nEdges and their data:")
#     # for source, target, data in G.graph.edges(data=True):
#     #     print(f"{source} -> {target} {data}")
#     # print(G)
#
#
#
#
#
#
#
#
#
#
#     # return await graph_client.create( user_id = user_id, custom_user_properties=custom_user_properties, required_layers=required_layers, default_fields=default_fields, existing_graph=existing_graph)
#
#
# if __name__ == "__main__":
#     import asyncio
#
#     user_id = 'user123'
#     custom_user_properties = {
#         'username': 'exampleUser',
#         'email': 'user@example.com'
#     }
#     asyncio.run(create_semantic_graph())