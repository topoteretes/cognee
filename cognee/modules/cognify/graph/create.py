""" This module is responsible for creating a semantic graph """
import logging
from typing import  Optional, Any
from pydantic import BaseModel

from cognee.infrastructure import infrastructure_config
from cognee.infrastructure.databases.graph.get_graph_client import get_graph_client
from cognee.shared.data_models import GraphDBType


async def generate_node_id(instance: BaseModel) -> str:
    for field in ["id", "doc_id", "location_id", "type_id", "node_id"]:
        if hasattr(instance, field):
            return f"{instance.__class__.__name__}:{getattr(instance, field)}"
    return f"{instance.__class__.__name__}_default"

async def add_node(client, parent_id: Optional[str], node_id: str, node_data: dict):


    if node_id != "Relationship_default":

        result = await client.add_node(node_id, **node_data)
        return result
    if parent_id:
        # Add an edge between the parent node and the current node with the correct relationship data
        if infrastructure_config.get_config()["graph_engine"] == GraphDBType.NETWORKX:
            if node_id != "Relationship_default":
                await client.add_edge(parent_id, node_id, **node_data)


async def add_edge(client, parent_id: Optional[str], node_id: str,node_data: dict, created_node_ids, custom_match = None):

    if (custom_match) or (node_id == "Relationship_default" and parent_id):
        # Initialize source and target variables outside the loop
        source, target = None, None

        # Assuming 'source' and 'target' are keys in node_data
        source_key = node_data.get('source', '').lower()
        target_key = node_data.get('target', '').lower()


        # Search for source and target nodes in the created_node_ids list
        for node in created_node_ids:
            node_id_lower = node['nodeId'].lower()
            if source_key in node_id_lower:
                source = node['nodeId']
                print("SOURCE FOUND", source)
            elif target_key in node_id_lower:
                target = node['nodeId']
                print("TARGET FOUND", target)

        # Check if both source and target are found
        if source and target:
            relationship_details = {
                'source': source,
                'target': target,
                'type': node_data.get('type')  # Safely access 'type' from node_data
            }
            # If you need to do something with relationship_details (e.g., add an edge), do it here
            # For demonstration, we'll just print the results
            print("RELATIONSHIP DETAILS", relationship_details)

            await client.add_edge(relationship_details['source'], relationship_details['target'], relationship_type=relationship_details['type'])




async def process_attribute(graph_client, parent_id: Optional[str], attribute: str, value: Any, created_node_ids=None):

    if created_node_ids is None:
        created_node_ids = []
    if isinstance(value, BaseModel):
        node_id = await generate_node_id(value)
        node_data = value.model_dump()

        # Use the specified default relationship for the edge between the parent node and the current node

        created_node_id = await add_node(graph_client, parent_id, node_id, node_data)

        created_node_ids.append(created_node_id)

        # await add_edge(graph_client, parent_id, node_id, node_data, relationship_data,created_node_ids)

        # Recursively process nested attributes to ensure all nodes and relationships are added to the graph
        for sub_attr, sub_val in value.__dict__.items():  # Access attributes and their values directly

            out = await process_attribute(graph_client, node_id, sub_attr, sub_val)

            created_node_ids.extend(out)

    elif isinstance(value, list) and all(isinstance(item, BaseModel) for item in value):
        # For lists of BaseModel instances, process each item in the list
        for item in value:
            out = await process_attribute(graph_client, parent_id, attribute, item, created_node_ids)
            created_node_ids.extend(out)

    return created_node_ids


async def process_attribute_edge(graph_client, parent_id: Optional[str], attribute: str, value: Any, created_node_ids=None):

    if isinstance(value, BaseModel):
        node_id = await generate_node_id(value)


        node_data = value.model_dump()
        relationship_data = {}

        await add_edge(graph_client, parent_id, node_id, node_data,created_node_ids)

        # Recursively process nested attributes to ensure all nodes and relationships are added to the graph
        for sub_attr, sub_val in value.__dict__.items():  # Access attributes and their values directly

            await process_attribute_edge(graph_client, node_id, sub_attr, sub_val, created_node_ids)

    elif isinstance(value, list) and all(isinstance(item, BaseModel) for item in value):
        # For lists of BaseModel instances, process each item in the list
        for item in value:
            await process_attribute_edge(graph_client, parent_id, attribute, item)

    return created_node_ids

async def create_dynamic(graph_model, graph_client) :
    root_id = await generate_node_id(graph_model)


    # node_data = graph_model.model_dump(exclude = {"default_relationship", "id"})
    node_data = graph_model.model_dump()


    root_id = root_id.replace(":", "_")

    _ = node_data.pop("node_id", None)

    created_node_ids = []
    out = await graph_client.add_node(root_id, **node_data)
    created_node_ids.append(out)
    for attribute_name, attribute_value in graph_model:
        # print("ATTRIB NAME", attribute_name)
        # print("ATTRIB VALUE", attribute_value)
        ids = await process_attribute(graph_client, root_id, attribute_name, attribute_value)
        created_node_ids.extend(ids)

    flattened_and_deduplicated = list({
                                          item['nodeId']: item
                                          # Use the 'nodeId' as the unique key in the dictionary comprehension
                                          for sublist in created_node_ids if sublist  # Ensure sublist is not None
                                          for item in sublist  # Iterate over items in the sublist
                                      }.values())

    for attribute_name, attribute_value in graph_model:
        ids = await process_attribute_edge(graph_client, root_id, attribute_name, attribute_value, flattened_and_deduplicated)

    return graph_client


async def create_semantic_graph(graph_model_instance, graph_client):
    # Dynamic graph creation based on the provided graph model instance
    graph = await create_dynamic(graph_model_instance, graph_client)

    return graph



# if __name__ == "__main__":
#     import asyncio
#
#     user_id = 'user123'
#     custom_user_properties = {
#         'username': 'exampleUser',
#         'email': 'user@example.com'
#     }
#     asyncio.run(create_semantic_graph())