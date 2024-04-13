""" This module is responsible for creating a semantic graph """
from typing import  Optional, Any
from pydantic import BaseModel

from cognee.infrastructure import infrastructure_config
from cognee.shared.data_models import GraphDBType


async def generate_node_id(instance: BaseModel) -> str:
    for field in ["id", "doc_id", "location_id", "type_id", "node_id"]:
        if hasattr(instance, field):
            return f"{instance.__class__.__name__}:{getattr(instance, field)}"
    return f"{instance.__class__.__name__}_default"


async def add_node(client, parent_id: Optional[str], node_id: str, node_data: dict) -> Optional[Any]:
    """
    Asynchronously adds a node to a graph database and, if applicable, an edge from a parent node to this new node.

    Parameters:
    - client: The database client used to interact with the graph database.
    - parent_id (Optional[str]): The unique identifier of the parent node. If provided, an edge will be added from the parent to the new node.
    - node_id (str): The unique identifier of the new node. If this is "Relationship_default", the node will not be added.
    - node_data (dict): A dictionary containing the data to be stored in the new node.

    Returns:
    - The result of the node addition operation if successful, otherwise None. This could be the newly added node object, a confirmation message, or any relevant data returned by the database client.

    Raises:
    - Exception: If there is an error during the node or edge addition process, it logs the error and continues without interrupting the execution flow.

    Note:
    - The function currently supports adding edges only if the graph database engine is NETWORKX, as specified in the global `infrastructure_config`.
    """

    # Initialize result to None to ensure a clear return path
    result = None

    # Proceed only if the node_id is not meant for default relationships
    if node_id != "Relationship_default":
        try:
            # Attempt to add the node to the graph database
            result = await client.add_node(node_id, node_properties = node_data)

            # Add an edge if a parent ID is provided and the graph engine is NETWORKX
            if parent_id and "default_relationship" in node_data and infrastructure_config.get_config()["graph_engine"] == GraphDBType.NETWORKX:
                await client.add_edge(parent_id, node_id, relationship_name = node_data["default_relationship"]["type"], edge_properties = node_data)
        except Exception as e:
            # Log the exception; consider a logging framework for production use
            print(f"Error adding node or edge: {e}")
            # Depending on requirements, you may want to handle the exception differently

    return result



# async def add_edge(client, parent_id: Optional[str], node_id: str, node_data: Dict[str, Any], created_node_ids: List[Dict[str, str]]) -> None:
#     """
#     Asynchronously adds an edge between two nodes in a graph based on provided node data and identifiers.
#
#     This function is specifically designed to handle cases where the edge represents a default relationship,
#     identified by 'node_id' being 'Relationship_default'. It searches for source and target nodes based on
#     'source' and 'target' keys in the provided 'node_data' dictionary, within a list of previously created nodes.
#
#     Parameters:
#     - client: The graph database client instance used to interact with the graph.
#     - parent_id (Optional[str]): The unique identifier of the parent node. This is currently not used in the function, but may be relevant for extended functionality.
#     - node_id (str): The identifier for the node. The function specifically checks for 'Relationship_default' to proceed with edge creation.
#     - node_data (Dict[str, Any]): A dictionary containing data related to the node, including keys for 'source', 'target', and 'type' to construct the edge.
#     - created_node_ids (List[Dict[str, str]]): A list of dictionaries, each containing 'nodeId' keys representing nodes previously added to the graph.
#
#     Returns:
#     None: The function performs an asynchronous operation to add an edge to the graph but does not return a value.
#
#     Raises:
#     - This function currently does not explicitly raise exceptions but relies on the calling context to handle exceptions raised by 'client.add_edge'.
#
#     Example:
#     ```
#     await add_edge(client, None, "Relationship_default", {'source': 'Node1', 'target': 'Node2', 'type': 'CONNECTS'}, [{'nodeId': 'Node1'}, {'nodeId': 'Node2'}])
#     ```
#     """
#     if node_id == "Relationship_default" and parent_id:
#         # Extract source and target node keys in lower case for case-insensitive matching
#         source_key = node_data.get('source', '').lower()
#         target_key = node_data.get('target', '').lower()
#
#         # Find the first matching node IDs for source and target using case-insensitive comparison
#         source = next((node['nodeId'] for node in created_node_ids if source_key in node['nodeId'].lower()), None)
#         target = next((node['nodeId'] for node in created_node_ids if target_key in node['nodeId'].lower()), None)
#
#         if source and target:
#             # Construct relationship details based on found source and target nodes
#             relationship_details = {
#                 'source': source,
#                 'target': target,
#                 'type': node_data.get('type')
#             }
#
#             # Add an edge between the source and target nodes using the client
#             await client.add_edge(source, target, relationship_name=relationship_details['type'])



async def add_edge(client, parent_id: Optional[str], node_id: str, node_data: dict, created_node_ids):

    if node_id == "Relationship_default" and parent_id:
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

            await client.add_edge(relationship_details['source'], relationship_details['target'], relationship_name = relationship_details['type'])




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

    out = await graph_client.add_node(root_id, node_properties = node_data)

    created_node_ids.append(out)

    for attribute_name, attribute_value in graph_model:
        ids = await process_attribute(graph_client, root_id, attribute_name, attribute_value)
        created_node_ids.extend(ids)

    flattened_and_deduplicated = list({
        item["nodeId"]: item
            # Use the 'nodeId' as the unique key in the dictionary comprehension
            for sublist in created_node_ids if sublist  # Ensure sublist is not None
            for item in sublist  # Iterate over items in the sublist
    }.values())

    for attribute_name, attribute_value in graph_model:
        ids = await process_attribute_edge(graph_client, root_id, attribute_name, attribute_value, flattened_and_deduplicated)

    return graph_client


async def create_semantic_graph(graph_model_instance, graph_client):
    # Dynamic graph creation based on the provided graph model instance
    return await create_dynamic(graph_model_instance, graph_client)
