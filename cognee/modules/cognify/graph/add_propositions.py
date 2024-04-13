""" Here we update semantic graph with content that classifier produced"""
import uuid
import json
from cognee.infrastructure import infrastructure_config
from cognee.infrastructure.databases.graph.get_graph_client import GraphDBType
from cognee.shared.encode_uuid import encode_uuid


async def add_propositions(
    graph_client,
    data_type,
    layer_name,
    layer_description,
    new_data,
    layer_uuid,
    layer_decomposition_uuid
):
    """ Add nodes and edges to the graph for the given LLM knowledge graph and the layer"""
    layer_node_id = None
    print(f"Looking for layer '{layer_name}' under category '{data_type}'")

    if infrastructure_config.get_config()["graph_engine"] == GraphDBType.NETWORKX:
        for node_id, __ in graph_client.graph.nodes(data = True):
            if layer_name in node_id:
                layer_node_id = node_id
    elif infrastructure_config.get_config()["graph_engine"] == GraphDBType.NEO4J:
        layer_node_id = await graph_client.filter_nodes(search_criteria = layer_name)
        layer_node_id = layer_node_id[0]["d"]["node_id"]

    if not layer_node_id:
        print(f"Subclass '{layer_name}' under category '{data_type}' not found in the graph.")
        return graph_client

    # Mapping from old node IDs to new node IDs
    node_id_mapping = {}

    # Add nodes from the Pydantic object
    for node in new_data.nodes:
        unique_node_id = uuid.uuid4()

        new_node_id = f"{node.description} - {str(layer_uuid)}  - {str(layer_decomposition_uuid)} - {str(unique_node_id)}"

        from cognee.utils import extract_pos_tags, extract_named_entities, extract_sentiment_vader

        extract_pos_tags = await extract_pos_tags(node.description)
        extract_named_entities = await extract_named_entities(node.description)
        extract_sentiment = await extract_sentiment_vader(node.description)

        await graph_client.add_node(
            new_node_id,
            name = node.description,
            description = node.description,
            layer_uuid = str(layer_uuid),
            layer_description = str(layer_description),
            layer_decomposition_uuid = str(layer_decomposition_uuid),
            unique_id = str(unique_node_id),
            pos_tags = extract_pos_tags,
            named_entities = extract_named_entities,
            sentiment = extract_sentiment,
            type="detail"
        )

        await graph_client.add_edge(layer_node_id, new_node_id, relationship_name = "detail")

        # Store the mapping from old node ID to new node ID
        node_id_mapping[node.id] = new_node_id

    # Add edges from the Pydantic object using the new node IDs
    for edge in new_data.edges:
        # Use the mapping to get the new node IDs
        source_node_id = node_id_mapping.get(edge.source)
        target_node_id = node_id_mapping.get(edge.target)

        if source_node_id and target_node_id:
            await graph_client.add_edge(source_node_id, target_node_id, relationship_name=edge.description)
        else:
            print(f"Could not find mapping for edge from {edge.source} to {edge.target}")

async def append_to_graph(graph_client, layer_graphs, required_layers):
    layer_uuid = uuid.uuid4()
    data_type = required_layers["data_type"]

    layer_name = required_layers["category_name"]

    for layer_ind in layer_graphs:
        for layer_json, knowledge_graph in layer_ind.items():
            layer_description = json.loads(layer_json)

            layer_decomposition_id = encode_uuid(uuid.uuid4())

            await add_propositions(
                graph_client,
                data_type,
                layer_name,
                layer_description,
                knowledge_graph,
                layer_uuid,
                layer_decomposition_id
            )
