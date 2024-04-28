from typing import List

async def add_classification_nodes(graph_client, parent_node_id: str, categories: List) -> None:
    for category in categories:
        data_type = category["data_type"].upper().replace(" ", "_")
        category_name = category["category_name"].upper().replace(" ", "_").replace("'", "").replace("/", "_")

        data_type_node_id = data_type

        data_type_node = await graph_client.extract_node(data_type_node_id)

        if not data_type_node:
            data_type_node = await graph_client.add_node(data_type_node_id, dict(name = data_type, type = "DataType"))

        await graph_client.add_edge(data_type_node_id, parent_node_id, "classified_as", dict(relationship_name = "classified_as"))

        category_node_id = category_name

        category_node = await graph_client.extract_node(category_node_id)

        if not category_node:
            category_node = await graph_client.add_node(category_node_id, dict(name = category_name, type = "DataCategory"))

        await graph_client.add_edge(category_node_id, parent_node_id, "classified_as", dict(relationship_name = "classified_as"))
