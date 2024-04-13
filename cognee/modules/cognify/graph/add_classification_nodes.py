from typing import List

async def add_classification_nodes(graph_client, parent_node_id: str, categories: List) -> None:
    for category in categories:
        data_type = category["data_type"]
        category_name = category["category_name"]

        data_type_node_id = f"DATA_TYPE-{data_type.upper().replace(' ', '_')}"

        data_type_node = await graph_client.extract_node(data_type_node_id)

        if not data_type_node:
            data_type_node = await graph_client.add_node(data_type_node_id, dict(name = data_type))

        await graph_client.add_edge(parent_node_id, data_type_node_id, relationship_name = "classified_as")

        category_node_id = f"DATA_CATEGORY-{category_name.upper().replace(' ', '_')}"

        category_node = await graph_client.extract_node(category_node_id)

        if not category_node:
            category_node = await graph_client.add_node(category_node_id, dict(name = category_name))

        await graph_client.add_edge(parent_node_id, category_node_id, relationship_name = "classified_as")
