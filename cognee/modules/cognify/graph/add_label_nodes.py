from uuid import uuid4
from typing import List
from cognee.infrastructure import infrastructure_config
from cognee.infrastructure.databases.vector import DataPoint

async def add_label_nodes(graph_client, parent_node_id: str, keywords: List[str]) -> None:
    vector_client = infrastructure_config.get_config("vector_engine")

    prepared_keywords = []

    for keyword in keywords:
        keyword_id = f"DATA_LABEL_{keyword.upper().replace(' ', '_')}"

        prepared_keywords.append((
            keyword_id,
            dict(
                keyword = keyword.lower(),
                label = keyword,
            ),
        ))

    # Add data to graph
    await graph_client.add_nodes(prepared_keywords)
    await graph_client.add_edges([
        (
            parent_node_id,
            keyword_id,
            dict(relationship_name = "refers_to"),
        ) for (keyword_id, __) in prepared_keywords
    ])

    # Add data to vector
    keyword_data_points = [
        DataPoint(
            id = str(uuid4()),
            payload = dict(
                keyword = keyword_data["keyword"],
                references = dict(
                    graph_id = keyword_node_id,
                    cognitive_layer = parent_node_id,
                ),
            ),
            embed_field = "keyword"
        ) for (keyword_node_id, keyword_data) in prepared_keywords
    ]

    try:
        await vector_client.create_collection(parent_node_id)
    except Exception as error:
        print(error)
        # It's ok if the collection already exists.
        pass

    await vector_client.create_data_points(parent_node_id, keyword_data_points)
