from uuid import uuid4
from typing import List
from datetime import datetime
from pydantic import BaseModel
from cognee.infrastructure import infrastructure_config
from cognee.infrastructure.databases.vector import DataPoint

async def add_label_nodes(graph_client, parent_node_id: str, keywords: List[str]) -> None:
    vector_client = infrastructure_config.get_config("vector_engine")

    keyword_nodes = []

    for keyword in keywords:
        keyword_id = f"DATA_LABEL_{keyword.upper().replace(' ', '_')}"

        keyword_nodes.append((
            keyword_id,
            dict(
                id = keyword_id,
                name = keyword.lower().capitalize(),
                keyword = keyword.lower(),
                entity_type = "Keyword",
                created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        ))

    # Add data to graph
    await graph_client.add_nodes(keyword_nodes)
    await graph_client.add_edges([
        (
            parent_node_id,
            keyword_id,
            "refers_to",
            dict(relationship_name = "refers_to"),
        ) for (keyword_id, __) in keyword_nodes
    ])

    class References(BaseModel):
        node_id: str
        cognitive_layer: str

    class PayloadSchema(BaseModel):
        value: str
        references: References

    # Add data to vector
    keyword_data_points = [
        DataPoint[PayloadSchema](
            id = str(uuid4()),
            payload = dict(
                value = keyword_data["keyword"],
                references = dict(
                    node_id = keyword_node_id,
                    cognitive_layer = parent_node_id,
                ),
            ),
            embed_field = "value"
        ) for (keyword_node_id, keyword_data) in keyword_nodes
    ]

    try:
        await vector_client.create_collection(parent_node_id, payload_schema = PayloadSchema)
    except Exception:
        # It's ok if the collection already exists.
        pass

    await vector_client.create_data_points(parent_node_id, keyword_data_points)
