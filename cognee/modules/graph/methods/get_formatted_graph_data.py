from uuid import UUID
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.context_global_variables import set_database_global_context_variables


async def get_formatted_graph_data(dataset_id: UUID, user_id: UUID):
    await set_database_global_context_variables(dataset_id, user_id)

    graph_client = await get_graph_engine()
    (nodes, edges) = await graph_client.get_graph_data()

    return {
        "nodes": list(
            map(
                lambda node: {
                    "id": str(node[0]),
                    "label": node[1]["name"]
                    if ("name" in node[1] and node[1]["name"] != "")
                    else f"{node[1]['type']}_{str(node[0])}",
                    "type": node[1]["type"],
                    "properties": {
                        key: value
                        for key, value in node[1].items()
                        if key not in ["id", "type", "name", "created_at", "updated_at"]
                        and value is not None
                    },
                },
                nodes,
            )
        ),
        "edges": list(
            map(
                lambda edge: {
                    "source": str(edge[0]),
                    "target": str(edge[1]),
                    "label": edge[2],
                },
                edges,
            )
        ),
    }
