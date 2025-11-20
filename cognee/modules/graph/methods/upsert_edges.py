from uuid import UUID, uuid5, NAMESPACE_OID
from typing import Any, Dict, List, Tuple
from fastapi.encoders import jsonable_encoder
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert
from cognee.modules.engine.utils import generate_edge_id

from cognee.modules.graph.models.Edge import Edge
from cognee.infrastructure.databases.relational.with_async_session import with_async_session


@with_async_session
async def upsert_edges(
    edges: List[Tuple[UUID, UUID, str, Dict[str, Any]]],
    tenant_id: UUID,
    user_id: UUID,
    data_id: UUID,
    dataset_id: UUID,
    session: AsyncSession,
):
    """
    Adds edges to the edges table.

    Parameters:
    -----------
        - edges (list): A list of edges to be added to the graph.
    """
    edges_to_add = []

    for edge in edges:
        edge_text = (
            edge[3]["edge_text"] if edge[2] == "contains" and "edge_text" in edge[3] else edge[2]
        )

        edges_to_add.append(
            {
                "id": uuid5(
                    NAMESPACE_OID,
                    str(tenant_id)
                    + str(user_id)
                    + str(dataset_id)
                    + str(edge[0])
                    + str(edge_text)
                    + str(edge[1]),
                ),
                "slug": generate_edge_id(edge_text),
                "user_id": user_id,
                "data_id": data_id,
                "dataset_id": dataset_id,
                "source_node_id": edge[0],
                "destination_node_id": edge[1],
                "relationship_name": edge_text,
                "label": edge[2],
                "attributes": jsonable_encoder(edge[3]),
            }
        )

    upsert_statement = (
        insert(Edge).values(edges_to_add).on_conflict_do_nothing(index_elements=["id"])
    )

    await session.execute(upsert_statement)

    await session.commit()
