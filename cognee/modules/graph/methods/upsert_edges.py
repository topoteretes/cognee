from uuid import UUID, uuid5, NAMESPACE_OID
from typing import Any, Dict, List, Tuple
from fastapi.encoders import jsonable_encoder
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert
from cognee.modules.engine.utils import generate_edge_id

from cognee.infrastructure.databases.relational import with_async_session
from cognee.modules.graph.models.Edge import Edge
from .set_current_user import set_current_user


@with_async_session
async def upsert_edges(
    edges: List[Tuple[UUID, UUID, str, Dict[str, Any]]],
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
    if session.get_bind().dialect.name == "postgresql":
        # Set the session-level RLS variable
        await set_current_user(session, user_id)

    upsert_statement = (
        insert(Edge)
        .values(
            [
                {
                    "id": uuid5(
                        NAMESPACE_OID,
                        str(user_id) + str(dataset_id) + str(edge[0]) + str(edge[2]) + str(edge[1]),
                    ),
                    "slug": generate_edge_id(edge[2]),
                    "user_id": user_id,
                    "data_id": data_id,
                    "dataset_id": dataset_id,
                    "source_node_id": edge[0],
                    "destination_node_id": edge[1],
                    "relationship_name": edge[2],
                    "label": edge[2],
                    "props": jsonable_encoder(edge[3]),
                }
                for edge in edges
            ]
        )
        .on_conflict_do_nothing(index_elements=["id"])
    )

    await session.execute(upsert_statement)

    await session.commit()
