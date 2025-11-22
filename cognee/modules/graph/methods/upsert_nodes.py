from typing import List
from uuid import NAMESPACE_OID, UUID, uuid5
from fastapi.encoders import jsonable_encoder
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert

from cognee.modules.graph.models import Node
from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.infrastructure.databases.relational.with_async_session import with_async_session


@with_async_session
async def upsert_nodes(
    nodes: List[DataPoint],
    tenant_id: UUID,
    user_id: UUID,
    dataset_id: UUID,
    data_id: UUID,
    session: AsyncSession,
):
    """
    Adds nodes to the nodes table.

    Parameters:
    -----------
        - nodes (list): A list of nodes to be added to the graph.
    """
    upsert_statement = (
        insert(Node)
        .values(
            [
                {
                    "id": uuid5(
                        NAMESPACE_OID,
                        str(tenant_id)
                        + str(user_id)
                        + str(dataset_id)
                        + str(data_id)
                        + str(node.id),
                    ),
                    "slug": node.id,
                    "user_id": user_id,
                    "data_id": data_id,
                    "dataset_id": dataset_id,
                    "type": node.type,
                    "indexed_fields": DataPoint.get_embeddable_property_names(node),
                    "label": getattr(node, "label", getattr(node, "name", str(node.id))),
                    "attributes": jsonable_encoder(node),
                }
                for node in nodes
            ]
        )
        .on_conflict_do_nothing(index_elements=["id"])
    )

    await session.execute(upsert_statement)

    await session.commit()
