from uuid import UUID
from typing import List, Tuple
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.relational import with_async_session
from cognee.infrastructure.environment.config.is_backend_access_control_enabled import (
    is_backend_access_control_enabled,
)
from cognee.modules.graph.models import Node
from .GraphRelationshipLedger import GraphRelationshipLedger


@with_async_session
async def has_nodes_in_legacy_ledger(nodes: List[Node], session: AsyncSession):
    node_ids = [node.slug for node in nodes]

    query = (
        select(
            GraphRelationshipLedger.node_label,
            GraphRelationshipLedger.source_node_id,
        )
        .where(
            and_(
                GraphRelationshipLedger.node_label.is_not(None),
                GraphRelationshipLedger.deleted_at.is_(None),
                GraphRelationshipLedger.source_node_id.in_(node_ids),
                GraphRelationshipLedger.source_node_id
                == GraphRelationshipLedger.destination_node_id,
            )
        )
        .distinct()
    )

    legacy_nodes = (await session.execute(query)).all()

    if len(legacy_nodes) == 0:
        return [False for __ in nodes]

    if is_backend_access_control_enabled():
        confirmed_nodes = await confirm_nodes_in_graph(legacy_nodes)
        return [node_id in confirmed_nodes for node_id in node_ids]
    else:
        found_ids = set()
        for __, node_id in legacy_nodes:
            found_ids.add(node_id)

        return [node_id in found_ids for node_id in node_ids]


async def confirm_nodes_in_graph(
    legacy_nodes: List[Tuple[str, UUID]],
):
    graph_engine = await get_graph_engine()

    graph_nodes = await graph_engine.get_nodes([str(node[1]) for node in legacy_nodes])
    graph_nodes_by_id = {node["id"]: node for node in graph_nodes}

    confirmed_nodes = set()
    for __, node_id in legacy_nodes:
        if str(node_id) in graph_nodes_by_id:
            confirmed_nodes.add(node_id)

    return confirmed_nodes
