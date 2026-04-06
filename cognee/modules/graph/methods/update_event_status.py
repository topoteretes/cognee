from typing import Dict, List
from uuid import UUID

from sqlalchemy import select, and_

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.graph.models import Node
from cognee.modules.users.models import User
from cognee.shared.data_models import EventStatus


VALID_STATUSES = {s.value for s in EventStatus}


async def update_event_status(
    event_ids: List[str],
    new_status: str,
    user: User,
    dataset_id: UUID,
) -> Dict[str, bool]:
    """Update the status property of Event nodes in the graph and relational DB.

    Args:
        event_ids: Graph node IDs (DataPoint UUIDs as strings) of the events to update.
        new_status: One of the EventStatus enum values.
        user: Authenticated user performing the update.
        dataset_id: Dataset the events belong to.

    Returns:
        Dict mapping each event_id to True if updated, False otherwise.
    """
    if new_status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid status '{new_status}'. Must be one of: {', '.join(sorted(VALID_STATUSES))}"
        )

    if not event_ids:
        return {}

    # 1. Update graph DB node properties
    graph_engine = await get_graph_engine()
    graph_result = await graph_engine.update_node_properties(
        {eid: {"status": new_status} for eid in event_ids}
    )

    # 2. Update relational DB attributes JSON column to stay in sync
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        rows = (
            await session.execute(
                select(Node).where(
                    and_(
                        Node.slug.in_([UUID(eid) for eid in event_ids]),
                        Node.dataset_id == dataset_id,
                    )
                )
            )
        ).scalars().all()

        for row in rows:
            attrs = dict(row.attributes) if row.attributes else {}
            attrs["status"] = new_status
            row.attributes = attrs

        await session.commit()

    return graph_result
