from datetime import datetime, timezone
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from cognee.modules.data.models import graph_relationship_ledger
from cognee.modules.users.models import User


async def create_relationship(
    session: AsyncSession,
    source_node_id: UUID,
    destination_node_id: UUID,
    creator_function: str,
    user: User,
) -> None:
    """Create a relationship between two nodes in the graph.

    Args:
        session: Database session
        source_node_id: ID of the source node
        destination_node_id: ID of the destination node
        creator_function: Name of the function creating the relationship
        user: User creating the relationship
    """
    relationship = graph_relationship_ledger(
        source_node_id=source_node_id,
        destination_node_id=destination_node_id,
        creator_function=creator_function,
        user_id=user.id,
    )
    session.add(relationship)
    await session.flush()


async def delete_relationship(
    session: AsyncSession,
    source_node_id: UUID,
    destination_node_id: UUID,
    user: User,
) -> None:
    """Mark a relationship as deleted.

    Args:
        session: Database session
        source_node_id: ID of the source node
        destination_node_id: ID of the destination node
        user: User deleting the relationship
    """
    relationship = await session.get(
        graph_relationship_ledger, (source_node_id, destination_node_id)
    )
    if relationship:
        relationship.deleted_at = datetime.now(timezone.utc)
        session.add(relationship)
        await session.flush()
