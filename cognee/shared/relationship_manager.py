from uuid import UUID
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
from cognee.modules.data.models.Relationship import Relationship


async def create_relationship(
    session: AsyncSession,
    parent_id: UUID,
    child_id: UUID,
    creator_function: str,
    user_id: UUID,
) -> Relationship:
    """Create a new relationship between two data items."""
    relationship = Relationship(
        parent_id=parent_id,
        child_id=child_id,
        creator_function=creator_function,
        user_id=user_id,
        created_at=datetime.now(timezone.utc),
        deleted_at=None,
    )
    session.add(relationship)
    return relationship


async def delete_relationship(
    session: AsyncSession,
    relationship_id: UUID,
) -> Relationship:
    """Soft delete a relationship by setting its deleted_at timestamp."""
    relationship = (
        await session.execute(select(Relationship).filter(Relationship.id == relationship_id))
    ).scalar_one_or_none()

    if relationship:
        relationship.deleted_at = datetime.now(timezone.utc)
        await session.merge(relationship)
        return relationship

    return None


async def find_children(
    session: AsyncSession, parent_id: UUID, include_deleted: bool = False
) -> List[tuple[UUID, UUID]]:
    """Find all child relationships for a given parent ID.

    Args:
        session: SQLAlchemy async session
        parent_id: UUID of the parent to find children for
        include_deleted: If True, includes soft-deleted relationships

    Returns:
        List of tuples (relationship_id, child_id)
    """
    query = select(Relationship.id, Relationship.child_id).filter(
        Relationship.parent_id == parent_id
    )

    if not include_deleted:
        query = query.filter(Relationship.deleted_at.is_(None))

    result = await session.execute(query)
    return result.all()  # Returns list of (id, child_id) tuples
