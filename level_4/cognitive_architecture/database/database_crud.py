
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)

@asynccontextmanager
async def session_scope(session):
    """Provide a transactional scope around a series of operations."""

    # session = AsyncSessionLocal()
    try:
        yield session
        await session.commit()
    except Exception as e:
        await session.rollback()
        logger.error(f"Session rollback due to: {str(e)}")
        raise
    finally:
        await session.close()


async def add_entity(session, entity):
    async with session_scope(session) as s:  # Use your async session_scope
        s.add(entity)  # No need to commit; session_scope takes care of it
        return "Successfully added entity"



async def update_entity(session, model, entity_id, new_value):
    async with session_scope(session) as s:
        # Retrieve the entity from the database
        entity = await s.get(model, entity_id)

        if entity:
            # Update the relevant column and 'updated_at' will be automatically updated
            entity.operation_status = new_value
            return "Successfully updated entity"
        else:
            return "Entity not found"