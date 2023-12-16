
from contextlib import asynccontextmanager
import logging
from .models.sessions import Session
from .models.memory import MemoryModel
from .models.user import User
from .models.operation import Operation
from .models.metadatas import MetaDatas
from .models.docs import DocsModel



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


async def fetch_job_id(session, user_id=None, memory_id=None, job_id=None):
    try:
        result = await session.execute(
            session.query(Session.id)
            .filter_by(user_id=user_id, id=job_id)
            .order_by(Session.created_at)
            .first()
        )
        return result.scalar_one_or_none()
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return None
