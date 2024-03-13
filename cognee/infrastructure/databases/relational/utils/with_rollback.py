import logging
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import async_scoped_session
logger = logging.getLogger(__name__)

@asynccontextmanager
async def with_rollback(session: async_scoped_session):
    """Provide a transactional scope around a series of operations."""

    try:
        # async with session.begin():
        yield session
        await session.commit()
        await session.remove()
    except Exception as exception:
        await session.rollback()
        logger.error("Session rolled back due to: %s", str(exception))
        raise exception
