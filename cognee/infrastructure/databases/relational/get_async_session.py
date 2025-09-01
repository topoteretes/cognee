from typing import AsyncGenerator
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession

from .get_relational_engine import get_relational_engine


@asynccontextmanager
async def get_async_session(auto_commit=False) -> AsyncGenerator[AsyncSession, None]:
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        yield session

        if auto_commit:
            await session.commit()
