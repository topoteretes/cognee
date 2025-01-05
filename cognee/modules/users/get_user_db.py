from typing import AsyncGenerator
from fastapi import Depends
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi_users.db import SQLAlchemyUserDatabase
from cognee.infrastructure.databases.relational import get_relational_engine
from .models.User import User


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        yield session


async def get_user_db(session: AsyncSession = Depends(get_async_session)):
    yield SQLAlchemyUserDatabase(session, User)


get_user_db_context = asynccontextmanager(get_user_db)
