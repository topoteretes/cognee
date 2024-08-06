# from fastapi import Depends
from fastapi_users.db import SQLAlchemyUserDatabase
# from cognee.infrastructure.databases.relational import get_relational_engine
from .models.User import User

async def get_user_db(session):
    yield SQLAlchemyUserDatabase(session, User)

from contextlib import asynccontextmanager
get_user_db_context = asynccontextmanager(get_user_db)
