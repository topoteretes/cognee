import os
import asyncio
from typing import Callable
from sqlalchemy.inspection import inspect
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncEngine, AsyncSession, async_scoped_session
from sqlalchemy.future import select
from cognee.infrastructure.files.storage.LocalStorage import LocalStorage
from ..DatabaseEngine import DatabaseEngine
from ..ModelBase import ModelBase
from ..utils import with_rollback

class SqliteEngine(DatabaseEngine):
    db_path: str = None
    db_name: str = None
    engine: AsyncEngine = None
    session_maker: Callable[[], async_scoped_session[AsyncSession]] = None
    is_db_done: bool = False

    def __init__(self, db_path: str, db_name: str):
        self.db_path = db_path
        self.db_name = db_name
        self.db_location = db_path + "/" + db_name
        self.engine = create_async_engine(
            f"sqlite+aiosqlite:///{self.db_location}",
            pool_recycle = 3600,
            echo = False
        )
        self.session_maker = lambda: async_scoped_session(
            async_sessionmaker(
                bind = self.engine,
                class_ = AsyncSession
            ),
            scopefunc = asyncio.current_task
        )

    async def ensure_tables(self):
        if not self.database_exists(self.db_name):
            self.create_database(self.db_name)

            await self.create_tables()

            self.is_db_done = True

            return True

    def database_exists(self, db_name: str) -> bool:
        return os.path.exists(self.db_path + "/" + db_name)

    def create_database(self, db_name: str):
        LocalStorage.ensure_directory_exists(self.db_path)

        with open(self.db_path + "/" + db_name, mode = "w+", encoding = "utf-8") as file:
            file.write("")

    def drop_database(self, db_name: str):
        os.remove(self.db_location)

    async def table_exists(self, table_name: str) -> bool:
        return inspect(self.engine).has_table(table_name)

    async def create_tables(self):
        async with self.engine.begin() as connection:
            return await connection.run_sync(ModelBase.metadata.create_all)

    async def create(self, data):
        async with with_rollback(self.session_maker()) as session:
            session.add(data)

    async def query(self, query_term):
        async with with_rollback(self.session_maker()) as session:
            return await session.execute(query_term)

    async def query_entity(self, entity):
        async with with_rollback(self.session_maker()) as session:
            return await session.execute(
                select(type(entity))
                    .where(type(entity).id == entity.id)
            )

    async def update(self, data_update_fn):
        async with with_rollback(self.session_maker()):
            data_update_fn()
