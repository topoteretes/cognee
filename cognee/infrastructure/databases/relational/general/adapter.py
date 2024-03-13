import uuid
from pathlib import Path
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from cognee.config import Config
# from ..relational_db_interface import RelationalDBInterface
from cognee.database.relationaldb.models.memory import MemoryModel

config = Config()
config.load()

class RelationalDBAdapter():
    session_maker: async_sessionmaker[AsyncSession]

    def __init__(self):
        engine = create_async_engine(
            self.get_database_url(),
            pool_recycle = 3600,
            echo = config.sqlalchemy_logging,
        )
        self.create_session = async_sessionmaker[AsyncSession](
            bind = engine,
            class_ = AsyncSession,
            expire_on_commit = False,
        )

    def get_database_url(
        self,
        db_type = config.db_type,
        db_name = config.db_name,
        db_path = config.db_path,
        user = config.db_user,
        password = config.db_password,
        host = config.db_host,
        port = config.db_port,
    ):
        if db_type == "sqlite":
            db_path = (Path(db_path) / db_name).absolute()
            return f"sqlite+aiosqlite:///{db_path}"  # SQLite uses file path
        elif db_type == "duckdb":
            db_path = (Path(db_path) / db_name).absolute()
            return f"duckdb+aiosqlite:///{db_path}"
        elif db_type == "postgresql":
            # Ensure optional parameters are handled gracefully
            port_str = f":{port}" if port else ""
            password_str = f":{password}" if password else ""
            if not all([user, host]):
                raise ValueError("User and host are required for PostgreSQL connections.")
            return f"postgresql+asyncpg://{user}{password_str}@{host}{port_str}/{db_name}"
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

    async def add_memory(self, user_id: str, memory_name: str):
        memory_id = uuid.uuid4()

        async with self.create_session() as session:
            async with session.begin():
                return session.add(MemoryModel(
                    id = memory_id,
                    user_id = user_id,
                    memory_name = memory_name,
                ))

    async def get_memory_by_name(self, memory_name: int):
        async with self.create_session() as session:
            async with session.begin():
                result = await session.execute(
                    select(MemoryModel.id)
                        .where(MemoryModel.memory_name == memory_name)
                )

                memory = result.scalars().one_or_none()

                return memory
