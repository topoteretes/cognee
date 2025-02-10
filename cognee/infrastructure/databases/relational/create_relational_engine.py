from .sqlalchemy.SqlAlchemyAdapter import SQLAlchemyAdapter
from functools import lru_cache


@lru_cache
def create_relational_engine(
    db_path: str,
    db_name: str,
    db_host: str,
    db_port: str,
    db_username: str,
    db_password: str,
    db_provider: str,
):
    if db_provider == "sqlite":
        connection_string = f"sqlite+aiosqlite:///{db_path}/{db_name}"

    if db_provider == "postgres":
        connection_string = (
            f"postgresql+asyncpg://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"
        )

    return SQLAlchemyAdapter(connection_string)
