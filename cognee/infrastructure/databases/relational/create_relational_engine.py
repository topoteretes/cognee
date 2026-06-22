from functools import lru_cache
from cognee.shared.lru_cache import DATABASE_MAX_LRU_CACHE_SIZE

from sqlalchemy import URL

from .sqlalchemy.SqlAlchemyAdapter import SQLAlchemyAdapter


@lru_cache(maxsize=DATABASE_MAX_LRU_CACHE_SIZE)
def create_relational_engine(
    db_path: str,
    db_name: str,
    db_host: str,
    db_port: str,
    db_username: str,
    db_password: str,
    db_provider: str,
    database_connect_args: tuple = None,
    pool_args: tuple = None,
) -> SQLAlchemyAdapter:
    """
    Create a relational database engine based on the specified parameters.

    Parameters:
    -----------

        - db_path (str): The file path to the database directory, applicable for SQLite.
        - db_name (str): The name of the database to be accessed or created.
        - db_host (str): The hostname or IP address of the database server, required for
          PostgreSQL.
        - db_port (str): The port number on which the database server is listening, required
          for PostgreSQL.
        - db_username (str): The username for database authentication, required for
          PostgreSQL.
        - db_password (str): The password for database authentication, required for
          PostgreSQL.
        - db_provider (str): The type of database provider (e.g., 'sqlite' or 'postgres').
        - database_connect_args (dict, optional): Database driver connection arguments.

    Returns:
    --------

        Returns a SQLAlchemyAdapter instance for the specified database connection.
    """
    # Transform pool_args and database_connect_args from tuple of key-value pairs back to dictionary format if they are provided
    database_connect_args = dict(database_connect_args) if database_connect_args else {}
    pool_args = dict(pool_args) if pool_args else {}

    if db_provider == "sqlite":
        connection_string = f"sqlite+aiosqlite:///{db_path}/{db_name}"

    elif db_provider == "postgres":
        try:
            # Test if asyncpg is available
            import asyncpg

            # Handle special characters in username and password like # or @
            connection_string = URL.create(
                "postgresql+asyncpg",
                username=db_username,
                password=db_password,
                host=db_host,
                port=int(db_port),
                database=db_name,
            )

        except ImportError:
            raise ImportError(
                "PostgreSQL dependencies are not installed. Please install with 'pip install cognee\"[postgres]\"' or 'pip install cognee\"[postgres-binary]\"' to use PostgreSQL functionality."
            )

    else:
        raise ConnectionError("unsupported DB type: " + db_provider)

    return SQLAlchemyAdapter(
        connection_string,
        connect_args=database_connect_args,
        pool_args=pool_args,
    )
