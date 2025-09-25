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

    Returns:
    --------

        Returns a SQLAlchemyAdapter instance for the specified database connection.
    """
    if db_provider == "sqlite":
        connection_string = f"sqlite+aiosqlite:///{db_path}/{db_name}"

    if db_provider == "postgres":
        try:
            # Test if asyncpg is available
            import asyncpg

            connection_string = (
                f"postgresql+asyncpg://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"
            )
        except ImportError:
            raise ImportError(
                "PostgreSQL dependencies are not installed. Please install with 'pip install cognee\"[postgres]\"' or 'pip install cognee\"[postgres-binary]\"' to use PostgreSQL functionality."
            )

    return SQLAlchemyAdapter(connection_string)
