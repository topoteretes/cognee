import os
import logging
import psycopg2
from dotenv import load_dotenv
from relationaldb.database import Base
from sqlalchemy import create_engine, text

from relationaldb.models import memory
from relationaldb.models import metadatas
from relationaldb.models import operation
from relationaldb.models import sessions
from relationaldb.models import user
from relationaldb.models import docs

load_dotenv()
logger = logging.getLogger(__name__)


import os
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager
from dotenv import load_dotenv
from relationaldb.database import (
    Base,
)  # Assuming all models are imported within this module
from relationaldb.database import (
    DatabaseConfig,
)  # Assuming DatabaseConfig is defined as before

load_dotenv()
logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.engine = create_engine(config.get_sqlalchemy_database_url())
        self.db_type = config.db_type

    @contextmanager
    def get_connection(self):
        if self.db_type in ["sqlite", "duckdb"]:
            # For SQLite and DuckDB, the engine itself manages connections
            yield self.engine
        else:
            connection = self.engine.connect()
            try:
                yield connection
            finally:
                connection.close()

    def database_exists(self, db_name):
        if self.db_type in ["sqlite", "duckdb"]:
            # For SQLite and DuckDB, check if the database file exists
            return os.path.exists(db_name)
        else:
            query = text(f"SELECT 1 FROM pg_database WHERE datname='{db_name}'")
            with self.get_connection() as connection:
                result = connection.execute(query).fetchone()
                return result is not None

    def create_database(self, db_name):
        if self.db_type not in ["sqlite", "duckdb"]:
            # For databases like PostgreSQL, create the database explicitly
            with self.get_connection() as connection:
                connection.execution_options(isolation_level="AUTOCOMMIT")
                connection.execute(f"CREATE DATABASE {db_name}")

    def drop_database(self, db_name):
        if self.db_type in ["sqlite", "duckdb"]:
            # For SQLite and DuckDB, simply remove the database file
            os.remove(db_name)
        else:
            with self.get_connection() as connection:
                connection.execution_options(isolation_level="AUTOCOMMIT")
                connection.execute(f"DROP DATABASE IF EXISTS {db_name}")

    def create_tables(self):
        Base.metadata.create_all(bind=self.engine)


if __name__ == "__main__":
    # Example usage with SQLite
    config = DatabaseConfig(db_type="sqlite", db_name="mydatabase.db")

    # For DuckDB, you would set db_type to 'duckdb' and provide the database file name
    # config = DatabaseConfig(db_type='duckdb', db_name='mydatabase.duckdb')

    db_manager = DatabaseManager(config=config)

    database_name = config.db_name

    if not db_manager.database_exists(database_name):
        logger.info(f"Database {database_name} does not exist. Creating...")
        db_manager.create_database(database_name)
        logger.info(f"Database {database_name} created successfully.")

    db_manager.create_tables()

#
# def create_admin_engine(username, password, host, database_name):
#     admin_url = f"postgresql://{username}:{password}@{host}:5432/{database_name}"
#     return create_engine(admin_url)
#
# def database_exists(connection, db_name):
#     query = text(f"SELECT 1 FROM pg_database WHERE datname='{db_name}'")
#     result = connection.execute(query).fetchone()
#     return result is not None
#
# def create_database(connection, db_name):
#     connection.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
#     cursor = connection.cursor()
#     cursor.execute(f"CREATE DATABASE {db_name}")
#     cursor.close()
#
# def drop_database(connection, db_name):
#     connection.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
#     cursor = connection.cursor()
#     cursor.execute(f"DROP DATABASE IF EXISTS {db_name}")
#     cursor.close()
#
#
#
# def create_tables(engine):
#     Base.metadata.create_all(bind = engine)
#
# if __name__ == "__main__":
#     host = os.environ.get('POSTGRES_HOST')
#     username = os.environ.get('POSTGRES_USER')
#     password = os.environ.get('POSTGRES_PASSWORD')
#     database_name = os.environ.get('POSTGRES_DB')
#
#     engine = create_admin_engine(username, password, host, database_name)
#     connection = engine.connect()
#
#     # print(Base.metadata.tables)
#
#     if not database_exists(connection, database_name):
#         logger.info(f"Database {database_name} does not exist. Creating...")
#         create_database(connection, database_name)
#         logger.info(f"Database {database_name} created successfully.")
#
#     connection.close()
#     engine.dispose()
#
#     create_tables(engine)
