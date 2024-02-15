import os
import logging
import psycopg2
from dotenv import load_dotenv
from postgres.database import Base
from sqlalchemy import create_engine, text

from postgres.models import memory
from postgres.models import metadatas
from postgres.models import operation
from postgres.models import sessions
from postgres.models import user
from postgres.models import docs

load_dotenv()
logger = logging.getLogger(__name__)

def create_admin_engine(username, password, host, database_name):
    admin_url = f"postgresql://{username}:{password}@{host}:5432/{database_name}"
    return create_engine(admin_url)

def database_exists(connection, db_name):
    query = text(f"SELECT 1 FROM pg_database WHERE datname='{db_name}'")
    result = connection.execute(query).fetchone()
    return result is not None

def create_database(connection, db_name):
    connection.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = connection.cursor()
    cursor.execute(f"CREATE DATABASE {db_name}")
    cursor.close()

def drop_database(connection, db_name):
    connection.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = connection.cursor()
    cursor.execute(f"DROP DATABASE IF EXISTS {db_name}")
    cursor.close()



def create_tables(engine):
    Base.metadata.create_all(bind = engine)

if __name__ == "__main__":
    host = os.environ.get('POSTGRES_HOST')
    username = os.environ.get('POSTGRES_USER')
    password = os.environ.get('POSTGRES_PASSWORD')
    database_name = os.environ.get('POSTGRES_DB')

    engine = create_admin_engine(username, password, host, database_name)
    connection = engine.connect()

    # print(Base.metadata.tables)

    if not database_exists(connection, database_name):
        logger.info(f"Database {database_name} does not exist. Creating...")
        create_database(connection, database_name)
        logger.info(f"Database {database_name} created successfully.")

    connection.close()
    engine.dispose()

    create_tables(engine)