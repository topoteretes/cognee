import json
import os
from pathlib import Path

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from contextlib import asynccontextmanager
from sqlalchemy.exc import OperationalError
import asyncio
import sys
from dotenv import load_dotenv

load_dotenv()


# this is needed to import classes from other modules
# script_dir = os.path.dirname(os.path.abspath(__file__))
# # Get the parent directory of your script and add it to sys.path
# parent_dir = os.path.dirname(script_dir)
# sys.path.append(parent_dir)
# from ...config import Config
# config = Config()
# config.load()


# in seconds
MAX_RETRIES = 3
RETRY_DELAY = 5

import os


class DatabaseConfig:
    def __init__(self, db_type=None, db_name=None, host=None, user=None, password=None, port=None, config_file=None):
        if config_file:
            self.load_from_file(config_file)
        else:
            # Load default values from environment variables or use provided values
            self.db_type = db_type or os.getenv('DB_TYPE', 'sqlite')
            self.db_name = db_name or os.getenv('DB_NAME', 'database.db')
            self.host = host or os.getenv('DB_HOST', 'localhost')
            self.user = user or os.getenv('DB_USER', 'user')
            self.password = password or os.getenv('DB_PASSWORD', 'password')
            self.port = port or os.getenv('DB_PORT', '5432')

    def load_from_file(self, file_path):
        with open(file_path, 'r') as file:
            config = json.load(file)
        self.db_type = config.get('db_type', 'sqlite')
        self.db_name = config.get('db_name', 'database.db')
        self.host = config.get('host', 'localhost')
        self.user = config.get('user', 'user')
        self.password = config.get('password', 'password')
        self.port = config.get('port', '5432')

    def get_sqlalchemy_database_url(self):
        if self.db_type == 'sqlite':
            db_path = Path(self.db_name).absolute()  # Ensure the path is absolute
            return f"sqlite+aiosqlite:///{db_path}"  # SQLite uses file path
        elif self.db_type == 'duckdb':
            db_path = Path(self.db_name).absolute()  # Ensure the path is absolute for DuckDB as well
            return f"duckdb+aiosqlite:///{db_path}"
        elif self.db_type == 'postgresql':
            # Ensure optional parameters are handled gracefully
            port_str = f":{self.port}" if self.port else ""
            password_str = f":{self.password}" if self.password else ""
            return f"postgresql+asyncpg://{self.user}{password_str}@{self.host}{port_str}/{self.db_name}"
        else:
            raise ValueError(f"Unsupported DB_TYPE: {self.db_type}")

# Example usage with a configuration file:
# config = DatabaseConfig(config_file='path/to/config.json')
# Or set them programmatically:
config = DatabaseConfig(db_type='postgresql', db_name='mydatabase', user='myuser', password='mypassword', host='myhost', port='5432')

SQLALCHEMY_DATABASE_URL = config.get_sqlalchemy_database_url()


engine = create_async_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_recycle=3600,
    echo=True  # Enable logging for tutorial purposes
)
# Use AsyncSession for the session
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

Base = declarative_base()

# Use asynccontextmanager to define an async context manager
@asynccontextmanager
async def get_db():
    db = AsyncSessionLocal()
    try:
        yield db
    finally:
        await db.close()

#
# if os.environ.get('AWS_ENV') == 'prd' or os.environ.get('AWS_ENV') == 'dev':
#     host = os.environ.get('POSTGRES_HOST')
#     username = os.environ.get('POSTGRES_USER')
#     password = os.environ.get('POSTGRES_PASSWORD')
#     database_name = os.environ.get('POSTGRES_DB')
# elif os.environ.get('AWS_ENV') == 'local':
#     host = os.environ.get('POSTGRES_HOST')
#     username = os.environ.get('POSTGRES_USER')
#     password = os.environ.get('POSTGRES_PASSWORD')
#     database_name = os.environ.get('POSTGRES_DB')
# else:
#     host = os.environ.get('POSTGRES_HOST')
#     username = os.environ.get('POSTGRES_USER')
#     password = os.environ.get('POSTGRES_PASSWORD')
#     database_name = os.environ.get('POSTGRES_DB')
#
# # host = config.postgres_host
# # username = config.postgres_user
# # password = config.postgres_password
# # database_name = config.postgres_db
#
#
#
#
#
# # Use the asyncpg driver for async operation
# SQLALCHEMY_DATABASE_URL = f"postgresql+asyncpg://{username}:{password}@{host}:5432/{database_name}"

