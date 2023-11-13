import os
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


# in seconds
MAX_RETRIES = 3
RETRY_DELAY = 5

username = os.getenv('POSTGRES_USER')
password = os.getenv('POSTGRES_PASSWORD')
database_name = os.getenv('POSTGRES_DB')
import os

environment = os.environ.get("ENVIRONMENT")

if environment == "local":
    host= os.getenv('POSTGRES_HOST')

elif environment == "docker":
    host= os.getenv('POSTGRES_HOST_DOCKER')
else:
    host= os.getenv('POSTGRES_HOST_DOCKER')



# Use the asyncpg driver for async operation
SQLALCHEMY_DATABASE_URL = f"postgresql+asyncpg://{username}:{password}@{host}:5432/{database_name}"


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

# Use async/await syntax for the async function
async def safe_db_operation(db_op, *args, **kwargs):
    for attempt in range(MAX_RETRIES):
        async with get_db() as db:
            try:
                # Ensure your db_op is also async
                return await db_op(db, *args, **kwargs)
            except OperationalError as e:
                await db.rollback()
                if "server closed the connection unexpectedly" in str(e) and attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    raise