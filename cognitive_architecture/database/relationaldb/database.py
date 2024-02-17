"""Database configuration and connection."""
from pathlib import Path
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv
from cognitive_architecture.config import Config
config = Config()

load_dotenv()



# in seconds
MAX_RETRIES = 3
RETRY_DELAY = 5

def get_sqlalchemy_database_url(db_type='sqlite', db_name=config.db_name, base_path=config.db_path, user=config.db_user, password=config.db_password, host=config.db_host, port=config.db_port):
    """Get the SQLAlchemy database URL based on parameters."""
    db_path = (Path(base_path) / db_name).absolute()
    if db_type == "sqlite":
        return f"sqlite+aiosqlite:///{db_path}"  # SQLite uses file path
    elif db_type == "duckdb":
        return f"duckdb+aiosqlite:///{db_path}"
    elif db_type == "postgresql":
        # Ensure optional parameters are handled gracefully
        port_str = f":{port}" if port else ""
        password_str = f":{password}" if password else ""
        if not all([user, host]):
            raise ValueError("User and host are required for PostgreSQL connections.")
        return f"postgresql+asyncpg://{user}{password_str}@{host}{port_str}/{db_name}"
    else:
        raise ValueError(f"Unsupported DB_TYPE: {db_type}")


# Example usage with a configuration file:
# config = DatabaseConfig(config_file='path/to/config.json')
# Or set them programmatically:


SQLALCHEMY_DATABASE_URL = get_sqlalchemy_database_url()


engine = create_async_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_recycle=3600,
    echo=True,  # Enable logging for tutorial purposes
)
# Use AsyncSession for the session
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

Base = declarative_base()

@asynccontextmanager
async def get_db():
    """Provide a database session to the context."""
    db = AsyncSessionLocal()
    try:
        yield db
    finally:
        await db.close()


