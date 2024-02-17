"""Database configuration and connection."""
from pathlib import Path
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv
from cognitive_architecture.config import Config


load_dotenv()



# in seconds
MAX_RETRIES = 3
RETRY_DELAY = 5

class DatabaseConfig:
    """Configuration for the database connection."""
    def __init__(
        self,
        db_type=None,
        db_name=None,
        host=None,
        user=None,
        password=None,
        port=None,
    ):
        self.config = Config()
        self.config.load()
        self.base_path = Path(self.config.db_path)
        # Load default values from environment variables or use provided values
        self.db_type = db_type or self.config.db_type
        self.db_name = db_name or self.config.db_name
        self.host = host or self.config.db_host
        self.user = user or self.config.db_user
        self.password = password or self.config.db_password
        self.port = port or self.config.db_port



    def get_sqlalchemy_database_url(self):
        """Get the SQLAlchemy database URL based on the configuration."""
        db_path = (self.base_path / self.db_name).absolute()
        if self.db_type == "sqlite":
            return f"sqlite+aiosqlite:///{db_path}"  # SQLite uses file path
        elif self.db_type == "duckdb":
            return f"duckdb+aiosqlite:///{db_path}"
        elif self.db_type == "postgresql":
            # Ensure optional parameters are handled gracefully
            port_str = f":{self.port}" if self.port else ""
            password_str = f":{self.password}" if self.password else ""
            return f"postgresql+asyncpg://{self.user}{password_str}@{self.host}{port_str}/{self.db_name}"
        else:
            raise ValueError(f"Unsupported DB_TYPE: {self.db_type}")


# Example usage with a configuration file:
# config = DatabaseConfig(config_file='path/to/config.json')
# Or set them programmatically:
config = DatabaseConfig(
)

SQLALCHEMY_DATABASE_URL = config.get_sqlalchemy_database_url()


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


