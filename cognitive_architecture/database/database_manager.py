"""This module is used to create the database and tables for the cognitive architecture."""
import asyncio
import logging

from dotenv import load_dotenv
from cognitive_architecture.config import Config
from cognitive_architecture.database.create_database import DatabaseManager


config = Config()
config.load()

load_dotenv()
logger = logging.getLogger(__name__)
async def main():
    """Runs as a part of startup docker scripts to create the database and tables."""


    db_manager = DatabaseManager()
    database_name = config.db_name

    if not await db_manager.database_exists(database_name):
        print(f"Database {database_name} does not exist. Creating...")
        await db_manager.create_database(database_name)
        print(f"Database {database_name} created successfully.")

    await db_manager.create_tables()

if __name__ == "__main__":
    asyncio.run(main())
