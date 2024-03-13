"""This module is used to create the database and tables for the cognitive architecture."""
import logging

logger = logging.getLogger(__name__)

async def main():
    """Runs as a part of startup docker scripts to create the database and tables."""
    from cognee.config import Config
    config = Config()
    config.load()

    from cognee.database.database_manager import DatabaseManager

    db_manager = DatabaseManager()
    database_name = config.db_name

    if not await db_manager.database_exists(database_name):
        print(f"Database {database_name} does not exist. Creating...")
        await db_manager.create_database(database_name)
        print(f"Database {database_name} created successfully.")

    await db_manager.create_tables()

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
