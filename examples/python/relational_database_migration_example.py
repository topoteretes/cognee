import asyncio
import cognee
import os
import logging

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.api.v1.visualize.visualize import visualize_graph
from cognee.infrastructure.databases.relational import (
    get_migration_relational_engine,
)

from cognee.modules.search.types import SearchType
from cognee.modules.users.methods import get_default_user

from cognee.infrastructure.databases.relational import (
    create_db_and_tables as create_relational_db_and_tables,
)
from cognee.infrastructure.databases.vector.pgvector import (
    create_db_and_tables as create_pgvector_db_and_tables,
)

# Prerequisites:
# 1. Copy `.env.template` and rename it to `.env`.
# 2. Add your OpenAI API key to the `.env` file in the `LLM_API_KEY` field:
#    LLM_API_KEY = "your_key_here"
# 3. Fill all relevant MIGRATION_DB information for the database you want to migrate to graph / Cognee


async def main():
    engine = get_migration_relational_engine()

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # Needed to create principals table
    # Create tables for databases
    await create_relational_db_and_tables()
    await create_pgvector_db_and_tables()

    print("\nExtracting schema of database to migrate.")
    schema = await engine.extract_schema()
    print(f"Migrated database schema:\n{schema}")

    graph = await get_graph_engine()
    print("Migrating relational database to graph database based on schema.")
    await graph.migrate_relational_database(schema=schema)
    print("Relational database migration complete.")

    # Define location where to store html visualization of graph of the migrated database
    home_dir = os.path.expanduser("~")
    destination_file_path = os.path.join(home_dir, "graph_visualization.html")

    # test.html is a file with visualized data migration
    print("Adding html visualization of graph database after migration.")
    await visualize_graph(destination_file_path)
    print(f"Visualization can be found at: {destination_file_path}")

    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION, query_text="What kind of data do you contain?"
    )
    print(f"Search results: {search_results}")


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
