import asyncio

import cognee
import os

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.api.v1.visualize.visualize import visualize_graph
from cognee.infrastructure.databases.relational import (
    get_migration_relational_engine,
)

from cognee.modules.search.types import SearchType

from cognee.infrastructure.databases.relational import (
    create_db_and_tables as create_relational_db_and_tables,
)
from cognee.infrastructure.databases.vector.pgvector import (
    create_db_and_tables as create_vector_db_and_tables,
)

# Prerequisites:
# 1. Copy `.env.template` and rename it to `.env`.
# 2. Add your OpenAI API key to the `.env` file in the `LLM_API_KEY` field:
#    LLM_API_KEY = "your_key_here"
# 3. Fill all relevant MIGRATION_DB information for the database you want to migrate to graph / Cognee

# NOTE: If you don't have a DB you want to migrate you can try it out with our
#       test database at the following location:
#           MIGRATION_DB_PATH="/{path_to_your_local_cognee}/cognee/tests/test_data"
#           MIGRATION_DB_NAME="migration_database.sqlite"
#           MIGRATION_DB_PROVIDER="sqlite"


async def main():
    engine = get_migration_relational_engine()

    # Clean all data stored in Cognee
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # Needed to create appropriate tables only on the Cognee side
    await create_relational_db_and_tables()
    await create_vector_db_and_tables()

    print("\nExtracting schema of database to migrate.")
    schema = await engine.extract_schema()
    print(f"Migrated database schema:\n{schema}")

    graph = await get_graph_engine()
    print("Migrating relational database to graph database based on schema.")
    from cognee.tasks.ingestion import migrate_relational_database

    await migrate_relational_database(graph, schema=schema)
    print("Relational database migration complete.")

    # Define location where to store html visualization of graph of the migrated database
    home_dir = os.path.expanduser("~")
    destination_file_path = os.path.join(home_dir, "graph_visualization.html")

    # Make sure to set top_k at a high value for a broader search, the default value is only 10!
    # top_k represent the number of graph tripplets to supply to the LLM to answer your question
    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="What kind of data do you contain?",
        top_k=1000,
    )
    print(f"Search results: {search_results}")

    # Having a top_k value set to too high might overwhelm the LLM context when specific questions need to be answered.
    # For this kind of question we've set the top_k to 30
    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION_COT,
        query_text="What invoices are related to Leonie Köhler?",
        top_k=30,
    )
    print(f"Search results: {search_results}")

    # test.html is a file with visualized data migration
    print("Adding html visualization of graph database after migration.")
    await visualize_graph(destination_file_path)
    print(f"Visualization can be found at: {destination_file_path}")


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
