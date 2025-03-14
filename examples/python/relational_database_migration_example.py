import asyncio
import cognee
import logging

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.relational import (
    get_migration_relational_engine,
)
from cognee.shared.utils import setup_logging

# Prerequisites:
# 1. Copy `.env.template` and rename it to `.env`.
# 2. Add your OpenAI API key to the `.env` file in the `LLM_API_KEY` field:
#    LLM_API_KEY = "your_key_here"
# 3. Fill all relevant MIGRATION_DB information for the database you want to migrate to graph / Cognee


async def main():
    engine = get_migration_relational_engine()
    schema = await engine.extract_schema()
    graph = await get_graph_engine()
    await graph.migrate_relational_database(schema=schema)
    from cognee.api.v1.visualize.visualize import visualize_graph

    # test.html is a file with visualized data migration
    await visualize_graph("/Users/igorilic/Desktop/cognee/test.html")


if __name__ == "__main__":
    setup_logging(logging.ERROR)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
