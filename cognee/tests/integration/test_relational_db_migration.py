import pytest
import pytest_asyncio
import os
from cognee.api.v1.visualize.visualize import visualize_graph
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.relational import (
    get_migration_relational_engine,
    create_db_and_tables as create_relational_db_and_tables,
)
from cognee.infrastructure.databases.vector.pgvector import (
    create_db_and_tables as create_pgvector_db_and_tables,
)
from cognee.tasks.ingestion import migrate_relational_database
from cognee.modules.search.types import SearchType
import cognee

@pytest_asyncio.fixture()
async def setup_test_db():

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    await create_relational_db_and_tables()
    await create_pgvector_db_and_tables()

    relational_engine = get_migration_relational_engine()

    return relational_engine

@pytest.mark.asyncio
async def test_relational_db_migration(setup_test_db):

    relational_engine = setup_test_db  # from our fixture

    # 1. Extract schema
    schema = await relational_engine.extract_schema()

    # 2. Migrate to the graph
    graph_engine = await get_graph_engine()
    await migrate_relational_database(graph_engine, schema=schema)

    await visualize_graph()
    #3. Search the graph
    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="Tell me about the artist AC/DC"
    )
    print("Search results:", search_results)

    #4. Assert that the search results contain "AC/DC"
    assert any("AC/DC" in r for r in search_results), "AC/DC not found in search results!"


    # Final success message
    assert True, "Data in the graph matches the relational DB"
