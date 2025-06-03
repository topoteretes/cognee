import dlt
import requests
import asyncio
import threading
import sqlalchemy as sa
import pathlib
import os
from dlt.destinations.impl.sqlalchemy.configuration import SqlalchemyCredentials


import cognee
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.api.v1.visualize.visualize import visualize_graph
from cognee.infrastructure.databases.relational import (
    get_migration_relational_engine,
    create_db_and_tables as create_relational_db_and_tables,
)
from cognee.infrastructure.databases.vector.pgvector import (
    create_db_and_tables as create_pgvector_db_and_tables,
)
from cognee.tasks.ingestion.migrate_relational_database import migrate_relational_database
from cognee.modules.search.types import SearchType

from cognee.root_dir import get_absolute_path


class PatchedSqlalchemyCredentials(SqlalchemyCredentials):
    def __init__(self, connection_string=None):
        super().__init__(connection_string)
        if not hasattr(self, "_conn_lock"):
            self._conn_lock = threading.Lock()


BASE_URL = "https://pokeapi.co/api/v2/"


@dlt.resource(write_disposition="replace")
def pokemon_list(limit: int = 5):
    """Fetch Pokémon list (first 5 Pokémon)."""
    response = requests.get(f"{BASE_URL}pokemon", params={"limit": limit})
    response.raise_for_status()
    yield response.json()["results"]


@dlt.transformer(data_from=pokemon_list)
def pokemon_details(pokemons):
    """Fetch full detail for each Pokémon."""
    for pokemon in pokemons:
        response = requests.get(pokemon["url"])
        response.raise_for_status()
        yield response.json()


async def setup_and_process_data():
    """
    Setup configuration and process Pokemon data into a SQLite database with dlt.
    """
    engine = sa.create_engine("sqlite:///pokemon_data.db")

    pipeline = dlt.pipeline(
        pipeline_name="pokemon_pipeline",
        destination=dlt.destinations.sqlalchemy(
            PatchedSqlalchemyCredentials("sqlite:///pokemon_data.db?timeout=15")
        ),
        dataset_name="main",
        dev_mode=False,
    )

    info = pipeline.run([pokemon_list, pokemon_details])
    print(f"[setup_and_process_data] Pipeline run complete. Pipeline info:\n{info}")

    # (Optional) Inspect tables for debugging
    print("[setup_and_process_data] Verifying data was written to the database.")
    with engine.connect() as conn:
        tables = conn.execute(
            sa.text("SELECT name FROM sqlite_master WHERE type='table';")
        ).fetchall()
        print(f"[setup_and_process_data] Tables in database: {tables}")
        # Example: if 'pokemon_details' is expected, we can see how many rows:
        for table_tuple in tables:
            table_name = table_tuple[0]
            row_count = conn.execute(sa.text(f"SELECT COUNT(*) FROM {table_name}")).fetchone()[0]
            print(f"    -> Table '{table_name}' has {row_count} row(s).")

    print("[setup_and_process_data] Data loading step finished.\n")
    return None


async def apply_foreign_key_fixes():
    """
    Apply foreign key fixes to the SQLite database after data processing.
    """
    engine = sa.create_engine("sqlite:///pokemon_data.db")
    with engine.connect() as conn:
        raw_conn = conn.connection.connection
        with open(
            "examples/relational_db_with_dlt/fix_foreign_keys.sql", "r", encoding="utf-8"
        ) as f:
            sql_script = f.read()
        raw_conn.executescript(sql_script)  # runs multiple statements
        print("[apply_foreign_key_fixes] Applied foreign key fixes")

    return None


async def migrate_to_cognee():
    """
    Migrate the data from the SQLite database to cognee's knowledge graph.
    """

    # Use cognee's absolute path function instead of relative paths
    data_directory_path = get_absolute_path(".data_storage")
    cognee.config.data_root_directory(data_directory_path)

    cognee_directory_path = get_absolute_path(".cognee_system")
    cognee.config.system_root_directory(cognee_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    engine = get_migration_relational_engine()

    await create_relational_db_and_tables()
    await create_pgvector_db_and_tables()

    schema = await engine.extract_schema()

    graph_engine = await get_graph_engine()

    await migrate_relational_database(graph_engine, schema=schema)


async def visualize_knowledge_graph():
    """
    Generate and save an HTML visualization of the knowledge graph.
    """
    home_dir = os.path.expanduser("~")
    html_path = os.path.join(home_dir, "graph_visualization.html")

    await visualize_graph(html_path)


async def search_knowledge_graph():
    """
    Perform a search query against the knowledge graph.
    """
    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION, query_text="What kind of data do you contain?"
    )
    print(search_results)


async def main():
    print("[main] Starting main function, running setup_and_process_data...")
    await setup_and_process_data()
    print("[main] Data loaded into SQLite.")
    await apply_foreign_key_fixes()
    print("[main] Foreign key fixes applied.")
    await migrate_to_cognee()
    print("[main] Migration to cognee finished.")
    await visualize_knowledge_graph()
    print("[main] Knowledge graph visualization created.")
    await search_knowledge_graph()
    print("[main] Knowledge graph search completed.")


if __name__ == "__main__":
    print("[__main__] Creating and running event loop.")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
    print("[__main__] Event loop closed. Exiting.")
