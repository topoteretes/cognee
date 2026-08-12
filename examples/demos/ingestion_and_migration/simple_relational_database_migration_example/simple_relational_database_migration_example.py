# ruff: noqa: E402
import asyncio
from pathlib import Path
import os

import sqlalchemy as sa

# This example uses a local Postgres migration database and no backend ACL.
# Set os.environ before importing Cognee: Cognee reads env-backed settings at import time, so values
# assigned later may not override defaults or `.env`. See https://docs.cognee.ai/setup-configuration/overview#using-os-environ
os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"
os.environ["MIGRATION_DB_PROVIDER"] = "postgres"
os.environ.setdefault("MIGRATION_DB_HOST", "127.0.0.1")
os.environ.setdefault("MIGRATION_DB_PORT", "5432")
os.environ.setdefault("MIGRATION_DB_NAME", "cognee_migration")
os.environ.setdefault("MIGRATION_DB_USERNAME", "cognee")
os.environ.setdefault("MIGRATION_DB_PASSWORD", "cognee")

import cognee
from cognee import SearchType
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.relational import (
    create_db_and_tables as create_relational_db_and_tables,
)
from cognee.infrastructure.databases.relational import (
    get_migration_relational_engine,
)
from cognee.infrastructure.databases.vector.pgvector import (
    create_db_and_tables as create_vector_db_and_tables,
)
from cognee.modules.ontology.ontology_config import Config
from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver
from cognee.run_migrations import (
    run_migrations,
)  # Keep local SQLite schema current before forget().
from cognee.tasks.ingestion import migrate_relational_database

DATA_DIR = Path(__file__).parent / "data"
TEXT_1 = (DATA_DIR / "german_car_manufacturers.txt").read_text()

TEXT_2 = (DATA_DIR / "us_tech_companies.txt").read_text()


def _get_postgres_engine() -> sa.Engine:
    # Requires a running Postgres database and a pre-created database (db_name).
    # URL.create safely encodes credentials that contain URL-reserved characters.
    connection_url = sa.URL.create(
        "postgresql+psycopg2",
        username=os.environ["MIGRATION_DB_USERNAME"],  # Read the defaults set at module import.
        password=os.environ["MIGRATION_DB_PASSWORD"],  # Keeps env overrides working.
        host=os.environ["MIGRATION_DB_HOST"],
        port=int(os.environ["MIGRATION_DB_PORT"]),
        database=os.environ["MIGRATION_DB_NAME"],
    )
    return sa.create_engine(connection_url)


def create_example_postgres_db() -> None:
    engine = _get_postgres_engine()

    with engine.begin() as conn:
        conn.execute(
            sa.text(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    body TEXT NOT NULL
                );
                """
            )
        )
        conn.execute(sa.text("TRUNCATE TABLE documents;"))
        conn.execute(
            sa.text("INSERT INTO documents (title, body) VALUES (:title, :body);"),
            [
                {"title": "German Car Manufacturers", "body": TEXT_1},
                {"title": "Tech Companies Overview", "body": TEXT_2},
            ],
        )


def fetch_texts_from_postgres() -> list[str]:
    engine = _get_postgres_engine()
    with engine.connect() as conn:
        rows = conn.execute(sa.text("SELECT body FROM documents;")).fetchall()
    return [row[0] for row in rows if row and row[0]]


async def main(ontology_path: str = None):
    # Create a small Postgres DB schema to migrate.
    create_example_postgres_db()

    # Ensure the local Cognee DB exists.
    await create_relational_db_and_tables()
    # Update a reused local Cognee DB so its tables match the current models.
    await run_migrations()
    await cognee.forget(everything=True)

    await create_vector_db_and_tables()

    engine = get_migration_relational_engine()
    schema = await engine.extract_schema()

    graph = await get_graph_engine()
    await migrate_relational_database(graph, schema=schema)

    # Second pass: remember text content from the relational DB (optional ontology).
    dataset_name = "migration_texts"
    db_texts = fetch_texts_from_postgres()

    if ontology_path:
        config: Config = {
            "ontology_config": {
                "ontology_resolver": RDFLibOntologyResolver(ontology_file=ontology_path)
            }
        }
        await cognee.remember(
            db_texts,
            dataset_name=dataset_name,
            config=config,
            self_improvement=False,
        )
        graph_visualization_path = os.path.join(
            os.path.dirname(__file__), ".artifacts", "simple_relational_db_ont.html"
        )

    else:
        await cognee.remember(db_texts, dataset_name=dataset_name, self_improvement=False)
        graph_visualization_path = os.path.join(
            os.path.dirname(__file__), ".artifacts", "simple_relational_db_no_ont.html"
        )

    results = await cognee.recall(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="Which companies are mentioned?",
        top_k=50,
    )
    print(results)

    await cognee.visualize_graph(graph_visualization_path)


async def _run():
    await main(ontology_path="data/basic_ontology.owl")
    await main()


if __name__ == "__main__":
    asyncio.run(_run())
