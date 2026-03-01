import asyncio
import os
from pathlib import Path
import sqlalchemy as sa

import cognee
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.relational import (
    create_db_and_tables as create_relational_db_and_tables,
    get_migration_relational_engine,
)
from cognee.infrastructure.databases.vector.pgvector import (
    create_db_and_tables as create_vector_db_and_tables,
)
from cognee.modules.search.types import SearchType
from cognee.modules.ontology.ontology_config import Config
from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver
from cognee.tasks.ingestion import migrate_relational_database


CAR_MANUFACTURERS = [
    (
        "Audi",
        "German car manufacturer known for engineering and the Quattro all-wheel drive system.",
    ),
    (
        "BMW",
        "German car manufacturer focused on performance and driving pleasure.",
    ),
    (
        "Mercedes-Benz",
        "German car manufacturer known for luxury sedans, SUVs, and commercial vehicles.",
    ),
    (
        "Porsche",
        "German car manufacturer specializing in high-performance sports cars like the 911.",
    ),
    (
        "Volkswagen",
        "German car manufacturer known for mass-market vehicles like the Golf.",
    ),
]

IT_COMPANIES = [
    (
        "Apple",
        "Technology company known for consumer electronics and software such as iPhone and macOS.",
    ),
    (
        "Google",
        "Technology company known for search, Android, and cloud services.",
    ),
    (
        "Microsoft",
        "Technology company known for Windows, Office, and Azure cloud services.",
    ),
    (
        "Amazon",
        "Technology company known for e-commerce and AWS cloud services.",
    ),
    (
        "Meta",
        "Technology company known for social platforms and virtual reality products.",
    ),
]

PRODUCTS = [
    # Car products
    ("Audi", "A4", "car", "sedan", 40000, 2023, "Audi A4 sedan with advanced safety features."),
    (
        "BMW",
        "3 Series",
        "car",
        "sedan",
        42000,
        2023,
        "BMW 3 Series sports sedan focused on driving dynamics.",
    ),
    (
        "Mercedes-Benz",
        "C-Class",
        "car",
        "sedan",
        45000,
        2023,
        "Mercedes-Benz C-Class luxury sedan.",
    ),
    (
        "Porsche",
        "911",
        "car",
        "sports car",
        120000,
        2024,
        "Porsche 911 high-performance sports car.",
    ),
    ("Volkswagen", "Golf", "car", "hatchback", 27000, 2022, "Volkswagen Golf compact hatchback."),
    # IT products
    ("Apple", "iPhone", "device", "smartphone", 999, 2023, "Apple iPhone smartphone."),
    ("Google", "Pixel", "device", "smartphone", 899, 2023, "Google Pixel smartphone."),
    ("Microsoft", "Surface", "device", "laptop", 1199, 2022, "Microsoft Surface laptop."),
    ("Amazon", "Kindle", "device", "e-reader", 139, 2021, "Amazon Kindle e-reader."),
    ("Meta", "Quest", "device", "vr headset", 499, 2022, "Meta Quest VR headset."),
]


def _get_postgres_engine() -> sa.Engine:
    db_host = os.environ.get("MIGRATION_DB_HOST", "127.0.0.1")
    db_port = os.environ.get("MIGRATION_DB_PORT", "5432")
    db_name = os.environ.get("MIGRATION_DB_NAME", "cognee_migration")
    db_user = os.environ.get("MIGRATION_DB_USERNAME", "cognee")
    db_password = os.environ.get("MIGRATION_DB_PASSWORD", "cognee")

    # Requires a running Postgres database and a pre-created database (db_name).
    connection_string = (
        f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    )
    return sa.create_engine(connection_string)


def create_example_postgres_db() -> None:
    engine = _get_postgres_engine()

    with engine.begin() as conn:
        conn.execute(
            sa.text(
                """
                CREATE TABLE IF NOT EXISTS companies (
                    id SERIAL PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    company_type TEXT NOT NULL
                );
                """
            )
        )
        conn.execute(
            sa.text(
                """
                CREATE TABLE IF NOT EXISTS car_manufacturers (
                    id SERIAL PRIMARY KEY,
                    company_id INTEGER UNIQUE NOT NULL REFERENCES companies(id),
                    name TEXT UNIQUE NOT NULL,
                    description TEXT NOT NULL
                );
                """
            )
        )
        conn.execute(
            sa.text(
                """
                CREATE TABLE IF NOT EXISTS it_companies (
                    id SERIAL PRIMARY KEY,
                    company_id INTEGER UNIQUE NOT NULL REFERENCES companies(id),
                    name TEXT UNIQUE NOT NULL,
                    description TEXT NOT NULL
                );
                """
            )
        )
        conn.execute(
            sa.text(
                """
                CREATE TABLE IF NOT EXISTS products (
                    id SERIAL PRIMARY KEY,
                    company_id INTEGER NOT NULL REFERENCES companies(id),
                    name TEXT NOT NULL,
                    product_type TEXT NOT NULL,
                    category TEXT NOT NULL,
                    price_usd INTEGER NOT NULL,
                    release_year INTEGER NOT NULL,
                    description TEXT NOT NULL
                );
                """
            )
        )

        # Reset tables for a clean run (handle FK constraints).
        conn.execute(
            sa.text("TRUNCATE TABLE products, car_manufacturers, it_companies, companies CASCADE;")
        )

        conn.execute(
            sa.text("INSERT INTO companies (name, company_type) VALUES (:name, :company_type);"),
            [{"name": name, "company_type": "car"} for name, _ in CAR_MANUFACTURERS]
            + [{"name": name, "company_type": "it"} for name, _ in IT_COMPANIES],
        )

        conn.execute(
            sa.text(
                """
                INSERT INTO car_manufacturers (company_id, name, description)
                SELECT c.id, :name, :description
                FROM companies c
                WHERE c.name = :name;
                """
            ),
            [{"name": name, "description": desc} for name, desc in CAR_MANUFACTURERS],
        )
        conn.execute(
            sa.text(
                """
                INSERT INTO it_companies (company_id, name, description)
                SELECT c.id, :name, :description
                FROM companies c
                WHERE c.name = :name;
                """
            ),
            [{"name": name, "description": desc} for name, desc in IT_COMPANIES],
        )
        conn.execute(
            sa.text(
                """
                INSERT INTO products (
                    company_id, name, product_type, category, price_usd, release_year, description
                ) VALUES (
                    (SELECT id FROM companies WHERE name = :company_name),
                    :name, :product_type, :category, :price_usd, :release_year, :description
                );
                """
            ),
            [
                {
                    "company_name": company,
                    "name": name,
                    "product_type": product_type,
                    "category": category,
                    "price_usd": price_usd,
                    "release_year": release_year,
                    "description": description,
                }
                for company, name, product_type, category, price_usd, release_year, description in PRODUCTS
            ],
        )


def fetch_texts_from_postgres() -> list[str]:
    engine = _get_postgres_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            sa.text(
                """
                SELECT name || ': ' || description FROM car_manufacturers
                UNION ALL
                SELECT name || ': ' || description FROM it_companies
                UNION ALL
                SELECT name || ': ' || description FROM products;
                """
            )
        ).fetchall()
    return [row[0] for row in rows if row and row[0]]


async def main(ontology_path: str = None):
    os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"

    # Create a small Postgres DB schema to migrate.
    create_example_postgres_db()

    # Point migration config to the example DB.
    os.environ["MIGRATION_DB_PROVIDER"] = "postgres"
    os.environ.setdefault("MIGRATION_DB_HOST", "127.0.0.1")
    os.environ.setdefault("MIGRATION_DB_PORT", "5432")
    os.environ.setdefault("MIGRATION_DB_NAME", "cognee_migration")
    os.environ.setdefault("MIGRATION_DB_USERNAME", "cognee")
    os.environ.setdefault("MIGRATION_DB_PASSWORD", "cognee")

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    await create_relational_db_and_tables()
    await create_vector_db_and_tables()

    engine = get_migration_relational_engine()
    schema = await engine.extract_schema()

    graph = await get_graph_engine()
    await migrate_relational_database(graph, schema=schema)

    # Second pass: ingest text content from the relational DB and run cognify (optional ontology).
    dataset_name = "migration_texts"
    db_texts = fetch_texts_from_postgres()
    await cognee.add(db_texts, dataset_name)

    if ontology_path:
        graph_visualization_path = os.path.join(
            os.path.dirname(__file__), ".artifacts", "complex_relational_db_ont.html"
        )
        config: Config = {
            "ontology_config": {
                "ontology_resolver": RDFLibOntologyResolver(ontology_file=ontology_path)
            }
        }
        await cognee.cognify([dataset_name], config=config)
    else:
        graph_visualization_path = os.path.join(
            os.path.dirname(__file__), ".artifacts", "complex_relational_db_no_ont.html"
        )
        await cognee.cognify([dataset_name])

    results = await cognee.search(
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
