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


TEXT_1 = """
1. Audi
Audi is known for its modern designs and advanced technology. Founded in the early 1900s, the brand has earned a reputation for precision engineering and innovation. With features like the Quattro all-wheel-drive system, Audi offers a range of vehicles from stylish sedans to high-performance sports cars.

2. BMW
BMW, short for Bayerische Motoren Werke, is celebrated for its focus on performance and driving pleasure. The company's vehicles are designed to provide a dynamic and engaging driving experience, and their slogan, "The Ultimate Driving Machine," reflects that commitment. BMW produces a variety of cars that combine luxury with sporty performance.

3. Mercedes-Benz
Mercedes-Benz is synonymous with luxury and quality. With a history dating back to the early 20th century, the brand is known for its elegant designs, innovative safety features, and high-quality engineering. Mercedes-Benz manufactures not only luxury sedans but also SUVs, sports cars, and commercial vehicles, catering to a wide range of needs.

4. Porsche
Porsche is a name that stands for high-performance sports cars. Founded in 1931, the brand has become famous for models like the iconic Porsche 911. Porsche cars are celebrated for their speed, precision, and distinctive design, appealing to car enthusiasts who value both performance and style.

5. Volkswagen
Volkswagen, which means "people's car" in German, was established with the idea of making affordable and reliable vehicles accessible to everyone. Over the years, Volkswagen has produced several iconic models, such as the Beetle and the Golf. Today, it remains one of the largest car manufacturers in the world, offering a wide range of vehicles that balance practicality with quality.

Each of these car manufacturer contributes to Germany's reputation as a leader in the global automotive industry, showcasing a blend of innovation, performance, and design excellence.
"""

TEXT_2 = """
1. Apple
Apple is renowned for its innovative consumer electronics and software. Its product lineup includes the iPhone, iPad, Mac computers, and wearables like the Apple Watch. Known for its emphasis on sleek design and user-friendly interfaces, Apple has built a loyal customer base and created a seamless ecosystem that integrates hardware, software, and services.

2. Google
Founded in 1998, Google started as a search engine and quickly became the go-to resource for finding information online. Over the years, the company has diversified its offerings to include digital advertising, cloud computing, mobile operating systems (Android), and various web services like Gmail and Google Maps. Google's innovations have played a major role in shaping the internet landscape.

3. Microsoft
Microsoft Corporation has been a dominant force in software for decades. Its Windows operating system and Microsoft Office suite are staples in both business and personal computing. In recent years, Microsoft has expanded into cloud computing with Azure, gaming with the Xbox platform, and even hardware through products like the Surface line. This evolution has helped the company maintain its relevance in a rapidly changing tech world.

4. Amazon
What began as an online bookstore has grown into one of the largest e-commerce platforms globally. Amazon is known for its vast online marketplace, but its influence extends far beyond retail. With Amazon Web Services (AWS), the company has become a leader in cloud computing, offering robust solutions that power websites, applications, and businesses around the world. Amazon's constant drive for innovation continues to reshape both retail and technology sectors.

5. Meta
Meta, originally known as Facebook, revolutionized social media by connecting billions of people worldwide. Beyond its core social networking service, Meta is investing in the next generation of digital experiences through virtual and augmented reality technologies, with projects like Oculus. The company's efforts signal a commitment to evolving digital interaction and building the metaverse-a shared virtual space where users can connect and collaborate.

Each of these companies has significantly impacted the technology landscape, driving innovation and transforming everyday life through their groundbreaking products and services.
"""


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
        config: Config = {
            "ontology_config": {
                "ontology_resolver": RDFLibOntologyResolver(ontology_file=ontology_path)
            }
        }
        await cognee.cognify([dataset_name], config=config)
        graph_visualization_path = os.path.join(
            os.path.dirname(__file__), ".artifacts", "simple_relational_db_ont.html"
        )

    else:
        await cognee.cognify([dataset_name])
        graph_visualization_path = os.path.join(
            os.path.dirname(__file__), ".artifacts", "simple_relational_db_no_ont.html"
        )

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
