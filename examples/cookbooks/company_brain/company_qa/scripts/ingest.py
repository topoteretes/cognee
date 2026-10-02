"""Remember your company's sources: a SQL database, a ticket export and a docs folder.

Each source goes into the dataset `company_brain` under its own node set, and is extracted
with one graph model (models.py). Nodes with the same identity merge, so a person in the
database, the assignee of a ticket and a name in the docs become one node. Every source is
optional; pass the ones you have.

Run alone: uv run python examples/cookbooks/company_brain/company_qa/scripts/ingest.py \
    [--database URL [--tables a,b]] [--tickets FILE] [--docs FOLDER]
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# One local store for these scripts, the API server and MCP, and no login (see README.md).
# A value in .env still wins.
os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")

import cognee
from cognee.shared.logging_utils import ERROR, setup_logging

sys.path.insert(0, str(Path(__file__).parent.parent))  # models.py sits next to company_qa.py
from models import EXTRACTION_PROMPT, CompanyGraph

DATASET = "company_brain"  # the same in every script


async def remember(data, node_set: str) -> None:
    await cognee.remember(
        data,
        dataset_name=DATASET,
        node_set=[node_set],
        graph_model=CompanyGraph,
        custom_prompt=EXTRACTION_PROMPT,
    )


async def ingest_database(url: str, tables: list[str] | None = None) -> None:
    """Every row of ``tables`` (all tables and views when None) becomes one document."""
    from dlt.sources.sql_database import sql_database

    database = sql_database(credentials=url, table_names=tables, include_views=True)
    # Send rows down the document path, where they are extracted with the graph model.
    # Without it, rows take the relational path, which builds its own table/row graph and
    # ignores graph_model, so nothing would link to the other sources.
    database.cognee_document_source = "database"
    await remember(database, "database")
    print(f"[ingest] Remembered the database ({', '.join(tables) if tables else 'all tables'})")


async def ingest_tickets(path: Path) -> None:
    """A ticket export: one JSON or CSV file."""
    await remember(str(path.expanduser()), "tickets")
    print(f"[ingest] Remembered the tickets in {path}")


async def ingest_docs(folder: Path) -> None:
    """Every document in a folder: meeting notes, postmortems, memos."""
    await remember(str(folder.expanduser()), "docs")
    print(f"[ingest] Remembered the docs in {folder}")


async def ingest(
    database: str | None = None,
    tables: list[str] | None = None,
    tickets: Path | None = None,
    docs: Path | None = None,
) -> None:
    if database:
        await ingest_database(database, tables)
    if tickets:
        await ingest_tickets(tickets)
    if docs:
        await ingest_docs(docs)


if __name__ == "__main__":
    setup_logging(log_level=ERROR)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--database", help="SQLAlchemy URL, e.g. postgresql://user:pw@host/db")
    parser.add_argument("--tables", help="comma-separated tables or views (default: all)")
    parser.add_argument("--tickets", type=Path, help="a JSON or CSV ticket export")
    parser.add_argument("--docs", type=Path, help="a folder of documents")
    args = parser.parse_args()
    tables = args.tables.split(",") if args.tables else None
    asyncio.run(ingest(args.database, tables, args.tickets, args.docs))
