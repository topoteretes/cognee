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
import hashlib
import os
import sys
from pathlib import Path

# One local store for these scripts, the API server and MCP, and no login (see README.md).
# A value in .env still wins.
os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")
os.environ.setdefault("LOG_LEVEL", "ERROR")  # quiet cognee's logs; set before importing it

import cognee

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


def as_document(table: str, primary_key: list[str]):
    """Map a row of ``table`` to the ``id``, ``title`` and ``content`` the document path reads.

    The document path builds each document from those three columns only, so a plain table
    row would become an empty document. A row that already has ``title`` and ``content``
    (like the sample's *_profiles views) is kept as it is; any other row is written out as
    one ``column: value`` line per column.
    """

    def to_document(row: dict) -> dict:
        if "title" in row and "content" in row:
            return row
        if "id" in row:
            row_id = row["id"]
        elif primary_key:
            row_id = ":".join(str(row[column]) for column in primary_key)
        else:
            row_id = hashlib.sha256(repr(sorted(row.items())).encode()).hexdigest()[:16]
        lines = [f"{column}: {value}" for column, value in row.items() if value is not None]
        return {"id": row_id, "title": f"{table} {row_id}", "content": "\n".join(lines)}

    return to_document


async def ingest_database(url: str, tables: list[str] | None = None) -> None:
    """Every row of ``tables`` (all tables and views when None) becomes one document."""
    from dlt.sources.sql_database import sql_database

    database = sql_database(credentials=url, table_names=tables, include_views=True)
    for name, resource in database.resources.items():
        columns = resource.compute_table_schema().get("columns", {})
        primary_key = [column for column, hints in columns.items() if hints.get("primary_key")]
        resource.add_map(as_document(name, primary_key))
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
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--database", help="SQLAlchemy URL, e.g. postgresql://user:pw@host/db")
    parser.add_argument("--tables", help="comma-separated tables or views (default: all)")
    parser.add_argument("--tickets", type=Path, help="a JSON or CSV ticket export")
    parser.add_argument("--docs", type=Path, help="a folder of documents")
    args = parser.parse_args()
    tables = args.tables.split(",") if args.tables else None
    asyncio.run(ingest(args.database, tables, args.tickets, args.docs))
