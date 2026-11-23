"""Search around one database column over HTTP and over ``cognee.serve()``.

A SQL source ingested through dlt becomes ``SchemaTable`` and ``SchemaColumn`` nodes
with the column values linked to them. ``node_type`` says what ``node_name`` names:
``NodeSet`` (the default) reads the names as node-set tags, while ``SchemaColumn``
scopes the search to the named columns (``"<table>:<column>"``) and their one-hop
neighbours. Over HTTP the type travels as its class name in ``nodeType``.

This guide asks the same column-scoped question twice: once with a raw
``POST /api/v1/search`` body, once with ``cognee.search()`` after ``cognee.serve()``.
Both return the same context: the ``users:name`` values, not the other column.

Prerequisite: a cognee server on this machine that may read local database files::

    ALLOW_DATABASE_INGESTION=true uv run python -m cognee.api.client

Set ``COGNEE_SERVICE_URL`` (default ``http://localhost:8000``) and, if the server
requires authentication, ``COGNEE_API_KEY``.
"""

import asyncio
import os
import sqlite3
import tempfile

import httpx

import cognee
from cognee.tasks.schema.models import SchemaColumn

SERVICE_URL = os.getenv("COGNEE_SERVICE_URL", "http://localhost:8000")
API_KEY = os.getenv("COGNEE_API_KEY", "")
DATASET = "dlt_column_search_guide"
QUESTION = "List every value in users:name."


def write_source_db(directory: str) -> str:
    path = os.path.join(directory, "shop.sqlite")
    connection = sqlite3.connect(path)
    connection.executescript(
        "CREATE TABLE users (name TEXT, user_code TEXT PRIMARY KEY);"
        "INSERT INTO users VALUES ('Ann', 'u1'), ('Bob', 'u2');"
    )
    connection.close()
    return path


async def main():
    with tempfile.TemporaryDirectory() as directory:
        source_db = write_source_db(directory)

        async with httpx.AsyncClient(
            base_url=SERVICE_URL, headers={"X-Api-Key": API_KEY}, timeout=600
        ) as http:
            # The server reads the database itself, so it needs the file's path.
            added = await http.post(
                "/api/v1/add",
                data={"raw_data": f"sqlite:///{source_db}", "datasetName": DATASET},
            )
            added.raise_for_status()
            cognified = await http.post("/api/v1/cognify", json={"datasets": [DATASET]})
            cognified.raise_for_status()

            # The typed scope as JSON: the class name, not the class.
            response = await http.post(
                "/api/v1/search",
                json={
                    "query": QUESTION,
                    "searchType": "GRAPH_COMPLETION",
                    "datasets": [DATASET],
                    "nodeType": "SchemaColumn",
                    "nodeName": ["users:name"],
                    "onlyContext": True,
                },
            )
            response.raise_for_status()
            print("POST /api/v1/search:\n", response.json()[0])

    # serve() routes search() to the server; node_type crosses as nodeType.
    await cognee.serve(url=SERVICE_URL, api_key=API_KEY)
    try:
        results = await cognee.search(
            QUESTION,
            query_type=cognee.SearchType.GRAPH_COMPLETION,
            datasets=[DATASET],
            node_type=SchemaColumn,
            node_name=["users:name"],
            only_context=True,
        )
        print("cognee.search() over serve():\n", results[0])
    finally:
        await cognee.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
