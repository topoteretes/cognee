"""
examples/example.py
-------------------
Ingest an IMAP mailbox into cognee and search it.

Setup
-----
Set environment variables (or add to .dlt/secrets.toml):

    export IMAP_HOST=imap.fastmail.com
    export IMAP_USERNAME=you@fastmail.com
    export IMAP_PASSWORD=your-app-password
    export IMAP_FOLDERS=INBOX,Sent         # comma-separated (optional)
    export IMAP_SINCE=2024-01-01           # ISO date (optional)

Run::

    python examples/example.py
"""

import asyncio
import os

import cognee
from cognee_community_connector_imap import imap_source


async def main() -> None:
    host = os.environ["IMAP_HOST"]
    username = os.environ["IMAP_USERNAME"]
    password = os.environ["IMAP_PASSWORD"]

    folders_raw = os.environ.get("IMAP_FOLDERS", "INBOX")
    folders = [f.strip() for f in folders_raw.split(",") if f.strip()]

    since_date = os.environ.get("IMAP_SINCE")  # e.g. "2024-01-01"

    print(f"Connecting to {host} as {username} …")
    print(f"Folders: {folders}")
    if since_date:
        print(f"Since: {since_date}")

    src = imap_source(
        host=host,
        username=username,
        password=password,
        folders=folders,
        since_date=since_date,
    )

    print("Ingesting messages into cognee …")
    await cognee.remember(src)
    print("Ingestion complete.\n")

    query = "budget approval"
    print(f"Searching: {query!r}")
    results = await cognee.recall(query)
    if results:
        for i, r in enumerate(results[:5], 1):
            print(f"[{i}] {r}")
    else:
        print("No results found.")


if __name__ == "__main__":
    asyncio.run(main())
