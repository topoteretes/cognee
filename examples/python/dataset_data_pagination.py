"""Read dataset documents in bounded HTTP pages.

The data endpoint now defaults to 100 rows (maximum limit 1000). Increasing
limit alone does not retrieve a larger dataset: advance offset until a short
page is returned. Use /data/count for totals. The Python datasets.list_data()
API still returns all rows, both locally and when connected to a server.

Usage: COGNEE_API_TOKEN=... python examples/python/dataset_data_pagination.py DATASET_UUID
Optional: COGNEE_API_URL=http://localhost:8000
Offset pagination assumes the dataset is not being changed during traversal.
"""

import asyncio
import os
import sys
from uuid import UUID

import httpx


async def main(dataset_id: UUID):
    base_url = os.getenv("COGNEE_API_URL", "http://localhost:8000").rstrip("/")
    token = os.getenv("COGNEE_API_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    path = f"/api/v1/datasets/{dataset_id}/data"
    async with httpx.AsyncClient(base_url=base_url, headers=headers) as client:
        count = await client.get(f"{path}/count")
        count.raise_for_status()
        print(f"Total documents: {count.json()['count']}")
        offset, limit = 0, 100
        while True:
            response = await client.get(path, params={"limit": limit, "offset": offset})
            response.raise_for_status()
            page = response.json()
            for row in page:
                print(row["id"], row["name"])
            if len(page) < limit:
                break
            offset += len(page)


if __name__ == "__main__":
    asyncio.run(main(UUID(sys.argv[1])))
