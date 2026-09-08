"""Discover sources and retrieve evidence with an existing Cognee API credential.

COGNEE_API_URL=http://localhost:8000 COGNEE_API_KEY=... python source_discovery.py
Credentials are read from the environment, never printed. This example is read-only.
"""

import asyncio
import os

import httpx


async def main():
    async with httpx.AsyncClient(
        base_url=os.environ.get("COGNEE_API_URL", "http://localhost:8000"),
        headers={"X-Api-Key": os.environ["COGNEE_API_KEY"]},
        timeout=180,
    ) as client:
        response = await client.post(
            "/api/v1/datasets/source-route",
            json={"query": "What did we decide about deployment?", "max_sources": 3},
        )
        response.raise_for_status()
        routing = response.json()
        print("Routing:", routing["status"], "Catalog complete:", routing["complete"])
        for target in routing["targets"]:
            print("Selected:", target["name"], "Reason:", target["reason"])
            result = await client.post(
                "/api/v1/search",
                json={
                    "query": "What did we decide about deployment?",
                    "search_type": "CHUNKS",
                    "dataset_ids": [target["dataset_id"]],
                    "node_name": target["node_sets"] or None,
                    "node_name_filter_operator": "AND",
                    "top_k": 5,
                },
            )
            result.raise_for_status()
            print(result.json())
        print("Only selected targets were searched; results are not an exhaustive export.")


if __name__ == "__main__":
    asyncio.run(main())

# Execute discovery and retrieval through the same API from an SDK client:
# await cognee.serve("http://localhost:8011", api_key="YOUR_AGENT_KEY")
# result = await cognee.sources.search("How many orders arrived this week?",
#                                     source_hint="company analytics")
# Inspect result["routing"], result["evidence"], result["errors"]. SQL results
# carry native query provenance; document results retain their source IDs.
