"""Import Slack history through a running Cognee API.

Set COGNEE_API_URL, COGNEE_API_KEY, SLACK_TEAM_ID, COGNEE_DATASET_ID and
SLACK_CHANNEL_IDS (comma-separated). Optional SLACK_IMPORT_DAYS defaults to 7.
The API owns the Slack credential and the databases; this client never opens
either. Run: uv run python examples/python/integrations/slack_history.py
"""

import asyncio
import os

import httpx


async def main():
    team = os.environ["SLACK_TEAM_ID"]
    async with httpx.AsyncClient(
        base_url=os.environ.get("COGNEE_API_URL", "http://localhost:8000"),
        headers={"Authorization": f"Bearer {os.environ['COGNEE_API_KEY']}"},
        timeout=httpx.Timeout(3600, connect=10),
    ) as client:
        response = await client.post(
            f"/api/v1/slack/history/{team}/import",
            json={
                "dataset_id": os.environ["COGNEE_DATASET_ID"],
                "channel_ids": [
                    value.strip()
                    for value in os.environ["SLACK_CHANNEL_IDS"].split(",")
                    if value.strip()
                ],
                "days": int(os.environ.get("SLACK_IMPORT_DAYS", "7")),
            },
        )
        response.raise_for_status()
        print(response.json())


if __name__ == "__main__":
    asyncio.run(main())
