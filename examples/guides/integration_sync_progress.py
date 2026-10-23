"""Read the current sync progress of an SDK-managed integration.

Set COGNEE_API_TOKEN to the connection owner's API token. For example:
  python examples/guides/integration_sync_progress.py google_drive --account-id ACCOUNT_ID

This makes one read-only request. It does not start a sync or call an LLM.
"""

import argparse
import asyncio
import json
import os
from urllib.parse import quote

import httpx


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provider", help="Registered provider, such as google_drive or github")
    parser.add_argument("--account-id", help="Provider account ID of your connected integration")
    parser.add_argument("--api-url", default="http://localhost:8000")
    args = parser.parse_args()
    params = {"account_id": args.account_id} if args.account_id else {}
    async with httpx.AsyncClient(
        base_url=args.api_url,
        headers={"Authorization": f"Bearer {os.environ['COGNEE_API_TOKEN']}"},
    ) as client:
        response = await client.get(
            f"/api/v1/integrations/{quote(args.provider, safe='')}/sync/progress", params=params
        )
        response.raise_for_status()
        print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
