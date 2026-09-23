"""Select Google resources and start a sync through Cognee's integration API.

Prerequisites:
  * A running Cognee API and a Google account connected through browser OAuth.
  * COGNEE_API_TOKEN set to the connected Cognee user's bearer token.
  * Google extras installed in the API environment: pip install 'cognee[gmail,google-drive]'.
    Both connectors ship in the SDK; no community package is required.
    From this checkout: uv sync --extra api --extra gmail --extra google-drive.

Configure the API with GOOGLE_DRIVE_* or GOOGLE_GMAIL_* settings: CLIENT_ID,
CLIENT_SECRET, REDIRECT_URI, STATE_SECRET, FRONTEND_BASE_URL. Also configure
INTEGRATION_CREDENTIALS_KEYS and INTEGRATION_CREDENTIALS_ACTIVE_KEY_ID for
encrypted token storage. Callback paths are /api/v1/integrations/google_drive/callback
and /api/v1/integrations/gmail/callback. Complete consent in the same browser
that starts authorization, so its OAuth nonce cookie reaches the callback.

Usage (omit --resource-id to list resources without starting ingestion):
  uv run python examples/guides/google_integration_sync.py gmail
  uv run python examples/guides/google_integration_sync.py gmail --resource-id INBOX
  uv run python examples/guides/google_integration_sync.py google_drive --resource-id FOLDER_ID

Sync is asynchronous. Re-run without --resource-id to inspect connection status.
Selected Gmail labels are combined using Gmail's intersection semantics. Gmail
starts with no labels selected; Drive selection may include shared-drive roots.
"""

import argparse
import asyncio
import json
import os

import aiohttp


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provider", choices=["google_drive", "gmail"])
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--resource-id", action="append", default=[])
    args = parser.parse_args()
    token = os.environ.get("COGNEE_API_TOKEN")
    if not token:
        parser.error("Set COGNEE_API_TOKEN to the connected Cognee user's bearer token")

    base = f"{args.api_url.rstrip('/')}/api/v1/integrations/{args.provider}"
    async with aiohttp.ClientSession(
        headers={"Authorization": f"Bearer {token}"}, raise_for_status=True
    ) as session:
        async with session.get(f"{base}/resources") as response:
            resources = await response.json()
        print(json.dumps(resources, indent=2))

        if args.resource_id:
            available = {resource["id"] for resource in resources["resources"]}
            missing = set(args.resource_id) - available
            if missing:
                parser.error(f"Resource IDs not in the account's picker: {sorted(missing)}")
            async with session.put(
                f"{base}/resources", json={"resource_ids": args.resource_id}
            ) as response:
                print("Selection:", await response.json())
            async with session.post(f"{base}/sync") as response:
                print("Sync request:", await response.json())

        async with session.get(f"{base}/connection") as response:
            print("Connection:", json.dumps(await response.json(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
