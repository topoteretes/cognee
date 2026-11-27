"""Xero connector demo — ask cognee about your invoices.

Step 1 (once): log in. This opens the browser, stores OAuth tokens (with a
rotating refresh token) at ``XERO_TOKEN_PATH`` (default ./xero_tokens.json):

    from cognee_community_connector_xero import xero_authenticate

    xero_authenticate()   # needs XERO_CLIENT_ID & XERO_CLIENT_SECRET

Step 2: sync. ``xero_source`` is a dlt source you hand to ``cognee.remember``.
Invoices become normal documents and flow through cognify.

Each run is a full snapshot: unchanged invoices keep a stable id, and invoices
deleted/voided out of the listing drop out of the snapshot, so cognee's orphan
cleanup forgets them on the next sync.
"""

import asyncio
import os

import cognee

from cognee_community_connector_xero import xero_authenticate, xero_source

DATASET_NAME = "xero"


async def main() -> None:
    if not (os.environ.get("XERO_CLIENT_ID") and os.environ.get("XERO_CLIENT_SECRET")):
        print("Set XERO_CLIENT_ID and XERO_CLIENT_SECRET to run this example.")
        return

    token_path = os.environ.get("XERO_TOKEN_PATH", "xero_tokens.json")
    if not os.path.exists(token_path):
        xero_authenticate(token_path=token_path)

    # Contacts are opt-in; invoice docs are the default.
    source = xero_source(token_path=token_path)

    print("Syncing Xero invoices into cognee ...")
    await cognee.remember(source, dataset_name=DATASET_NAME)

    answer = await cognee.search(
        query_text="Which customers have open invoices, and for what?",
        query_type=cognee.SearchType.GRAPH_COMPLETION,
        datasets=[DATASET_NAME],
    )
    print("\nSearch result:\n", answer)


if __name__ == "__main__":
    asyncio.run(main())
