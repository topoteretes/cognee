"""Sync one Dropbox folder into cognee memory, incrementally.

Text, Markdown, CSV and PDF files and Paper docs of the folder (and its subfolders) become
documents; a re-run fetches only what changed, and files deleted or moved out of the folder
are forgotten. Install with `pip install "cognee-connector-dropbox[cognee]"`.

Setup:
    1. Create a Dropbox app (https://www.dropbox.com/developers/apps) with the scopes
       files.metadata.read, files.content.read and account_info.read, and generate an
       access token for your account on its settings page (it lasts about four hours).
    2. Find the folder's id: it is the "id" that /2/files/get_metadata answers for the
       folder's path, like "id:a4ayc_80_OEAAAAAAAAAXw". Leave it empty for the whole Dropbox.
    3. Set DROPBOX_ACCESS_TOKEN and DROPBOX_FOLDER_ID, plus cognee's usual LLM_API_KEY.

Run:
    python connectors/dropbox/examples/dropbox.py
"""

import asyncio
import os

import cognee

from cognee_connector_dropbox import dropbox_account_id, dropbox_source


async def main():
    token = os.environ["DROPBOX_ACCESS_TOKEN"]
    folder_id = os.environ.get("DROPBOX_FOLDER_ID", "")

    source = dropbox_source(
        folder_id=folder_id,
        # Binds the sync state to the account, so another account can never reuse it.
        account_id=dropbox_account_id(token),
        token=token,
        # One name per folder and account: it names the staging table and the cursor.
        resource_name="dropbox_example",
    )
    await cognee.remember(
        source,
        dataset_name="my_dropbox",
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )
    print("Sync counts:", source.cognee_sync_stats)

    for answer in await cognee.recall(
        "What did I decide about the Q4 plan?", datasets=["my_dropbox"]
    ):
        print(answer)


if __name__ == "__main__":
    asyncio.run(main())
