"""Ingest a Google Drive folder into cognee memory, incrementally.

Demonstrates the Google Drive DLT connector: Docs, Sheets, PDFs, and plain-text
files in a folder (recursively, by default) are extracted, chunked, and
cognified like any other document. Re-running this script only re-processes
files that changed since the last run, and files removed from the folder
(deleted or trashed) are forgotten automatically.

Install:
    pip install "cognee[google-drive]"

Auth setup — pick ONE of the two modes below.

1) Service account (recommended — no interactive step, good for scheduled
   re-syncs):
   - In the Google Cloud Console, create a project (or reuse one) and
     enable the "Google Drive API".
   - Create a Service Account, then create + download a JSON key for it.
   - Share the target Drive folder with the service account's email
     address (found in the JSON key as "client_email"), Viewer access is
     enough.
   - Set:
       GOOGLE_DRIVE_AUTH_MODE=service_account
       GOOGLE_DRIVE_CREDENTIALS_PATH=/path/to/service-account.json
       GOOGLE_DRIVE_FOLDER_ID=<the folder's Drive ID, from its URL>

2) OAuth user credentials (for a folder in your own "My Drive"):
   - In the Google Cloud Console, enable the "Google Drive API" and
     create an OAuth Client ID of type "Desktop app"; download its JSON.
   - Set:
       GOOGLE_DRIVE_AUTH_MODE=oauth
       GOOGLE_DRIVE_CREDENTIALS_PATH=/path/to/oauth-client-secret.json
       GOOGLE_DRIVE_TOKEN_PATH=/path/to/cache-the-user-token.json
       GOOGLE_DRIVE_FOLDER_ID=<the folder's Drive ID, from its URL>
   - The first run opens a browser for one-time consent; subsequent runs
     reuse the cached, auto-refreshed token. For headless / CI use,
     pre-authorize the token file once on a machine with a browser.

Also set the usual cognee LLM_API_KEY (see .env.template) — this example
calls cognee.recall(), which needs an LLM for the final completion.
"""

import asyncio

import cognee
from cognee.tasks.ingestion.connectors import google_drive_source


async def main():
    drive_source = google_drive_source()  # reads GOOGLE_DRIVE_* env vars

    print("=== Initial sync ===")
    result = await cognee.remember(
        drive_source,
        dataset_name="google_drive_demo",
        primary_key="file_id",
        # "merge" is required: it's what makes re-runs incremental and what
        # makes deletions propagate via orphan cleanup. The default
        # ("replace") would re-extract every file on every run instead.
        write_disposition="merge",
        # Routes file content through normal chunking + LLM graph
        # extraction instead of the relational-row treatment.
        dlt_content_column="content",
        # The DLT ingestion default caps a table at 50 rows; Drive folders
        # commonly exceed that, so lift the cap.
        max_rows_per_table=0,
    )
    print(result)

    answer = await cognee.recall("Summarize what's in the Drive folder.")
    print("Recall:", answer)

    print("\n=== Incremental re-sync (only changed/removed files are processed) ===")
    result = await cognee.remember(
        google_drive_source(),
        dataset_name="google_drive_demo",
        primary_key="file_id",
        write_disposition="merge",
        dlt_content_column="content",
        max_rows_per_table=0,
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
