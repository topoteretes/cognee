"""Remember the Docs, Sheets, PDFs and text files in one Google Drive folder (node set `drive`).

Drive comes in through cognee's Google Drive connector, `google_drive_source`. It reads the
folder and its subfolders. Docs export as text and Sheets as CSV (first sheet only); other
file types are skipped, and so are PDFs and text files over 25 MB (Docs and Sheets report no
size). GOOGLE_DRIVE_INCLUDE_SUBFOLDERS and GOOGLE_DRIVE_MAX_FILE_SIZE_MB in .env change
those defaults. Needs
credentials.json (a Google OAuth Desktop client) in the cookbook folder and
GOOGLE_DRIVE_FOLDER_ID in .env (the ID at the end of the folder's URL); drive_token.json is
written in the cookbook folder on the first run. Later runs process only changed files. With
--sample, it reads the orders sheet setup.py wrote into the dataset
`google_workspace_memory_sample` instead.

Run alone: uv run python examples/cookbooks/google_workspace_memory/scripts/ingest_drive.py [--sample]
"""

import argparse
import asyncio
import os
from pathlib import Path

os.environ.setdefault("LOG_LEVEL", "ERROR")  # quiet cognee's logs; set before importing it

import cognee
from cognee.tasks.ingestion.connectors import google_drive_source

DATASET = "google_workspace_memory"  # the same in every script
SAMPLE_DATASET = "google_workspace_memory_sample"  # the sample never mixes with your data
COOKBOOK_DIR = Path(__file__).parent.parent
CREDENTIALS, TOKEN = COOKBOOK_DIR / "credentials.json", COOKBOOK_DIR / "drive_token.json"
SAMPLE = COOKBOOK_DIR / "sample" / "drive"


async def ingest_drive(sample: bool = False) -> None:
    if sample:
        files = [path.read_text() for path in sorted(SAMPLE.glob("*.txt"))]
        await cognee.remember(
            files, dataset_name=SAMPLE_DATASET, node_set=["drive"], self_improvement=False
        )
        print(f"[ingest_drive] Remembered {len(files)} sample Drive file{'s' * (len(files) != 1)}")
        return
    folder_id = os.environ.get("GOOGLE_DRIVE_FOLDER_ID")
    if not CREDENTIALS.exists() or not folder_id:
        raise SystemExit(
            f"[ingest_drive] MISSING: Google OAuth client at {CREDENTIALS} or GOOGLE_DRIVE_FOLDER_ID"
        )
    print(
        "[ingest_drive] Remembering your Drive folder. The first run reads every file in it…",
        flush=True,
    )
    source = google_drive_source(
        folder_id,
        auth_mode="oauth",
        credentials_path=str(CREDENTIALS),
        token_path=str(TOKEN),
    )
    await cognee.remember(
        source,
        dataset_name=DATASET,
        node_set=["drive"],
        write_disposition="merge",  # the connector's rows are merged by file id
        primary_key="id",
        self_improvement=False,
    )
    stats = source.cognee_sync_stats
    print(
        f"[ingest_drive] Remembered your Drive folder: {stats.get('scanned', 0)} new or "
        "changed files checked"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", action="store_true", help="use the sample from setup.py")
    asyncio.run(ingest_drive(parser.parse_args().sample))
