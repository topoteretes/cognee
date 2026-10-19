"""Remember the newest emails in your Gmail inbox, 50 by default (node set `email`).

Gmail comes in through cognee's Gmail connector, `gmail_source`. Needs credentials.json (a
Google OAuth Desktop client) in the cookbook folder; gmail_token.json is written there on
the first run. With --sample, it reads the emails setup.py wrote instead.

Run alone: uv run python examples/cookbooks/google_workspace_memory/scripts/ingest_gmail.py [--emails N] [--sample]
"""

import argparse
import asyncio
import os
from pathlib import Path

os.environ.setdefault("LOG_LEVEL", "ERROR")  # quiet cognee's logs; set before importing it

import cognee
from cognee.tasks.ingestion.connectors import gmail_source

DATASET = "google_workspace_memory"  # the same in every script
COOKBOOK_DIR = Path(__file__).parent.parent
CREDENTIALS, TOKEN = COOKBOOK_DIR / "credentials.json", COOKBOOK_DIR / "gmail_token.json"
SAMPLE = COOKBOOK_DIR / "sample" / "gmail"


async def ingest_gmail(count: int = 50, sample: bool = False) -> None:
    if sample:
        emails = [path.read_text() for path in sorted(SAMPLE.glob("*.txt"))]
        await cognee.remember(
            emails, dataset_name=DATASET, node_set=["email"], self_improvement=False
        )
        print(f"[ingest_gmail] Remembered {len(emails)} sample emails")
        return
    if not CREDENTIALS.exists():
        raise SystemExit(f"[ingest_gmail] MISSING: Google OAuth client at {CREDENTIALS}")
    print(f"[ingest_gmail] Remembering your newest {count} inbox emails…", flush=True)
    await cognee.remember(
        gmail_source(
            label_ids=["INBOX"],
            max_results=count,  # keeps the first try small
            credentials_path=str(CREDENTIALS),
            token_path=str(TOKEN),
        ),
        dataset_name=DATASET,
        node_set=["email"],
        write_disposition="merge",  # the connector's rows are merged by message id
        primary_key="id",
        max_rows_per_table=0,
        self_improvement=False,
    )
    print(f"[ingest_gmail] Remembered your newest {count} inbox emails")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--emails", type=int, default=50, help="inbox emails to remember")
    parser.add_argument("--sample", action="store_true", help="use the sample from setup.py")
    args = parser.parse_args()
    asyncio.run(ingest_gmail(args.emails, args.sample))
