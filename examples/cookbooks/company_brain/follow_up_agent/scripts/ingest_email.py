"""Remember your newest 50 Gmail inbox emails (node set `email`).

Emails carry deadlines a call doesn't mention. Gmail comes in through cognee's Gmail
connector, `gmail_source`. Needs credentials.json (a Gmail OAuth Desktop client) in the
cookbook folder; token.json is written there on the first run. With --sample, it reads
the emails setup.py wrote instead.

Run alone: uv run python examples/cookbooks/company_brain/follow_up_agent/scripts/ingest_email.py [--emails N] [--sample]
"""

import argparse
import asyncio
import os
from pathlib import Path

os.environ.setdefault("LOG_LEVEL", "ERROR")  # quiet cognee's logs; set before importing it

import cognee
from cognee.tasks.ingestion.connectors import gmail_source

DATASET = "follow_up_agent"  # the same in every script
COOKBOOK_DIR = Path(__file__).parent.parent
CREDENTIALS, TOKEN = COOKBOOK_DIR / "credentials.json", COOKBOOK_DIR / "token.json"
SAMPLE = COOKBOOK_DIR / "sample"


async def ingest_email(count: int = 50, sample: bool = False) -> None:
    if sample:
        emails = [path.read_text() for path in sorted((SAMPLE / "email").glob("*.txt"))]
        await cognee.remember(
            emails, dataset_name=DATASET, node_set=["email"], self_improvement=False
        )
        print(f"[ingest_email] Remembered {len(emails)} sample inbox emails")
        return
    if not CREDENTIALS.exists():
        raise SystemExit(f"[ingest_email] MISSING: Gmail OAuth client at {CREDENTIALS}")
    await cognee.remember(
        gmail_source(
            credentials_path=str(CREDENTIALS),
            token_path=str(TOKEN),
            label_ids=["INBOX"],
            max_results=count,  # keeps the first try small
        ),
        dataset_name=DATASET,
        node_set=["email"],
        write_disposition="merge",  # the connector's rows are merged by message id
        primary_key="id",
        max_rows_per_table=0,
        self_improvement=False,
    )
    print(f"[ingest_email] Remembered your newest {count} inbox emails")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--emails", type=int, default=50, help="inbox emails to remember")
    parser.add_argument("--sample", action="store_true", help="use the sample from setup.py")
    args = parser.parse_args()
    asyncio.run(ingest_email(args.emails, args.sample))
