"""Remember the newest emails in your Gmail inbox, 50 by default (node set `email`).

Gmail comes in through cognee's Gmail connector, `gmail_source`. Needs credentials.json (a
Google OAuth Desktop client) in the cookbook folder; gmail_token.json is written there on
the first run. Each run is a snapshot of the newest emails: those already remembered are
skipped, and emails that left the newest N (archived, deleted or pushed out by new mail)
are forgotten. With --sample, it reads the emails setup.py wrote into the dataset
`google_workspace_memory_sample` instead.

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
SAMPLE_DATASET = "google_workspace_memory_sample"  # the sample never mixes with your data
COOKBOOK_DIR = Path(__file__).parent.parent
CREDENTIALS, TOKEN = COOKBOOK_DIR / "credentials.json", COOKBOOK_DIR / "gmail_token.json"
SAMPLE = COOKBOOK_DIR / "sample" / "gmail"


def email_count(value: str) -> int:
    """argparse type for --emails: the connector reads 0 as "the whole inbox"."""
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("--emails must be 1 or more")
    return count


async def ingest_gmail(count: int = 50, sample: bool = False) -> None:
    if sample:
        emails = [path.read_text() for path in sorted(SAMPLE.glob("*.txt"))]
        await cognee.remember(
            emails, dataset_name=SAMPLE_DATASET, node_set=["email"], self_improvement=False
        )
        print(f"[ingest_gmail] Remembered {len(emails)} sample emails")
        return
    if not CREDENTIALS.exists():
        raise SystemExit(f"[ingest_gmail] MISSING: Google OAuth client at {CREDENTIALS}")
    print(f"[ingest_gmail] Remembering your newest {count} inbox emails…", flush=True)
    source = gmail_source(
        label_ids=["INBOX"],
        max_results=count,  # capped, so every run re-reads the newest emails and saves no cursor
        credentials_path=str(CREDENTIALS),
        token_path=str(TOKEN),
    )
    await cognee.remember(
        source,
        dataset_name=DATASET,
        node_set=["email"],
        # A capped read is a snapshot, so replace it: emails no longer among the newest are
        # forgotten. (An uncapped, incremental Gmail sync must use "merge" instead.)
        write_disposition="replace",
        self_improvement=False,
    )
    stats = source.cognee_sync_stats
    print(f"[ingest_gmail] Remembered your newest inbox emails: {stats.get('scanned', 0)} read")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--emails", type=email_count, default=50, help="inbox emails to remember")
    parser.add_argument("--sample", action="store_true", help="use the sample from setup.py")
    args = parser.parse_args()
    asyncio.run(ingest_gmail(args.emails, args.sample))
