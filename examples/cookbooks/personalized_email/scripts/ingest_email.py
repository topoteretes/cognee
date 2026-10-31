"""Remember your Gmail inbox and sent mail, the newest 50 of each (node sets `inbox`, `sent_mail`).

The inbox gives the facts; the sent mail is a sample of how you write. Gmail comes in
through cognee's Gmail connector, `gmail_source`. Needs credentials.json (a Gmail OAuth
Desktop client) in the cookbook folder; token.json is written there on the first run.
With --sample, it reads the emails setup.py wrote instead.

Run alone: uv run python examples/cookbooks/personalized_email/scripts/ingest_email.py [--emails N] [--sample]
"""

import argparse
import asyncio
import os
from pathlib import Path

os.environ.setdefault("LOG_LEVEL", "ERROR")  # quiet cognee's logs; set before importing it

import cognee
from cognee.tasks.ingestion.connectors import gmail_source

DATASET = "personalized_email"  # the same in every script
COOKBOOK_DIR = Path(__file__).parent.parent
CREDENTIALS, TOKEN = COOKBOOK_DIR / "credentials.json", COOKBOOK_DIR / "token.json"
SAMPLE = COOKBOOK_DIR / "sample"


def sample_emails(node_set: str) -> list[str]:
    """The sample emails setup.py wrote for one node set, each as text."""
    return [path.read_text() for path in sorted((SAMPLE / node_set).glob("*.txt"))]


async def ingest_email(count: int = 50, sample: bool = False) -> None:
    if not sample and not CREDENTIALS.exists():
        raise SystemExit(f"[ingest_email] MISSING: Gmail OAuth client at {CREDENTIALS}")
    for label, node_set in (("INBOX", "inbox"), ("SENT", "sent_mail")):
        if sample:
            emails = sample_emails(node_set)
            await cognee.remember(
                emails, dataset_name=DATASET, node_set=[node_set], self_improvement=False
            )
            print(f"[ingest_email] Remembered {len(emails)} sample {label.lower()} emails")
            continue
        await cognee.remember(
            gmail_source(
                resource_name=f"gmail_{node_set}",
                label_ids=[label],
                max_results=count,  # keeps the first try small
                credentials_path=str(CREDENTIALS),
                token_path=str(TOKEN),
            ),
            dataset_name=DATASET,
            node_set=[node_set],
            write_disposition="merge",  # the connector's rows are merged by message id
            primary_key="id",
            max_rows_per_table=0,
            self_improvement=False,
        )
        print(f"[ingest_email] Remembered your newest {count} {label.lower()} emails")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--emails", type=int, default=50, help="emails to remember per label")
    parser.add_argument("--sample", action="store_true", help="use the sample from setup.py")
    args = parser.parse_args()
    asyncio.run(ingest_email(args.emails, args.sample))
