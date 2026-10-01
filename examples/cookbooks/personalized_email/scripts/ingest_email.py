"""Remember your Gmail inbox and sent mail, the newest 50 of each (node sets `inbox`, `sent_mail`).

The inbox gives the facts; the sent mail is a sample of how you write. Gmail comes in
through cognee's Gmail connector, `gmail_source`. Needs credentials.json (a Gmail OAuth
Desktop client) in the cookbook folder; token.json is written there on the first run.

Run alone: uv run python examples/cookbooks/personalized_email/scripts/ingest_email.py [--emails N]
"""

import argparse
import asyncio
from pathlib import Path

import cognee
from cognee.shared.logging_utils import ERROR, setup_logging
from cognee.tasks.ingestion.connectors import gmail_source

DATASET = "personalized_email"  # the same in every script
COOKBOOK_DIR = Path(__file__).parent.parent
CREDENTIALS, TOKEN = COOKBOOK_DIR / "credentials.json", COOKBOOK_DIR / "token.json"


async def ingest_email(count: int = 50) -> None:
    if not CREDENTIALS.exists():
        raise SystemExit(f"[ingest_email] MISSING: Gmail OAuth client at {CREDENTIALS}")
    for label, node_set in (("INBOX", "inbox"), ("SENT", "sent_mail")):
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
    setup_logging(log_level=ERROR)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--emails", type=int, default=50, help="emails to remember per label")
    asyncio.run(ingest_email(parser.parse_args().emails))
