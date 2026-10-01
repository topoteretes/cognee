"""Remember your newest inbox emails (20 by default) and your last 20 sent emails.

The inbox emails (node set `email`) are what answer.py answers. The sent emails (node set
`sent`) are a sample of how you write, so a drafted reply sounds like you. Needs
credentials.json (a Gmail OAuth Desktop client) in the skill folder; token.json is written
there on the first run.

Run alone: uv run python examples/cookbooks/inbox_qa/scripts/ingest_email.py [--emails N]
"""

import argparse
import asyncio
from pathlib import Path

import cognee
from cognee.shared.logging_utils import ERROR, setup_logging
from cognee.tasks.ingestion.connectors.gmail import build_gmail_service, parse_message

DATASET = "inbox_qa_skill"  # the same in every script
SENT_COUNT = 20
SKILL_DIR = Path(__file__).parent.parent
CREDENTIALS, TOKEN = SKILL_DIR / "credentials.json", SKILL_DIR / "token.json"


def fetch_emails(label: str, count: int) -> list[str]:
    """The newest ``count`` emails under a Gmail label, each as text."""
    if not CREDENTIALS.exists():
        raise SystemExit(f"[ingest_email] MISSING: Gmail OAuth client at {CREDENTIALS}")
    messages = build_gmail_service(str(CREDENTIALS), str(TOKEN)).users().messages()
    listed = messages.list(userId="me", labelIds=[label], maxResults=count).execute()
    emails = []
    for ref in listed.get("messages", []):
        email = parse_message(messages.get(userId="me", id=ref["id"]).execute())
        emails.append(f"Subject: {email['title']}\n{email['content']}")
    return emails


async def ingest_email(count: int = 20) -> None:
    inbox = fetch_emails("INBOX", count)
    if not inbox:
        raise SystemExit("[ingest_email] Your Gmail inbox is empty.")
    await cognee.remember(inbox, dataset_name=DATASET, node_set=["email"], self_improvement=False)
    print(
        f"[ingest_email] Remembered {len(inbox)} inbox emails, newest: {inbox[0].splitlines()[0]}"
    )

    sent = fetch_emails("SENT", SENT_COUNT)
    if sent:
        await cognee.remember(sent, dataset_name=DATASET, node_set=["sent"], self_improvement=False)
    print(f"[ingest_email] Remembered {len(sent)} sent emails as your writing style")


if __name__ == "__main__":
    setup_logging(log_level=ERROR)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--emails", type=int, default=20, help="inbox emails to remember")
    asyncio.run(ingest_email(parser.parse_args().emails))
