"""Step 1: remember the newest email in your Gmail inbox, plus your last 20 sent emails.

The sent emails (node set `sent`) are a sample of how you write, so step 3 can draft a
reply in your style. Needs credentials.json (a Gmail OAuth Desktop client) in this folder;
token.json is written here on the first run.

Run alone: uv run python .agents/skills/inbox-qa/1_ingest_email/main.py
"""

import asyncio
from pathlib import Path

import cognee
from cognee.shared.logging_utils import ERROR, setup_logging
from cognee.tasks.ingestion.connectors.gmail import build_gmail_service, parse_message

DATASET = "inbox_qa_skill"  # the same in every step
SENT_COUNT = 20
HERE = Path(__file__).parent
CREDENTIALS, TOKEN = HERE / "credentials.json", HERE / "token.json"


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


async def main() -> None:
    inbox = fetch_emails("INBOX", 1)
    if not inbox:
        raise SystemExit("[ingest_email] Your Gmail inbox is empty.")
    await cognee.remember(inbox, dataset_name=DATASET, node_set=["email"], self_improvement=False)
    print(f"[ingest_email] Remembered: {inbox[0].splitlines()[0]}")

    sent = fetch_emails("SENT", SENT_COUNT)
    if sent:
        await cognee.remember(sent, dataset_name=DATASET, node_set=["sent"], self_improvement=False)
    print(f"[ingest_email] Remembered {len(sent)} sent emails as your writing style")


if __name__ == "__main__":
    setup_logging(log_level=ERROR)
    asyncio.run(main())
