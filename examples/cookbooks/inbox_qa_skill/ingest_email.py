"""Step 1: remember the newest email in your Gmail inbox.

Run alone: uv run python examples/cookbooks/inbox_qa_skill/ingest_email.py
"""

import asyncio

from common import DATASET, GMAIL_CREDENTIALS, GMAIL_TOKEN

import cognee
from cognee.tasks.ingestion.connectors.gmail import build_gmail_service, parse_message


def newest_email() -> str:
    """The newest email in your inbox, as text."""
    messages = build_gmail_service(str(GMAIL_CREDENTIALS), str(GMAIL_TOKEN)).users().messages()
    listed = messages.list(userId="me", labelIds=["INBOX"], maxResults=1).execute()
    if not listed.get("messages"):
        raise SystemExit("Your Gmail inbox is empty.")
    email = parse_message(messages.get(userId="me", id=listed["messages"][0]["id"]).execute())
    return f"Subject: {email['title']}\n{email['content']}"


async def run() -> str:
    email = newest_email()
    await cognee.remember(email, dataset_name=DATASET, node_set=["email"], self_improvement=False)
    subject = email.splitlines()[0]
    print(f"[ingest_email] Remembered: {subject}")
    return subject


if __name__ == "__main__":
    asyncio.run(run())
