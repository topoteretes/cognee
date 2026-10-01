"""Step 1: remember the newest email in your Gmail inbox.

Needs credentials.json (a Gmail OAuth Desktop client) in this folder; token.json is
written here on the first run.

Run alone: uv run python .agents/skills/inbox-qa/1_ingest_email/main.py
"""

import asyncio
from pathlib import Path

import cognee
from cognee.shared.logging_utils import ERROR, setup_logging
from cognee.tasks.ingestion.connectors.gmail import build_gmail_service, parse_message

DATASET = "inbox_qa_skill"  # the same in every step
HERE = Path(__file__).parent
CREDENTIALS, TOKEN = HERE / "credentials.json", HERE / "token.json"


def newest_email() -> str:
    """The newest email in your inbox, as text."""
    if not CREDENTIALS.exists():
        raise SystemExit(f"[ingest_email] MISSING: Gmail OAuth client at {CREDENTIALS}")
    messages = build_gmail_service(str(CREDENTIALS), str(TOKEN)).users().messages()
    listed = messages.list(userId="me", labelIds=["INBOX"], maxResults=1).execute()
    if not listed.get("messages"):
        raise SystemExit("[ingest_email] Your Gmail inbox is empty.")
    email = parse_message(messages.get(userId="me", id=listed["messages"][0]["id"]).execute())
    return f"Subject: {email['title']}\n{email['content']}"


async def main() -> None:
    email = newest_email()
    await cognee.remember(email, dataset_name=DATASET, node_set=["email"], self_improvement=False)
    print(f"[ingest_email] Remembered: {email.splitlines()[0]}")


if __name__ == "__main__":
    setup_logging(log_level=ERROR)
    asyncio.run(main())
