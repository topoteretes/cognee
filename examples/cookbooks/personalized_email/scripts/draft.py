"""Draft a reply to the newest email in your inbox. Printed, never sent.

The facts come from memory: who the sender is, what was decided in your meetings, and what
you promised them. The tone comes from a few of your own sent emails. Set MY_NAME to your
name as it appears in your email.

Run alone: uv run python examples/cookbooks/personalized_email/scripts/draft.py
"""

import argparse
import asyncio
import os
from pathlib import Path

import cognee
from cognee.modules.search.types import SearchType
from cognee.shared.logging_utils import ERROR, setup_logging
from cognee.tasks.ingestion.connectors.gmail import build_gmail_service, parse_message

DATASET = "personalized_email"  # the same in every script
ME = os.environ.get("MY_NAME", "me")  # your name, as it appears in your email
COOKBOOK_DIR = Path(__file__).parent.parent
CREDENTIALS, TOKEN = COOKBOOK_DIR / "credentials.json", COOKBOOK_DIR / "token.json"

DRAFT_PROMPT = f"""You write the reply {ME} would send to an email.
- Answer every question with facts from the context. Never invent a date, price or promise.
- If {ME} owes the sender something, say plainly whether it was sent.
- Match the greeting, length, tone and sign-off of {ME}'s own emails."""


def newest_email() -> str:
    """The newest email in your inbox, as text."""
    if not CREDENTIALS.exists():
        raise SystemExit(f"[draft] MISSING: Gmail OAuth client at {CREDENTIALS}")
    messages = build_gmail_service(str(CREDENTIALS), str(TOKEN)).users().messages()
    listed = messages.list(userId="me", labelIds=["INBOX"], maxResults=1).execute()
    if not listed.get("messages"):
        raise SystemExit("[draft] Your Gmail inbox is empty.")
    email = parse_message(messages.get(userId="me", id=listed["messages"][0]["id"]).execute())
    return f"Subject: {email['title']}\n{email['content']}"


async def draft() -> None:
    email = newest_email()
    print(f"[draft] Email: {email.splitlines()[0]}")
    own_emails = await cognee.recall(
        f"Emails written by {ME}",
        query_type=SearchType.CHUNKS,
        datasets=[DATASET],
        node_name=["sent_mail"],
        top_k=3,
    )
    reply = await cognee.recall(
        f"Write {ME}'s reply to this email:\n{email}\n\nExamples of {ME}'s own emails:\n"
        + "\n---\n".join(str(chunk.text) for chunk in own_emails),
        query_type=SearchType.GRAPH_COMPLETION,
        datasets=[DATASET],
        system_prompt=DRAFT_PROMPT,
    )
    text = str(reply[0].text) if reply else "Nothing found. Run the ingest scripts first."
    print(f"[draft] Reply:\n{text}")


if __name__ == "__main__":
    setup_logging(log_level=ERROR)
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    asyncio.run(draft())
