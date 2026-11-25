"""Draft a reply to the newest email in your inbox. Printed, never sent.

The facts come from memory: who the sender is, what was decided in your meetings, and what
you promised them. The tone comes from a few of your own sent emails. Set MY_NAME to your
name as it appears in your email. With --sample, it answers the newest sample email as
Mira Lang, whose mailbox the sample is.

Run alone: uv run python examples/cookbooks/personalized_email/scripts/draft.py [--sample]
"""

import argparse
import asyncio
import os
import re
from pathlib import Path

os.environ.setdefault("LOG_LEVEL", "ERROR")  # quiet cognee's logs; set before importing it

import cognee
from cognee.modules.search.types import SearchType
from cognee.tasks.ingestion.connectors.gmail import build_gmail_service, parse_message

DATASET = "personalized_email"  # the same in every script
ME = os.environ.get("MY_NAME", "me")  # your name, as it appears in your email
SAMPLE_OWNER = "Mira Lang"  # the sample mailbox is hers
COOKBOOK_DIR = Path(__file__).parent.parent
CREDENTIALS, TOKEN = COOKBOOK_DIR / "credentials.json", COOKBOOK_DIR / "token.json"
SAMPLE = COOKBOOK_DIR / "sample"

DRAFT_PROMPT = """You write the reply {me} would send to an email.
- Answer every question with facts from the context. Never invent a date, price or promise.
- If {me} owes the sender something, say plainly whether it was sent.
- Match the greeting, length, tone and sign-off of {me}'s own emails, in the language of
  the email being answered.
- Return only the email body, from the greeting to the sign-off: no subject line, no notes,
  no explanation and no markdown."""


def newest_email_text(sample: bool = False) -> str:
    """The newest email in your inbox (or in the sample), as text: headers, then the body."""
    if sample:
        return max((SAMPLE / "inbox").glob("*.txt")).read_text()  # files are named in date order
    if not CREDENTIALS.exists():
        raise SystemExit(f"[draft] MISSING: Gmail OAuth client at {CREDENTIALS}")
    messages = build_gmail_service(str(CREDENTIALS), str(TOKEN)).users().messages()
    listed = messages.list(userId="me", labelIds=["INBOX"], maxResults=1).execute()
    if not listed.get("messages"):
        raise SystemExit("[draft] Your Gmail inbox is empty.")
    email = parse_message(messages.get(userId="me", id=listed["messages"][0]["id"]).execute())
    return f"Subject: {email['title']}\n{email['content']}"


def header(text: str, field: str) -> str:
    """One header line of an email (Subject, From, ...), or ""."""
    match = re.search(rf"^{field}: (.*)$", text, re.MULTILINE)
    return match.group(1).strip() if match else ""


async def draft(sample: bool = False) -> None:
    text = newest_email_text(sample)
    me = SAMPLE_OWNER if sample else ME
    email = {"sender": header(text, "From"), "subject": header(text, "Subject"), "text": text}
    print(f"[draft] Answering: {email['subject']} (from {email['sender']})")
    own_emails = await cognee.recall(
        f"Emails written by {me}",
        query_type=SearchType.CHUNKS,
        datasets=[DATASET],
        node_name=["sent_mail"],
        top_k=3,
    )
    reply = await cognee.recall(
        f"Write {me}'s reply to this email:\n{email['text']}\n\nExamples of {me}'s own emails:\n"
        + "\n---\n".join(str(chunk.text) for chunk in own_emails),
        query_type=SearchType.HYBRID_COMPLETION,
        datasets=[DATASET],
        system_prompt=DRAFT_PROMPT.format(me=me),
    )
    if not reply:
        raise SystemExit("[draft] Nothing found. Run the ingest scripts first.")
    subject = email["subject"]
    if not subject.lower().startswith("re:"):
        subject = f"Re: {subject}"
    print(f"[draft] Reply:\n\nTo: {email['sender']}\nSubject: {subject}\n\n{reply[0].text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", action="store_true", help="answer the newest sample email")
    asyncio.run(draft(parser.parse_args().sample))
