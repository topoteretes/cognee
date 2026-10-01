"""Answer a question about one remembered inbox email.

Facts come from that email and your Granola meetings: the newest email from --sender when
given, else the newest email. When the question asks for a reply, a few of your own sent
emails are passed in so the draft sounds like you.

Run alone: uv run python .agents/skills/inbox-qa/scripts/answer.py [--sender NAME] ["question"]
"""

import argparse
import asyncio
import re

import cognee
from cognee.modules.search.types import SearchType
from cognee.shared.logging_utils import ERROR, setup_logging

DATASET = "inbox_qa_skill"  # the same in every script
DEFAULT_QUESTION = "Draft my reply to this email, using what my meetings say about it."
STYLE_PROMPT = """Answer from the context. Never invent a date, price or promise.
When asked to write or draft a reply, write it as the user would: match the greeting,
length, tone and sign-off of the user's own emails given below. Write in the language of
the email being answered, not the language of the examples."""


def header(text: str, field: str) -> str:
    """One header line of a remembered email (From, Received, ...), or ""."""
    match = re.search(rf"^{field}: (.*)$", text, re.MULTILINE)
    return match.group(1).strip() if match else ""


async def pick_email(sender: str | None) -> str:
    """The newest inbox email, or the newest one whose From line contains ``sender``."""
    # Similarity search can't tell which email is newest, so take every inbox chunk and
    # choose by the headers ingest_email.py stored. Only an email's first chunk carries them.
    chunks = await cognee.recall(
        sender or "email",
        query_type=SearchType.CHUNKS,
        datasets=[DATASET],
        node_name=["email"],
        top_k=100,
    )
    emails = [str(chunk.text) for chunk in chunks if header(str(chunk.text), "From")]
    if sender:
        emails = [text for text in emails if sender.lower() in header(text, "From").lower()]
    if not emails:
        who = f" from {sender}" if sender else ""
        raise SystemExit(f"[answer] No inbox email{who}. Run ingest_email.py first.")
    return max(emails, key=lambda text: header(text, "Received"))


async def answer(question: str = DEFAULT_QUESTION, sender: str | None = None) -> None:
    email = await pick_email(sender)
    print(f"[answer] Email: {email.splitlines()[0]} (from {header(email, 'From')})")
    own_emails = await cognee.recall(
        "Emails I wrote to a person",
        query_type=SearchType.CHUNKS,
        datasets=[DATASET],
        node_name=["sent"],
        top_k=3,
    )
    prompt = f"{question}\n\nThe email to answer:\n{email}"
    if own_emails:
        examples = "\n---\n".join(str(chunk.text) for chunk in own_emails)
        prompt += f"\n\nExamples of my own emails, for style only:\n{examples}"
    results = await cognee.recall(
        prompt,
        query_type=SearchType.GRAPH_COMPLETION,
        datasets=[DATASET],
        system_prompt=STYLE_PROMPT,
    )
    reply = str(results[0].text) if results else "Nothing found. Run the ingest scripts first."
    print(f"[answer] Q: {question}\n[answer] A: {reply}")


if __name__ == "__main__":
    setup_logging(log_level=ERROR)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question", nargs="*", help="what to ask (default: draft a reply)")
    parser.add_argument("--sender", help="answer the newest email from this name or address")
    args = parser.parse_args()
    asyncio.run(answer(" ".join(args.question) or DEFAULT_QUESTION, args.sender))
