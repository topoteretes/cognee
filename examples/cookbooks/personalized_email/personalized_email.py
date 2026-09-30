"""Personalized email: draft a reply that knows your meetings, your promises and your style.

cognee remembers your Granola meeting notes and your Gmail inbox and sent mail into one
knowledge graph. Then it drafts a reply to the newest email in your inbox: the facts come
from memory, and the tone from your own sent mail. The draft is printed, never sent.

Run: uv run python examples/cookbooks/personalized_email/personalized_email.py
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

import cognee
from cognee.modules.search.types import SearchType
from cognee.shared.logging_utils import ERROR, setup_logging
from cognee.tasks.ingestion.connectors import gmail_source
from cognee.tasks.ingestion.connectors.gmail import build_gmail_service, parse_message

DATASET = "personalized_email"
ME = os.environ.get("MY_NAME", "me")  # your name, as it appears in your email
HERE = Path(__file__).parent
GMAIL = {"credentials_path": str(HERE / "credentials.json"), "token_path": str(HERE / "token.json")}

DRAFT_PROMPT = f"""You write the reply {ME} would send to an email.
- Answer every question with facts from the context. Never invent a date, price or promise.
- If {ME} owes the sender something, say plainly whether it was sent.
- Match the greeting, length, tone and sign-off of {ME}'s own emails."""


def granola_notes(days: int = 30) -> list[str]:
    """Your Granola meeting notes from the last ``days`` days, each as text."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    headers = {"Authorization": f"Bearer {os.environ['GRANOLA_API_KEY']}"}
    api = httpx.Client(base_url="https://public-api.granola.ai/v1/", headers=headers, timeout=60)
    notes, params = [], {"created_after": since, "page_size": 30}
    while True:
        page = api.get("notes", params=params).raise_for_status().json()
        for listed in page["notes"]:
            note = api.get(f"notes/{listed['id']}", params={"include": "transcript"}).json()
            attendees = ", ".join(a.get("name") or a["email"] for a in note.get("attendees") or [])
            transcript = "\n".join(
                f"{(turn['speaker'] or {}).get('name') or 'Speaker'}: {turn['text']}"
                for turn in note.get("transcript") or []
            )
            notes.append(
                f"Meeting: {note.get('title')}\nDate: {note.get('created_at')}\n"
                f"Attendees: {attendees}\n\n{note.get('summary_text') or ''}\n\n{transcript}"
            )
        if not page.get("hasMore"):
            return notes
        params["cursor"] = page["cursor"]


def newest_email() -> str:
    """The newest email in your inbox, as text."""
    messages = build_gmail_service(**GMAIL).users().messages()
    newest = messages.list(userId="me", labelIds=["INBOX"], maxResults=1).execute()["messages"][0]
    email = parse_message(messages.get(userId="me", id=newest["id"], format="full").execute())
    return f"Subject: {email['title']}\n{email['content']}"


async def main(ui: bool = False) -> None:
    # 1. Remember your meetings, inbox and sent mail, each in its own node set.
    notes = granola_notes() if os.environ.get("GRANOLA_API_KEY") else []
    if notes:
        print(f"Remembering {len(notes)} Granola meeting notes...")
        await cognee.remember(
            notes, dataset_name=DATASET, node_set=["meetings"], self_improvement=False
        )
    for label, node_set in (("INBOX", "inbox"), ("SENT", "sent_mail")):
        print(f"Remembering your Gmail {node_set}...")
        await cognee.remember(
            # cognee's Gmail connector; max_results keeps the first try small.
            gmail_source(
                resource_name=f"gmail_{node_set}", label_ids=[label], max_results=50, **GMAIL
            ),
            dataset_name=DATASET,
            node_set=[node_set],
            write_disposition="merge",  # the connector's rows are merged by message id
            primary_key="id",
            max_rows_per_table=0,
            self_improvement=False,
        )

    # 2. Draft a reply to the newest email: your own emails for tone, the graph for facts.
    email = newest_email()
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
    print(f"\n== Incoming ==\n{email}\n\n== Draft reply ==\n{reply[0].text}")

    if ui:
        await open_ui()


async def open_ui() -> None:
    """Browse the graph: cognee's API server runs in this process, next to the databases it
    already has open, and the UI runs as its own process. Ctrl+C stops both."""
    import uvicorn

    from cognee.api.client import app
    from cognee.api.v1.ui.ui import remove_ui_container, stop_ui_pid

    server = uvicorn.Server(uvicorn.Config(app, port=8000, log_level="warning"))
    api = asyncio.create_task(server.serve())
    while not server.started:
        if api.done():
            return await api  # raises the startup error, such as a port in use
        await asyncio.sleep(0.2)
    spawned: list = []  # a PID, or (PID, container) when the UI runs in Docker
    await asyncio.to_thread(cognee.start_ui, spawned.append, auto_download=True)
    print("Browse the graph at http://localhost:3000. Press Ctrl+C to stop.")
    try:
        await api  # returns once uvicorn has handled Ctrl+C
    finally:
        for item in spawned:
            pid, container = item if isinstance(item, tuple) else (item, None)
            if container:
                remove_ui_container(container)
            stop_ui_pid(pid)


if __name__ == "__main__":
    if not Path(GMAIL["credentials_path"]).exists():
        sys.exit(
            "Save your Gmail OAuth client as credentials.json next to this script (see README)."
        )
    setup_logging(log_level=ERROR)
    asyncio.run(main("--ui" in sys.argv))
