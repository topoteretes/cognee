"""Personalized email with cognee: draft replies that know your meetings, promises and style.

Granola meeting notes and your Gmail inbox and sent mail are remembered into one graph
(models.py). A draft agent then answers an email in four steps: it asks memory who the sender
is, what you still owe them, and what the answers to their questions are, reads a few of your
own emails for tone, and writes the reply from what it found.

Commands:
  ingest   Build the memory from scratch.
  sync     Load only what is new since the last sync; --interval repeats it.
  serve    Run the API server and the UI, and keep syncing in the background.
  draft    Draft a reply to one email.

Every command takes --sample to use the files in sample_data/ instead of your accounts.

Requires: LLM_API_KEY. Real data also needs GRANOLA_API_KEY and/or cognee[gmail] with a
Gmail OAuth client (see README.md). The UI needs Node.js and npm (or Docker).
Run: uv run python examples/cookbooks/personalized_email/personalized_email.py ingest --sample
     uv run python examples/cookbooks/personalized_email/personalized_email.py draft --sample
     uv run python examples/cookbooks/personalized_email/personalized_email.py sync --sample
     uv run python examples/cookbooks/personalized_email/personalized_email.py serve --sample
"""

import argparse
import asyncio
import json
import os
import socket
import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

# Local single-user mode: the SDK and the API server share one set of databases and the API
# needs no login. Load .env first so an explicit setting there still wins.
load_dotenv(override=False)
os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")

import cognee  # noqa: E402
from cognee.infrastructure.llm.LLMGateway import LLMGateway  # noqa: E402
from cognee.modules.search.types import SearchType  # noqa: E402
from cognee.shared.logging_utils import ERROR, setup_logging  # noqa: E402

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import sources  # noqa: E402
from models import EXTRACTION_PROMPT, InboxGraph  # noqa: E402

DATASET = "personalized_email"
GRANOLA_NOTES, INBOX, SENT_MAIL = "granola_notes", "inbox", "sent_mail"
ME = os.environ.get("MY_NAME", "Alex Rivera")
DRAFTS = HERE / "drafts"

# The cookbook keeps its memory in its own folder, so `ingest` can rebuild it from scratch
# without touching anything else cognee has stored on this machine.
STORE = HERE / ".cognee_system"
cognee.config.system_root_directory(str(STORE))
cognee.config.data_root_directory(str(STORE / "data"))


# ---------------------------------------------------------------- 1. ingestion


async def remember(data, node_set: str, **kwargs) -> None:
    """Remember one source into its own node set, extracted with the inbox graph model."""
    await cognee.remember(
        data,
        dataset_name=DATASET,
        node_set=[node_set],
        graph_model=InboxGraph,
        custom_prompt=EXTRACTION_PROMPT,
        self_improvement=False,
        **kwargs,
    )


async def remember_batch(notes: list[dict], emails: list[dict]) -> None:
    inbox, sent = sources.split_by_label(emails)
    if notes:
        await remember([sources.render_note(note) for note in notes], GRANOLA_NOTES)
    if inbox:
        await remember([sources.render_email(email) for email in inbox], INBOX)
    if sent:
        await remember([sources.render_email(email) for email in sent], SENT_MAIL)


async def ingest(sample: bool) -> None:
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    sources.reset_state()

    if sample:
        notes, emails = sources.sample_batch(sources.SAMPLE_DATA)
        print(f"Remembering {len(notes)} sample meeting notes and {len(emails)} sample emails...")
        await remember_batch(notes, emails)
    else:
        print(await sync_once(sample=False))
    print("Memory is ready. Try: draft --sample" if sample else "Memory is ready.")


# ---------------------------------------------------------------- 2. keeping memory live


async def sync_once(sample: bool) -> str:
    """Remember only what arrived since the last sync, and report what that was."""
    state = sources.load_state()

    if sample:
        # The sample stands in for a live account: its incoming/ folder is what "arrives"
        # after the first ingest, and it arrives once.
        if state.get("sample_incoming_synced"):
            return "sync: nothing new"
        notes, emails = sources.sample_batch(sources.SAMPLE_DATA / "incoming")
        await remember_batch(notes, emails)
        state["sample_incoming_synced"] = True
        sources.save_state(state)
        return f"sync: {len(notes)} new meeting notes, {len(emails)} new emails"

    report = []
    if os.environ.get("GRANOLA_API_KEY"):
        notes = await sources.fetch_granola_notes(sources.granola_since(state))
        if notes:
            await remember([sources.render_note(note) for note in notes], GRANOLA_NOTES)
        sources.advance_granola_watermark(state, notes)
        report.append(f"{len(notes)} new meeting notes")
    if Path(sources.GMAIL_CREDENTIALS).exists():
        # The Gmail connector remembers where it stopped, so this loads only new, changed and
        # deleted messages after the first run.
        inbox, sent = sources.gmail_sources()
        await remember(inbox, INBOX, **sources.GMAIL_REMEMBER_KWARGS)
        await remember(sent, SENT_MAIL, **sources.GMAIL_REMEMBER_KWARGS)
        report.append(f"Gmail: {inbox.cognee_sync_stats}")
    sources.save_state(state)
    if not report:
        return "sync: no source configured (set GRANOLA_API_KEY or add Gmail credentials.json)"
    return "sync: " + ", ".join(report)


async def sync_forever(sample: bool, interval: int) -> None:
    while True:
        try:
            print(await sync_once(sample), flush=True)
        except Exception as error:  # noqa: BLE001 - a failed sync must not stop the next one.
            print(f"sync failed: {error}", flush=True)
        await asyncio.sleep(interval)


# ---------------------------------------------------------------- 3. API server and UI


async def serve(sample: bool, interval: int, api_port: int, ui_port: int) -> None:
    """Run the API server, the UI and the sync loop in this one process.

    The graph database is embedded and only one process may open it, so the API server runs
    in-process (uvicorn) instead of as a child process. Syncs and UI requests then share it.
    """
    import uvicorn

    from cognee.api.client import app
    from cognee.api.v1.ui.ui import remove_ui_container, stop_ui_pid

    for port in (api_port, ui_port):
        with socket.socket() as probe:
            if probe.connect_ex(("localhost", port)) == 0:
                sys.exit(f"Port {port} is already in use: pass --api-port / --ui-port.")

    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=api_port, log_level="warning")
    )
    api = asyncio.create_task(server.serve())
    while not server.started:
        if api.done():
            return await api  # Surfaces a startup error.
        await asyncio.sleep(0.2)

    # The UI assumes the API is on port 8000 unless COGNEE_BACKEND_URL says otherwise.
    os.environ["COGNEE_BACKEND_URL"] = f"http://localhost:{api_port}"
    spawned: list = []
    ui = await asyncio.to_thread(
        cognee.start_ui, pid_callback=spawned.append, port=ui_port, auto_download=True
    )
    print(
        f"\nUI:  http://localhost:{ui_port}"
        if ui
        else "\nThe UI did not start; the API still runs."
    )
    print(f"API: http://localhost:{api_port}\nSyncing every {interval}s. Press Ctrl+C to stop.\n")

    syncing = asyncio.create_task(sync_forever(sample, interval))
    try:
        await api  # Returns when uvicorn handles Ctrl+C.
    finally:
        syncing.cancel()
        for item in spawned:
            pid, container = item if isinstance(item, tuple) else (item, None)
            if container:
                remove_ui_container(container)
            stop_ui_pid(pid)


# ---------------------------------------------------------------- 4. the draft agent


class EmailDraft(BaseModel):
    subject: str
    body: str
    facts_used: list[str]


DRAFT_PROMPT = f"""
You write email replies for {ME}. Write the reply {ME} would send to the incoming email.

- Answer every question in the incoming email with the facts you are given. Never invent a
  date, a price, a document or a promise that the facts do not contain.
- If {ME} owes the sender something, say plainly whether it was sent. Do not claim it was,
  and do not promise a new date: write [WHEN] where {ME} should fill one in.
- Match the greeting, length, tone and sign-off of {ME}'s own emails.
- In facts_used, list each fact from memory the reply relies on.
"""


def load_email(sample: bool, message_id: str | None) -> dict:
    if not sample:
        if not message_id:
            sys.exit("draft needs --message-id <Gmail message id> (or --sample).")
        return sources.fetch_gmail_message(message_id)

    emails = [
        *sources.sample_batch(sources.SAMPLE_DATA)[1],
        *sources.sample_batch(sources.SAMPLE_DATA / "incoming")[1],
    ]
    wanted = message_id or "msg_inbox_brightline_questions"
    email = next((e for e in emails if e["id"] == wanted), None)
    if email is None:
        sys.exit(f"No sample email with id {wanted}.")
    return {**email, "text": sources.render_email(email)}


async def draft(email: dict) -> EmailDraft:
    sender = email["from"]
    # One session per draft: each question sees the ones before it, and no other draft's.
    session_id = f"draft-{email['id']}"

    async def ask(question: str) -> str:
        results = await cognee.recall(
            question,
            query_type=SearchType.GRAPH_COMPLETION,
            datasets=[DATASET],
            session_id=session_id,
        )
        return results[0].text if results else "(nothing in memory)"

    print(f"Researching {sender}...")
    about_sender = await ask(
        f"Who is {sender}? Give their role and organization, the meetings {ME} had with them, "
        "and what was decided in those meetings."
    )
    open_commitments = await ask(
        f"Which commitments did {ME} make to {sender} or their organization? For each, give "
        "the due date and whether any email shows it was done."
    )
    answers = await ask(
        "Answer each question asked in this email, using only facts from meetings and "
        f"emails:\n\n{email['text']}"
    )

    # Raw text, read from one node set: the meeting notes closest to the email, so dates
    # and terms are copied exactly, and your own sent emails, for tone.
    async def closest(query: str, node_set: str) -> str:
        chunks = await cognee.recall(
            query,
            query_type=SearchType.CHUNKS,
            datasets=[DATASET],
            node_name=[node_set],
            top_k=3,
        )
        return "\n---\n".join(str(chunk.text) for chunk in chunks) or "(none)"

    print("Writing the reply...")
    return await LLMGateway.acreate_structured_output(
        f"Incoming email:\n{email['text']}\n\n"
        f"What memory says about the sender:\n{about_sender}\n\n"
        f"Commitments towards the sender:\n{open_commitments}\n\n"
        f"Answers found in memory:\n{answers}\n\n"
        f"Meeting notes closest to this email:\n{await closest(email['text'], GRANOLA_NOTES)}\n\n"
        f"Examples of {ME}'s own emails:\n{await closest(f'Emails written by {ME}', SENT_MAIL)}",
        DRAFT_PROMPT,
        EmailDraft,
    )


async def draft_command(sample: bool, message_id: str | None) -> None:
    email = load_email(sample, message_id)
    reply = await draft(email)

    print(f"\n== Draft reply ==\nSubject: {reply.subject}\n\n{reply.body}\n")
    print("== Facts from memory ==")
    for fact in reply.facts_used:
        print(f"  - {fact}")

    DRAFTS.mkdir(exist_ok=True)
    path = DRAFTS / f"{email['id']}.json"
    path.write_text(json.dumps({"in_reply_to": email["id"], **reply.model_dump()}, indent=2))
    print(f"\nSaved to {path.relative_to(HERE)}")


# ---------------------------------------------------------------- command line


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("ingest", "sync", "serve", "draft"):
        command = commands.add_parser(name)
        command.add_argument("--sample", action="store_true", help="use sample_data/")
        if name in ("sync", "serve"):
            command.add_argument("--interval", type=int, default=300, help="seconds between syncs")
        if name == "sync":
            command.add_argument(
                "--watch", action="store_true", help="keep syncing every --interval"
            )
        if name == "serve":
            command.add_argument("--api-port", type=int, default=8000)
            command.add_argument("--ui-port", type=int, default=3000)
        if name == "draft":
            command.add_argument("--message-id", help="the email to answer")
    args = parser.parse_args()

    setup_logging(log_level=ERROR)
    if args.command == "ingest":
        asyncio.run(ingest(args.sample))
    elif args.command == "sync":
        if args.watch:
            asyncio.run(sync_forever(args.sample, args.interval))
        else:
            print(asyncio.run(sync_once(args.sample)))
    elif args.command == "serve":
        asyncio.run(serve(args.sample, args.interval, args.api_port, args.ui_port))
    else:
        asyncio.run(draft_command(args.sample, args.message_id))


if __name__ == "__main__":
    main()
