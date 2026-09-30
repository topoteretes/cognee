"""The two sources this cookbook reads: Granola meeting notes and Gmail.

Gmail uses the connector that ships with cognee (``cognee[gmail]``): it is a dlt resource that
tracks its own position, so passing it to ``remember`` again only loads what changed.

Granola has no connector in cognee, so ``fetch_granola_notes`` calls its public API directly.
Its position is a watermark kept in ``.state.json`` next to this file.

Both sources become plain text documents. The ``--sample`` files in ``sample_data/`` use the
same shapes as the real APIs and go through the same rendering.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiohttp

HERE = Path(__file__).parent
SAMPLE_DATA = HERE / "sample_data"
STATE_FILE = HERE / ".state.json"

GRANOLA_API = os.environ.get("GRANOLA_API_BASE_URL", "https://public-api.granola.ai/v1")
GMAIL_CREDENTIALS = os.environ.get("GMAIL_CREDENTIALS_PATH", str(HERE / "credentials.json"))
GMAIL_TOKEN = os.environ.get("GMAIL_TOKEN_PATH", str(HERE / "token.json"))

# The same remember() arguments on every Gmail sync. write_disposition="merge" keeps earlier
# messages; the default "replace" would drop them on the next sync.
GMAIL_REMEMBER_KWARGS = {"primary_key": "id", "write_disposition": "merge", "max_rows_per_table": 0}


# ---------------------------------------------------------------- state between syncs


def load_state() -> dict:
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def reset_state() -> None:
    STATE_FILE.unlink(missing_ok=True)


# ---------------------------------------------------------------- Granola


async def fetch_granola_notes(since: datetime) -> list[dict]:
    """Every Granola note created after ``since``, each with its transcript."""
    headers = {"Authorization": f"Bearer {os.environ['GRANOLA_API_KEY']}"}
    notes = []
    async with aiohttp.ClientSession(
        base_url=GRANOLA_API.rstrip("/") + "/", headers=headers
    ) as http:
        cursor = None
        while True:
            params = {"created_after": since.isoformat().replace("+00:00", "Z")}
            if cursor:
                params["cursor"] = cursor
            async with http.get("notes", params=params) as response:
                response.raise_for_status()
                page = await response.json()
            for listed in page.get("notes", []):
                async with http.get(f"notes/{listed['id']}", params={"include": "transcript"}) as r:
                    r.raise_for_status()
                    notes.append(await r.json())
            cursor = page.get("cursor")
            if not page.get("hasMore") or not cursor:
                return notes


def granola_since(state: dict) -> datetime:
    """Where the last Granola sync stopped, or INGEST_SINCE_DAYS ago on the first run."""
    if "granola_since" in state:
        return datetime.fromisoformat(state["granola_since"])
    days = int(os.environ.get("INGEST_SINCE_DAYS", "30"))
    return datetime.now(timezone.utc) - timedelta(days=days)


def advance_granola_watermark(state: dict, notes: list[dict]) -> None:
    if notes:
        state["granola_since"] = max(note["created_at"] for note in notes).replace("Z", "+00:00")


def _person(person: dict | str | None) -> str:
    if isinstance(person, dict):
        name, email = person.get("name"), person.get("email")
        return f"{name} <{email}>" if name and email else (name or email or "")
    return person or ""


def render_note(note: dict) -> str:
    """One Granola note as the text cognee extracts from."""
    lines = [
        f"Meeting: {note.get('title', 'Untitled meeting')}",
        f"Date: {note.get('created_at', '')}",
        "Attendees: " + ", ".join(_person(p) for p in note.get("attendees", [])),
    ]
    summary = note.get("summary_markdown") or note.get("summary_text") or note.get("summary")
    if summary:
        lines += ["", summary]
    transcript = note.get("transcript") or []
    if transcript:
        lines += ["", "Transcript:"]
        lines += [f"{_person(t.get('speaker')) or 'Speaker'}: {t['text']}" for t in transcript]
    return "\n".join(lines)


# ---------------------------------------------------------------- email


def render_email(email: dict) -> str:
    """One email as the text cognee extracts from, in the same layout as the Gmail connector."""
    return "\n".join(
        [
            f"Subject: {email['subject']}",
            f"From: {email['from']}",
            f"To: {email['to']}",
            f"Date: {email['date']}",
            f"Labels: {', '.join(email.get('labels', []))}",
            "",
            email["body"],
        ]
    )


def gmail_sources():
    """Two Gmail connector resources: the inbox, and the mail you sent (your writing style)."""
    from cognee.tasks.ingestion.connectors import gmail_source

    common = {"credentials_path": GMAIL_CREDENTIALS, "token_path": GMAIL_TOKEN}
    max_results = (
        int(os.environ["GMAIL_MAX_RESULTS"]) if os.environ.get("GMAIL_MAX_RESULTS") else None
    )
    inbox = gmail_source(
        resource_name="gmail_inbox", label_ids=["INBOX"], max_results=max_results, **common
    )
    sent = gmail_source(
        resource_name="gmail_sent", label_ids=["SENT"], max_results=max_results, **common
    )
    return inbox, sent


def fetch_gmail_message(message_id: str) -> dict:
    """One Gmail message, for the draft agent to answer: sender, subject and full text."""
    from cognee.tasks.ingestion.connectors.gmail import build_gmail_service, parse_message

    service = build_gmail_service(GMAIL_CREDENTIALS, GMAIL_TOKEN)
    message = service.users().messages().get(userId="me", id=message_id, format="full").execute()
    headers = {h["name"].lower(): h["value"] for h in message["payload"].get("headers", [])}
    row = parse_message(message)
    return {
        "id": message_id,
        "from": headers.get("from", ""),
        "subject": row["title"],
        "text": f"Subject: {row['title']}\n{row['content']}",
    }


# ---------------------------------------------------------------- sample data


def sample_batch(folder: Path) -> tuple[list[dict], list[dict]]:
    """The Granola notes and emails in one sample folder."""
    notes = json.loads((folder / "granola_notes.json").read_text())
    emails = json.loads((folder / "emails.json").read_text())
    return notes, emails


def split_by_label(emails: list[dict]) -> tuple[list[dict], list[dict]]:
    inbox = [e for e in emails if "SENT" not in e.get("labels", [])]
    sent = [e for e in emails if "SENT" in e.get("labels", [])]
    return inbox, sent
