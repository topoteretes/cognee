"""The three sources the follow-up agent reads: Granola calls, Gmail and Linear.

Gmail uses the connector that ships with cognee (``cognee[gmail]``): it is a dlt resource that
tracks its own position, so passing it to ``remember`` again only loads what changed.

Granola and Linear have no connector in cognee, so this module calls their APIs directly and
keeps a watermark for each in ``.state.json`` next to this file: the newest note or issue
update it has seen.

Every source becomes plain text. The ``--sample`` files in ``sample_data/`` use the same
shapes as the real APIs and go through the same rendering.
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
LINEAR_API = "https://api.linear.app/graphql"
GMAIL_CREDENTIALS = os.environ.get("GMAIL_CREDENTIALS_PATH", str(HERE / "credentials.json"))
GMAIL_TOKEN = os.environ.get("GMAIL_TOKEN_PATH", str(HERE / "token.json"))

# The same remember() arguments on every Gmail sync. write_disposition="merge" keeps earlier
# messages; the default "replace" would drop them on the next sync.
GMAIL_REMEMBER_KWARGS = {"primary_key": "id", "write_disposition": "merge", "max_rows_per_table": 0}


# ---------------------------------------------------------------- state between syncs


def load_state() -> dict:
    state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
    state.setdefault("followed_up_calls", [])
    state.setdefault("pending", {})
    return state


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def reset_state() -> None:
    STATE_FILE.unlink(missing_ok=True)


def since(state: dict, key: str) -> datetime:
    """Where the last sync of one source stopped, or INGEST_SINCE_DAYS ago on the first run."""
    if key in state:
        return datetime.fromisoformat(state[key])
    return datetime.now(timezone.utc) - timedelta(
        days=int(os.environ.get("INGEST_SINCE_DAYS", "30"))
    )


def advance(state: dict, key: str, timestamps: list[str]) -> None:
    if timestamps:
        state[key] = max(timestamps).replace("Z", "+00:00")


def _iso(moment: datetime) -> str:
    return moment.isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------- Granola


async def fetch_granola_notes(created_after: datetime) -> list[dict]:
    """Every Granola note created after ``created_after``, each with its transcript."""
    headers = {"Authorization": f"Bearer {os.environ['GRANOLA_API_KEY']}"}
    notes = []
    async with aiohttp.ClientSession(
        base_url=GRANOLA_API.rstrip("/") + "/", headers=headers
    ) as http:
        cursor = None
        while True:
            params = {"created_after": _iso(created_after)}
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


# ---------------------------------------------------------------- Linear


LINEAR_ISSUES_QUERY = """
query Issues($since: DateTimeOrDuration!, $after: String) {
  issues(filter: {updatedAt: {gt: $since}}, first: 100, after: $after) {
    nodes {
      identifier title description url dueDate updatedAt
      state { name }
      assignee { name email }
      team { name key }
      project { name }
    }
    pageInfo { hasNextPage endCursor }
  }
}
"""


async def linear_graphql(query: str, variables: dict | None = None) -> dict:
    """Run one Linear GraphQL operation with a personal API key and return its data."""
    headers = {"Authorization": os.environ["LINEAR_API_KEY"], "Content-Type": "application/json"}
    body = {"query": query, "variables": variables or {}}
    async with (
        aiohttp.ClientSession(headers=headers) as http,
        http.post(LINEAR_API, json=body) as r,
    ):
        r.raise_for_status()
        payload = await r.json()
    if payload.get("errors"):
        raise RuntimeError(f"Linear API error: {payload['errors'][0].get('message')}")
    return payload["data"]


async def fetch_linear_issues(updated_after: datetime) -> list[dict]:
    """Every Linear issue created or changed after ``updated_after``."""
    issues, after = [], None
    while True:
        data = await linear_graphql(
            LINEAR_ISSUES_QUERY, {"since": _iso(updated_after), "after": after}
        )
        issues += data["issues"]["nodes"]
        if not data["issues"]["pageInfo"]["hasNextPage"]:
            return issues
        after = data["issues"]["pageInfo"]["endCursor"]


def render_issue(issue: dict) -> str:
    """One Linear issue as the text cognee extracts from."""
    assignee = issue.get("assignee") or {}
    return "\n".join(
        [
            f"Linear issue {issue['identifier']}: {issue['title']}",
            f"Status: {(issue.get('state') or {}).get('name', '')}",
            f"Team: {(issue.get('team') or {}).get('name', '')}",
            f"Project: {(issue.get('project') or {}).get('name', '')}",
            f"Assignee: {_person(assignee) or 'unassigned'}",
            f"Due: {issue.get('dueDate') or 'none'}",
            f"URL: {issue.get('url', '')}",
            "",
            issue.get("description") or "",
        ]
    )


# ---------------------------------------------------------------- email


def render_email(email: dict) -> str:
    """One email as the text cognee extracts from, in the same layout as the Gmail connector."""
    return "\n".join(
        [
            f"Subject: {email['subject']}",
            f"From: {email['from']}",
            f"To: {email['to']}",
            f"Date: {email['date']}",
            "",
            email["body"],
        ]
    )


def gmail_inbox():
    from cognee.tasks.ingestion.connectors import gmail_source

    max_results = (
        int(os.environ["GMAIL_MAX_RESULTS"]) if os.environ.get("GMAIL_MAX_RESULTS") else None
    )
    return gmail_source(
        credentials_path=GMAIL_CREDENTIALS,
        token_path=GMAIL_TOKEN,
        label_ids=["INBOX"],
        max_results=max_results,
    )


# ---------------------------------------------------------------- sample data


def sample_batch(folder: Path) -> dict[str, list[dict]]:
    """The Granola notes, emails and Linear issues in one sample folder."""
    batch = {}
    for name in ("granola_notes", "emails", "linear_issues"):
        path = folder / f"{name}.json"
        batch[name] = json.loads(path.read_text()) if path.exists() else []
    return batch
