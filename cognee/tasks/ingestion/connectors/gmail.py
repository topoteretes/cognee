"""Gmail connector for cognee — a ``dlt`` source that turns your inbox into memory.

Pull Gmail messages (optionally label-scoped) into cognee, incrementally
and with forget-on-deletion — "ask my inbox".  This builds entirely on the
existing DLT ingestion subsystem; the source produced here is meant to be
handed directly to :func:`cognee.remember`::

    import cognee
    from cognee.tasks.ingestion.connectors import gmail_source

    await cognee.remember(
        gmail_source(label_ids=["INBOX"]),
        dataset_name="my_inbox",
        primary_key="id",
        write_disposition="merge",   # incremental upsert by message id
        max_rows_per_table=0,        # 0 = no row cap (see note below)
    )

Design
------
* **Auth** — OAuth2 *installed-app* flow.  Point ``credentials_path`` at the
  client-secret JSON you download from Google Cloud Console; the resulting
  user token is cached at ``token_path`` and refreshed automatically.  Scope is
  read-only (``gmail.readonly``).
* **Primary key** — the Gmail message ``id``.  Combined with
  ``write_disposition="merge"`` this gives idempotent upserts.
* **Incremental cursor** — Gmail's ``historyId``.  The first run does a full
  (label-scoped) backfill and records the mailbox ``historyId``; subsequent
  runs call ``users.history.list(startHistoryId=...)`` and emit only the delta
  (added / changed / deleted messages).  The cursor is persisted in dlt's
  per-resource state, so re-running ``remember`` resumes where it left off.
* **Forget-on-delete** — messages reported as deleted/trashed by the History
  API are emitted with the ``_deleted`` hard-delete marker.  dlt removes those
  rows from its destination on ``merge``; they then fall out of the freshly
  read row set and cognee's existing ``orphan_cleanup`` purges them from the
  graph + vector + relational stores.  A full backfill reconciles the same way.

.. note::
   cognee's ``ingest_dlt_source`` reads at most ``max_rows_per_table`` rows
   from the dlt destination (default 50).  For a real inbox pass
   ``max_rows_per_table=0`` (unlimited) so orphan-cleanup compares against the
   *whole* synced corpus rather than a truncated window.

Privacy
-------
This connector reads the content of your email.  It is **opt-in**: nothing is
fetched until you explicitly construct a source and call ``remember``.  Use
``label_ids`` to scope what leaves your mailbox, keep the OAuth
token file (``token.json``) private, and prefer a dedicated dataset so you can
``cognee.forget`` the inbox in one call.
"""

from __future__ import annotations

import base64
from typing import Any, Dict, Iterator, List, Optional

from cognee.shared.logging_utils import get_logger

logger = get_logger("gmail_connector")

# Read-only access — the connector never modifies the mailbox.
GMAIL_READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"

# History event types we care about for incremental sync.
_HISTORY_TYPES = ["messageAdded", "messageDeleted", "labelAdded", "labelRemoved"]


# ---------------------------------------------------------------------------
# Auth / service construction
# ---------------------------------------------------------------------------
def build_gmail_service(
    credentials_path: str = "credentials.json",
    token_path: str = "token.json",
) -> Any:
    """Build an authenticated Gmail API client via the OAuth2 installed-app flow.

    On first run this opens a browser to consent and caches the resulting token
    at ``token_path``.  Later runs reuse / silently refresh that token.

    The Google client libraries are imported lazily so they remain an optional
    dependency (``pip install "cognee[gmail]"``).
    """
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "The Gmail connector requires the 'gmail' extra. Install it with:\n"
            '    pip install "cognee[gmail]"\n'
            "(provides google-api-python-client, google-auth, google-auth-oauthlib)."
        ) from exc

    import os

    scopes = [GMAIL_READONLY_SCOPE]
    creds = None

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, scopes)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(credentials_path):
                raise FileNotFoundError(
                    f"Gmail OAuth client secrets not found at '{credentials_path}'. "
                    "Download an OAuth 2.0 Client ID (Desktop app) from the Google "
                    "Cloud Console and point credentials_path at the JSON file."
                )
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, scopes)
            creds = flow.run_local_server(port=0)
        with open(token_path, "w", encoding="utf-8") as token_file:
            token_file.write(creds.to_json())

    return build("gmail", "v1", credentials=creds, cache_discovery=False)


# ---------------------------------------------------------------------------
# Message parsing
# ---------------------------------------------------------------------------
def _decode_body(data: Optional[str]) -> str:
    """Decode a base64url-encoded Gmail body part into text."""
    if not data:
        return ""
    try:
        return base64.urlsafe_b64decode(data.encode("utf-8")).decode("utf-8", errors="replace")
    except Exception:  # pragma: no cover - defensive
        return ""


def _extract_plaintext(payload: Optional[dict]) -> str:
    """Walk a Gmail message payload and return its ``text/plain`` body.

    Depth-first search for the first ``text/plain`` part anywhere in the MIME
    tree.  HTML-only messages return "" (the snippet still carries a preview),
    which keeps ingested text clean for entity extraction rather than feeding
    raw HTML markup into the graph.
    """
    if not payload:
        return ""

    mime_type = payload.get("mimeType", "")
    body = payload.get("body", {}) or {}

    if mime_type == "text/plain" and body.get("data"):
        return _decode_body(body.get("data"))

    for part in payload.get("parts") or []:
        text = _extract_plaintext(part)
        if text:
            return text

    return ""


def _headers_to_dict(payload: Optional[dict]) -> Dict[str, str]:
    """Index a message's headers case-insensitively by name."""
    headers = {}
    for header in (payload or {}).get("headers", []) or []:
        name = header.get("name", "").lower()
        if name:
            headers[name] = header.get("value", "")
    return headers


def parse_message(message: dict) -> Dict[str, Any]:
    """Flatten a Gmail ``users.messages.get`` resource into a dlt row.

    Lists (label ids) are flattened to a comma-separated string so dlt does not
    spawn a child table per message; this keeps the row 1:1 with a cognee
    ``DataItem`` and the orphan-cleanup bookkeeping simple.
    """
    payload = message.get("payload", {}) or {}
    headers = _headers_to_dict(payload)
    label_ids = message.get("labelIds", []) or []

    internal_date_raw = message.get("internalDate")
    try:
        internal_date = int(internal_date_raw) if internal_date_raw is not None else 0
    except (TypeError, ValueError):
        internal_date = 0

    return {
        "id": message.get("id"),
        "thread_id": message.get("threadId"),
        "labels": ", ".join(label_ids),
        "subject": headers.get("subject", ""),
        "from": headers.get("from", ""),
        "to": headers.get("to", ""),
        "cc": headers.get("cc", ""),
        "date": headers.get("date", ""),
        "snippet": message.get("snippet", ""),
        "body": _extract_plaintext(payload),
        "internal_date": internal_date,
        # Hard-delete marker (always False for live messages). Deleted/trashed
        # messages are emitted separately with _deleted=True.
        "_deleted": False,
    }


def _deleted_row(message_id: str) -> Dict[str, Any]:
    """Build a minimal row that instructs dlt to hard-delete a message by id."""
    return {"id": message_id, "_deleted": True}


# ---------------------------------------------------------------------------
# Gmail API helpers (paginated)
# ---------------------------------------------------------------------------
def _list_message_ids(
    service: Any,
    label_ids: Optional[List[str]],
    max_results: Optional[int],
) -> Iterator[str]:
    """Yield message ids matching the given labels, following pagination."""
    page_token = None
    fetched = 0
    while True:
        request = (
            service.users()
            .messages()
            .list(
                userId="me",
                labelIds=label_ids or None,
                pageToken=page_token,
            )
        )
        response = request.execute()
        for ref in response.get("messages", []) or []:
            yield ref["id"]
            fetched += 1
            if max_results and fetched >= max_results:
                return
        page_token = response.get("nextPageToken")
        if not page_token:
            return


def _get_message(service: Any, message_id: str) -> Optional[dict]:
    """Fetch a full message; return None if it has since vanished (404)."""
    try:
        return service.users().messages().get(userId="me", id=message_id, format="full").execute()
    except Exception as exc:  # pragma: no cover - network/permission dependent
        logger.warning("Gmail: failed to fetch message %s: %s", message_id, exc)
        return None


def _mailbox_history_id(service: Any) -> Optional[str]:
    """Return the mailbox-wide ``historyId`` used as the incremental baseline."""
    try:
        profile = service.users().getProfile(userId="me").execute()
        history_id = profile.get("historyId")
        return str(history_id) if history_id is not None else None
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("Gmail: failed to read profile historyId: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Sync strategies (pure given a service + state dict — unit-testable)
# ---------------------------------------------------------------------------
def full_backfill(
    service: Any,
    state: dict,
    *,
    label_ids: Optional[List[str]] = None,
    max_results: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield every matching message and record the incremental baseline.

    The mailbox ``historyId`` is captured *before* listing so no change is
    missed in the window between backfill start and finish; it is written to
    ``state['last_history_id']`` for the next incremental run.
    """
    baseline_history_id = _mailbox_history_id(service)

    count = 0
    for message_id in _list_message_ids(service, label_ids, max_results):
        message = _get_message(service, message_id)
        if message is None:
            continue
        count += 1
        yield parse_message(message)

    if baseline_history_id is not None:
        state["last_history_id"] = baseline_history_id
    logger.info("Gmail: full backfill yielded %d message(s).", count)


def incremental_fetch(
    service: Any,
    state: dict,
    *,
    label_ids: Optional[List[str]] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield only changes since ``state['last_history_id']`` via the History API.

    Added / changed messages are fetched and emitted normally; deleted (or
    trashed) messages are emitted as hard-delete markers.  ``last_history_id``
    is advanced to the newest history id returned so the next run is a no-op if
    nothing changed.

    If the stored history id is too old (Gmail expires history after ~a week),
    the API raises 404; we fall back to a full backfill so memory re-syncs
    rather than silently stalling.
    """
    start_history_id = state.get("last_history_id")
    if not start_history_id:
        # No cursor yet — caller should have backfilled. Be defensive.
        yield from full_backfill(service, state, label_ids=label_ids)
        return

    page_token = None
    newest_history_id = start_history_id
    seen_added: set = set()
    seen_deleted: set = set()

    while True:
        try:
            response = (
                service.users()
                .history()
                .list(
                    userId="me",
                    startHistoryId=start_history_id,
                    historyTypes=_HISTORY_TYPES,
                    labelId=(label_ids[0] if label_ids else None),
                    pageToken=page_token,
                )
                .execute()
            )
        except Exception as exc:
            # A 404 means the cursor expired — recover with a full backfill.
            status = getattr(getattr(exc, "resp", None), "status", None)
            if status == 404 or "404" in str(exc):
                logger.warning(
                    "Gmail: history id %s expired; falling back to full backfill.",
                    start_history_id,
                )
                state.pop("last_history_id", None)
                yield from full_backfill(service, state, label_ids=label_ids)
                return
            raise

        for record in response.get("history", []) or []:
            record_history_id = record.get("id")
            if record_history_id and str(record_history_id) > str(newest_history_id):
                newest_history_id = str(record_history_id)

            for deleted in record.get("messagesDeleted", []) or []:
                msg_id = (deleted.get("message") or {}).get("id")
                if msg_id and msg_id not in seen_deleted:
                    seen_deleted.add(msg_id)

            # messagesAdded plus label changes both mean "(re)fetch this message".
            for key in ("messagesAdded", "labelsAdded", "labelsRemoved"):
                for change in record.get(key, []) or []:
                    msg_id = (change.get("message") or {}).get("id")
                    if msg_id and msg_id not in seen_added:
                        seen_added.add(msg_id)

        page_token = response.get("nextPageToken")
        if response.get("historyId"):
            candidate = str(response["historyId"])
            if candidate > str(newest_history_id):
                newest_history_id = candidate
        if not page_token:
            break

    # A message that was added and then deleted within the same delta window is
    # a net deletion — don't bother fetching it.
    seen_added -= seen_deleted

    added_count = 0
    for msg_id in seen_added:
        message = _get_message(service, msg_id)
        if message is None:
            # Disappeared between history and fetch — treat as a deletion.
            seen_deleted.add(msg_id)
            continue
        added_count += 1
        yield parse_message(message)

    for msg_id in seen_deleted:
        yield _deleted_row(msg_id)

    state["last_history_id"] = str(newest_history_id)
    logger.info(
        "Gmail: incremental sync yielded %d added/changed and %d deleted message(s).",
        added_count,
        len(seen_deleted),
    )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------
def gmail_source(
    *,
    credentials_path: str = "credentials.json",
    token_path: str = "token.json",
    label_ids: Optional[List[str]] = None,
    max_results: Optional[int] = None,
    service: Any = None,
):
    """Return a ``dlt`` resource that yields Gmail messages for ``remember``.

    Args:
        credentials_path: Path to the OAuth client-secret JSON (Desktop app).
        token_path: Where the cached user token is read/written.
        label_ids: Restrict to these Gmail label ids (e.g. ``["INBOX"]``).
        max_results: Cap the number of messages pulled in a full backfill
            (handy for demos/tests). ``None`` = no cap.
        service: Pre-built Gmail API client. Mainly an injection point for
            tests; when omitted an OAuth client is built from the paths above.

    Returns:
        A ``dlt`` resource (``gmail_messages``) configured with
        ``primary_key="id"``, ``write_disposition="merge"`` and an ``_deleted``
        hard-delete column. Hand it to ``cognee.remember(...)``.
    """
    try:
        import dlt
    except ImportError as exc:
        raise ImportError(
            'The Gmail connector requires the dlt extra: pip install "cognee[dlt]".'
        ) from exc

    @dlt.resource(
        name="gmail_messages",
        primary_key="id",
        write_disposition="merge",
        # _deleted is a boolean hard-delete marker: rows where it is True are
        # removed from the dlt destination on merge, which propagates the
        # deletion through cognee's orphan_cleanup.
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def gmail_messages():
        client = service or build_gmail_service(credentials_path, token_path)
        resource_state = dlt.current.resource_state()

        if resource_state.get("last_history_id"):
            yield from incremental_fetch(client, resource_state, label_ids=label_ids)
        else:
            yield from full_backfill(
                client,
                resource_state,
                label_ids=label_ids,
                max_results=max_results,
            )

    return gmail_messages
