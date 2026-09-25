# Adapted from topoteretes/cognee-community (Apache-2.0), commit 21d0b36.
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
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
        write_disposition="merge",   # REQUIRED (see .. important:: below)
        max_rows_per_table=0,        # REQUIRED for a real inbox (see .. note:: below)
    )

.. important::
   ``write_disposition="merge"`` is **mandatory**.  The add pipeline defaults to
   ``"replace"`` (drop + reload the table each run); on the second, incremental
   sync that would wipe the entire synced inbox.  Always pass ``"merge"``.

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
* **Quota** — every API call is paced against Gmail's per-user quota
  (6,000 units/minute; fetching one message costs 20 units) and rate-limit
  errors are retried with Google's recommended exponential backoff. A full
  backfill is therefore bounded at roughly 250 messages/minute by default.
* **Forget-on-delete** — messages reported as deleted/trashed by the History
  API are emitted with the ``_deleted`` hard-delete marker.  dlt removes those
  rows from its destination on ``merge``; they then fall out of the freshly
  read row set and cognee's existing ``orphan_cleanup`` purges them from the
  graph + vector + relational stores.  A full backfill reconciles the same way.

.. note::
   cognee's ``ingest_dlt_source`` reads at most ``max_rows_per_table`` rows
   from the dlt destination (default ``0``, unlimited, unless
   ``DLT_MAX_ROWS_PER_TABLE`` is set).  Keep it unlimited for a real inbox so
   orphan-cleanup compares against the *whole* synced corpus rather than a
   truncated window.

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
import json
import os
import time
from collections.abc import Callable, Iterator
from typing import Any

from limits import RateLimitItemPerMinute
from limits.storage import MemoryStorage
from limits.strategies import MovingWindowRateLimiter
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR

logger = get_logger("gmail_connector")

# Read-only access — the connector never modifies the mailbox.
GMAIL_READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"

# History event types we care about for incremental sync.
_HISTORY_TYPES = ["messageAdded", "messageDeleted", "labelAdded", "labelRemoved"]

# Gmail allows 6,000 quota units per user per minute; each method has a fixed
# cost. https://developers.google.com/workspace/gmail/api/reference/quota
GMAIL_USER_QUOTA_UNITS_PER_MINUTE = 6_000
# Stay below the hard limit so other apps on the same account keep working.
DEFAULT_QUOTA_UNITS_PER_MINUTE = 5_000
_COST_MESSAGES_GET = 20
_COST_MESSAGES_LIST = 5
_COST_HISTORY_LIST = 2
_COST_GET_PROFILE = 1

# Exponential backoff per Google's guidance: start at >= 1s, double each
# retry with up to 1s of random jitter, cap at 64s.
# https://developers.google.com/workspace/gmail/api/guides/handle-errors
_MAX_ATTEMPTS = 8
_MAX_BACKOFF_SECONDS = 64.0
_RATE_LIMIT_REASONS = {"rateLimitExceeded", "userRateLimitExceeded"}


# ---------------------------------------------------------------------------
# Quota pacing and retries
# ---------------------------------------------------------------------------
def _error_reason(exc: Exception) -> str | None:
    """Return the Google error ``reason`` (e.g. ``rateLimitExceeded``), if any."""
    content = getattr(exc, "content", None)
    if not content:
        return None
    try:
        data = json.loads(content.decode("utf-8") if isinstance(content, bytes) else content)
        errors = data["error"].get("errors") or []
        return errors[0].get("reason") if errors else None
    except (ValueError, KeyError, TypeError, AttributeError, IndexError):
        return None


def _is_retryable(exc: Exception) -> bool:
    """Rate limits, server errors and dropped connections are worth retrying.

    Anything else (auth, bad request, 404/410) is raised immediately: a caller
    treats 404/410 as "message gone", so it must never be delayed or masked.
    """
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    status = getattr(getattr(exc, "resp", None), "status", None)
    if status == 429 or (isinstance(status, int) and status >= 500):
        return True
    return status == 403 and _error_reason(exc) in _RATE_LIMIT_REASONS


class GmailQuota:
    """Paces Gmail API calls to a per-user quota budget and retries rate limits.

    Pacing uses a ``limits`` moving window, so no 60-second window spends more
    than ``units_per_minute``; retries use ``tenacity`` with Google's backoff.
    One instance should be shared by every call made for the same mailbox.
    """

    def __init__(
        self,
        units_per_minute: int = DEFAULT_QUOTA_UNITS_PER_MINUTE,
        *,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if units_per_minute < _COST_MESSAGES_GET:
            raise ValueError(
                f"units_per_minute must be at least {_COST_MESSAGES_GET} "
                "(the cost of fetching one message)."
            )
        self._item = RateLimitItemPerMinute(units_per_minute)
        self._limiter = MovingWindowRateLimiter(MemoryStorage())
        self._clock = clock
        self._sleep = sleep

    def acquire(self, cost: int) -> None:
        """Block until ``cost`` units fit in the current window, then spend them."""
        while not self._limiter.hit(self._item, "gmail", cost=cost):
            reset_at, _ = self._limiter.get_window_stats(self._item, "gmail")
            # The window frees up one entry at a time, so this may loop a few
            # times before ``cost`` units are available.
            self._sleep(max(reset_at - self._clock(), 0.05))

    def execute(self, request: Any, cost: int) -> Any:
        """Run a googleapiclient request within the quota, retrying rate limits."""

        def _log_retry(retry_state) -> None:
            logger.warning(
                "Gmail API call failed (%s); retry %d of %d in %.1fs.",
                retry_state.outcome.exception(),
                retry_state.attempt_number,
                _MAX_ATTEMPTS - 1,
                retry_state.next_action.sleep,
            )

        def _attempt() -> Any:
            self.acquire(cost)
            return request.execute()

        retrying = Retrying(
            retry=retry_if_exception(_is_retryable),
            wait=wait_exponential_jitter(initial=1, max=_MAX_BACKOFF_SECONDS),
            stop=stop_after_attempt(_MAX_ATTEMPTS),
            sleep=self._sleep,
            before_sleep=_log_retry,
            reraise=True,
        )
        return retrying(_attempt)


def _execute(request: Any, cost: int, quota: GmailQuota | None) -> Any:
    """Execute through ``quota`` when given; a bare call otherwise (tests)."""
    return quota.execute(request, cost) if quota is not None else request.execute()


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
        # token.json holds the long-lived OAuth refresh token — keep it private.
        # Create it 0600 (no group/world read) rather than the default 0644.
        fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as token_file:
            token_file.write(creds.to_json())
        os.chmod(token_path, 0o600)  # tighten a pre-existing token file too

    return build("gmail", "v1", credentials=creds, cache_discovery=False)


def build_gmail_service_from_access_token(access_token: str) -> Any:
    """Build a Gmail client from a short-lived token supplied by a host."""
    try:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError('The Gmail connector requires the "gmail" extra.') from exc
    credentials = Credentials(token=access_token, scopes=[GMAIL_READONLY_SCOPE])
    return build("gmail", "v1", credentials=credentials, cache_discovery=False)


# ---------------------------------------------------------------------------
# Message parsing
# ---------------------------------------------------------------------------
def _decode_body(data: str | None) -> str:
    """Decode a base64url-encoded Gmail body part into text."""
    if not data:
        return ""
    try:
        return base64.urlsafe_b64decode(data.encode("utf-8")).decode("utf-8", errors="replace")
    except (ValueError, UnicodeError):  # pragma: no cover - defensive
        return ""


def _extract_plaintext(payload: dict | None) -> str:
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


def _headers_to_dict(payload: dict | None) -> dict[str, str]:
    """Index a message's headers case-insensitively by name."""
    headers = {}
    for header in (payload or {}).get("headers", []) or []:
        name = header.get("name", "").lower()
        if name:
            headers[name] = header.get("value", "")
    return headers


def _document_content(headers: dict[str, str], body: str, snippet: str) -> str:
    """Render the text cognify sees for a message: header lines, then the body.

    Document-source rows are turned into text from their ``title``/``content``
    columns only (see ``resolve_dlt_sources._build_document_data_item``), so the
    sender, recipients and date must be folded into ``content`` to reach the
    graph. HTML-only messages have no plain-text body and fall back to the
    snippet.
    """
    lines = [
        f"{label}: {headers[key]}"
        for key, label in (("from", "From"), ("to", "To"), ("cc", "Cc"), ("date", "Date"))
        if headers.get(key)
    ]
    text = body.strip() or snippet.strip()
    if text:
        lines.extend(["", text])
    return "\n".join(lines).strip()


def parse_message(message: dict) -> dict[str, Any]:
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

    body = _extract_plaintext(payload)
    snippet = message.get("snippet", "")

    return {
        "id": message.get("id"),
        "thread_id": message.get("threadId"),
        # title/content are what the document ingestion path turns into text.
        "title": headers.get("subject", ""),
        "content": _document_content(headers, body, snippet),
        "labels": ", ".join(label_ids),
        "subject": headers.get("subject", ""),
        "from": headers.get("from", ""),
        "to": headers.get("to", ""),
        "cc": headers.get("cc", ""),
        "date": headers.get("date", ""),
        "snippet": snippet,
        "body": body,
        "internal_date": internal_date,
        # Hard-delete marker (always False for live messages). Deleted/trashed
        # messages are emitted separately with _deleted=True.
        "_deleted": False,
    }


def _deleted_row(message_id: str) -> dict[str, Any]:
    """Build a minimal row that instructs dlt to hard-delete a message by id."""
    return {"id": message_id, "_deleted": True}


# ---------------------------------------------------------------------------
# Gmail API helpers (paginated)
# ---------------------------------------------------------------------------
def _list_message_ids(
    service: Any,
    label_ids: list[str] | None,
    max_results: int | None,
    quota: GmailQuota | None = None,
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
        response = _execute(request, _COST_MESSAGES_LIST, quota)
        for ref in response.get("messages", []) or []:
            yield ref["id"]
            fetched += 1
            if max_results and fetched >= max_results:
                return
        page_token = response.get("nextPageToken")
        if not page_token:
            return


def _get_message(
    service: Any,
    message_id: str,
    stats: dict[str, int] | None = None,
    quota: GmailQuota | None = None,
) -> dict | None:
    """Fetch a full message; return None only if it is genuinely gone (404/410).

    A transient failure (5xx / rate-limit / network) is re-raised rather than
    swallowed. On the incremental path a ``None`` result is interpreted as a
    deletion, so masking a transient error here would hard-delete a live
    message from memory. Re-raising instead aborts the sync before the cursor
    advances, so the next run safely retries from the same point.
    """
    if stats is None:
        stats = {}
    stats["scanned"] = stats.get("scanned", 0) + 1
    try:
        return _execute(
            service.users().messages().get(userId="me", id=message_id, format="full"),
            _COST_MESSAGES_GET,
            quota,
        )
    except Exception as exc:
        # Trust only the structured HTTP status: str(exc) embeds the request
        # URL, and a hex message id can spuriously contain "404"/"410".
        if getattr(getattr(exc, "resp", None), "status", None) in (404, 410):
            stats["skipped"] = stats.get("skipped", 0) + 1
            stats["skipped_unavailable"] = stats.get("skipped_unavailable", 0) + 1
            return None
        stats["failed"] = stats.get("failed", 0) + 1
        stats["failed_message_fetch"] = stats.get("failed_message_fetch", 0) + 1
        raise


def _mailbox_history_id(service: Any, quota: GmailQuota | None = None) -> str | None:
    """Return the mailbox-wide ``historyId`` used as the incremental baseline."""
    try:
        profile = _execute(service.users().getProfile(userId="me"), _COST_GET_PROFILE, quota)
        history_id = profile.get("historyId")
        return str(history_id) if history_id is not None else None
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("Failed to read profile historyId: %s", exc, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Sync strategies (pure given a service + state dict — unit-testable)
# ---------------------------------------------------------------------------
def full_backfill(
    service: Any,
    state: dict,
    *,
    label_ids: list[str] | None = None,
    max_results: int | None = None,
    stats: dict[str, int] | None = None,
    quota: GmailQuota | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield every matching message and record the incremental baseline.

    The mailbox ``historyId`` is captured *before* listing so no change is
    missed in the window between backfill start and finish; it is written to
    ``state['last_history_id']`` for the next incremental run.
    """
    baseline_history_id = _mailbox_history_id(service, quota)

    count = 0
    known_ids = set(state.get("known_ids", []))
    present_ids = set()
    for message_id in _list_message_ids(service, label_ids, max_results, quota):
        message = _get_message(service, message_id, stats, quota)
        if message is None:
            continue
        count += 1
        present_ids.add(message_id)
        yield parse_message(message)

    # A capped or failed scan is not evidence of absence. Only reconcile after
    # exhausting a complete listing; exceptions never reach this checkpoint.
    if max_results is None:
        for message_id in sorted(known_ids - present_ids):
            if stats is not None:
                stats["deleted"] = stats.get("deleted", 0) + 1
            yield _deleted_row(message_id)
        state["known_ids"] = sorted(present_ids)
    else:
        state["known_ids"] = sorted(known_ids | present_ids)
        state.pop("last_history_id", None)
    if baseline_history_id is not None and max_results is None:
        state["last_history_id"] = baseline_history_id
    logger.info("Full backfill yielded %d message(s).", count)


def incremental_fetch(
    service: Any,
    state: dict,
    *,
    label_ids: list[str] | None = None,
    stats: dict[str, int] | None = None,
    quota: GmailQuota | None = None,
) -> Iterator[dict[str, Any]]:
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
        yield from full_backfill(service, state, label_ids=label_ids, stats=stats, quota=quota)
        return

    page_token = None
    newest_history_id = start_history_id
    seen_added: set = set()
    seen_deleted: set = set()

    while True:
        try:
            response = _execute(
                service.users()
                .history()
                .list(
                    userId="me",
                    startHistoryId=start_history_id,
                    historyTypes=_HISTORY_TYPES,
                    labelId=(label_ids[0] if label_ids else None),
                    pageToken=page_token,
                ),
                _COST_HISTORY_LIST,
                quota,
            )
        except Exception as exc:
            # A 404 means the cursor expired — recover with a full backfill.
            status = getattr(getattr(exc, "resp", None), "status", None)
            if status == 404 or "404" in str(exc):
                logger.warning(
                    "History id %s expired; falling back to full backfill.",
                    start_history_id,
                )
                state.pop("last_history_id", None)
                yield from full_backfill(
                    service, state, label_ids=label_ids, stats=stats, quota=quota
                )
                return
            raise

        for record in response.get("history", []) or []:
            record_history_id = record.get("id")
            # historyIds are monotonically increasing integers — compare
            # numerically, not lexicographically ("1000" < "999" as strings).
            if record_history_id and int(record_history_id) > int(newest_history_id):
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
        if response.get("historyId") and int(response["historyId"]) > int(newest_history_id):
            newest_history_id = str(response["historyId"])
        if not page_token:
            break

    # A message that was added and then deleted within the same delta window is
    # a net deletion — don't bother fetching it.
    seen_added -= seen_deleted

    added_count = 0
    known_ids = set(state.get("known_ids", []))
    for msg_id in seen_added:
        message = _get_message(service, msg_id, stats, quota)
        if message is None:
            # Genuinely gone (404/410) — treat as a deletion.
            seen_deleted.add(msg_id)
            continue
        # A message that moved out of scope (e.g. trashed) must be forgotten,
        # not re-ingested as live: trashing INBOX mail fires labelsRemoved(INBOX)
        # (not messagesDeleted), so without this check it would be fetched and
        # upserted. Re-check the message against the SAME scope full_backfill
        # applies — history.list can only pre-filter by a single labelId, so the
        # authoritative decision is made here after the full label set is known.
        labels = set(message.get("labelIds", []) or [])
        if label_ids:
            # full_backfill lists with labelIds=label_ids, which ANDs them.
            in_scope = set(label_ids).issubset(labels)
        else:
            # Unscoped: mirror messages.list, which excludes SPAM/TRASH.
            in_scope = "TRASH" not in labels and "SPAM" not in labels
        if not in_scope:
            seen_deleted.add(msg_id)
            continue
        added_count += 1
        known_ids.add(msg_id)
        yield parse_message(message)

    for msg_id in seen_deleted:
        if stats is not None:
            stats["deleted"] = stats.get("deleted", 0) + 1
        yield _deleted_row(msg_id)

    state["known_ids"] = sorted(known_ids - seen_deleted)
    state["last_history_id"] = str(newest_history_id)
    logger.info(
        "Incremental sync yielded %d added/changed and %d deleted message(s).",
        added_count,
        len(seen_deleted),
    )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------
def gmail_source(
    *,
    resource_name: str = "gmail_messages",
    check_active: Callable[[], None] | None = None,
    credentials_path: str = "credentials.json",
    token_path: str = "token.json",
    label_ids: list[str] | None = None,
    max_results: int | None = None,
    service: Any = None,
    quota_units_per_minute: int = DEFAULT_QUOTA_UNITS_PER_MINUTE,
):
    """Return a ``dlt`` resource that yields Gmail messages for ``remember``.

    Args:
        resource_name: Stable, account-specific name when sharing a dlt pipeline.
        check_active: Optional host authorization checkpoint during extraction.
        credentials_path: Path to the OAuth client-secret JSON (Desktop app).
        token_path: Where the cached user token is read/written.
        label_ids: Restrict to these Gmail label ids (e.g. ``["INBOX"]``).
        max_results: Cap the number of messages pulled in a full backfill
            (handy for demos/tests). ``None`` = no cap.
        service: Pre-built Gmail API client. Mainly an injection point for
            tests; when omitted an OAuth client is built from the paths above.
        quota_units_per_minute: Gmail API quota budget for this source.
            Gmail allows 6,000 units per user per minute and fetching one
            message costs 20, so the default of 5,000 fetches about 250
            messages a minute. Lower it if other apps share the account.

    Returns:
        A ``dlt`` resource (``gmail_messages``) configured with
        ``primary_key="id"``, ``write_disposition="merge"`` and an ``_deleted``
        hard-delete column. Hand it to ``cognee.remember(...)``.
    """
    try:
        import dlt
    except ImportError as exc:
        raise ImportError(
            "The Gmail connector requires the 'gmail' extra. Install it with:\n"
            '    pip install "cognee[gmail]"\n'
            "(the gmail extra bundles dlt)."
        ) from exc

    if getattr(dlt_utils, "DOCUMENT_SYNC_VERSION", 0) < 1:
        raise RuntimeError(
            "Gmail sync requires a Cognee build with table-scoped DLT document cleanup. "
            "Upgrade Cognee so deleting the final message also removes its stored content."
        )

    stats: dict[str, int] = {}
    # One budget per source (i.e. per mailbox), shared by every call in a sync.
    quota = GmailQuota(quota_units_per_minute)

    @dlt.resource(
        name=resource_name,
        primary_key="id",
        write_disposition="merge",
        # _deleted is a boolean hard-delete marker: rows where it is True are
        # removed from the dlt destination on merge, which propagates the
        # deletion through cognee's orphan_cleanup.
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def gmail_messages():
        stats.clear()
        stats.update(scanned=0, skipped=0, failed=0, deleted=0)
        client = service or build_gmail_service(credentials_path, token_path)
        resource_state = dlt.current.resource_state()
        # A history cursor describes changes after a point in time, not the
        # contents of a newly selected label. Backfill when the selection
        # changes so older messages entering the scope are not lost. Sorting
        # avoids a backfill when the picker only reorders the same labels.
        scope = sorted(set(label_ids or []))
        same_scope = resource_state.get("label_scope") == scope

        if same_scope and resource_state.get("last_history_id"):
            rows = incremental_fetch(
                client, resource_state, label_ids=label_ids, stats=stats, quota=quota
            )
        else:
            rows = full_backfill(
                client,
                resource_state,
                label_ids=label_ids,
                max_results=max_results,
                stats=stats,
                quota=quota,
            )
        yield from dlt_utils.guarded_rows(rows, check_active)
        # Do not persist the new scope if extraction fails midway through.
        resource_state["label_scope"] = scope

    resource = gmail_messages()
    # Gmail rows are prose documents, not a relational manifest. Without this
    # marker Cognee would ingest the DLT table as one structured object and
    # never run normal document cognification or orphan cleanup.
    setattr(resource, DOCUMENT_SOURCE_ATTR, "gmail")
    setattr(resource, dlt_utils.PIPELINE_SCOPE_ATTR, resource_name)
    resource.cognee_sync_stats = stats
    return resource
