# Adapted from topoteretes/cloud-backend, CLO-504, cognee-saas-pod/modules/integrations/sources/granola.py at aa05be08.
# Modified to ship in the Cognee SDK: SDK run bounds, check_active per request, include_transcript, a daily reconcile.
"""Granola connector for cognee: a ``dlt`` source that turns one Granola scope into memory.

``granola_source(scope=..., api_key=...)`` reads the notes a Granola API key can see in one
scope (``all``, a folder ``fol_...`` or a space ``spa_...``) through Granola's public REST API
and yields one document per note (title, meeting date, summary and, by default, the
transcript), row id = the note id. Hand it to ``cognee.remember`` with
``write_disposition="merge"``. Sync model, deletion, limits and privacy: ``granola.md`` next to
this file.

Private scratch notes and attachments are never read into a row: the render is a whitelist.
"""

import bisect
import json
import math
import re
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any

from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR, with_retries

logger = get_logger("granola_connector")

API_BASE = "https://public-api.granola.ai"
_RESOURCE = re.compile(r"^(fol|spa)_[a-zA-Z0-9]{14}$", re.IGNORECASE)
_NOTE = re.compile(r"^not_[a-zA-Z0-9]{14}$", re.IGNORECASE)
# Every path the client may request; all are built here, never from a provider URL or cursor.
_PATH = re.compile(r"/v1/notes(?:/not_[a-zA-Z0-9]{14}(?:/transcript)?)?", re.IGNORECASE)
# The bounds below are the Cloud pod's (CLO-504, shipped in v1.6.8-001); changing one is a
# decision, not a tuning.
MAX_BODY = 8 * 1024 * 1024
MAX_TRANSCRIPT_CHARS = 2_000_000
MAX_NOTE_RETRIES = 3
MAX_FAILED_NOTES = 500
# Leave room for listing requests and other users of the same rate pool: Granola documents
# 5 requests a second sustained (burst 25 in 5 s) per user or workspace.
MIN_REQUEST_INTERVAL = 0.25
MAX_RETRY_AFTER_SECONDS = 7 * 24 * 60 * 60
# 5xx/timeouts in a row after which a cycle's retries by id stop for this run.
MAX_UNAVAILABLE_RETRIES = 3
DEFAULT_MAX_REQUESTS = 200
DEFAULT_DEADLINE_SECONDS = 240.0
DEFAULT_MAX_ROWS = 200
# Granola's page size: the incremental listing keeps the pod's 10 (its pending pages are kept
# in state), the reconcile listing reads ids only and takes the documented maximum, 30.
LIST_PAGE_SIZE = 10
RECONCILE_PAGE_SIZE = 30
TRANSCRIPT_PAGE_SIZE = 100
# A complete listing of the scope at most once a day (the blueprint's D3). It only delays a
# deletion; nothing is dropped by it.
RECONCILE_INTERVAL = timedelta(hours=24)
# The watermark trails the cycle's start, so an edit the local clock is ahead of is read again.
WATERMARK_OVERLAP = timedelta(minutes=5)
_CONNECT_TIMEOUT = 10
_READ_TIMEOUT = 30
# Positions of the incremental walk; a changed ``include_transcript`` starts them over, so
# every note is rendered again.
_LISTING_KEYS = ("watermark", "cursor", "pending", "next_cursor", "cycle_start")
# Positions of the walk and of the reconcile; a replaced key (``key_revision``) starts them
# over. ``known`` and ``reconciled_at`` stay, and unconfirmed candidates go back to ``known``.
_WALK_KEYS = (
    "watermark",
    "cursor",
    "pending",
    "next_cursor",
    "cycle_start",
    "note_failures",
    "retry_notes",
    "reconcile_cursor",
    "reconcile_seen",
    "reconcile_candidates",
)


class GranolaError(RuntimeError):
    """A Granola request or answer that failed. Messages carry an HTTP status at most, never
    the key, a note id, a URL or a body."""

    def __init__(self, message: str, status: int | None = None, *, retry_after: int = 0):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


class GranolaNoteError(GranolaError):
    """A permanent-looking error confined to one note, not the connection."""


class GranolaBudgetError(GranolaError):
    """The run's request budget or deadline is spent."""


def retry_after_seconds(value: str | None, *, now: datetime | None = None) -> int:
    """``Retry-After`` as whole seconds (seconds or an HTTP date), 0 when absent or unreadable,
    at most a week."""
    if not isinstance(value, str):
        return 0
    try:
        seconds = float(value)
    except ValueError:
        try:
            deadline = parsedate_to_datetime(value)
            seconds = (deadline - (now or datetime.now(timezone.utc))).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return 0
    if not math.isfinite(seconds):
        return 0
    return min(MAX_RETRY_AFTER_SECONDS, max(0, math.ceil(seconds)))


def validate_scope(scope: Any) -> None:
    if not isinstance(scope, str) or (scope != "all" and not _RESOURCE.fullmatch(scope)):
        raise ValueError("Invalid Granola folder or space ID")


class _BearerAuth:
    """``Authorization: Bearer`` as a requests auth object.

    A header passed in ``headers`` is replaced by a ``~/.netrc`` entry for the host, because
    requests consults netrc when no ``auth`` is given; an auth object turns that off.
    """

    def __init__(self, api_key: str):
        self._api_key = api_key

    def __call__(self, request: Any) -> Any:
        request.headers["Authorization"] = f"Bearer {self._api_key}"
        return request

    def __repr__(self) -> str:
        return "_BearerAuth(<redacted>)"


class GranolaClient:
    """Minimal synchronous client for Granola's public API: a fixed origin, no redirects, a
    capped streamed body and paced requests.

    A ``service`` handed to :func:`granola_source` instead must offer the same ``get`` and call
    ``check_active`` before each attempt it makes.
    """

    def __init__(
        self,
        api_key: str,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        # requests names the value of a header it rejects, so every key it would reject is
        # refused here, without echoing it.
        key = api_key if isinstance(api_key, str) else ""
        if not key or not key.isascii() or not key.isprintable() or any(c.isspace() for c in key):
            raise ValueError("The Granola API key is empty or has an invalid format")
        self._api_key = key
        self._sleep = sleep
        self._clock = clock
        self._last_request_at: float | None = None

    def __repr__(self) -> str:
        return "GranolaClient(api_key=<redacted>)"

    def _pace(self) -> None:
        """Keep below Granola's sustained five requests a second.

        Only the remainder of the interval since the previous request is slept, so a run that
        is already slower never sleeps under the sync locks.
        """
        now = self._clock()
        if self._last_request_at is not None:
            wait = MIN_REQUEST_INTERVAL - (now - self._last_request_at)
            if wait > 0:
                self._sleep(wait)
                now = self._clock()
        self._last_request_at = now

    def get(
        self,
        path: str,
        params: dict | None = None,
        *,
        check_active: Callable[[], None] | None = None,
        time_left: Callable[[], float] | None = None,
    ) -> dict:
        """GET ``path`` and return its JSON object.

        A 5xx or no answer is tried once more at once, unless ``time_left`` says the run's
        deadline has passed. ``check_active`` runs before every attempt, and what it raises
        propagates as it is.
        """
        import requests  # lazy: importing the connectors package loads no client library

        if not isinstance(path, str) or not _PATH.fullmatch(path):
            raise ValueError("Invalid Granola API path")
        auth = _BearerAuth(self._api_key)

        def remaining() -> float:
            return math.inf if time_left is None else time_left()

        def attempt() -> Any:
            self._pace()  # a retried attempt is paced too
            try:
                response = requests.get(
                    API_BASE + path,
                    params=params or {},
                    auth=auth,
                    allow_redirects=False,  # a 3xx is an error answer, never a new target
                    timeout=(_CONNECT_TIMEOUT, min(_READ_TIMEOUT, max(1, remaining()))),
                    stream=True,
                )
            except requests.RequestException as error:
                # Only the class: a requests error quotes the URL. Raised outside this block,
                # so the provider's error is no one's __context__.
                failure = type(error).__name__
            else:
                if response.status_code < 500:
                    return response
                failure = f"HTTP {response.status_code}"
                status = response.status_code
                response.close()
                raise GranolaError(f"Granola request failed: {failure}", status)
            raise GranolaError(f"Granola request failed: {failure}")

        if check_active is not None:
            check_active()
        response = with_retries(
            attempt,
            retry_on=(GranolaError,),
            delay=lambda _error, _number: 0 if remaining() > 0 else None,
            attempts=2,
            check_active=check_active,
            sleep=self._sleep,
        )
        return self._read(response)

    @staticmethod
    def _read(response: Any) -> dict:
        try:
            if response.status_code != 200:
                raise GranolaError(
                    f"Granola request failed: HTTP {response.status_code}",
                    response.status_code,
                    retry_after=retry_after_seconds(response.headers.get("Retry-After"))
                    if response.status_code == 429
                    else 0,
                )
            body = bytearray()
            for chunk in response.iter_content(65536):
                body.extend(chunk)
                if len(body) > MAX_BODY:
                    raise GranolaNoteError("Granola response exceeded the size limit")
            data = json.loads(body)
            if not isinstance(data, dict):
                raise ValueError
            return data
        except (ValueError, RecursionError):  # UnicodeDecodeError is a ValueError
            raise GranolaNoteError("Invalid Granola response") from None
        except OSError:  # requests' RequestException is an OSError
            raise GranolaError("Could not read Granola response") from None
        finally:
            response.close()


def build_granola_service(api_key: str) -> GranolaClient:
    """The client the integration layer injects (``_PROVIDERS`` service builder).

    ``api_key`` is a Granola personal or workspace API key (``grn_...``), sent as
    ``Authorization: Bearer``. Granola has no OAuth for its REST API.
    """
    return GranolaClient(api_key)


class _RunBounds:
    """The request budget and the deadline of one run, in front of the client.

    Kept here rather than in the client so a host-built ``service`` is bounded too. A request
    the client retries counts once, in ``stats["requests"]``.
    """

    def __init__(
        self,
        client: Any,
        stats: dict[str, int],
        *,
        max_requests: int,
        deadline_seconds: float,
        check_active: Callable[[], None] | None,
        clock: Callable[[], float],
    ):
        self._client = client
        self._stats = stats
        self._max_requests = max_requests
        self._deadline_seconds = deadline_seconds
        self._check_active = check_active
        self._clock = clock
        self._started = clock()

    def time_left(self) -> float:
        return self._deadline_seconds - (self._clock() - self._started)

    def get(self, path: str, params: dict | None = None) -> dict:
        if self.time_left() <= 0:
            logger.warning(
                "Granola run deadline (%.0f s) passed; stopping before the next request.",
                self._deadline_seconds,
            )
            raise GranolaBudgetError("Granola run deadline passed")
        if self._stats["requests"] >= self._max_requests:
            raise GranolaBudgetError("Granola request budget exhausted")
        self._stats["requests"] += 1
        return self._client.get(
            path, params, check_active=self._check_active, time_left=self.time_left
        )


def _page(page: dict, collection: str) -> tuple[list[dict], str | None]:
    entries = page.get(collection)
    if not isinstance(entries, list) or not all(isinstance(item, dict) for item in entries):
        raise GranolaNoteError("Invalid Granola page")
    more = page.get("hasMore")
    cursor = page.get("cursor")
    if more is False:
        return entries, None
    if more is not True or not isinstance(cursor, str) or not cursor or len(cursor) > 8192:
        raise GranolaNoteError("Invalid Granola cursor")
    return entries, cursor


def _transcript_size(transcript: Any) -> int:
    if transcript is None:
        return 0
    if not isinstance(transcript, list):
        raise GranolaNoteError("Invalid Granola transcript")
    total = 0
    for part in transcript:
        if not isinstance(part, dict) or not isinstance(part.get("text"), str):
            raise GranolaNoteError("Invalid Granola transcript")
        if part.get("speaker") is not None and not isinstance(part["speaker"], dict):
            raise GranolaNoteError("Invalid Granola speaker")
        total += len(part["text"])
        if total > MAX_TRANSCRIPT_CHARS:
            raise GranolaNoteError("Granola transcript exceeded the size limit")
    return total


def _get_note(api: Any, note_id: str, include_transcript: bool = True) -> dict:
    if not include_transcript:
        # Never ``include=transcript`` and never the transcript endpoint; a transcript the
        # answer carries anyway is not rendered.
        note = api.get(f"/v1/notes/{note_id}")
        note.pop("transcript", None)
        return note
    try:
        note = api.get(f"/v1/notes/{note_id}", {"include": "transcript"})
        _transcript_size(note.get("transcript"))
        return note
    except GranolaError as error:
        # 413: the transcript is too large to inline. 403: transcripts may be disabled by the
        # workspace admin; the bare note below still carries the summary, and a 403 on that
        # note itself stays this note's error.
        if error.status not in {403, 413}:
            raise
    # Large transcripts use the documented paginated endpoint. Never publish a truncated
    # meeting as complete: an interrupted transcript retries this note.
    note = api.get(f"/v1/notes/{note_id}")
    parts: list[dict] = []
    cursor = None
    seen = set()
    size = 0
    while True:
        params: dict[str, Any] = {"page_size": TRANSCRIPT_PAGE_SIZE}
        if cursor:
            params["cursor"] = cursor
        try:
            entries, next_cursor = _page(
                api.get(f"/v1/notes/{note_id}/transcript", params), "transcript"
            )
        except GranolaError as error:
            if error.status == 403:
                note["transcript"] = None  # admin disabled transcripts; summary remains usable
                return note
            raise
        size += _transcript_size(entries)
        if size > MAX_TRANSCRIPT_CHARS:
            raise GranolaNoteError("Granola transcript exceeded the size limit")
        parts.extend(entries)
        if next_cursor is None:
            note["transcript"] = parts
            return note
        if next_cursor in seen:
            raise GranolaNoteError("Granola repeated a transcript cursor")
        seen.add(next_cursor)
        cursor = next_cursor


def _render(note: dict, scope: str) -> dict:
    """The note's row. Frozen: Cloud has synced with this exact text since v1.6.8-001."""
    title = str(note.get("title") or "Untitled meeting")
    lines = [title]
    if note.get("created_at"):
        lines.append(f"Meeting date: {note['created_at']}")
    summary = note.get("summary_markdown") or note.get("summary_text")
    if summary:
        lines.extend(["", str(summary)])
    transcript = note.get("transcript")
    if isinstance(transcript, list) and transcript:
        lines.extend(["", "Transcript:"])
        for part in transcript:
            if not isinstance(part, dict) or not isinstance(part.get("text"), str):
                raise GranolaNoteError("Invalid Granola transcript")
            speaker = part.get("speaker") or {}
            label = (
                speaker.get("diarization_label")
                or speaker.get("attribution")
                or speaker.get("source")
                or "Speaker"
            )
            lines.append(f"{label}: {part['text']}")
    # Whitelist fields: private_notes_* must never reach this workspace dataset.
    row = {
        "id": note["id"],
        "title": title,
        "content": "\n".join(lines),
        "_deleted": False,
        dlt_utils.NODE_SET_COLUMN: [f"granola:{scope}"],
    }
    url = note.get("web_url")
    if isinstance(url, str) and url.startswith("https://notes.granola.ai/"):
        row["url"] = url
    return row


def _is_note_error(error: GranolaError) -> bool:
    """An error that belongs to one note, not to the connection or the run."""
    return isinstance(error, GranolaNoteError) or (
        error.status is not None and 400 <= error.status < 500 and error.status not in {401, 429}
    )


def _is_unavailable(error: GranolaError) -> bool:
    """A 5xx or a request that never got an answer (after the client's one retry)."""
    return not isinstance(error, GranolaBudgetError) and (
        error.status is None or error.status >= 500
    )


def _api_answers(api: Any) -> bool:
    """Whether Granola answers a one-note listing, the cheapest request there is.

    Tells a note that keeps failing with a 5xx or a timeout apart from an outage: only the
    first may be charged to the note, or an outage would use up the retries of every note it
    meets.
    """
    try:
        api.get("/v1/notes", {"page_size": 1})
    except GranolaBudgetError:
        raise
    except GranolaError:
        return False
    return True


def _in_scope(note: dict, scope: str) -> bool:
    """Whether a fetched note still belongs to ``scope``, from its own membership.

    A retry by id bypasses the scope's listing, so it must not import a note that has since
    left the folder or space. A note in a folder's direct child counts as in the folder;
    Granola does not say whether deeper ancestors are listed, so a deeper one is treated as
    outside, never imported into the wrong scope.
    """
    if scope == "all":
        return True
    wanted = scope.lower()
    folders = note.get("folder_membership")
    folders = (
        [item for item in folders if isinstance(item, dict)] if isinstance(folders, list) else []
    )
    if wanted.startswith("fol_"):
        keys = ("id", "parent_folder_id")
        ids = {str(item.get(key) or "").lower() for item in folders for key in keys}
    else:
        spaces = note.get("space_membership")
        spaces = spaces if isinstance(spaces, list) else []
        ids = {str(item.get("id") or "").lower() for item in spaces if isinstance(item, dict)}
        ids |= {str(item.get("space_id") or "").lower() for item in folders}
    return wanted in ids


def _read_note(api: Any, note_id: str, include_transcript: bool) -> dict:
    note = _get_note(api, note_id, include_transcript)
    if note.get("id") != note_id:
        raise GranolaNoteError("Granola returned a different note")
    return note


def _row(note: dict, scope: str) -> dict:
    if note.get("deleted_at"):
        return {"id": note["id"], "_deleted": True}
    return _render(note, scope)


def _record_failure(
    state: dict, stats: dict, note_id: str, stamp: Any, *, counts: bool = True
) -> bool:
    """Record a failed note for a retry by id; ``True`` once it is given up.

    Only an error that belongs to the note (``counts``: a 4xx, an invalid shape, the size cap)
    uses up one of its :data:`MAX_NOTE_RETRIES` attempts; a 5xx or a timeout does not, since a
    listing that answers cannot see an outage of the note endpoints alone. A given-up note is
    dropped from the list, and an edit lists it again with a fresh allowance.

    Raises :class:`GranolaError` (a pause, ``failed_unavailable``) when a 5xx note finds the
    list full: it is never given up, so the run waits instead.
    """
    failures = state.setdefault("note_failures", {})
    previous = failures.get(note_id) or {}
    attempts = previous.get("attempts", 0) if previous.get("stamp") == stamp else 0
    if counts:
        attempts += 1
    full = note_id not in failures and len(failures) >= MAX_FAILED_NOTES
    if full and not counts:
        if not failures:
            state.pop("note_failures")
        raise GranolaError("Granola retry list is full; waiting for notes to read")
    stats["failed"] += 1
    if attempts < MAX_NOTE_RETRIES and not full:
        failures[note_id] = {"attempts": attempts, "stamp": stamp}
        return False
    failures.pop(note_id, None)
    if not failures:
        state.pop("note_failures")
    stats["skipped_unavailable"] += 1
    logger.warning(
        "Gave up on a Granola note after %d attempt(s)%s; an edit to it retries it",
        attempts,
        " (retry list full)" if full else "",
    )
    return True


def _clear_failure(state: dict, note_id: str) -> None:
    failures = state.get("note_failures")
    if failures is not None:
        failures.pop(note_id, None)
        if not failures:
            state.pop("note_failures")


def _scope_params(scope: str) -> dict[str, str]:
    if scope == "all":
        return {}
    return {"folder_id" if scope.lower().startswith("fol_") else "space_id": scope}


def _note_id(item: dict) -> str:
    note_id = item.get("id")
    if not isinstance(note_id, str) or not _NOTE.fullmatch(note_id):
        raise GranolaError("Invalid Granola note ID")
    return note_id


def _add_sorted(ids: list[str], note_id: str) -> None:
    at = bisect.bisect_left(ids, note_id)
    if at == len(ids) or ids[at] != note_id:
        ids.insert(at, note_id)


def _drop_sorted(ids: list[str], note_id: str) -> None:
    at = bisect.bisect_left(ids, note_id)
    if at < len(ids) and ids[at] == note_id:
        del ids[at]


def _track(state: dict, row: dict) -> None:
    """Keep ``known`` (the note ids this table holds) and a running reconcile's ``reconcile_seen``
    and ``reconcile_candidates`` in step with a row the run emits.

    A live row emitted while a reconcile is in progress is evidence the note is in the scope:
    it may have been created or moved in after the listing passed its place, and must not be
    tombstoned for missing from the pages already read. Either way it is no longer a candidate.
    """
    known = state.setdefault("known", [])
    seen = state.get("reconcile_seen")
    candidates = state.get("reconcile_candidates")
    if candidates is not None:
        _drop_sorted(candidates, row["id"])
    if row["_deleted"]:
        _drop_sorted(known, row["id"])
        if seen is not None:
            _drop_sorted(seen, row["id"])
        return
    _add_sorted(known, row["id"])
    if seen is not None:
        _add_sorted(seen, row["id"])


def _reconcile_due(state: dict, now: datetime) -> bool:
    # ``reconciled_at`` moves only when a reconcile completes, so one in progress stays due.
    reconciled_at = state.get("reconciled_at")
    return reconciled_at is None or now - datetime.fromisoformat(reconciled_at) >= (
        RECONCILE_INTERVAL
    )


def _reconcile(
    api: Any,
    scope: str,
    state: dict,
    stats: dict,
    take: Callable[[dict], dict],
    now: datetime,
) -> Iterator[dict]:
    """List the whole scope, ids only, then confirm each known note it no longer shows and
    tombstone the ones Granola confirms are gone.

    The listing is carried across runs in ``reconcile_cursor`` and ``reconcile_seen``, so a cut
    resumes it; only a listing that finished names candidates. A complete listing of zero notes
    while notes are known deletes nothing and reports ``failed_access``: that is a key that lost
    access, not a scope emptied by hand.

    A listing that pages while notes change can skip a live note, so absence only names a
    candidate (``reconcile_candidates``). Each is read by id: a 404, ``deleted_at`` or
    membership outside the scope tombstones it, a note still in the scope stays known. Any
    other error stops the run as usual; the candidates not yet confirmed stay for the next run.
    """
    if "reconcile_candidates" not in state:
        listed = _list_scope(api, scope, state)
        known = list(state.get("known") or [])
        if not listed and known:
            # reconciled_at stays: the next run lists again (one request while it stays empty).
            stats["failed"] += 1
            stats["failed_access"] = 1
            logger.warning(
                "Granola listed no note in the scope while %d are known; nothing is forgotten",
                len(known),
            )
            return
        state["known"] = sorted(listed)
        state["reconcile_candidates"] = [note_id for note_id in known if note_id not in listed]
    candidates = state["reconcile_candidates"]
    while candidates:
        # Off the list before its row: ``take`` keeps the candidates in step with every row.
        note_id = candidates[0]
        try:
            note = api.get(f"/v1/notes/{note_id}")
        except GranolaError as error:
            if error.status != 404:
                raise
            note = None
        if note is not None and note.get("id") != note_id:
            raise GranolaNoteError("Granola returned a different note")
        gone = note is None or bool(note.get("deleted_at")) or not _in_scope(note, scope)
        candidates.pop(0)
        if gone:
            yield take({"id": note_id, "_deleted": True})
        else:
            _add_sorted(state.setdefault("known", []), note_id)
    del state["reconcile_candidates"]
    state["reconciled_at"] = now.isoformat()


def _list_scope(api: Any, scope: str, state: dict) -> set[str]:
    """Every note id the scope lists, read across runs in ``reconcile_cursor`` and
    ``reconcile_seen``."""
    seen = state.setdefault("reconcile_seen", [])
    restarted = False
    cursors: set[str] = set()
    while True:
        cursor = state.get("reconcile_cursor")
        params: dict[str, Any] = {"page_size": RECONCILE_PAGE_SIZE, **_scope_params(scope)}
        if cursor:
            params["cursor"] = cursor
        try:
            entries, next_cursor = _page(api.get("/v1/notes", params), "notes")
        except GranolaError as error:
            if error.status == 400 and cursor and not restarted:
                # An expired cursor: list the scope again from its first page, once.
                state.pop("reconcile_cursor")
                seen.clear()
                restarted = True
                continue
            raise
        if next_cursor is not None and (next_cursor == cursor or next_cursor in cursors):
            raise GranolaError("Granola repeated a notes cursor")
        for item in entries:
            note_id = _note_id(item)
            if not item.get("deleted_at"):
                _add_sorted(seen, note_id)
        if next_cursor is None:
            break
        cursors.add(next_cursor)
        state["reconcile_cursor"] = next_cursor
    state.pop("reconcile_cursor", None)
    return set(state.pop("reconcile_seen"))


def new_stats() -> dict[str, int]:
    """The counts of a run, every key present from its start so a host finds them on any exit.

    ``failed`` counts failed notes plus one for a run that stopped early; the reason of the
    stop is the one ``failed_*`` key set to 1 (none for a listing Granola refused or answered
    malformed). ``skipped_unavailable`` counts notes given up after their retries.
    ``retry_after_seconds`` is the wait a 429 asked for. ``requests`` counts the run's
    requests, a retried one once.
    """
    return dict.fromkeys(
        (
            "scanned",
            "skipped",
            "deleted",
            "failed",
            "skipped_unavailable",
            "failed_auth",
            "failed_rate_limit",
            "failed_unavailable",
            "failed_budget",
            "failed_access",
            "retry_after_seconds",
            "requests",
        ),
        0,
    )


def _iter_rows(
    client: Any,
    scope: str,
    state: dict,
    stats: dict,
    *,
    check_active: Callable[[], None] | None = None,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_requests: int = DEFAULT_MAX_REQUESTS,
    deadline_seconds: float = DEFAULT_DEADLINE_SECONDS,
    include_transcript: bool = True,
    now: datetime | None = None,
    key_revision: str | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> Iterator[dict]:
    """Yield the changed rows of one scope. Pure of dlt, so tests drive it with a dict."""
    validate_scope(scope)
    stats.clear()
    stats.update(new_stats())
    now = now or datetime.now(timezone.utc)
    api = _RunBounds(
        client,
        stats,
        max_requests=max_requests,
        deadline_seconds=deadline_seconds,
        check_active=check_active,
        clock=clock,
    )
    if key_revision is not None and state.get("key_revision") != key_revision:
        # A replacement key may grant access to older notes. Re-enumerate from the beginning
        # while preserving already imported documents (``known`` included); candidates the old
        # key never confirmed go back to ``known`` for the new key's reconcile.
        for note_id in state.get("reconcile_candidates") or []:
            _add_sorted(state.setdefault("known", []), note_id)
        for key in _WALK_KEYS:
            state.pop(key, None)
        state["key_revision"] = key_revision
    # Tables synced before the option existed rendered transcripts. A change lists every note
    # again so each is rendered with or without its transcript; nothing is deleted by it.
    if state.get("include_transcript", True) != include_transcript:
        for key in _LISTING_KEYS:
            state.pop(key, None)
    state["include_transcript"] = include_transcript
    state.pop("retry_notes", None)  # superseded by direct retries of note_failures
    # A pending page stores identifiers only, never meeting content or secrets. The high
    # watermark advances whenever a walk completes; a note that failed is retried by id at the
    # start of the next cycle instead of holding it back.
    new_cycle = "cycle_start" not in state
    state.setdefault("cycle_start", (now - WATERMARK_OVERLAP).isoformat())
    emitted = 0
    answered = False  # Granola answered a request in this run
    reset_cursor = False
    seen: set[str] = set()

    def set_aside(error: GranolaError) -> bool:
        """Whether the note may leave the page for a retry by id, not pause the run.

        A note's own error always may. A 5xx or timeout may while Granola otherwise answers;
        it is then kept for retries without using up any attempt.
        """
        if _is_note_error(error):
            return True
        return _is_unavailable(error) and (answered or _api_answers(api))

    def take(row: dict) -> dict:
        nonlocal emitted
        emitted += 1
        stats["scanned"] += 1
        stats["deleted"] += int(row["_deleted"])
        _track(state, row)
        return row

    try:
        if new_cycle:
            unavailable = 0
            for note_id, failure in sorted((state.get("note_failures") or {}).items()):
                if emitted >= max_rows:
                    raise GranolaBudgetError("Granola row budget exhausted")
                try:
                    note = _read_note(api, note_id, include_transcript)
                    # A retry by id bypasses the listing, so the note's own membership decides.
                    in_scope = bool(note.get("deleted_at")) or _in_scope(note, scope)
                    row = _row(note, scope) if in_scope else None
                except GranolaError as error:
                    if _is_note_error(error):
                        stats["skipped"] += 1
                        _record_failure(state, stats, note_id, failure.get("stamp"))
                        continue
                    if not _is_unavailable(error):
                        raise
                    # Kept as it is, attempts untouched. A few in a row means the note
                    # endpoints are down: leave the rest for the next cycle rather than spend
                    # the run's budget on them.
                    stats["skipped"] += 1
                    stats["failed"] += 1
                    unavailable += 1
                    if unavailable >= MAX_UNAVAILABLE_RETRIES:
                        break
                    continue
                answered = True
                _clear_failure(state, note_id)
                if row is None:
                    stats["skipped"] += 1
                    logger.info("A retried Granola note left the scope; not imported there")
                    continue
                yield take(row)
        while True:
            pending = state.get("pending")
            if pending is None:
                params: dict[str, Any] = {"page_size": LIST_PAGE_SIZE, **_scope_params(scope)}
                if state.get("watermark"):
                    params["updated_after"] = state["watermark"]
                cursor = state.get("cursor")
                if cursor:
                    params["cursor"] = cursor
                try:
                    entries, next_cursor = _page(api.get("/v1/notes", params), "notes")
                except GranolaError as error:
                    if error.status == 400 and cursor and not reset_cursor:
                        state.pop("cursor", None)
                        reset_cursor = True
                        continue
                    raise
                answered = True
                if next_cursor is not None and (next_cursor == cursor or next_cursor in seen):
                    raise GranolaError("Granola repeated a notes cursor")
                if next_cursor:
                    seen.add(next_cursor)
                listed: list[dict[str, Any]] = [
                    {
                        "id": _note_id(item),
                        "deleted": bool(item.get("deleted_at")),
                        "stamp": item.get("updated_at")
                        if isinstance(item.get("updated_at"), str)
                        else None,
                    }
                    for item in entries
                ]
                state["pending"] = pending = listed
                state["next_cursor"] = next_cursor
            while pending:
                if emitted >= max_rows:
                    raise GranolaBudgetError("Granola row budget exhausted")
                item = pending[0]
                note_id = item["id"]
                try:
                    if item["deleted"]:
                        row = {"id": note_id, "_deleted": True}
                    else:
                        row = _row(_read_note(api, note_id, include_transcript), scope)
                except GranolaError as error:
                    # Never poison a page with one unreadable note: a 4xx, an invalid shape,
                    # the size cap, or a 5xx/timeout while Granola otherwise answers. The note
                    # leaves the page and is retried by id later (a 5xx without using up an
                    # attempt); an outage still pauses the run here.
                    if not set_aside(error):
                        raise
                    _record_failure(
                        state,
                        stats,
                        note_id,
                        item.get("stamp"),
                        counts=_is_note_error(error),
                    )
                    pending.pop(0)
                    stats["skipped"] += 1
                    continue
                answered = True
                _clear_failure(state, note_id)
                yield take(row)
                pending.pop(0)
            next_cursor = state.pop("next_cursor")
            state.pop("pending")
            if next_cursor is None:
                state["watermark"] = state.pop("cycle_start")
                state.pop("cursor", None)
                break
            state["cursor"] = next_cursor
        if _reconcile_due(state, now):
            yield from _reconcile(api, scope, state, stats, take, now)
    except GranolaError as error:
        stats["failed"] += 1
        if isinstance(error, GranolaBudgetError):
            reason = "budget"
        elif error.status == 401:
            reason = "auth"
        elif error.status == 429:
            reason = "rate_limit"
            stats["retry_after_seconds"] = error.retry_after
        elif _is_unavailable(error):
            reason = "unavailable"
        else:
            reason = None  # a listing Granola refused or answered malformed
        if reason:
            stats[f"failed_{reason}"] = 1
        logger.warning(
            "Granola sync paused (%s); progress will resume on the next sync",
            reason or "listing refused",
        )


def _positive(value: Any, name: str, kind: type | tuple[type, ...] = int) -> None:
    if isinstance(value, bool) or not isinstance(value, kind) or not value > 0:
        raise ValueError(f"granola_source needs {name} as a positive number")


def granola_source(
    *,
    scope: str,
    resource_name: str | None = None,
    api_key: str | None = None,
    key_revision: str | None = None,
    include_transcript: bool = True,
    check_active: Callable[[], None] | None = None,
    service: Any = None,
    max_requests: int = DEFAULT_MAX_REQUESTS,
    deadline_seconds: float = DEFAULT_DEADLINE_SECONDS,
    max_rows: int = DEFAULT_MAX_ROWS,
):
    """Return a ``dlt`` resource that yields one Granola scope's notes for ``remember``.

    Args:
        scope: ``"all"`` (every note the key can see, unfiled ones included), a folder id
            (``fol_...``, child folders included) or a space id (``spa_...``).
        resource_name: Stable name of this connection and scope; it names the staging table
            and scopes the sync state. Defaults to ``granola_notes_<scope lowercased>``, which
            two keys syncing one scope into same-named datasets would share: hosts pass their
            own (Cloud names it from the owner, the account and the scope).
        api_key: A Granola API key (``grn_...``), sent as ``Authorization: Bearer``. Used to
            build the client when ``service`` is omitted. Never read from the environment.
        key_revision: Any string that changes when the key is replaced. A change starts the
            walk over (a new key may see older notes) and deletes nothing by itself.
        include_transcript: Render each meeting's transcript (default). ``False`` never asks
            Granola for a transcript and renders the same text without the ``Transcript:``
            block. A change from the previous sync lists the scope again, so every synced note
            is rendered again; it deletes nothing by itself.
        check_active: Host authorization checkpoint, called before each request attempt (by
            the client) and around each row. What it raises stops the run as it is.
        service: Pre-built client (see :func:`build_granola_service`). It must accept
            ``get(path, params, *, check_active, time_left)`` and call ``check_active`` before
            each attempt it makes.
        max_requests, deadline_seconds, max_rows: Bounds of one run (Cloud's: 200 requests,
            240 s, 200 rows). A run that reaches one stops with ``failed_budget``, keeps what it
            emitted and resumes on the next run.
    """
    import dlt

    # In tree the constant always exists; a packaged copy runs against any installed cognee.
    if getattr(dlt_utils, "DOCUMENT_SYNC_VERSION", 0) < 3:
        raise RuntimeError(
            "Granola sync requires a Cognee build that reads per-row node sets "
            "(DOCUMENT_SYNC_VERSION >= 3). Upgrade Cognee before syncing."
        )
    validate_scope(scope)
    if resource_name is None:
        resource_name = f"granola_notes_{scope.lower()}"
    if not isinstance(resource_name, str) or not resource_name.strip():
        raise ValueError("granola_source needs a resource_name, one per connection and scope")
    if key_revision is not None and not isinstance(key_revision, str):
        raise ValueError("key_revision must be a string or None")
    if not isinstance(include_transcript, bool):
        raise ValueError("include_transcript must be True or False")
    _positive(max_requests, "max_requests")
    _positive(deadline_seconds, "deadline_seconds", (int, float))
    _positive(max_rows, "max_rows")
    if service is None and not api_key:
        raise ValueError("granola_source needs a service or an api_key")
    # Built here so a malformed key fails at construction and the closure holds no raw key.
    client = service if service is not None else GranolaClient(api_key or "")
    stats: dict[str, int] = new_stats()

    @dlt.resource(
        name=resource_name,
        primary_key="id",
        write_disposition="merge",
        columns={
            "_deleted": {"data_type": "bool", "hard_delete": True},
            # dlt creates no column that is None in every row; without the hint the first note
            # with a URL would add the column and change every older row's identity.
            "url": {"data_type": "text", "nullable": True},
        },
    )
    def granola_notes():
        rows = _iter_rows(
            client,
            scope,
            dlt.current.resource_state(),
            stats,
            check_active=check_active,
            max_rows=max_rows,
            max_requests=max_requests,
            deadline_seconds=deadline_seconds,
            include_transcript=include_transcript,
            key_revision=key_revision,
        )
        yield from dlt_utils.guarded_rows(rows, check_active)

    resource = granola_notes()
    # The _PROVIDERS key, the document source tag and the Cloud key: host retire finds the
    # tables by it.
    setattr(resource, DOCUMENT_SOURCE_ATTR, "granola")
    setattr(resource, dlt_utils.PIPELINE_SCOPE_ATTR, resource_name)
    resource.cognee_sync_stats = stats
    # Read by the Cloud pod: a 401 asks for a reconnect of the key, and a 429 cools down the
    # whole workspace (Granola's rate pool is per user or workspace).
    resource.cognee_api_key_source = True
    resource.cognee_shared_rate_pool = True
    return resource
