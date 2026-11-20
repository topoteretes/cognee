# Adapted from topoteretes/cognee-community (Apache-2.0), PR #559, commit 25dc254 (blvckShawn).
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""Evernote connector for cognee: a ``dlt`` source that turns an account's notes into memory.

One source is one Evernote token (a developer token, or an OAuth 1.0a access token obtained
elsewhere: this module runs no OAuth flow) and an optional selection by notebook and tag names.
It yields one document per note and is meant to be handed to :func:`cognee.remember`::

    import cognee
    from cognee.tasks.ingestion.connectors import evernote_source

    await cognee.remember(
        evernote_source(resource_name="evernote_notes", token="<token>"),
        dataset_name="evernote",
        primary_key="id",
        write_disposition="merge",   # REQUIRED, the add pipeline defaults to "replace"
        max_rows_per_table=0,
    )

Design
------
* **Rows** are flat ``{id, title, content, _deleted}``; ``id`` is ``note:<guid>``. The content is
  the note's ENML as plain text, after a ``Notebook:`` line and a sorted ``Tags:`` line.
* **Incremental** through Evernote's update sequence numbers. A run reads the sync chunks after
  the stored ``last_usn`` (pass 1, metadata only), then fetches the body of every changed note in
  the selection (pass 2). A run cut in either pass keeps its progress and the next run continues
  where it stopped; pass 1 is saved chunk by chunk, so a large account always moves forward.
* **Deletion** only from a complete pass 1: an expunged note or notebook, a trashed note, a note
  that left the selection, and on a full pass every note that is no longer live and selected.
  A note whose body Evernote no longer finds is deleted too.
* **Full pass** on the first run, when the selection changes, when ``fullSyncBefore`` is newer
  than the last sync (Evernote's own signal that ``last_usn`` is no longer usable) and when the
  account's ``updateCount`` went backwards.
"""

import copy
import time
from collections.abc import Callable, Iterator
from html.parser import HTMLParser
from types import SimpleNamespace
from typing import Any

from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR, with_retries

logger = get_logger("evernote_connector")

STATE_VERSION = 1

# Entries per getFilteredSyncChunk call, the value the community connector used.
CHUNK_SIZE = 250

# Attempts of one call: a transport error is retried after 1s, then 2s, then 4s. The community
# connector's max_retries=3.
_RETRIES = 4

# A rate limit is waited out only while the waits of one call stay within this many seconds;
# a longer one cuts the run, which resumes next time. The community connector's bound: the walk
# holds cognee's process-wide dlt staging lock.
MAX_RATE_LIMIT_WAIT_SECONDS = 30

# Socket timeout of one Thrift request, the community connector's timeout_seconds=30.
_TIMEOUT_MS = 30_000


class EvernoteError(RuntimeError):
    """Base error of the Evernote connector. Messages carry no token, guid, URL or body."""


class EvernoteAuthError(EvernoteError):
    """Evernote rejected the token: INVALID_AUTH or AUTH_EXPIRED."""


class EvernotePermissionError(EvernoteError):
    """Evernote answered PERMISSION_DENIED: the token is valid but may not read this."""


class EvernoteUnavailableError(EvernoteError):
    """Evernote did not answer after the retries, or answered with a system error."""


class NoteNotFoundError(EvernoteError):
    """The note no longer exists at Evernote."""


class EvernoteRateLimitError(EvernoteError):
    """Evernote's rate limit is spent; ``retry_after`` is the wait in seconds it named."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class _TransportError(EvernoteError):
    """Internal: no answer from Evernote, retried by the client."""


_MISSING_EXTRA = 'The Evernote connector needs its client library: pip install "cognee[evernote]"'


def _edam() -> SimpleNamespace:
    """The Thrift classes the client uses, imported only when a call is made."""
    try:
        from evernote.edam.error.ttypes import (
            EDAMErrorCode,
            EDAMNotFoundException,
            EDAMSystemException,
            EDAMUserException,
        )
        from evernote.edam.notestore.ttypes import SyncChunkFilter
        from thrift.Thrift import TException
        from thrift.transport.TTransport import TTransportException
    except ImportError as exc:
        raise ImportError(_MISSING_EXTRA) from exc

    return SimpleNamespace(
        chunk_filter=SyncChunkFilter,
        codes=EDAMErrorCode,
        user=EDAMUserException,
        system=EDAMSystemException,
        not_found=EDAMNotFoundException,
        thrift=TException,
        transport=TTransportException,
    )


def _code_name(codes: Any, code: Any) -> str:
    """``NAME (value)`` of an EDAM error code, or the value alone."""
    for name in dir(codes):
        if name.isupper() and getattr(codes, name) == code:
            return f"{name} ({int(code)})"
    return str(code)


class EvernoteApi:
    """Synchronous client over Evernote's Thrift UserStore and NoteStore.

    The token is an argument of every call, never a header. Nothing is imported or contacted at
    construction; ``user_store`` and ``note_store`` replace the Thrift clients (tests).
    """

    def __init__(
        self,
        token: str,
        *,
        sandbox: bool = False,
        user_store: Any = None,
        note_store: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(token, str) or not token.strip():
            raise EvernoteAuthError("An Evernote token is required: pass a non-empty string")
        self._token = token.strip()
        self.sandbox = sandbox
        self._user_store = user_store
        self._note_store = note_store
        self._sleep = sleep
        self._user_id: int | None = None

    def __repr__(self) -> str:
        return f"EvernoteApi(token=<redacted>, sandbox={self.sandbox!r})"

    def _host(self) -> str:
        return "sandbox.evernote.com" if self.sandbox else "www.evernote.com"

    def _thrift_client(self, url: str, module: str) -> Any:
        try:
            from thrift.protocol.TBinaryProtocol import TBinaryProtocol
            from thrift.transport.THttpClient import THttpClient

            if module == "user":
                from evernote.edam.userstore import UserStore as service
            else:
                from evernote.edam.notestore import NoteStore as service
        except ImportError as exc:
            raise ImportError(_MISSING_EXTRA) from exc

        transport = THttpClient(url)
        transport.setTimeout(_TIMEOUT_MS)
        return service.Client(TBinaryProtocol(transport))

    def _users(self) -> Any:
        if self._user_store is None:
            self._user_store = self._thrift_client(f"https://{self._host()}/edam/user", "user")
        return self._user_store

    def _notes(self, check_active: Callable[[], None] | None) -> Any:
        if self._note_store is None:
            urls = self._call(self._users().getUserUrls, check_active)
            url = str(getattr(urls, "noteStoreUrl", "") or "")
            if not url.startswith(f"https://{self._host()}/"):
                # The note store must be on the host the token was issued for.
                raise EvernoteError("Evernote returned a note store URL on an unexpected host")
            self._note_store = self._thrift_client(url, "note")
        return self._note_store

    def _call(self, method: Callable[..., Any], check_active, *args: Any) -> Any:
        """One Thrift call with the token as its first argument, mapped to the connector's errors.

        A transport error is retried, a rate limit is waited out while the waits of this call
        stay within ``MAX_RATE_LIMIT_WAIT_SECONDS``. ``check_active`` runs before each attempt
        and before each wait; what it raises propagates as it is.
        """
        edam = _edam()
        waited = [0.0]

        def attempt() -> Any:
            try:
                return method(self._token, *args)
            except edam.system as exc:
                code = exc.errorCode
                if code == edam.codes.RATE_LIMIT_REACHED:
                    duration = float(getattr(exc, "rateLimitDuration", None) or 1)
                    raise EvernoteRateLimitError(
                        f"Evernote rate limit reached, retry after {duration:.0f}s",
                        retry_after=duration,
                    ) from None
                failure = EvernoteUnavailableError(
                    f"Evernote system error {_code_name(edam.codes, code)}"
                )
            except edam.user as exc:
                code, name = exc.errorCode, _code_name(edam.codes, exc.errorCode)
                if code in (edam.codes.INVALID_AUTH, edam.codes.AUTH_EXPIRED):
                    failure = EvernoteAuthError(f"Evernote rejected the token ({name})")
                elif code == edam.codes.PERMISSION_DENIED:
                    failure = EvernotePermissionError(f"Evernote denied access ({name})")
                else:
                    failure = EvernoteError(f"Evernote user error {name}")
            except edam.not_found:
                # The exception names the guid it did not find: the message never does.
                failure = NoteNotFoundError("Evernote note not found")
            except (edam.transport, OSError, EOFError) as exc:
                failure = _TransportError(f"Evernote transport failed: {type(exc).__name__}")
            except edam.thrift as exc:
                failure = EvernoteError(f"Evernote API call failed: {type(exc).__name__}")
            raise failure from None

        def delay(error: BaseException, number: int) -> float | None:
            if isinstance(error, EvernoteRateLimitError):
                wait = float(error.retry_after or 1)
                if waited[0] + wait > MAX_RATE_LIMIT_WAIT_SECONDS:
                    return None
                waited[0] += wait
                return wait
            return float(2**number)

        if check_active is not None:
            check_active()
        try:
            return with_retries(
                attempt,
                retry_on=(_TransportError, EvernoteRateLimitError),
                delay=delay,
                attempts=_RETRIES,
                check_active=check_active,
                sleep=self._sleep,
            )
        except _TransportError:
            raise EvernoteUnavailableError(
                f"Evernote did not answer after {_RETRIES} attempts"
            ) from None

    def get_user_id(self, check_active: Callable[[], None] | None = None) -> int:
        """The account's user id, read once per client."""
        if self._user_id is None:
            user = self._call(self._users().getUser, check_active)
            self._user_id = int(user.id)
        return self._user_id

    def get_sync_state(self, check_active: Callable[[], None] | None = None) -> Any:
        """``SyncState``: ``currentTime``, ``fullSyncBefore`` and ``updateCount``."""
        return self._call(self._notes(check_active).getSyncState, check_active)

    def get_filtered_sync_chunk(
        self, after_usn: int, max_entries: int, check_active: Callable[[], None] | None = None
    ) -> Any:
        """The notes, notebooks, tags and expunged guids changed after ``after_usn``."""
        chunk_filter = _edam().chunk_filter(
            includeNotes=True, includeNotebooks=True, includeTags=True, includeExpunged=True
        )
        store = self._notes(check_active)
        return self._call(
            store.getFilteredSyncChunk, check_active, after_usn, max_entries, chunk_filter
        )

    def get_note(self, guid: str, check_active: Callable[[], None] | None = None) -> Any:
        """The note with its ENML ``content``, ``title``, ``notebookGuid`` and ``tagGuids``,
        without resource data."""
        store = self._notes(check_active)
        return self._call(store.getNote, check_active, guid, True, False, False, False)


def build_evernote_service(token: str, *, sandbox: bool = False) -> EvernoteApi:
    """Wrap a host-owned Evernote token in a client. No network call is made here.

    A ``service`` handed to :func:`evernote_source` instead must offer ``get_user_id``,
    ``get_sync_state``, ``get_filtered_sync_chunk(after_usn, max_entries)`` and
    ``get_note(guid)``, each taking ``check_active`` and calling it before each attempt.
    """
    return EvernoteApi(token, sandbox=sandbox)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
class ENMLToTextParser(HTMLParser):
    """Turns Evernote ENML into plain text: headings, lists, to-dos, attachments as markers."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.pieces: list[str] = []
        self._ignore_depth = 0
        self._crypt_depth = 0
        self._cells = 0  # cells opened so far in the current table row

    def _ensure_newline(self) -> None:
        if self.pieces and not self.pieces[-1].endswith("\n"):
            self.pieces.append("\n")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attributes = dict(attrs)
        if tag in ("script", "style"):
            self._ignore_depth += 1
            return
        if tag == "en-crypt":
            self.pieces.append("[encrypted content]")
            self._crypt_depth += 1
            return
        if self._ignore_depth or self._crypt_depth:
            return
        if tag in ("td", "th"):
            if self._cells:
                self.pieces.append(" | ")
            self._cells += 1
            return
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6", "div", "p", "tr"):
            self._cells = 0
            self._ensure_newline()
            self.pieces.append({"h1": "# ", "h2": "## ", "h3": "### "}.get(tag, ""))
        elif tag == "li":
            self._ensure_newline()
            self.pieces.append("- ")
        elif tag == "br":
            self.pieces.append("\n")
        elif tag == "en-todo":
            checked = str(attributes.get("checked", "")).lower() == "true"
            self.pieces.append("[x] " if checked else "[ ] ")
        elif tag == "en-media":
            self.pieces.append(f"[attachment: {attributes.get('type', '')}]")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in ("script", "style"):
            self._ignore_depth = max(0, self._ignore_depth - 1)
            return
        if tag == "en-crypt":
            self._crypt_depth = max(0, self._crypt_depth - 1)
            return
        if self._ignore_depth or self._crypt_depth:
            return
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6", "div", "p", "tr", "li"):
            self._ensure_newline()

    def handle_data(self, data: str) -> None:
        if not (self._ignore_depth or self._crypt_depth):
            self.pieces.append(data)


def enml_to_text(enml: str) -> str:
    """ENML as plain text, runs of blank lines collapsed to one. Malformed markup never raises."""
    if not enml or not isinstance(enml, str):
        return ""
    parser = ENMLToTextParser()
    parser.feed(enml)
    parser.close()
    collapsed: list[str] = []
    for line in (line.rstrip() for line in "".join(parser.pieces).splitlines()):
        if line.strip() or (collapsed and collapsed[-1]):
            collapsed.append(line if line.strip() else "")
    while collapsed and not collapsed[-1]:
        collapsed.pop()
    return "\n".join(collapsed)


def note_to_row(
    note: dict[str, Any], content_text: str, notebooks: dict[str, str], tags: dict[str, str]
) -> dict[str, Any]:
    """The document row of a note (its stored metadata, see ``_meta``) and its text.

    The notebook and tag names are part of the text, so a rename reaches a note's document only
    when the note itself changes.
    """
    header = []
    notebook = notebooks.get(note["notebook"] or "")
    if notebook:
        header.append(f"Notebook: {notebook}")
    names = sorted((tags[g] for g in note["tags"] if tags.get(g)), key=str.casefold)
    if names:
        header.append(f"Tags: {', '.join(names)}")
    content = "\n".join(header) + "\n\n" + content_text if header else content_text
    return {
        "id": f"note:{note['guid']}",
        "title": note["title"],
        "content": content,
        "_deleted": False,
    }


def _tombstone(guid: str) -> dict[str, Any]:
    return {"id": f"note:{guid}", "title": "", "content": "", "_deleted": True}


def _meta(note: Any) -> dict[str, Any]:
    """What the selection needs of a listed note, as plain data that dlt state can hold. No
    title or other text: the state of a large cut sync holds one of these per note."""
    return {
        "guid": note.guid,
        "notebook": getattr(note, "notebookGuid", None),
        "tags": sorted(getattr(note, "tagGuids", None) or []),
        "usn": int(getattr(note, "updateSequenceNum", None) or 0),
        "live": getattr(note, "active", True) is not False
        and getattr(note, "deleted", None) is None,
    }


# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------
# The stats key each cut reason sets. Every key is one Cloud counts (resource_health
# _COUNT_KEYS); a permission denial has none there, so it counts in ``failed`` alone.
_CUT_KEYS = {
    "rate_limit": "failed_rate_limit",
    "auth": "failed_auth",
    "unavailable": "failed_unavailable",
    "permission": None,
}
_CUT_MESSAGES = {
    "rate_limit": "rate limit",
    "auth": "Evernote rejected the token",
    "unavailable": "Evernote is unavailable",
    "permission": "Evernote denied access",
}


class _CutShort(Exception):
    """Internal: stop the run here, keeping the state reached so far."""

    def __init__(self, reason: str | None, error: EvernoteError | None = None):
        super().__init__(reason or "cap")
        self.reason = reason
        self.error = error


def new_stats() -> dict[str, int]:
    """The counts of a run, every key present from its start so a host finds them on any exit.

    ``emitted`` counts note documents, ``deleted`` tombstones. A run stopped by the rate limit, a
    rejected token or an outage adds one to ``failed`` and sets ``failed_rate_limit``,
    ``failed_auth`` or ``failed_unavailable``; a permission denial or a note that could not be read
    adds to ``failed`` alone.
    """
    return dict.fromkeys(
        ("emitted", "deleted", "failed", "failed_rate_limit", "failed_auth", "failed_unavailable"),
        0,
    )


class _Walker:
    """One run, working on a copy of the state that is written back only by ``commit``.

    State (version ``STATE_VERSION``): ``account`` (``production:<user id>``), ``last_usn`` and
    ``last_sync_time`` (of the last complete sync), ``notebooks`` and ``tags`` (guid -> name),
    ``note_ids`` (guid -> notebook guid of every note emitted and current), ``scope`` (the
    selected names and the guids they resolved to), ``retry`` (guid -> metadata of the notes whose
    body Evernote failed to serve, fetched again next run) and, while a sync is unfinished,
    ``listing``:
    the pass in progress, its maps, the notes it found and, once pass 1 is complete, the notes
    whose bodies are still to fetch.
    """

    def __init__(
        self,
        service: Any,
        state: dict,
        stats: dict[str, int],
        *,
        notebook_names: list[str] | None,
        tag_names: list[str] | None,
        sandbox: bool,
        full_resync: bool,
        max_notes_per_run: int | None,
        check_active: Callable[[], None] | None,
    ):
        self.service = service
        self.state = state
        self.work = copy.deepcopy(state)
        self.stats = stats
        self.notebook_names = notebook_names
        self.tag_names = tag_names
        self.sandbox = sandbox
        self.full_resync = full_resync
        self.max_notes_per_run = max_notes_per_run
        self.check_active = check_active
        self.rows_out = 0
        self.failed_notes = 0
        self.fetched = 0  # getNote calls of this run, against max_notes_per_run
        self.answered = False  # a getNote of this run returned a note or "not found"
        # In the retry tail the run's progress is already in ``work``, so a refusal there is a
        # clean cut, never a raise that would lose it.
        self.in_tail = False
        # The retry tail reads the notes that failed in earlier runs, not one this run's pass
        # has just failed.
        self.retry_before = set(state.get("retry") or {})

    def _signature(self) -> dict[str, Any]:
        return {"notebooks": self.notebook_names, "tags": self.tag_names}

    def _needs_full(self, sync_state: Any) -> bool:
        work = self.work
        if self.full_resync or "last_usn" not in work:
            return True
        scope = work.get("scope") or {}
        if {"notebooks": scope.get("notebooks"), "tags": scope.get("tags")} != self._signature():
            logger.info("Evernote selection changed: a full sync runs.")
            return True
        before, last = getattr(sync_state, "fullSyncBefore", None), work.get("last_sync_time")
        if before is not None and last is not None and before > last:
            logger.info("Evernote asks for a full sync (fullSyncBefore is newer than the last).")
            return True
        count = getattr(sync_state, "updateCount", None)
        return count is not None and count < work["last_usn"]

    def _new_listing(self, full: bool, sync_state: Any) -> dict[str, Any]:
        work = self.work
        if full:
            # Until this pass completes, the stored selection no longer describes what is
            # emitted (its tombstones may already be out): whatever the next run is asked for,
            # it must be a full pass too. _finish writes the selection back.
            work.pop("scope", None)
        return {
            "full": full,
            "scope": self._signature(),
            "started": getattr(sync_state, "currentTime", None),
            "usn": 0 if full else work["last_usn"],
            "notebooks": {} if full else dict(work.get("notebooks") or {}),
            "tags": {} if full else dict(work.get("tags") or {}),
            "notes": {},
            "expunged": [],
            "expunged_notebooks": [],
            "pending": None,
        }

    def _call(self, name: str, *args: Any, per_note: bool = False) -> Any:
        """One service call; a rate limit or a rejected token becomes a cut.

        With ``per_note`` a PERMISSION_DENIED or an unavailable answer is raised as it is, for
        ``_fetch`` to decide: it may be about one note, not the account.
        """
        try:
            return getattr(self.service, name)(*args, check_active=self.check_active)
        except EvernoteRateLimitError as exc:
            raise _CutShort("rate_limit", exc) from None
        except EvernotePermissionError as exc:
            if per_note:
                raise
            raise _CutShort("permission", exc) from None
        except EvernoteAuthError as exc:
            raise _CutShort("auth", exc) from None
        except EvernoteUnavailableError as exc:
            if per_note:
                raise
            raise _CutShort("unavailable", exc) from None

    def rows(self) -> Iterator[dict[str, Any]]:
        user_id = self._call("get_user_id")
        bind = f"{'sandbox' if self.sandbox else 'production'}:{user_id}"
        version = self.work.get("v")
        if version is not None and version != STATE_VERSION:
            raise EvernoteError(f"Evernote sync state version {version} is not supported")
        if self.work.get("account", bind) != bind:
            raise EvernoteError(
                "This resource_name is bound to another Evernote account or environment; "
                "use another resource_name"
            )
        self.work.update(v=STATE_VERSION, account=bind)
        sync_state = self._call("get_sync_state")

        listing = self.work.get("listing")
        if listing is not None and self._listing_is_stale(listing, sync_state):
            listing = None
        if listing is None:
            full = self._needs_full(sync_state)
            if not full and getattr(sync_state, "updateCount", None) == self.work["last_usn"]:
                # Nothing changed at Evernote: only notes that failed before are read.
                yield from self._retry_tail(self.work)
                return
            listing = self._new_listing(full, sync_state)
        self.work["listing"] = listing

        while listing["pending"] is None:
            self._list(listing)
            if self._restart_as_full(listing):
                listing = self.work["listing"] = self._new_listing(True, sync_state)
                continue
            yield from self._reconcile(listing)
        try:
            yield from self._fetch(listing)
        except _CutShort as cut:
            if cut.reason is not None:
                raise
            # max_notes_per_run stopped the pass one read short: that read is the tail's. The
            # pass is not stored yet, so the current names are the listing's.
            yield from self._retry_tail(listing)
            raise cut from None
        self._finish(listing)
        yield from self._retry_tail(self.work)

    def _listing_is_stale(self, listing: dict, sync_state: Any) -> bool:
        """A saved pass for another selection, an incremental one when a full one is asked for,
        or one older than Evernote's ``fullSyncBefore``."""
        if listing["scope"] != self._signature() or (self.full_resync and not listing["full"]):
            return True
        before, started = getattr(sync_state, "fullSyncBefore", None), listing["started"]
        return before is not None and started is not None and before > started

    def _list(self, listing: dict) -> None:
        """Pass 1: read sync chunks from ``listing["usn"]``, saving progress after each chunk."""
        while True:
            chunk = self._call("get_filtered_sync_chunk", listing["usn"], CHUNK_SIZE)
            high = getattr(chunk, "chunkHighUSN", None)
            if high is None:
                # No object of the filter is left after listing["usn"], so the pass has read
                # everything up to the account's updateCount. Taking it keeps a change the
                # filter does not include (a saved search, a resource) from making every idle
                # run list again.
                count = getattr(chunk, "updateCount", None)
                if count is not None and count > listing["usn"]:
                    listing["usn"] = count
                break
            if high <= listing["usn"]:
                raise EvernoteError(
                    f"Evernote sync chunk did not advance ({high} <= {listing['usn']})"
                )
            for notebook in getattr(chunk, "notebooks", None) or []:
                listing["notebooks"][notebook.guid] = notebook.name
            for tag in getattr(chunk, "tags", None) or []:
                listing["tags"][tag.guid] = tag.name
            for note in getattr(chunk, "notes", None) or []:
                if isinstance(note.guid, str) and note.guid:
                    listing["notes"][note.guid] = _meta(note)
            for guid in getattr(chunk, "expungedNotes", None) or []:
                listing["notes"].pop(guid, None)
                listing["expunged"].append(guid)
            for guid in getattr(chunk, "expungedNotebooks", None) or []:
                listing["notebooks"].pop(guid, None)
                listing["expunged_notebooks"].append(guid)
            for guid in getattr(chunk, "expungedTags", None) or []:
                listing["tags"].pop(guid, None)
            listing["usn"] = high
            count = getattr(chunk, "updateCount", None)
            if count is not None and high >= count:
                break

    def _resolve(self, listing: dict) -> tuple[list[str] | None, list[str] | None]:
        """The guids the selected names stand for. A name Evernote does not know fails the run."""
        resolved = []
        for kind, names, found in (
            ("Notebook", self.notebook_names, listing["notebooks"]),
            ("Tag", self.tag_names, listing["tags"]),
        ):
            if names is None:
                resolved.append(None)
                continue
            missing = sorted(set(names) - set(found.values()))
            if missing:
                raise EvernoteError(
                    f"{kind} name(s) not found in the Evernote account: {missing}. A renamed "
                    f"{kind.lower()} must be renamed in the selection too."
                )
            resolved.append(sorted(g for g, name in found.items() if name in names))
        return resolved[0], resolved[1]

    def _restart_as_full(self, listing: dict) -> bool:
        """An incremental pass starts over as full when its names resolve to other guids, or
        when a notebook was expunged: the expunge takes its notes with it, and a note moved
        into it since the last sync is recorded under its old notebook."""
        if listing["full"]:
            return False
        if listing["expunged_notebooks"]:
            logger.info("Evernote notebook expunged: a full sync runs.")
            return True
        notebook_guids, tag_guids = self._resolve(listing)
        scope = self.work.get("scope") or {}
        if notebook_guids == scope.get("notebook_guids") and tag_guids == scope.get("tag_guids"):
            return False
        logger.info("Evernote selection resolves to other notebooks or tags: a full sync runs.")
        return True

    def _reconcile(self, listing: dict) -> Iterator[dict[str, Any]]:
        """Pass 1 is complete: emit the tombstones it proves and keep the notes to fetch."""
        notebook_guids, tag_guids = self._resolve(listing)
        listing["guids"] = {"notebooks": notebook_guids, "tags": tag_guids}

        def selected(note: dict) -> bool:
            if not note["live"]:
                return False
            if notebook_guids is not None and note["notebook"] not in notebook_guids:
                return False
            return tag_guids is None or bool(set(note["tags"]) & set(tag_guids))

        known = self.work.setdefault("note_ids", {})
        notes = listing["notes"]
        keep = {guid for guid, note in notes.items() if selected(note)}
        if listing["full"]:
            gone = set(known) - keep
        else:
            gone = set(listing["expunged"]) | (set(notes) - keep)
        # A note waiting for a retry that this pass lists again, or proves gone, is this pass's
        # to read or forget; a full pass lists every live note, so it settles them all.
        retry = self.work.pop("retry", {})
        left = {} if listing["full"] else {g: n for g, n in retry.items() if g not in notes}
        left = {g: n for g, n in left.items() if g not in gone}
        if left:
            self.work["retry"] = left
        for guid in sorted(gone & set(known)):
            known.pop(guid)
            self.stats["deleted"] += 1
            self.rows_out += 1
            yield _tombstone(guid)
        pending = [{key: notes[guid][key] for key in ("guid", "usn", "notebook")} for guid in keep]
        listing["pending"] = sorted(pending, key=lambda note: (note["usn"], note["guid"]))
        listing["notes"] = {}
        listing["expunged"] = []
        listing["expunged_notebooks"] = []

    def _budget_left(self, reserve: int = 0) -> bool:
        return self.max_notes_per_run is None or self.fetched < self.max_notes_per_run - reserve

    def _tail_reserve(self) -> int:
        """One read of a capped run's budget kept for the retry tail when notes wait there, so
        a run whose new notes fill the cap still retries one. A cap of 1 keeps it for the pass."""
        waiting = any(guid in self.retry_before for guid in self.work.get("retry") or {})
        return 1 if waiting and (self.max_notes_per_run or 0) >= 2 else 0

    def _read(self, guid: str) -> Any:
        """``getNote`` for one note, counted against the run's budget. ``None``: not found."""
        self.fetched += 1
        try:
            found = self._call("get_note", guid, per_note=True)
        except NoteNotFoundError:
            found = None
        self.answered = True
        return found

    def _emit(
        self, guid: str, found: Any, notebook: str | None, names: dict
    ) -> Iterator[dict[str, Any]]:
        """The row of a note as read now (its title, notebook and tags come with the body), or
        its tombstone when it is gone or empty and was emitted before."""
        known = self.work.setdefault("note_ids", {})
        title = (getattr(found, "title", None) or "") if found is not None else ""
        text = enml_to_text(getattr(found, "content", None) or "") or title
        if not text:
            if known.pop(guid, None) is not None:
                self.stats["deleted"] += 1
                self.rows_out += 1
                yield _tombstone(guid)
            return
        current = {
            "guid": guid,
            "title": title,
            "notebook": getattr(found, "notebookGuid", None) or notebook,
            "tags": sorted(getattr(found, "tagGuids", None) or []),
        }
        known[guid] = current["notebook"] or ""
        self.stats["emitted"] += 1
        self.rows_out += 1
        yield note_to_row(current, text, names["notebooks"], names["tags"])

    def _to_retry(self, note: dict) -> None:
        """Keep a note for the retry tail of later runs, counted as a failure of this run."""
        self.work.setdefault("retry", {})[note["guid"]] = note
        self.stats["failed"] += 1
        self.failed_notes += 1

    def _fetch(self, listing: dict) -> Iterator[dict[str, Any]]:
        """Pass 2: read the notes this pass listed, in USN order, dropping each one once done.

        A note Evernote will not hand over goes to ``retry``. An unavailable answer is held
        until the next note is read: if that one fails the same way, it is an outage, and the
        run is cut with both notes in ``retry`` and the rest pending; otherwise the held note
        goes to ``retry``. A held last note with nothing answered in the run is an outage too.
        """
        pending = listing["pending"]
        held: dict | None = None
        reserve = self._tail_reserve()
        try:
            while pending:
                if not self._budget_left(reserve):
                    logger.info(
                        "Evernote: max_notes_per_run=%d reached, %d note(s) left for the next run.",
                        self.max_notes_per_run,
                        len(pending) + (held is not None),
                    )
                    raise _CutShort(None)
                note = pending[0]
                try:
                    found = self._read(note["guid"])
                except EvernoteUnavailableError as exc:
                    if held is not None:
                        # Two in a row: an outage. Both notes wait in retry, so two notes
                        # that always fail cannot keep the pass from completing.
                        self._to_retry(held)
                        self._to_retry(pending.pop(0))
                        held = None
                        raise _CutShort("unavailable", exc) from None
                    held = pending.pop(0)
                    continue
                except EvernoteError:
                    self._to_retry(pending.pop(0))
                    continue
                pending.pop(0)
                if held is not None:
                    self._to_retry(held)
                    held = None
                yield from self._emit(note["guid"], found, note["notebook"], listing)
        except _CutShort:
            if held is not None:
                pending.insert(0, held)
            raise
        if held is not None:
            self._to_retry(held)
            if not self.answered:
                # The only note this run tried answered unavailable: an outage, reported as
                # one. The note waits in retry like a pair would, and the next run completes.
                raise _CutShort("unavailable")

    def _retry_tail(self, names: dict) -> Iterator[dict[str, Any]]:
        """Read the notes that failed before, after the pass is written.

        A failure keeps the note in ``retry`` and counts in ``failed``; two unavailable answers
        in a row end the tail with ``failed_unavailable``. Nothing here holds the pass: new
        notes and edits are read whatever these notes do.
        """
        self.in_tail = True
        retry = self.work.get("retry") or {}
        down = 0
        # Insertion order, and a note that fails again goes to the end, so each run starts with
        # the notes tried least recently: notes that always fail cannot spend the budget of
        # every run ahead of one that would be read.
        attempts = [guid for guid in retry if guid in self.retry_before]
        for guid in attempts:
            if not self._budget_left():
                return
            try:
                found = self._read(guid)
            except EvernoteError as exc:
                retry[guid] = retry.pop(guid)
                self.stats["failed"] += 1
                self.failed_notes += 1
                down = down + 1 if isinstance(exc, EvernoteUnavailableError) else 0
                if down >= 2 or (down and not self.answered and guid == attempts[-1]):
                    # Two in a row, or every note this run tried answered unavailable.
                    self.stats["failed_unavailable"] = 1
                    return
                continue
            down = 0
            note = retry.pop(guid)
            yield from self._emit(guid, found, note["notebook"], names)
        if not retry:
            self.work.pop("retry", None)

    def _finish(self, listing: dict) -> None:
        work = self.work
        work.pop("listing", None)
        work["last_usn"] = listing["usn"]
        if listing["started"] is not None:
            work["last_sync_time"] = int(listing["started"])
        work["notebooks"] = listing["notebooks"]
        work["tags"] = listing["tags"]
        work["scope"] = {
            "notebooks": self.notebook_names,
            "tags": self.tag_names,
            "notebook_guids": listing["guids"]["notebooks"],
            "tag_guids": listing["guids"]["tags"],
        }

    def commit(self) -> None:
        """Write the work copy into dlt state, touching only the keys that changed."""
        for key in [key for key in self.state if key not in self.work]:
            del self.state[key]
        for key, value in self.work.items():
            old = self.state.get(key)
            # Dict equality ignores order, and the order of "retry" is the tail's rotation.
            if old != value or (isinstance(old, dict) and list(old) != list(value)):
                self.state[key] = copy.deepcopy(value)


def _iter_rows(
    service: Any,
    state: dict,
    stats: dict[str, int],
    *,
    notebook_names: list[str] | None = None,
    tag_names: list[str] | None = None,
    sandbox: bool = False,
    full_resync: bool = False,
    max_notes_per_run: int | None = None,
    check_active: Callable[[], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield the changed rows. Pure of dlt, so tests drive it with a dict.

    A run cut by the rate limit, by ``max_notes_per_run`` or by a token rejected after rows were
    emitted keeps its progress. A token rejected before any row raises ``EvernoteAuthError``;
    any other error raises and leaves ``state`` as it was. What ``check_active`` raises
    propagates as it is and counts nothing.
    """
    walker = _Walker(
        service,
        state,
        stats,
        notebook_names=notebook_names,
        tag_names=tag_names,
        sandbox=sandbox,
        full_resync=full_resync,
        max_notes_per_run=max_notes_per_run,
        check_active=check_active,
    )
    try:
        yield from walker.rows()
    except _CutShort as cut:
        key = _CUT_KEYS.get(cut.reason or "")
        if (
            isinstance(cut.error, (EvernoteAuthError, EvernotePermissionError))
            and not walker.rows_out
            and not walker.in_tail
        ):
            # Nothing was emitted, so the caller learns of the refusal from the raise.
            stats["failed"] += 1
            if key:
                stats[key] = 1
            raise cut.error from None
        walker.commit()
        if cut.reason is not None:
            stats["failed"] += 1
            if key:
                stats[key] = 1
            logger.warning(
                "Evernote sync stopped early (%s); the next run resumes where it stopped.",
                _CUT_MESSAGES[cut.reason],
            )
        return
    except EvernoteError:
        stats["failed"] += 1
        raise
    finally:
        if walker.failed_notes:
            # A count only: a guid is not for a log.
            logger.warning(
                "Evernote: %d note(s) could not be fetched; they are tried again next run.",
                walker.failed_notes,
            )
    walker.commit()


def _name_list(names: Any, argument: str) -> list[str] | None:
    """A selection sorted and without repeats; ``None`` or ``[]`` selects the whole account."""
    if names is None:
        return None
    if not isinstance(names, (list, tuple)) or not all(
        isinstance(name, str) and name.strip() for name in names
    ):
        raise ValueError(f"{argument} must be a list or tuple of non-blank name strings")
    return sorted(set(names)) or None


def evernote_source(
    *,
    resource_name: str,
    token: str | None = None,
    sandbox: bool = False,
    notebook_names: list[str] | tuple[str, ...] | None = None,
    tag_names: list[str] | tuple[str, ...] | None = None,
    service: Any = None,
    full_resync: bool = False,
    max_notes_per_run: int | None = None,
    check_active: Callable[[], None] | None = None,
):
    """Return a ``dlt`` resource that yields an Evernote account's notes as documents.

    Args:
        resource_name: Stable name of this connection. It names the staging table and scopes the
            sync state, so two accounts in one dataset must never share one. No default.
        token: Evernote developer token or OAuth 1.0a access token. Never read from the
            environment.
        sandbox: Use ``sandbox.evernote.com`` instead of ``www.evernote.com``.
        notebook_names: Notebooks to sync, by name. ``None`` or empty: every notebook.
        tag_names: Only notes carrying one of these tags, by name. ``None`` or empty: any note.
            A name the account does not have (a renamed notebook or tag) fails the run.
        service: Pre-built client (see :func:`build_evernote_service`), for tests and hosts.
        full_resync: Read the whole account again and forget what is no longer there.
        max_notes_per_run: Note bodies fetched per run at most. ``None`` (default): no limit. A
            run stopped by it, or by Evernote's rate limit, continues on the next run.
        check_active: Optional host authorization checkpoint, called before each request attempt.
            What it raises stops the run as it is.

    Returns:
        A ``dlt`` resource with ``primary_key="id"``, ``write_disposition="merge"`` and a
        ``_deleted`` hard-delete column; ``cognee_sync_stats`` holds the counts of its last run.
    """
    import dlt
    from dlt.common.normalizers.naming.snake_case import NamingConvention

    if getattr(dlt_utils, "DOCUMENT_SYNC_VERSION", 0) < 1:
        raise RuntimeError(
            "Evernote sync requires a Cognee build with document sync "
            "(DOCUMENT_SYNC_VERSION >= 1). Upgrade Cognee before syncing."
        )
    if not isinstance(resource_name, str) or not resource_name.strip():
        raise ValueError("evernote_source needs a resource_name, one per Evernote account")
    table = NamingConvention().normalize_table_identifier(resource_name)
    if table.startswith("_dlt_"):
        raise ValueError("resource_name must not start with _dlt_, which dlt keeps for itself")
    if table != resource_name:
        # dlt names the staging table after the normalized name, while the account bind and
        # the pipeline scope use the name as given: "Work Notes" and "work_notes" would share a
        # table under two binds.
        raise ValueError(f"resource_name must be a dlt table name; use {table!r}")
    if max_notes_per_run is not None and (
        not isinstance(max_notes_per_run, int)
        or isinstance(max_notes_per_run, bool)
        or max_notes_per_run < 1
    ):
        raise ValueError("max_notes_per_run must be a positive int or None")
    notebooks = _name_list(notebook_names, "notebook_names")
    tags = _name_list(tag_names, "tag_names")
    if service is None and not (isinstance(token, str) and token.strip()):
        raise ValueError("evernote_source needs a service or a non-empty token")

    service_sandbox = getattr(service, "sandbox", None)
    if isinstance(service_sandbox, bool) and service_sandbox != sandbox:
        # The account bind names the environment, so it must be the one the client talks to.
        raise ValueError("sandbox differs from the sandbox setting of the given service")
    client = service if service is not None else EvernoteApi(token or "", sandbox=sandbox)
    stats: dict[str, int] = new_stats()

    @dlt.resource(
        name=resource_name,
        primary_key="id",
        write_disposition="merge",
        columns={
            **{
                field: {"data_type": "text", "nullable": False}
                for field in ("id", "title", "content")
            },
            "_deleted": {"data_type": "bool", "hard_delete": True},
        },
    )
    def evernote_notes():
        stats.clear()
        stats.update(new_stats())
        rows = _iter_rows(
            client,
            dlt.current.resource_state(),
            stats,
            notebook_names=notebooks,
            tag_names=tags,
            sandbox=sandbox,
            full_resync=full_resync,
            max_notes_per_run=max_notes_per_run,
            check_active=check_active,
        )
        yield from dlt_utils.guarded_rows(rows, check_active)

    resource = evernote_notes()
    setattr(resource, DOCUMENT_SOURCE_ATTR, "evernote")
    setattr(resource, dlt_utils.PIPELINE_SCOPE_ATTR, resource_name)
    resource.cognee_sync_stats = stats
    return resource
