# Adapted from topoteretes/cognee-community (Apache-2.0), PR #316, commit a154bd4.
# Modified to ship in the Cognee SDK with an httpx client, one source per folder id and SDK-local imports.
"""Dropbox connector for cognee: a ``dlt`` source that syncs one Dropbox folder into memory.

One source is one selected folder, named by its Dropbox folder id (``id:...``), so a renamed
folder keeps its table, cursor and documents. It yields one document row per new or changed
file and a hard-delete tombstone per file that left the folder, and is meant to be handed to
:func:`cognee.remember`::

    await cognee.remember(
        dropbox_source(
            folder_id="id:...",
            account_id="dbid:...",
            token="<OAuth access token>",
            resource_name="dropbox_notes",
        ),
        dataset_name="my_dropbox",
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )

See ``dropbox.md`` for what is read, the incremental model, deletion and its gaps, limits and
privacy.
"""

import codecs
import json
import math
import posixpath
import re
import tempfile
import time
import zlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import date
from typing import Any, Protocol, cast

from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.dlt_utils import (
    DOCUMENT_SOURCE_ATTR,
    PIPELINE_SCOPE_ATTR,
    STRUCTURE_COLUMN,
    STRUCTURE_COLUMN_HINT,
)

logger = get_logger("dropbox_connector")


class TokenSource(Protocol):
    """Where the client takes its token from before every request attempt.

    ``invalidate()`` is optional: a source that has it is told when Dropbox answered 401 for
    the token it gave, and the client then asks it once more and retries that request once.
    A plain zero-argument callable is a ``TokenSource`` without ``invalidate``.
    """

    def __call__(self) -> str: ...


Token = str | TokenSource | Callable[[], str]

_API = "https://api.dropboxapi.com/2/"
_CONTENT = "https://content.dropboxapi.com/2/"
_STATE_VERSION = 1
# How a file becomes a row's text. Bumping it relists every folder whole on its next run and
# re-emits every file under the new rendering: one full re-extraction per synced account.
_RENDER_VERSION = 1

TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".csv"}
PDF_EXTENSION = ".pdf"
# Paper docs are exported as markdown; a file that offers only other formats is skipped.
TEXT_EXPORT_FORMATS = ("markdown",)
# The page size list_folder documents as its maximum ("approximate", Dropbox API reference).
LIST_FOLDER_LIMIT = 2000
# The walk runs under cognee's process-wide dlt staging lock, which blocks every other dlt
# ingestion in the process while it is held, so all retry waits of a run together are bounded
# to two minutes (the value of the community connector, PR #316). A wait that does not fit what
# is left cuts the run with failed_rate_limit, and the next run resumes. A 429 waits its
# Retry-After; a 5xx or a network error waits 1, 2, then 4 seconds.
RUN_WAIT_BUDGET_SECONDS = 120
RETRIES = 3
WAIT_STEP_SECONDS = 1
# A download stays in memory up to this size and spills to a temporary file beyond it.
DOWNLOAD_SPOOL_BYTES = 8 * 1024 * 1024
# Lowercase, digits and single underscores: a name dlt keeps unchanged as a table name, and
# within the 63 characters the provider prefixes allow.
_RESOURCE_NAME = re.compile(r"dropbox(_[a-z0-9]+)+")
_TAG = re.compile(r"[a-z_]{1,64}")

STATS_KEYS = (
    "scanned",
    "unchanged",
    "deleted",
    "skipped",
    "skipped_too_large",
    "skipped_unsupported_type",
    "skipped_unparseable",
    "skipped_empty_content",
    "skipped_not_found",
    "skipped_restricted",
    "failed",
    "failed_auth",
    "failed_account",
    "failed_listing",
    "failed_rate_limit",
    "failed_retry",
    "team_space_unsupported",
    "complete",
)


class DropboxError(RuntimeError):
    """A Dropbox failure. Carries the route and the HTTP status only, never a token, an id, a
    path or a response body. ``tags`` is Dropbox's error tag chain, like ``("path",
    "not_found")``."""

    def __init__(self, message: str, *, status: int = 0, endpoint: str = "", tags=()):
        super().__init__(message)
        self.status = status
        self.endpoint = endpoint
        self.tags = tuple(tags)


class DropboxAuthError(DropboxError):
    """Dropbox refused the access token (HTTP 401), or the token is not a valid header value.

    ``tags`` holds Dropbox's AuthError tag when the 401 carried one (``expired_access_token``,
    ``invalid_access_token``, ``user_suspended``, ...)."""


class DropboxScopeError(DropboxAuthError):
    """The token lacks a scope or a route the connector needs (``missing_scope``,
    ``route_access_denied``). Renewing the token does not help: the account must reconnect
    and grant files.metadata.read, files.content.read and account_info.read."""


# Dropbox's AuthError tags (API reference, "AuthError" union), and what a 401 with each one
# leads to. A token source that can renew (``TokenSource.invalidate``) gets one renewal and
# one retry for a token Dropbox calls expired or invalid, and for a tag it does not list here
# or a 401 without a readable body. The other tags are about the account or the app, not the
# token: no renewal, no retry.
_SCOPE_TAGS = frozenset({"missing_scope", "route_access_denied"})
_ACCOUNT_TAGS = frozenset({"user_suspended", "invalid_select_user", "invalid_select_admin"})


class DropboxRateLimitedError(DropboxError):
    """Dropbox kept answering 429, or asked for a longer wait than the stated bound."""


class DropboxApiError(DropboxError):
    """Any other refusal: an endpoint error (HTTP 409), a 4xx, a 5xx or the network after the
    retries, or a body that is not what the route documents."""


class DropboxAccountMismatchError(DropboxError):
    """The token belongs to another Dropbox account than ``account_id``. Not an expired token:
    a later run fails the same way until the host passes the right account's token."""


class DropboxFolderNotFoundError(DropboxError):
    """The selected folder does not exist or is not a folder, and this source never listed it."""


class _Transient(DropboxError):
    """One failed attempt worth retrying."""

    def __init__(self, *, status: int, endpoint: str, retry_after=None):
        super().__init__(
            f"Dropbox {endpoint} failed: HTTP {status}", status=status, endpoint=endpoint
        )
        self.retry_after = retry_after


def _clean_token(token: Any) -> str:
    value = token.strip() if isinstance(token, str) else ""
    if not value or not value.isascii() or not value.isprintable() or " " in value:
        raise DropboxAuthError("The Dropbox access token is empty or not a valid header value")
    return value


def _error_tags(response: Any) -> tuple[str, ...]:
    """Dropbox's error tag chain (``{"error": {".tag": "path", "path": {".tag": ...}}}``)."""
    try:
        node = response.json().get("error")
    except (ValueError, AttributeError):
        return ()
    tags: list[str] = []
    while isinstance(node, dict) and isinstance(node.get(".tag"), str) and len(tags) < 4:
        tag = node[".tag"]
        if not _TAG.fullmatch(tag):
            break
        tags.append(tag)
        node = node.get(tag)
    return tuple(tags)


class _Budget:
    """Seconds one run may still spend waiting for retries."""

    def __init__(self) -> None:
        self.left: float = RUN_WAIT_BUDGET_SECONDS


class DropboxClient:
    """A small synchronous client for the Dropbox routes the connector uses.

    ``token`` is an OAuth access token or a zero-argument callable that returns a fresh one; it
    is read before every attempt and sent as ``Authorization: Bearer``. ``http`` is an
    ``httpx.Client`` to send through (tests pass one with a mock transport); without it the
    client owns one and ``close()`` closes it.
    """

    def __init__(
        self, token: Token, *, http: Any = None, sleep: Callable[[float], None] = time.sleep
    ):
        import httpx  # lazy: importing the connectors package loads no client library

        if not callable(token):
            token = _clean_token(token)
        self._token = token
        self._http = http if http is not None else httpx.Client(timeout=60)
        self._owned = http is None
        self._sleep = sleep

    def __repr__(self) -> str:
        return "DropboxClient(token=<redacted>)"

    def close(self) -> None:
        if self._owned:
            self._http.close()

    def _headers(self) -> dict[str, str]:
        token = self._token() if callable(self._token) else self._token
        return {"Authorization": f"Bearer {_clean_token(token)}"}

    def rpc(self, endpoint: str, body: Any, *, check_active=None, budget=None) -> dict:
        """The JSON object a ``200`` answer of an RPC route carries."""

        def attempt():
            headers = {**self._headers(), "Content-Type": "application/json"}
            with self._send(endpoint, _API, headers, json.dumps(body).encode()) as response:
                try:
                    result = response.json()
                except ValueError:
                    result = None
            if not isinstance(result, dict):
                raise DropboxApiError(
                    f"Dropbox {endpoint} answered with a malformed body", endpoint=endpoint
                )
            return result

        return self._retried(endpoint, attempt, check_active, budget)

    def download(self, endpoint: str, arg: dict, *, check_active=None, budget=None) -> Any:
        """A rewound spooled temporary file holding the body of a content route; close it."""

        import httpx

        def attempt():
            headers = {**self._headers(), "Dropbox-API-Arg": json.dumps(arg)}
            spool = tempfile.SpooledTemporaryFile(max_size=DOWNLOAD_SPOOL_BYTES)  # noqa: SIM115 - the caller closes it
            broken = False
            try:
                with self._send(endpoint, _CONTENT, headers, b"") as response:
                    for chunk in response.iter_bytes():
                        spool.write(chunk)
            except httpx.TransportError:
                broken = True
            except BaseException:
                spool.close()
                raise
            if broken:
                # Raised here so the httpx error, which names the URL, is not its context.
                spool.close()
                raise _Transient(status=0, endpoint=endpoint)
            spool.seek(0)
            return spool

        return self._retried(endpoint, attempt, check_active, budget)

    @contextmanager
    def _send(self, endpoint: str, base: str, headers: dict, content: bytes):
        """One request; a non-200 answer raises a typed error before the body is read."""
        import httpx

        failure = None
        request = self._http.build_request(
            "POST", base + endpoint, headers=headers, content=content
        )
        try:
            response = self._http.send(request, stream=True, follow_redirects=False)
        except httpx.LocalProtocolError:
            failure = "protocol"
        except httpx.TransportError:
            failure = "transport"
        # Raised outside the except blocks so no httpx error (which names the URL) stays
        # attached as __context__.
        if failure == "protocol":
            raise DropboxAuthError(f"Dropbox {endpoint} could not be sent", endpoint=endpoint)
        if failure == "transport":
            raise _Transient(status=0, endpoint=endpoint)
        try:
            status = response.status_code
            if status == 200:
                yield response
                return
            if status == 429 or status >= 500:
                # A number of seconds; kept for the retry policy, never put in a message.
                retry_after = response.headers.get("Retry-After")
                raise _Transient(status=status, endpoint=endpoint, retry_after=retry_after)
            if status == 401:
                response.read()
                tags = _error_tags(response)
                if tags[:1] and tags[0] in _SCOPE_TAGS:
                    raise DropboxScopeError(
                        f"Dropbox {endpoint} refused the app's access (HTTP 401, {tags[0]}); "
                        "reconnect the account and grant the scopes files.metadata.read, "
                        "files.content.read and account_info.read",
                        status=status,
                        endpoint=endpoint,
                        tags=tags,
                    )
                raise DropboxAuthError(
                    f"Dropbox {endpoint} refused the access token: HTTP 401",
                    status=status,
                    endpoint=endpoint,
                    tags=tags,
                )
            tags = ()
            if status == 409:
                response.read()
                tags = _error_tags(response)
            raise DropboxApiError(
                f"Dropbox {endpoint} failed: HTTP {status}",
                status=status,
                endpoint=endpoint,
                tags=tags,
            )
        finally:
            response.close()

    def _retried(self, endpoint, attempt, check_active, budget):
        budget = budget if budget is not None else _Budget()
        if check_active is not None:
            check_active()

        def delay(error: BaseException, number: int) -> float | None:
            wait = float(2**number)
            if isinstance(error, _Transient) and error.status == 429:
                try:
                    wait = float(error.retry_after or "")
                except (TypeError, ValueError):
                    wait = float(2**number)
                if not math.isfinite(wait) or wait < 0:
                    wait = float(2**number)
            if wait > budget.left:
                return None
            budget.left -= wait
            return wait

        def sleep(seconds: float) -> None:
            # In short steps, so a host that revokes the connection stops a long wait.
            while seconds > 0:
                step = min(seconds, WAIT_STEP_SECONDS)
                self._sleep(step)
                seconds -= step
                if seconds > 0 and check_active is not None:
                    check_active()

        # A token source that can drop its cached token (a host's cache) gets one more try
        # after a 401: the cached token may be stale while the store holds a renewed one.
        renew = getattr(self._token, "invalidate", None)
        renewed = False

        def attempt_with_renewal():
            nonlocal renewed
            try:
                return attempt()
            except DropboxAuthError as error:
                if (
                    renew is None
                    or renewed
                    or error.status != 401
                    or isinstance(error, DropboxScopeError)
                    or (error.tags[:1] and error.tags[0] in _ACCOUNT_TAGS)
                ):
                    raise
            renewed = True
            renew()
            if check_active is not None:
                check_active()
            return attempt()

        failure = None
        try:
            return dlt_utils.with_retries(
                attempt_with_renewal,
                retry_on=(_Transient,),
                delay=delay,
                attempts=RETRIES + 1,
                check_active=check_active,
                sleep=sleep,
            )
        except _Transient as error:
            failure = error
        if failure.status == 429:
            raise DropboxRateLimitedError(
                f"Dropbox {endpoint} is rate limited", status=429, endpoint=endpoint
            ) from None
        raise DropboxApiError(
            f"Dropbox {endpoint} failed after retries: HTTP {failure.status}",
            status=failure.status,
            endpoint=endpoint,
        ) from None


def build_dropbox_service(access_token: str) -> DropboxClient:
    """A client that sends a host-owned OAuth access token as ``Authorization: Bearer``."""
    return DropboxClient(access_token)


def dropbox_account_id(access_token: Token, *, http: Any = None) -> str:
    """The ``dbid:...`` account id of the token's account (``users/get_current_account``)."""
    client = DropboxClient(access_token, http=http)
    try:
        account = client.rpc("users/get_current_account", None)
    finally:
        client.close()
    account_id = account.get("account_id")
    if not isinstance(account_id, str) or not account_id.startswith("dbid:"):
        raise DropboxApiError(
            "Dropbox users/get_current_account answered without an account id",
            endpoint="users/get_current_account",
        )
    return account_id


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------
class _Session:
    """One run's access to Dropbox: the client, the host checkpoint and the wait budget."""

    def __init__(self, client: Any, check_active: Callable[[], None] | None) -> None:
        self.client = client
        self.check_active = check_active
        self.budget = _Budget()

    def rpc(self, endpoint: str, body: Any) -> dict:
        return self.client.rpc(endpoint, body, check_active=self.check_active, budget=self.budget)

    def download(self, endpoint: str, arg: dict) -> Any:
        return self.client.download(
            endpoint, arg, check_active=self.check_active, budget=self.budget
        )


def _entries(page: dict, endpoint: str) -> list[dict]:
    entries = page.get("entries")
    if (
        not isinstance(entries, list)
        or not isinstance(page.get("cursor"), str)
        or not isinstance(page.get("has_more"), bool)
    ):
        raise DropboxApiError(
            f"Dropbox {endpoint} answered without entries and a cursor", endpoint=endpoint
        )
    for entry in entries:
        if not isinstance(entry, dict) or entry.get(".tag") not in ("file", "folder", "deleted"):
            raise DropboxApiError(f"Dropbox {endpoint} listed a malformed entry", endpoint=endpoint)
    return entries


def _drain(session: _Session, page: dict, endpoint: str, seen: set[str]) -> tuple[list, str]:
    """Follow ``has_more`` to the end, refusing a cursor Dropbox already gave."""
    entries = _entries(page, endpoint)
    while page["has_more"]:
        if page["cursor"] in seen:
            raise DropboxApiError(
                "Dropbox returned the same cursor twice", endpoint="files/list_folder/continue"
            )
        seen.add(page["cursor"])
        page = session.rpc("files/list_folder/continue", {"cursor": page["cursor"]})
        entries.extend(_entries(page, "files/list_folder/continue"))
    return entries, page["cursor"]


def _list_folder(session: _Session, folder_id: str) -> tuple[list, str]:
    page = session.rpc(
        "files/list_folder",
        {
            "path": folder_id,
            "recursive": True,
            "limit": LIST_FOLDER_LIMIT,
            "include_non_downloadable_files": True,
        },
    )
    return _drain(session, page, "files/list_folder", set())


def _list_changes(session: _Session, cursor: str) -> tuple[list, str]:
    page = session.rpc("files/list_folder/continue", {"cursor": cursor})
    return _drain(session, page, "files/list_folder/continue", {cursor})


def _resolve_folder(session: _Session, folder_id: str) -> tuple[str, str] | None:
    """The folder's current ``(path_lower, path_display)``, or None when it is gone.

    The whole Dropbox (``""``) has no metadata and is ``("", "")``.
    """
    if not folder_id:
        return "", ""
    try:
        meta = session.rpc("files/get_metadata", {"path": folder_id})
    except DropboxApiError as error:
        if error.tags[:2] in (("path", "not_found"), ("path", "not_folder")):
            return None
        raise
    if meta.get(".tag") != "folder":
        return None
    if not isinstance(meta.get("path_lower"), str) or not isinstance(meta.get("path_display"), str):
        raise DropboxApiError(
            "Dropbox files/get_metadata answered without a path", endpoint="files/get_metadata"
        )
    return meta["path_lower"], meta["path_display"]


def _failure_key(error: DropboxError) -> str:
    """The stats key a run that stops on ``error`` is counted under."""
    if isinstance(error, DropboxAuthError):
        return "failed_auth"
    if isinstance(error, DropboxAccountMismatchError):
        return "failed_account"
    return "failed_listing"


def _check_account(session: _Session, account_id: str, stats: dict[str, int]) -> None:
    """Refuse a token of another account, and report a team space (not supported).

    A team member's token sees only their member folder without ``Dropbox-API-Path-Root``,
    which this connector does not send: a team account's root namespace differs from its
    home namespace (``root_info``, users/get_current_account).
    """
    account = session.rpc("users/get_current_account", None)
    if account.get("account_id") != account_id:
        raise DropboxAccountMismatchError(
            "The access token belongs to another Dropbox account than account_id",
            endpoint="users/get_current_account",
        )
    root_info = account.get("root_info")
    if isinstance(root_info, dict) and root_info.get("root_namespace_id") != root_info.get(
        "home_namespace_id"
    ):
        stats["team_space_unsupported"] = 1
        logger.warning(
            "Dropbox: the account is in a team space, which is not supported; only its "
            "member folder is synced."
        )


def _is_under(path: str, folder: str) -> bool:
    return folder == "" or path == folder or path.startswith(folder + "/")


def _ancestors(entry: dict, root: tuple[str, str], folder_id: str, folders: dict) -> list[dict]:
    """The file's folders, nearest first, up to and including the selected folder."""
    lower_parts = entry["path_lower"].split("/")[:-1]
    display_parts = entry["path_display"].split("/")[:-1]
    if len(lower_parts) != len(display_parts):
        display_parts = lower_parts
    root_lower = root[0]
    ancestors = []
    for depth in range(len(lower_parts), 1, -1):
        path = "/".join(lower_parts[:depth])
        if path == root_lower:
            break
        if path not in folders:
            raise DropboxApiError(
                "Dropbox listed a file before its folder", endpoint="files/list_folder"
            )
        ancestors.append({"kind": "folder", "id": folders[path], "name": display_parts[depth - 1]})
    if root_lower:
        ancestors.append(
            {"kind": "folder", "id": folder_id, "name": posixpath.basename(root[1]) or root[1]}
        )
    return ancestors


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------
_RETRY = object()
# What pypdf (and the decoder) raise on a malformed file. The same bytes fail the same way
# every run, so the file is skipped until it changes.
_PARSE_ERRORS = (
    ValueError,
    KeyError,
    TypeError,
    AttributeError,
    IndexError,
    AssertionError,
    RecursionError,
    zlib.error,
)
_UNSUPPORTED_TAGS = {"unsupported_file", "non_exportable", "invalid_export_format"}
# Download lookup errors that a retry cannot cure: content Dropbox blocks (legal or policy
# restriction) and a path that is no longer a file.
_RESTRICTED_LOOKUPS = {("path", "restricted_content"), ("path", "not_file")}


def _count_skip(stats: dict[str, int], reason: str) -> None:
    stats["skipped"] += 1
    stats[f"skipped_{reason}"] += 1


def _decode(stream: Any) -> str:
    """Text of a .txt/.md/.csv file: UTF-16 with a BOM, else UTF-8. NUL bytes mean binary."""
    data = stream.read()
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        text = data.decode("utf-16")
    else:
        text = data.decode("utf-8-sig", errors="replace")
    if "\x00" in text:
        raise ValueError("binary content")
    return text


def _extract_pdf_text(stream: Any) -> str:
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError

    try:
        reader = PdfReader(stream, strict=False)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except PyPdfError as error:
        raise ValueError("unreadable PDF") from error
    # NUL bytes in a text layer break a Postgres staging load.
    return text.replace("\x00", "")


def _modified_day(entry: dict) -> str | None:
    """``YYYY-MM-DD`` of the file's last edit, from the listing entry (UTC, as Dropbox sends).

    ``client_modified`` is when the content was last changed by its author; a Paper export
    has no client file, so it and an entry without one use ``server_modified``. Day
    precision: a second save on the same day leaves the text, and so the document, as it was.
    """
    keys = ("client_modified", "server_modified")
    if not entry["is_downloadable"]:
        keys = ("server_modified",)
    for key in keys:
        value = entry.get(key)
        if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}T[0-9:.]+Z", value):
            try:
                date.fromisoformat(value[:10])
            except ValueError:
                # An impossible day would be stated as a fact; no line instead.
                return None
            return value[:10]
    return None


def _render(entry: dict, text: str) -> str:
    day = _modified_day(entry)
    return f"Modified: {day}\n\n{text}" if day else text


def _export_format(entry: dict) -> str | None:
    info = entry.get("export_info")
    if not isinstance(info, dict):
        return None
    offered = [info.get("export_as"), *(info.get("export_options") or [])]
    return next((fmt for fmt in TEXT_EXPORT_FORMATS if fmt in offered), None)


def _file_to_row(
    session: _Session, entry: dict, ancestors: list[dict], max_bytes: float | None, stats
) -> Any:
    """A row, None when the file is skipped, or ``_RETRY`` for a failure worth retrying."""
    stats["scanned"] += 1
    size = entry.get("size")
    # Checked first, so an exported Paper doc is held to the limit too and never downloaded.
    if max_bytes is not None and isinstance(size, int) and size > max_bytes:
        _count_skip(stats, "too_large")
        return None
    if entry["is_downloadable"]:
        extension = posixpath.splitext(entry["name"])[1].lower()
        if extension in TEXT_EXTENSIONS:
            parse = _decode
        elif extension == PDF_EXTENSION:
            parse = _extract_pdf_text
        else:
            _count_skip(stats, "unsupported_type")
            return None
        endpoint, arg = "files/download", {"path": entry["id"]}
    else:
        export_format = _export_format(entry)
        if export_format is None:
            _count_skip(stats, "unsupported_type")
            return None
        parse = _decode
        endpoint, arg = "files/export", {"path": entry["id"], "export_format": export_format}

    try:
        stream = session.download(endpoint, arg)
    except (DropboxAuthError, DropboxRateLimitedError):
        raise
    except DropboxApiError as error:
        if error.tags[:2] == ("path", "not_found"):
            # A direct check of the record: the file is gone. A stored copy is tombstoned in
            # this run (the caller forgets a stored file that yields no row); the next
            # listing reports the delete as well.
            _count_skip(stats, "not_found")
            return None
        if error.tags[:2] in _RESTRICTED_LOOKUPS:
            _count_skip(stats, "restricted")
            return None
        if error.tags[:1] and error.tags[0] in _UNSUPPORTED_TAGS:
            _count_skip(stats, "unsupported_type")
            return None
        return _RETRY
    with stream:
        try:
            text = parse(stream)
        except _PARSE_ERRORS:
            _count_skip(stats, "unparseable")
            return None
    if not text.strip():
        _count_skip(stats, "empty_content")
        return None
    return {
        "id": entry["id"],
        "title": entry["name"],
        # Where the file sits is in the structure column, which is not hashed, so moving the
        # file or renaming a folder above it extracts nothing again. The title is hashed, so
        # renaming the file itself extracts it once more. The date is in the text, at day
        # precision.
        "content": _render(entry, text),
        # Dropbox has no web link for a file without creating a shared link, which is a write.
        "url": None,
        STRUCTURE_COLUMN: {"ancestors": ancestors},
        "_deleted": False,
    }


def _valid_file(entry: dict) -> bool:
    return isinstance(entry.get("is_downloadable"), bool) and all(
        isinstance(entry.get(key), str) and entry[key]
        for key in ("id", "name", "path_lower", "path_display")
    )


def valid_folder_id(value: object) -> bool:
    """A Dropbox folder id: ``id:`` followed by at least one character and no whitespace.

    Shared by ``dropbox_source`` and the integration module's selection check, so a selection
    the picker stores is one the source accepts.
    """
    return (
        isinstance(value, str)
        and value.startswith("id:")
        and len(value) > 3
        and not any(char.isspace() for char in value)
    )


# ---------------------------------------------------------------------------
# Sync state machine
# ---------------------------------------------------------------------------
def _iter_rows(
    session: _Session,
    *,
    folder_id: str,
    account_id: str,
    state: dict[str, Any],
    stats: dict[str, int],
    max_bytes: float | None,
) -> Iterator[dict[str, Any]]:
    """Yield one row per new or changed file of the folder, plus hard-delete tombstones.

    ``state`` keys: ``v``, ``account`` and ``folder`` (what the state belongs to), ``path``
    (the folder's path at the last listing), ``cursor``, ``index`` (file ``path_lower`` to id,
    since a delete carries only a path), ``folders`` (folder ``path_lower`` to id, for the
    structure column), ``stored`` (ids that reached cognee) and ``retry`` (ids whose last
    fetch failed).
    Nothing is written to ``state`` until the listing is complete.
    """
    if state.get("account", account_id) != account_id:
        raise ValueError("This resource_name already holds another Dropbox account's sync")
    if state.get("folder", folder_id) != folder_id:
        raise ValueError("This resource_name already holds another Dropbox folder's sync")

    old_index: dict[str, str] = dict(state.get("index", {}))
    index = dict(old_index)
    folders: dict[str, str] = dict(state.get("folders", {}))
    stored: set[str] = set(state.get("stored", []))
    retry: set[str] = set(state.get("retry", []))
    removed: set[str] = set()
    changed: dict[str, dict] = {}

    def forget(path: str) -> None:
        for key in [p for p in index if _is_under(p, path)]:
            removed.add(index.pop(key))
        for key in [p for p in folders if _is_under(p, path)]:
            del folders[key]

    def apply(entry: dict) -> None:
        path = entry.get("path_lower")
        if not isinstance(path, str):
            # Unmounted entries carry no path; there is nothing to index.
            logger.warning("Dropbox: skipped a change without a path.")
            return
        tag = entry[".tag"]
        if tag == "file":
            if not _valid_file(entry):
                raise DropboxApiError(
                    "Dropbox listed a malformed file", endpoint="files/list_folder"
                )
            previous = index.get(path)
            if previous is not None and previous != entry["id"]:
                # A new file written over the path of a deleted one.
                removed.add(previous)
            index[path] = entry["id"]
            changed[entry["id"]] = entry
        elif tag == "deleted":
            forget(path)
        else:
            if path in index:
                # A folder created where a file used to be.
                removed.add(index.pop(path))
            if isinstance(entry.get("id"), str):
                folders[path] = entry["id"]

    try:
        _check_account(session, account_id, stats)
        root = _resolve_folder(session, folder_id)
        cursor = state.get("cursor")
        if root is None:
            if cursor is None:
                # Never listed: most likely a wrong id, which would otherwise be an empty table.
                raise DropboxFolderNotFoundError(
                    "The selected Dropbox folder was not found or is not a folder; check folder_id",
                    endpoint="files/get_metadata",
                )
            # Listed before and now deleted, or out of this account's reach: forget it.
            logger.warning("Dropbox: the selected folder is gone; forgetting its files.")
            forget("")
            new_cursor = None
        else:
            # A rendering change lists the folder whole, so every file is re-emitted under it.
            full = (
                cursor is None
                or root[0] != state.get("path")
                or state.get("render") != _RENDER_VERSION
            )
            if not full:
                try:
                    entries, new_cursor = _list_changes(session, cursor)
                except DropboxApiError as error:
                    if error.tags[:1] not in (("reset",), ("path",)):
                        raise
                    logger.info("Dropbox: the folder cursor is no longer valid; listing again.")
                    full = True
            if full:
                # First run, a reset cursor or a renamed folder: list it whole and reconcile
                # against the index like a snapshot.
                entries, new_cursor = _list_folder(session, folder_id)
                forget("")
            for entry in entries:
                apply(entry)
            if not all(_is_under(entry["path_lower"], root[0]) for entry in changed.values()):
                # The folder was renamed or moved between files/get_metadata and the listing.
                # Dropping those files would leave them unsynced, so the run stops before
                # the state changes and the next one resolves the new path.
                raise DropboxApiError(
                    "The selected Dropbox folder moved while it was listed; sync again",
                    endpoint="files/list_folder",
                )
            work_ancestors = {
                file_id: _ancestors(entry, root, folder_id, folders)
                for file_id, entry in changed.items()
            }
    except DropboxRateLimitedError:
        stats["failed"] += 1
        stats["failed_rate_limit"] += 1
        logger.warning("Dropbox: rate limited while listing; the next run lists again.")
        return
    except DropboxError as error:
        # One failure, one reason key, so a host that classifies the raised error the same
        # way (``dropbox/sync.py``) does not count it a second time.
        stats["failed"] += 1
        stats[_failure_key(error)] += 1
        raise

    rows_stored = 0
    present = set(index.values())

    def tombstone(file_id: str) -> dict[str, Any]:
        if file_id in stored:
            stored.discard(file_id)
            stats["deleted"] += 1
        return {"id": file_id, "_deleted": True}

    # A tombstone for an id never stored is harmless, and after a failed run the id may have
    # been stored without the state being saved, so it is sent anyway.
    for file_id in sorted(removed - present):
        yield tombstone(file_id)

    work: list[tuple[str, dict | None]] = []
    if root is not None:
        work += sorted(changed.items())
        work += [(file_id, None) for file_id in sorted(retry) if file_id not in changed]
        work = [(file_id, entry) for file_id, entry in work if file_id in present]

    next_retry: set[str] = set()
    cut = None
    scope = root or ("", "")  # work is empty when the folder is gone
    row: Any = None
    for position, (file_id, entry) in enumerate(work):
        try:
            if entry is None:
                entry = _retry_metadata(session, file_id, scope[0])
                if entry is None:
                    continue
                ancestors = _ancestors(entry, scope, folder_id, folders)
            else:
                ancestors = work_ancestors[file_id]
            row = _file_to_row(session, entry, ancestors, max_bytes, stats)
        except (DropboxAuthError, DropboxRateLimitedError) as error:
            if isinstance(error, DropboxAuthError) and not (rows_stored or stats["deleted"]):
                # Nothing done yet: a bad token, said as such.
                stats["failed"] += 1
                stats["failed_auth"] += 1
                raise
            # Every later request would fail the same way. Keep what was stored and fetch the
            # rest next run.
            next_retry.update(fid for fid, _ in work[position:])
            cut = "auth" if isinstance(error, DropboxAuthError) else "rate_limit"
            break
        except DropboxApiError:
            # Only the metadata read of a retried file gets here.
            row = _RETRY
        if row is _RETRY:
            # Retried every run until it is fetched or Dropbox reports it changed or gone, and
            # counted every run, so the connection stays degraded while it waits. Its stored
            # text stays: a failure is no evidence the file changed or went away.
            stats["failed"] += 1
            stats["failed_retry"] += 1
            next_retry.add(file_id)
        elif row is not None:
            stored.add(file_id)
            rows_stored += 1
            yield cast(dict, row)
        elif file_id in stored:
            # Stored before and skipped now (emptied, too large, unsupported): its old text
            # must not stay in memory.
            yield tombstone(file_id)

    if cut is not None:
        stats["failed"] += 1
        stats[f"failed_{cut}"] += 1
        logger.warning("Dropbox: sync stopped early (%s); it resumes on the next run.", cut)

    # The cursor moves on; failed files wait by id in ``retry``.
    state["v"] = _STATE_VERSION
    state["render"] = _RENDER_VERSION
    state["account"] = account_id
    state["folder"] = folder_id
    state["index"] = index
    state["folders"] = folders
    state["stored"] = sorted(stored)
    state["retry"] = sorted(next_retry)
    if root is None:
        for key in ("cursor", "path"):
            state.pop(key, None)
    else:
        state["cursor"] = new_cursor
        state["path"] = root[0]
    stats["unchanged"] = len(present) - len(changed)
    stats["complete"] = int(cut is None)
    logger.info("Dropbox: sync stats %s", stats)


def _retry_metadata(session: _Session, file_id: str, root_lower: str) -> dict | None:
    """Current metadata of a file kept for retry, or None once it is gone or moved away."""
    try:
        meta = session.rpc("files/get_metadata", {"path": file_id})
    except DropboxApiError as error:
        if error.tags[:2] == ("path", "not_found"):
            return None
        raise
    if meta.get(".tag") != "file" or not _valid_file(meta) or meta["id"] != file_id:
        return None
    return meta if _is_under(meta["path_lower"], root_lower) else None


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------
def dropbox_source(
    *,
    folder_id: str,
    account_id: str,
    resource_name: str,
    token: Token | None = None,
    check_active: Callable[[], None] | None = None,
    service: Any = None,
    max_file_size_mb: float | None = None,
):
    """Return a ``dlt`` merge resource that syncs one Dropbox folder.

    Args:
        folder_id: The folder's Dropbox id (``id:...``), or ``""`` for the whole Dropbox. A
            path is refused: a renamed folder would become a new source. A folder this source
            never listed that does not exist raises ``DropboxFolderNotFoundError``; one listed
            before and gone now is forgotten like a delete.
        account_id: The Dropbox account id (``dbid:...``) the token belongs to. The state is
            bound to it, and another account on the same ``resource_name`` raises.
        resource_name: Required: ``dropbox_`` followed by lowercase letters, digits and single
            underscores. It names the staging table and the sync state, so every folder of
            every account needs its own.
        token: OAuth access token, or a ``TokenSource`` (a zero-argument callable returning the
            token, called before every request attempt, optionally with ``invalidate()`` for
            a renewal and one retry after a 401). Give either this or ``service``.
        check_active: Host checkpoint, called before every request attempt and every wait, and
            around every row. What it raises stops the run as it is.
        service: A ``DropboxClient`` (see :func:`build_dropbox_service`); never closed here.
        max_file_size_mb: Leave out files larger than this. Default: no limit. A file over it
            is counted in ``skipped_too_large`` and forgotten if it was stored.

    ``cognee_sync_stats`` holds counts only, every key from the start of a run. A run is a
    finished sync when ``failed == 0`` and ``complete == 1``.
    """
    import dlt

    if dlt_utils.DOCUMENT_SYNC_VERSION < 3:
        raise RuntimeError(
            "Dropbox sync requires a Cognee build that reads per-row structure "
            "(DOCUMENT_SYNC_VERSION >= 3). Upgrade Cognee before syncing."
        )
    if (
        not isinstance(resource_name, str)
        or len(resource_name) > 63
        or not _RESOURCE_NAME.fullmatch(resource_name)
    ):
        raise ValueError(
            "resource_name must start with dropbox_ and hold only lowercase letters, digits "
            'and single underscores, at most 63 characters, like "dropbox_notes".'
        )
    if not isinstance(account_id, str) or not account_id.strip():
        raise ValueError("dropbox_source needs the Dropbox account_id the token belongs to")
    if folder_id != "" and not valid_folder_id(folder_id):
        raise ValueError(
            'folder_id must be a Dropbox folder id like "id:...", or "" for the whole Dropbox; '
            "a path is not accepted"
        )
    if (token is None) == (service is None):
        raise ValueError("Give dropbox_source exactly one of token and service")
    if max_file_size_mb is not None and (
        isinstance(max_file_size_mb, bool)
        or not isinstance(max_file_size_mb, (int, float))
        or not math.isfinite(max_file_size_mb)
        or max_file_size_mb <= 0
    ):
        # 0 or less would skip every file and forget everything stored.
        raise ValueError("max_file_size_mb must be a number above 0, or None for no limit")
    max_bytes = None if max_file_size_mb is None else max_file_size_mb * 1024 * 1024
    if token is not None and not callable(token):
        # Fails at construction, not halfway through a pipeline run.
        _clean_token(token)
    stats: dict[str, int] = dict.fromkeys(STATS_KEYS, 0)
    run_token: Token = token if token is not None else ""

    @dlt.resource(
        name=resource_name,
        primary_key="id",
        write_disposition="merge",
        columns={
            **STRUCTURE_COLUMN_HINT,
            "_deleted": {"data_type": "bool", "hard_delete": True},
        },
    )
    def dropbox_files():
        stats.update(dict.fromkeys(STATS_KEYS, 0))
        client = service if service is not None else DropboxClient(run_token)
        try:
            rows = _iter_rows(
                _Session(client, check_active),
                folder_id=folder_id,
                account_id=account_id,
                state=dlt.current.resource_state(),
                stats=stats,
                max_bytes=max_bytes,
            )
            yield from dlt_utils.guarded_rows(rows, check_active)
        finally:
            if service is None:
                client.close()

    resource = dropbox_files()
    setattr(resource, DOCUMENT_SOURCE_ATTR, "dropbox")
    setattr(resource, PIPELINE_SCOPE_ATTR, resource_name)
    resource.cognee_sync_stats = stats
    return resource
