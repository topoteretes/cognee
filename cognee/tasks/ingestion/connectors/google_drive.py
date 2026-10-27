# Adapted from topoteretes/cognee-community (Apache-2.0), commit 21d0b36.
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""Google Drive connector for cognee — a ``dlt`` source that turns a Drive folder into memory.

Sync a Google Drive folder (Docs, Sheets, PDFs, and plain-text files) into
cognee, incrementally and with forget-on-deletion.  Built entirely on the
existing DLT ingestion subsystem; the resource produced here is handed directly
to :func:`cognee.remember`::

    import cognee
    from cognee.tasks.ingestion.connectors import google_drive_source

    await cognee.remember(
        google_drive_source(folder_id="<folder id from the Drive URL>"),
        dataset_name="my_drive_folder",
        primary_key="id",
        write_disposition="merge",   # incremental upsert by file id
        max_rows_per_table=0,        # 0 = no row cap (folders often exceed the default 50)
    )

Design
------
* **Auth**: service account (default) or OAuth installed-app, read-only scope.
* **Rows**: one per Drive file, keyed by the file ``id``, upserted with ``merge``.
* **Cursor**: the first run captures a Changes API token and lists the folder; later runs
  read the changes feed and emit only added/changed files plus ``_deleted`` tombstones.
* **State**: ``versions`` (file id to ``modifiedTime``, kept only for files that produced a
  row) lets a rerun fetch just what failed or changed. A run stopped by a rate limit or an
  expired token keeps its progress and resumes.
* **Content**: see ``google_drive.md`` for the formats, the limits and every reason a file
  is not ingested.

Limitations
-----------
* The Drive Changes API is account-wide; deleted-file events carry no metadata
  and cannot be scope-filtered, so an incremental run may emit harmless no-op
  delete rows for files removed outside the configured folder.
* ``auth_mode="oauth"`` uses an interactive browser flow on first run.  For
  headless / CI use, pre-authorize a token file and point ``token_path`` at it.
"""

import asyncio
import codecs
import io
import os
import tempfile
import threading
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR

logger = get_logger("google_drive_connector")

# Read-only access — the connector never modifies Drive.
DRIVE_READONLY_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

GOOGLE_DOC_MIME_TYPE = "application/vnd.google-apps.document"
GOOGLE_SHEET_MIME_TYPE = "application/vnd.google-apps.spreadsheet"
GOOGLE_FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
PDF_MIME_TYPE = "application/pdf"
GOOGLE_SLIDES_MIME_TYPE = "application/vnd.google-apps.presentation"
# Read as UTF-8 text: every ``text/*`` type except those cognee's loaders parse (HTML), and
# the structured text types below.
PLAIN_TEXT_MIME_TYPES = {"application/json", "application/xml"}

# Office and document files cognee's loader engine converts to text (the same loaders
# ``add()`` uses), with the extension the loader picks its parser by.
_LOADER_EXTENSIONS = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/vnd.oasis.opendocument.text": ".odt",
    "application/vnd.oasis.opendocument.spreadsheet": ".ods",
    "application/vnd.oasis.opendocument.presentation": ".odp",
    "application/rtf": ".rtf",
    "text/html": ".html",
    "application/epub+zip": ".epub",
}

# Native Google formats we know how to export as text.  Other native types
# (Drawings, Forms, ...) are counted as unsupported rather than failing the whole sync.
_EXPORT_MIME_MAP = {
    GOOGLE_DOC_MIME_TYPE: "text/plain",
    # Drive's export only returns the first sheet as CSV — a documented
    # limitation; multi-sheet extraction would require the Sheets API.
    GOOGLE_SHEET_MIME_TYPE: "text/csv",
    GOOGLE_SLIDES_MIME_TYPE: "text/plain",
}

# googleapiclient retries 429, 5xx and rate-limit 403 answers up to five times, sleeping a random
# time of up to 2**n seconds before retry n: about 31 s in expectation and 62 s at most before
# a rate limit cuts the run short.
API_RETRIES = 5
# files.list returns at most 1000 entries per page.
LIST_PAGE_SIZE = 1000
# A handful of threads: each fetch is one or two requests, and a rate limit that outlasts the
# retries cuts the run short instead of failing it, so the count does not need to be exact.
DEFAULT_MAX_WORKERS = 8
# Files fetched at once are also bounded by their reported size, because each result holds
# its whole extracted text, and the staging load needs about twelve times that in memory
# (measured 12 to 17 times with 25 to 300 MB of text). 256 MB of files in flight is then 3 to
# 4 GB of staging memory at the worst. A file over the bound is not skipped: it is fetched
# alone, and the pool refills behind it.
INFLIGHT_FETCH_BYTES = 256 * 1024 * 1024
# A file that produced no row for one of these reasons is forgotten if an earlier run
# ingested it. ``failed_loader_unavailable`` is not here: it depends on the environment (a
# missing extra), and a file must not lose its memory because a package is not installed.
_FORGET_REASONS = (
    "skipped_too_large",
    "skipped_empty_content",
    "skipped_unsupported_type",
    "failed_export_too_large",
    "failed_unreadable",
)
# A download stays in memory up to this size and spills to a temporary file beyond it, so
# the memory of a fetch depends on the extracted text, not on the size of the file. Most
# Docs, PDFs and text files are smaller, so they never touch the disk. The chunk size is
# the unit of retry.
DOWNLOAD_SPOOL_BYTES = 8 * 1024 * 1024
DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024

FILE_FIELDS = "id, name, mimeType, modifiedTime, webViewLink, parents, driveId, trashed, size"


@dataclass(frozen=True)
class _DriveConfig:
    folder_id: str
    auth_mode: str
    credentials_path: str | None
    token_path: str | None
    include_subfolders: bool
    max_file_size_mb: int | None = None
    shared_drive_id: str | None = None
    max_workers: int = DEFAULT_MAX_WORKERS


# ---------------------------------------------------------------------------
# Auth / service construction
# ---------------------------------------------------------------------------
def load_drive_credentials(
    *,
    auth_mode: str = "service_account",
    credentials_path: str | None = None,
    token_path: str | None = None,
) -> Any:
    """Load Google credentials for the Drive API (service account or OAuth user token).

    The Google client libraries are imported lazily so they remain an optional
    dependency (``pip install "cognee[google-drive]"``).
    """
    try:
        from google.auth.transport.requests import Request
        from google.oauth2 import service_account
        from google.oauth2.credentials import Credentials as UserCredentials
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            'The Google Drive connector requires the "google-drive" extra. '
            'Install it with: pip install "cognee[google-drive]"'
        ) from exc

    if auth_mode == "service_account":
        if not credentials_path or not os.path.exists(credentials_path):
            raise FileNotFoundError(
                f"Google Drive service-account key not found at {credentials_path!r}. "
                "Set GOOGLE_DRIVE_CREDENTIALS_PATH to a valid service-account JSON key file."
            )
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path, scopes=DRIVE_READONLY_SCOPES
        )
    elif auth_mode == "oauth":
        credentials = _load_oauth_credentials(
            UserCredentials, InstalledAppFlow, Request, credentials_path, token_path
        )
    else:
        raise ValueError(
            f"Unsupported Google Drive auth_mode: {auth_mode!r}. "
            "Must be 'service_account' or 'oauth'."
        )

    return credentials


def _build_client(credentials: Any) -> Any:
    try:
        from googleapiclient.discovery import build
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            'The Google Drive connector requires the "google-drive" extra. '
            'Install it with: pip install "cognee[google-drive]"'
        ) from exc
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def build_drive_service(
    *,
    auth_mode: str = "service_account",
    credentials_path: str | None = None,
    token_path: str | None = None,
) -> Any:
    """Build an authenticated Drive v3 API client."""
    return _build_client(
        load_drive_credentials(
            auth_mode=auth_mode, credentials_path=credentials_path, token_path=token_path
        )
    )


def build_drive_service_from_access_token(access_token: str) -> Any:
    """Build a Drive client from a short-lived token supplied by a host.

    Cloud hosts keep refresh tokens in their control plane and pass only an
    access token to the ingestion worker. This helper deliberately does not
    persist the token or attempt an interactive OAuth flow.
    """
    try:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError('The Google Drive connector requires the "google-drive" extra.') from exc
    credentials = Credentials(token=access_token, scopes=DRIVE_READONLY_SCOPES)
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


# The first three parameters are injected google-auth classes (dependency
# injection for testability); classmethods are called on them below.
def _load_oauth_credentials(
    user_credentials, installed_app_flow, request, client_secret_path, token_path
):
    creds = None
    if token_path and os.path.exists(token_path):
        creds = user_credentials.from_authorized_user_file(token_path, DRIVE_READONLY_SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(request())
        else:
            if not client_secret_path or not os.path.exists(client_secret_path):
                raise FileNotFoundError(
                    f"Google Drive OAuth client secret not found at {client_secret_path!r}. "
                    "Set GOOGLE_DRIVE_CREDENTIALS_PATH to an OAuth client-secret JSON file "
                    "(Desktop app) downloaded from the Google Cloud Console."
                )
            # Interactive: opens a browser on first run. For headless / CI use,
            # pre-authorize a token file and point token_path at it.
            flow = installed_app_flow.from_client_secrets_file(
                client_secret_path, DRIVE_READONLY_SCOPES
            )
            creds = flow.run_local_server(port=0)
        if token_path:
            # The cached token is a credential — write it private (0600).
            with open(os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
                f.write(creds.to_json())

    return creds


# ---------------------------------------------------------------------------
# Content extraction (mime-type dispatch)
# ---------------------------------------------------------------------------
def _is_plain_text(mime_type: str) -> bool:
    return mime_type.startswith("text/") or mime_type in PLAIN_TEXT_MIME_TYPES


def is_supported_mime_type(mime_type: str) -> bool:
    return (
        mime_type in _EXPORT_MIME_MAP
        or mime_type == PDF_MIME_TYPE
        or mime_type in _LOADER_EXTENSIONS
        or _is_plain_text(mime_type)
    )


class DriveAuthError(Exception):
    """Drive rejected the access token (HTTP 401) before the run produced anything."""


class _CutShort(Exception):
    """Internal: stop the run here, keeping the state reached so far."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


_RATE_LIMIT_REASONS = (
    "rateLimitExceeded",
    "userRateLimitExceeded",
    "dailyLimitExceeded",
    "sharingRateLimitExceeded",
)


def _cut_reason(exc: Exception) -> str | None:
    """``rate_limit`` or ``auth`` for a Drive error that outlasted the retries, else None."""
    status = getattr(getattr(exc, "resp", None), "status", None)
    if status == 429:
        return "rate_limit"
    if status == 401:
        return "auth"
    if status == 403 and any(reason in str(exc) for reason in _RATE_LIMIT_REASONS):
        return "rate_limit"
    return None


def _raise_if_cut(exc: Exception) -> None:
    reason = _cut_reason(exc)
    if reason:
        raise _CutShort(reason) from exc


def _record_cut(stats: dict[str, int], cut: _CutShort, *, emitted: bool) -> None:
    """Report a run that stopped early; it resumes on the next run.

    A token refused before anything came out is raised instead: the caller must reconnect.
    """
    if cut.reason == "auth" and not emitted:
        raise DriveAuthError("Google Drive rejected the access token (HTTP 401).") from cut
    stats["failed"] += 1
    stats[f"failed_{cut.reason}"] = 1
    logger.warning("Google Drive sync stopped early (%s); it resumes on the next run.", cut.reason)


class ExportTooLargeError(Exception):
    """Drive refused to export a Google Doc or Sheet because it is over Google's export limit."""


def _is_export_too_large(exc: Exception) -> bool:
    return getattr(getattr(exc, "resp", None), "status", None) == 403 and (
        "exportSizeLimitExceeded" in str(exc)
    )


def _stream_media(request: Any, buffer: Any) -> None:
    """Download ``request`` into ``buffer`` chunk by chunk, retrying each chunk."""
    from googleapiclient.http import MediaIoBaseDownload

    downloader = MediaIoBaseDownload(buffer, request, chunksize=DOWNLOAD_CHUNK_BYTES)
    done = False
    while not done:
        _, done = downloader.next_chunk(num_retries=API_RETRIES)


@contextmanager
def _download_media(service: Any, file_id: str):
    """Yield the file's bytes as a rewound spooled temporary file, closed on exit."""
    with tempfile.SpooledTemporaryFile(max_size=DOWNLOAD_SPOOL_BYTES) as buffer:
        _stream_media(service.files().get_media(fileId=file_id, supportsAllDrives=True), buffer)
        buffer.seek(0)
        yield buffer


class LoaderUnavailableError(Exception):
    """No cognee loader is installed for this file's format (it needs an optional extra)."""


class UnreadableFileError(Exception):
    """pypdf could not read the downloaded PDF (password protected, corrupt, or a feature it lacks).

    The same bytes give the same answer, so a rerun cannot cure it until the file changes.
    """


# Failures a rerun in the same environment cannot cure, so they must not hold the Changes
# cursor back. (A missing package is cured by installing it, and the file is then read when
# it next changes or when the source's state is cleared.)
_PERMANENT_REASONS = {
    ExportTooLargeError: "failed_export_too_large",
    LoaderUnavailableError: "failed_loader_unavailable",
    UnreadableFileError: "failed_unreadable",
}
_PERMANENT_FAILURES = tuple(_PERMANENT_REASONS.values())


def _read_pdf(media) -> str:
    """pypdf works on bytes already downloaded, with no network and no state between calls, so
    what it raises it raises again. What may pass is the environment: memory, the disk under the
    spool, and ``cryptography`` for an AES file (pypdf's ``DependencyError``)."""
    from pypdf.errors import DependencyError

    try:
        return _extract_pdf_text(media)
    except (MemoryError, OSError, DependencyError):
        raise
    except Exception as exc:
        raise UnreadableFileError(f"{type(exc).__name__}: {exc}") from exc


@contextmanager
def _download_file(service: Any, file_id: str, suffix: str):
    """Yield the path of a temporary file holding the download, removed on exit.

    The extension is one of the signals the loader engine picks its parser by, with the
    file's content.
    """
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "drive_file" + suffix)
        with open(path, "wb") as target:
            _stream_media(service.files().get_media(fileId=file_id, supportsAllDrives=True), target)
        yield path


def _load_with_loader_engine(path: str) -> str:
    """Convert a downloaded file to text with the loader ``add()`` would use."""
    from cognee.infrastructure.loaders.get_loader_engine import get_loader_engine

    engine = get_loader_engine()
    if engine.get_loader(path, None) is None:
        raise LoaderUnavailableError(
            f"No cognee loader for '{os.path.splitext(path)[1]}' files; they need the docling "
            'or unstructured extra (pip install "cognee[docling]").'
        )
    # dlt extracts on a worker thread, which has no event loop; a loop already running here
    # (the inline path called from async code) fails the file as a transient error.
    coroutine = engine.load_file(path, persist=False)
    try:
        return asyncio.run(coroutine)
    except ImportError as exc:
        raise LoaderUnavailableError(str(exc)) from exc
    finally:
        coroutine.close()  # a loop already running on this thread leaves it unawaited


def extract_file_content(service: Any, file_id: str, mime_type: str, name: str) -> str | None:
    """Return a Drive file's text, or None when reading it failed for a reason that may pass.

    The causes a rerun cannot cure are raised, so the caller can name them:
    :class:`ExportTooLargeError`, :class:`LoaderUnavailableError` (the format's loader needs an
    extra) and :class:`UnreadableFileError` (a PDF that pypdf cannot read).
    """
    try:
        if mime_type in _EXPORT_MIME_MAP:
            data = (
                service.files()
                .export(fileId=file_id, mimeType=_EXPORT_MIME_MAP[mime_type])
                .execute(num_retries=API_RETRIES)
            )
            return _decode(data)

        if mime_type == PDF_MIME_TYPE:
            with _download_media(service, file_id) as media:
                return _read_pdf(media)

        if mime_type in _LOADER_EXTENSIONS:
            with _download_file(service, file_id, _LOADER_EXTENSIONS[mime_type]) as path:
                return _load_with_loader_engine(path)

        if _is_plain_text(mime_type):
            with _download_media(service, file_id) as media:
                # Not io.TextIOWrapper: before Python 3.11 a SpooledTemporaryFile has no
                # readable(), which TextIOWrapper needs.
                return codecs.getreader("utf-8")(media, errors="replace").read()
    except LoaderUnavailableError:
        logger.warning("No loader for Drive file '%s' (%s).", name, file_id)
        raise
    except UnreadableFileError as exc:
        logger.warning("Drive file '%s' (%s) cannot be read: %s", name, file_id, exc)
        raise
    except Exception as exc:
        _raise_if_cut(exc)
        if _is_export_too_large(exc):
            logger.warning(
                "Drive file '%s' (%s) is over Google's export size limit.", name, file_id
            )
            raise ExportTooLargeError(file_id) from exc
        logger.warning(
            "Skipping Drive file '%s' (%s): content extraction failed: %s",
            name,
            file_id,
            exc,
            exc_info=True,
        )
        return None

    # No matching branch: unsupported type. The folder-sync caller (_file_to_row)
    # guards with is_supported_mime_type and logs the skip, so don't log twice.
    return None


def _decode(data) -> str:
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return str(data)


def _extract_pdf_text(stream) -> str:
    from pypdf import PdfReader

    reader = PdfReader(stream, strict=False)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


# ---------------------------------------------------------------------------
# Sync state machine (pure given a service + state dict — unit-testable)
# ---------------------------------------------------------------------------
def _ordered_map(fn, items, service, service_factory, workers, weight=None):
    """Yield ``(item, fn(client, item))`` in input order, ``fn`` running on ``workers`` threads.

    Each thread builds its own client with ``service_factory`` (googleapiclient clients are not
    thread-safe), and all are closed at the end; without a factory ``fn`` runs inline on
    ``service``. At most ``workers`` items are in flight, results not yet yielded included, and
    ``weight(item)`` bytes of them at most ``INFLIGHT_FETCH_BYTES`` (a heavier item runs alone).
    """
    if service_factory is None or workers == 1:
        for item in items:
            yield item, fn(service, item)
        return

    local = threading.local()
    clients = []
    lock = threading.Lock()

    def call(item):
        if not hasattr(local, "client"):
            client = service_factory()
            with lock:
                clients.append(client)
            local.client = client
        return fn(local.client, item)

    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="google-drive")
    pending = deque()
    held = 0
    try:
        for item in items:
            size = weight(item) if weight else 0
            while pending and (len(pending) >= workers or held + size > INFLIGHT_FETCH_BYTES):
                head, future, head_size = pending.popleft()
                held -= head_size
                yield head, future.result()
            pending.append((item, pool.submit(call, item), size))
            held += size
        while pending:
            head, future, _ = pending.popleft()
            yield head, future.result()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        for client in clients:
            client.close()


def _blocks_cursor(stats: dict[str, int]) -> bool:
    """True when a failure a rerun can cure keeps the Changes cursor where it was."""
    return stats["failed"] > sum(stats.get(key, 0) for key in _PERMANENT_FAILURES)


def _merge_stats(stats: dict[str, int], local: dict[str, int]) -> None:
    for key, value in local.items():
        stats[key] = stats.get(key, 0) + value


def _over_limit(config: _DriveConfig, file_meta: dict[str, Any]) -> bool:
    size = file_meta.get("size")
    return (
        config.max_file_size_mb is not None
        and bool(size)
        and int(size) > config.max_file_size_mb * 1024 * 1024
    )


def _is_unchanged(versions: dict[str, str], file_meta: dict[str, Any], config) -> bool:
    """True when this exact version of the file already produced a row and is still allowed.

    A file over the size limit is never unchanged, so a limit lowered after the fetch
    reaches ``_file_to_row`` and forgets it.
    """
    modified = file_meta.get("modifiedTime")
    return (
        bool(modified)
        and versions.get(file_meta["id"]) == modified
        and not _over_limit(config, file_meta)
    )


def _left_out(local: dict[str, int]) -> bool:
    return any(local.get(reason) for reason in _FORGET_REASONS)


def _iter_rows(
    service,
    config: _DriveConfig,
    state: dict,
    stats: dict[str, int] | None = None,
    *,
    service_factory: Callable[[], Any] | None = None,
):
    """Yield one row per in-scope file (full listing on the first run, deltas after) and tombstones.

    Files are fetched on ``config.max_workers`` threads with ``service_factory`` (one client per
    thread), else one by one on ``service``. ``state["versions"]`` maps a file id to the
    ``modifiedTime`` that produced its row, so a repeated run fetches only what failed or changed.
    Driven by tests with a plain dict as state and a fake Drive service.
    """
    if stats is None:
        stats = {}
    stats.clear()
    stats.update(scanned=0, skipped=0, failed=0, deleted=0, unchanged=0)
    page_token = state.get("page_token")
    known_ids = set(state.get("known_ids", []))
    versions = dict(state.get("versions", {}))

    def fetch_row(client, file_meta):
        local: dict[str, int] = {}
        return _file_to_row(client, file_meta, config, local), local

    if page_token is None:
        yielded = 0
        present_ids = set()
        fetched_versions = {}
        forgotten = set()

        def files_to_fetch():
            for file_meta in _list_files_in_scope(service, config):
                # Extraction failure does not mean the file disappeared.
                present_ids.add(file_meta["id"])
                if _is_unchanged(versions, file_meta, config):
                    stats["scanned"] += 1
                    stats["unchanged"] += 1
                    fetched_versions[file_meta["id"]] = file_meta["modifiedTime"]
                    continue
                yield file_meta

        try:
            # Capture the changes cursor BEFORE the full listing, so a file edited
            # during a slow initial sync is caught on the next incremental run
            # rather than missed. Re-processing such a file is idempotent under
            # write_disposition="merge".
            start_token = _get_start_page_token(service, config.shared_drive_id)
            for file_meta, (row, local) in _ordered_map(
                fetch_row,
                files_to_fetch(),
                service,
                service_factory,
                config.max_workers,
                weight=lambda file_meta: int(file_meta.get("size") or 0),
            ):
                _merge_stats(stats, local)
                if row is not None:
                    yielded += 1
                    if file_meta.get("modifiedTime"):
                        fetched_versions[file_meta["id"]] = file_meta["modifiedTime"]
                    yield row
                elif _left_out(local) and file_meta["id"] in versions:
                    # An earlier run ingested it; it now yields nothing to keep, so its old
                    # content must not go on answering queries.
                    forgotten.add(file_meta["id"])
                    stats["deleted"] += 1
                    yield {"id": file_meta["id"], "_deleted": True}
        except _CutShort as cut:
            _record_cut(stats, cut, emitted=bool(yielded or stats["deleted"]))
            # The listing is incomplete, so nothing is reconciled as absent; what was
            # fetched stays fetched, and the next run goes on from there.
            state["known_ids"] = sorted(known_ids | present_ids)
            kept = {**versions, **fetched_versions}
            state["versions"] = {
                file_id: version for file_id, version in kept.items() if file_id not in forgotten
            }
            return
        for file_id in sorted(known_ids - present_ids):
            stats["deleted"] += 1
            yield {"id": file_id, "_deleted": True}
        state["known_ids"] = sorted(present_ids)
        state["versions"] = fetched_versions
        if not _blocks_cursor(stats):
            state["page_token"] = start_token
        logger.info("Google Drive: initial sync yielded %d file(s).", yielded)
        return

    try:
        sizes: dict[str, int] = {}
        changed_ids, deleted_ids, new_page_token = _list_changed_file_ids(
            service, page_token, config.shared_drive_id, sizes
        )
    except _CutShort as cut:
        _record_cut(stats, cut, emitted=False)
        return

    yielded = 0
    tombstoned = 0
    for file_id in deleted_ids:
        known_ids.discard(file_id)
        versions.pop(file_id, None)
        tombstoned += 1
        stats["deleted"] += 1
        yield {"id": file_id, "_deleted": True}

    # The scope set is only needed to check whether changed files are still in
    # the folder, so skip the subfolder walk entirely when nothing changed.
    if changed_ids:

        def fetch_changed(client, file_id):
            """Return ``(file_meta, row, local_stats)``; ``file_meta`` is None for a file
            that is gone, trashed or out of scope, and ``row`` None when nothing is emitted."""
            try:
                file_meta = (
                    client.files()
                    .get(fileId=file_id, fields=FILE_FIELDS, supportsAllDrives=True)
                    .execute(num_retries=API_RETRIES)
                )
            except Exception as e:
                _raise_if_cut(e)
                if _is_not_found(e):
                    return None, None, {}
                raise RuntimeError(
                    f"Google Drive: failed to fetch metadata for file '{file_id}': {e}"
                ) from e

            if file_meta.get("trashed") or not _is_in_scope(
                file_meta,
                scope_folder_ids,
                config.shared_drive_id if config.include_subfolders else None,
            ):
                return None, None, {}
            if _is_unchanged(versions, file_meta, config):
                return file_meta, None, {"scanned": 1, "unchanged": 1}
            row, local = fetch_row(client, file_meta)
            return file_meta, row, local

        try:
            scope_folder_ids = _get_scope_folder_ids(service, config)
            if not config.shared_drive_id:
                # Drive reports a file's parents by real id, never by the "root" alias the
                # scope may hold, so a file directly in the folder would read as out of scope.
                scope_folder_ids = scope_folder_ids | {
                    _resolve_folder_id(service, config.folder_id)
                }
            for file_id, (file_meta, row, local) in _ordered_map(
                fetch_changed,
                sorted(changed_ids),
                service,
                service_factory,
                config.max_workers,
                weight=lambda file_id: sizes.get(file_id, 0),
            ):
                if file_meta is None:
                    known_ids.discard(file_id)
                    versions.pop(file_id, None)
                    tombstoned += 1
                    stats["deleted"] += 1
                    yield {"id": file_id, "_deleted": True}
                    continue
                _merge_stats(stats, local)
                known_ids.add(file_id)
                if row is not None:
                    yielded += 1
                    if file_meta.get("modifiedTime"):
                        versions[file_id] = file_meta["modifiedTime"]
                    yield row
                elif _left_out(local) and file_id in versions:
                    del versions[file_id]
                    stats["deleted"] += 1
                    yield {"id": file_id, "_deleted": True}
                elif not local.get("unchanged"):
                    # No row and nothing forgotten (a failure): no version, as on a full scan.
                    versions.pop(file_id, None)
        except _CutShort as cut:
            # The cursor stays: what was fetched is kept by its version, and the changes
            # feed is read again from the same token on the next run.
            _record_cut(stats, cut, emitted=bool(yielded or tombstoned))

    # Retry transient extraction failures on the next run instead of losing
    # them behind a successfully advanced changes cursor.
    state["known_ids"] = sorted(known_ids)
    state["versions"] = versions
    if not _blocks_cursor(stats):
        state["page_token"] = new_page_token
    logger.info(
        "Google Drive: incremental sync yielded %d changed file(s), %d deletion(s).",
        yielded,
        tombstoned,
    )


def _get_scope_folder_ids(service, config: _DriveConfig) -> set[str]:
    root = config.folder_id
    if config.shared_drive_id:
        # Incremental scope checks use ``driveId`` below; do not walk a shared
        # drive root as if it were a normal My Drive folder.
        return {root}
    if not config.include_subfolders:
        return {root}

    scope_ids = {root}
    queue = [root]
    while queue:
        current = queue.pop(0)
        page_token = None
        while True:
            try:
                response = (
                    service.files()
                    .list(
                        q=(
                            f"'{current}' in parents and "
                            f"mimeType='{GOOGLE_FOLDER_MIME_TYPE}' and trashed=false"
                        ),
                        fields="nextPageToken, files(id)",
                        pageSize=LIST_PAGE_SIZE,
                        pageToken=page_token,
                        supportsAllDrives=True,
                        includeItemsFromAllDrives=True,
                    )
                    .execute(num_retries=API_RETRIES)
                )
            except Exception as e:
                _raise_if_cut(e)
                raise RuntimeError(
                    f"Google Drive: failed to list subfolders of folder '{current}': {e}"
                ) from e

            for f in response.get("files", []):
                if f["id"] not in scope_ids:
                    scope_ids.add(f["id"])
                    queue.append(f["id"])

            page_token = response.get("nextPageToken")
            if not page_token:
                break

    return scope_ids


def _list_files_in_scope(service, config: _DriveConfig):
    if config.shared_drive_id and config.include_subfolders:
        # A shared-drive root is a corpus, not a normal My Drive folder. The
        # Drive API can enumerate that corpus directly, which also includes
        # nested folders without a separate folder-tree walk.
        yield from _list_files_in_folder(
            service,
            config.folder_id,
            shared_drive_id=config.shared_drive_id,
            all_files=True,
        )
        return
    for folder_id in _get_scope_folder_ids(service, config):
        yield from _list_files_in_folder(service, folder_id, shared_drive_id=config.shared_drive_id)


def _list_files_in_folder(
    service,
    folder_id: str,
    *,
    shared_drive_id: str | None = None,
    all_files: bool = False,
):
    page_token = None
    while True:
        try:
            params = {
                "q": "trashed=false"
                if all_files
                else f"'{folder_id}' in parents and trashed=false",
                "fields": f"nextPageToken, files({FILE_FIELDS})",
                "pageSize": LIST_PAGE_SIZE,
                "pageToken": page_token,
                "includeItemsFromAllDrives": True,
                "supportsAllDrives": True,
            }
            if shared_drive_id:
                params.update(
                    {
                        "corpora": "drive",
                        "driveId": shared_drive_id,
                        "includeItemsFromAllDrives": True,
                        "supportsAllDrives": True,
                    }
                )
            response = service.files().list(**params).execute(num_retries=API_RETRIES)
        except Exception as e:
            _raise_if_cut(e)
            raise RuntimeError(
                f"Google Drive: failed to list files in folder '{folder_id}': {e}"
            ) from e

        for file_meta in response.get("files", []):
            if file_meta.get("mimeType") != GOOGLE_FOLDER_MIME_TYPE:
                yield file_meta

        page_token = response.get("nextPageToken")
        if not page_token:
            break


def _get_start_page_token(service, shared_drive_id: str | None = None) -> str:
    try:
        params = {"supportsAllDrives": True}
        if shared_drive_id:
            params["driveId"] = shared_drive_id
        response = service.changes().getStartPageToken(**params).execute(num_retries=API_RETRIES)
    except Exception as e:
        _raise_if_cut(e)
        raise RuntimeError(f"Google Drive: failed to get changes start page token: {e}") from e
    return response["startPageToken"]


def _list_changed_file_ids(
    service,
    page_token: str,
    shared_drive_id: str | None = None,
    sizes: dict[str, int] | None = None,
) -> tuple[set[str], set[str], str]:
    """Return the changed and the deleted file ids since ``page_token``, and the next token.

    ``sizes`` is filled with the size the feed reports for each changed file, if given.
    """
    # The feed is chronological and may report one file several times (trashed, then
    # restored), so the last change of a file decides its state.
    gone_by_id: dict[str, bool] = {}
    current_token = page_token
    new_start_token = page_token
    params = {"supportsAllDrives": True, "includeItemsFromAllDrives": True}
    if shared_drive_id:
        params["driveId"] = shared_drive_id

    while True:
        try:
            response = (
                service.changes()
                .list(
                    pageToken=current_token,
                    fields=(
                        "nextPageToken, newStartPageToken, "
                        "changes(fileId, removed, file(trashed, size))"
                    ),
                    **params,
                )
                .execute(num_retries=API_RETRIES)
            )
        except Exception as e:
            _raise_if_cut(e)
            raise RuntimeError(f"Google Drive: failed to list changes: {e}") from e

        for change in response.get("changes", []):
            file_id = change.get("fileId")
            if not file_id:
                # The account feed also includes shared-drive membership events.
                continue
            file_resource = change.get("file") or {}
            gone_by_id[file_id] = bool(change.get("removed") or file_resource.get("trashed"))
            if sizes is not None:
                sizes[file_id] = int(file_resource.get("size") or 0)

        current_token = response.get("nextPageToken")
        if response.get("newStartPageToken"):
            new_start_token = response["newStartPageToken"]
        if not current_token:
            break

    changed_ids = {file_id for file_id, gone in gone_by_id.items() if not gone}
    deleted_ids = {file_id for file_id, gone in gone_by_id.items() if gone}
    return changed_ids, deleted_ids, new_start_token


def _resolve_folder_id(service, folder_id: str) -> str:
    try:
        response = (
            service.files()
            .get(fileId=folder_id, fields="id", supportsAllDrives=True)
            .execute(num_retries=API_RETRIES)
        )
    except Exception as e:
        _raise_if_cut(e)
        raise RuntimeError(f"Google Drive: failed to resolve folder '{folder_id}': {e}") from e
    return response["id"]


def _is_in_scope(
    file_meta: dict[str, Any], scope_folder_ids: set[str], shared_drive_id: str | None = None
) -> bool:
    if shared_drive_id:
        return file_meta.get("driveId") == shared_drive_id
    return bool(set(file_meta.get("parents", [])) & scope_folder_ids)


def _is_not_found(e: Exception) -> bool:
    return getattr(getattr(e, "resp", None), "status", None) == 404


def _file_to_row(
    service, file_meta: dict[str, Any], config: _DriveConfig, stats: dict[str, int] | None = None
) -> dict | None:
    if stats is None:
        stats = {}
    stats["scanned"] = stats.get("scanned", 0) + 1
    mime_type = file_meta.get("mimeType", "")
    name = file_meta.get("name", "")
    file_id = file_meta["id"]

    if not is_supported_mime_type(mime_type):
        _count_skip(stats, "unsupported_type")
        logger.debug(
            "Skipping unsupported Drive file '%s' (%s): mime type '%s'.", name, file_id, mime_type
        )
        return None

    if _over_limit(config, file_meta):
        _count_skip(stats, "too_large")
        logger.warning(
            "Skipping Drive file '%s' (%s): size exceeds max_file_size_mb=%d.",
            name,
            file_id,
            config.max_file_size_mb,
        )
        return None

    try:
        content = extract_file_content(service, file_id, mime_type, name)
    except (ExportTooLargeError, LoaderUnavailableError, UnreadableFileError) as exc:
        reason = _PERMANENT_REASONS[type(exc)]
        stats["failed"] = stats.get("failed", 0) + 1
        stats[reason] = stats.get(reason, 0) + 1
        return None
    if content is None:
        stats["failed"] = stats.get("failed", 0) + 1
        stats["failed_content_extraction"] = stats.get("failed_content_extraction", 0) + 1
        return None
    if not content.strip():
        _count_skip(stats, "empty_content")
        return None

    # Document-mode row contract: {id, title, content, url}. resolve_dlt_sources
    # tags these rows system_metadata["source"]="google_drive" (see the
    # DOCUMENT_SOURCE_ATTR marker below), so each file flows through normal
    # cognify (LLM graph extraction) rather than the relational schema path.
    return {
        "id": file_id,
        "title": name,
        "content": content,
        "url": file_meta.get("webViewLink"),
        "_deleted": False,
    }


def _count_skip(stats: dict[str, int], reason: str) -> None:
    stats["skipped"] = stats.get("skipped", 0) + 1
    key = f"skipped_{reason}"
    stats[key] = stats.get(key, 0) + 1


def _first_set(*values):
    return next(value for value in values if value is not None)


def _env_int(name: str) -> int | None:
    value = os.getenv(name)
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from None


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------
def google_drive_source(
    folder_id: str | None = None,
    *,
    resource_name: str = "google_drive_files",
    check_active: Callable[[], None] | None = None,
    auth_mode: str | None = None,
    credentials_path: str | None = None,
    token_path: str | None = None,
    include_subfolders: bool | None = None,
    max_file_size_mb: int | None = None,
    shared_drive_id: str | None = None,
    max_workers: int | None = None,
    service: Any = None,
    service_factory: Callable[[], Any] | None = None,
):
    """Return a ``dlt`` resource yielding one row per in-scope Google Drive file.

    Any argument left as ``None`` falls back to the matching ``GOOGLE_DRIVE_*``
    environment variable.  Hand the result to ``cognee.remember(...)`` with
    ``write_disposition="merge"`` and ``primary_key="id"``.

    Args:
        folder_id: Drive folder ID to sync (found in the folder's URL).
        resource_name: Stable dlt resource name. Hosts syncing several folders
            in one dataset should give each folder its own name so each
            folder's Changes API cursor is persisted independently.
        check_active: Optional host authorization checkpoint during extraction.
        auth_mode: ``"service_account"`` (default) or ``"oauth"``.
        credentials_path: Path to the service-account key or OAuth client-secret JSON.
        token_path: Where the cached OAuth user token is read/written (oauth mode).
        include_subfolders: Recurse into subfolders (default True).
        max_file_size_mb: Leave out files larger than this. Default: no limit. A file over
            an explicit limit is counted as ``skipped_too_large`` and forgotten if it was
            ingested before, like a file that left the folder.
        shared_drive_id: Shared-drive id when ``folder_id`` is a shared-drive root.
        max_workers: Threads fetching file content in parallel (default 8). Each thread
            uses its own Drive client, so values above 1 need ``service_factory`` or the
            auth settings above; a bare ``service`` is used one file at a time.
        service: Pre-built Drive API client, used for listing (and for fetching when no
            ``service_factory`` is given). When omitted a client is built from the auth
            settings above.
        service_factory: Zero-argument callable returning a fresh Drive client. Called
            once per fetch thread, because googleapiclient clients are not thread-safe.
            Hosts that supply their own credentials pass this instead of ``service``.
    """
    try:
        import dlt
    except ImportError as exc:
        raise ImportError(
            "The Google Drive connector requires the google-drive extra: "
            'pip install "cognee[google-drive]".'
        ) from exc

    if getattr(dlt_utils, "DOCUMENT_SYNC_VERSION", 0) < 1:
        raise RuntimeError(
            "Google Drive sync requires a Cognee build with table-scoped DLT document cleanup. "
            "Upgrade Cognee before syncing to avoid deleting another folder's data."
        )

    resolved_folder_id = folder_id or os.getenv("GOOGLE_DRIVE_FOLDER_ID")
    if not resolved_folder_id:
        raise ValueError(
            "folder_id is required (pass it explicitly or set GOOGLE_DRIVE_FOLDER_ID)."
        )

    if include_subfolders is None:
        # Env-bool parsed inline, matching the codebase convention (e.g.
        # migrations/startup.py) — default on, unset via false/0/no.
        include_subfolders = os.getenv(
            "GOOGLE_DRIVE_INCLUDE_SUBFOLDERS", "true"
        ).strip().lower() not in (
            "false",
            "0",
            "no",
        )

    config = _DriveConfig(
        folder_id=resolved_folder_id,
        auth_mode=auth_mode or os.getenv("GOOGLE_DRIVE_AUTH_MODE", "service_account"),
        credentials_path=credentials_path or os.getenv("GOOGLE_DRIVE_CREDENTIALS_PATH"),
        token_path=token_path or os.getenv("GOOGLE_DRIVE_TOKEN_PATH"),
        include_subfolders=include_subfolders,
        max_file_size_mb=(
            max_file_size_mb
            if max_file_size_mb is not None
            else _env_int("GOOGLE_DRIVE_MAX_FILE_SIZE_MB")
        ),
        shared_drive_id=shared_drive_id,
        max_workers=_first_set(
            max_workers, _env_int("GOOGLE_DRIVE_MAX_WORKERS"), DEFAULT_MAX_WORKERS
        ),
    )
    if config.max_workers < 1:
        raise ValueError(
            f"max_workers must be at least 1, got {config.max_workers} "
            "(the max_workers argument or GOOGLE_DRIVE_MAX_WORKERS)."
        )
    if config.max_file_size_mb is not None and config.max_file_size_mb < 1:
        raise ValueError(
            f"max_file_size_mb must be at least 1, got {config.max_file_size_mb} "
            "(the max_file_size_mb argument or GOOGLE_DRIVE_MAX_FILE_SIZE_MB)."
        )

    stats: dict[str, int] = {}

    @dlt.resource(
        name=resource_name,
        write_disposition="merge",
        primary_key="id",
        # `_deleted` is a boolean hard-delete marker (matching gmail.py): rows
        # where it is True are removed from the dlt destination on merge, which
        # propagates the deletion through cognee's orphan_cleanup.
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def google_drive_files():
        factory = service_factory
        if service is None and factory is None:

            def factory():
                return build_drive_service(
                    auth_mode=config.auth_mode,
                    credentials_path=config.credentials_path,
                    token_path=config.token_path,
                )

        client = service or factory()
        try:
            yield from dlt_utils.guarded_rows(
                _iter_rows(
                    client,
                    config,
                    dlt.current.resource_state(),
                    stats,
                    service_factory=factory,
                ),
                check_active,
            )
        finally:
            if service is None:
                client.close()

    resource = google_drive_files()
    # Opt into the document ingestion path: each file row (id/title/content/url)
    # becomes a text document that flows through normal cognify (LLM graph
    # extraction). resolve_dlt_sources reads this marker; it never imports this
    # connector. Sync stays incremental — hand this to remember() with
    # write_disposition="merge" (the Changes-API delta + _deleted hard-delete).
    setattr(resource, DOCUMENT_SOURCE_ATTR, "google_drive")
    setattr(resource, dlt_utils.PIPELINE_SCOPE_ATTR, resource_name)
    # Host-readable diagnostics contain counts only, never file names or content.
    resource.cognee_sync_stats = stats
    return resource
