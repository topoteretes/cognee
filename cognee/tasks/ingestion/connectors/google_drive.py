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
* **Auth** — service account (default, non-interactive) or OAuth installed-app.
  Point ``credentials_path`` at the service-account key or OAuth client-secret
  JSON; scope is read-only (``drive.readonly``).
* **Primary key** — the Drive file ``id``.  With ``write_disposition="merge"``
  this gives idempotent upserts.
* **Throughput** — file content is fetched on ``max_workers`` threads (default 8, one
  Drive client per thread) and every API call retries 429/5xx/rate-limit answers with
  backoff.  A bare injected ``service`` is used one file at a time; pass
  ``service_factory`` to parallelise.
* **Per-file versions** — the state keeps each fetched file's ``modifiedTime``.  A run
  repeated after a failed file (or a changes event for an unchanged file) skips what was
  already fetched instead of downloading the folder again.
* **Incremental cursor** — the Drive Changes API page token.  The first run
  captures a start token, does a full folder listing, and records the token;
  later runs call ``changes().list(pageToken=...)`` and emit only added/changed
  files plus hard-delete tombstones for removed/trashed/out-of-scope files.  The
  cursor is persisted in dlt's per-resource state, so re-running ``remember``
  resumes where it left off.
* **Forget-on-delete** — removed files are emitted with the ``deleted``
  hard-delete marker; dlt drops them from its destination on ``merge`` and
  cognee's existing ``orphan_cleanup`` purges them from the graph, vector, and
  relational stores.
* **Content** — Google Docs/Slides export to text and Sheets to CSV, PDFs are parsed
  with the core ``pypdf`` dependency, ``text/*``, JSON and XML are downloaded as UTF-8
  text, and Office/HTML/EPUB files go through cognee's loader engine (the loaders
  ``add()`` uses; they need the ``docling`` or ``unstructured`` extra).  A file that
  can't be parsed is skipped with a warning rather than failing the whole sync.
* **Self-describing** — the resource declares its content column, so a plain
  ``remember()`` call routes file content through normal chunking + LLM graph
  extraction; no ``dlt_content_column`` kwarg is required.

Limitations
-----------
* The Drive Changes API is account-wide; deleted-file events carry no metadata
  and cannot be scope-filtered, so an incremental run may emit harmless no-op
  delete rows for files removed outside the configured folder.
* ``auth_mode="oauth"`` uses an interactive browser flow on first run.  For
  headless / CI use, pre-authorize a token file and point ``token_path`` at it.
"""

import asyncio
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
# (Slides, Drawings, Forms, ...) are skipped with a warning rather than failing
# the whole sync.
_EXPORT_MIME_MAP = {
    GOOGLE_DOC_MIME_TYPE: "text/plain",
    # Drive's export only returns the first sheet as CSV — a documented
    # limitation; multi-sheet extraction would require the Sheets API.
    GOOGLE_SHEET_MIME_TYPE: "text/csv",
    GOOGLE_SLIDES_MIME_TYPE: "text/plain",
}

# googleapiclient retries 429, 5xx and rate-limit 403 answers with exponential backoff.
API_RETRIES = 5
# files.list returns at most 1000 entries per page.
LIST_PAGE_SIZE = 1000
DEFAULT_MAX_WORKERS = 8
# A download stays in memory up to this size and spills to a temporary file beyond it, so
# the memory of a fetch depends on the extracted text, not on the size of the file.
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


@contextmanager
def _download_file(service: Any, file_id: str, suffix: str):
    """Yield the path of a temporary file holding the download, removed on exit.

    The extension matters: the loader engine picks its parser by it.
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
        raise LoaderUnavailableError(engine._no_loader_message(path))
    # Extraction runs on a worker thread without an event loop (see guarded_rows).
    return asyncio.run(engine.load_file(path, persist=False))


def extract_file_content(service: Any, file_id: str, mime_type: str, name: str) -> str | None:
    """Return extracted text for a Drive file, or None to skip it.

    A file that can't be parsed (corrupt PDF, transient per-file hiccup) is logged and
    skipped rather than aborting the whole folder sync. Genuine auth / connectivity
    failures surface from the listing calls instead. A Google Doc or Sheet over Google's
    export limit raises :class:`ExportTooLargeError`, and a format whose cognee loader is
    not installed raises :class:`LoaderUnavailableError`, so the caller can name the cause.
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
                return _extract_pdf_text(media)

        if mime_type in _LOADER_EXTENSIONS:
            with _download_file(service, file_id, _LOADER_EXTENSIONS[mime_type]) as path:
                return _load_with_loader_engine(path)

        if _is_plain_text(mime_type):
            with _download_media(service, file_id) as media:
                return io.TextIOWrapper(media, encoding="utf-8", errors="replace").read()
    except LoaderUnavailableError:
        logger.warning("No loader for Drive file '%s' (%s).", name, file_id)
        raise
    except Exception as exc:
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
def _ordered_map(fn, items, service, service_factory, workers):
    """Yield ``(item, fn(client, item))`` in input order, running ``fn`` on up to ``workers`` threads.

    googleapiclient clients are not thread-safe, so each thread builds its own with
    ``service_factory``. Without a factory the calls run inline on ``service``. At most
    ``2 * workers`` items are in flight, which bounds the file contents held in memory.
    """
    if service_factory is None or workers == 1:
        for item in items:
            yield item, fn(service, item)
        return

    local = threading.local()

    def call(item):
        if not hasattr(local, "client"):
            local.client = service_factory()
        return fn(local.client, item)

    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="google-drive")
    pending = deque()
    try:
        for item in items:
            pending.append((item, pool.submit(call, item)))
            if len(pending) >= 2 * workers:
                head, future = pending.popleft()
                yield head, future.result()
        while pending:
            head, future = pending.popleft()
            yield head, future.result()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def _merge_stats(stats: dict[str, int], local: dict[str, int]) -> None:
    for key, value in local.items():
        stats[key] = stats.get(key, 0) + value


def _is_unchanged(versions: dict[str, str], file_meta: dict[str, Any]) -> bool:
    """True when this exact version of the file was already fetched without failing."""
    modified = file_meta.get("modifiedTime")
    return bool(modified) and versions.get(file_meta["id"]) == modified


def _iter_rows(
    service,
    config: _DriveConfig,
    state: dict,
    stats: dict[str, int] | None = None,
    *,
    service_factory: Callable[[], Any] | None = None,
):
    """Yield one row per in-scope file (full listing on the first run, deltas
    thereafter), plus hard-delete tombstones for removed files.

    Files are fetched on ``config.max_workers`` threads when ``service_factory`` is
    given (one client per thread), otherwise one by one on ``service``. ``state``
    keeps ``versions`` (file id to the ``modifiedTime`` fetched without failing), so a
    run repeated after a failure fetches only what failed or changed.

    Decoupled from dlt's resource-state machinery (which needs an active
    pipeline context) so it is directly unit-testable with a fake Drive service
    and a plain dict standing in for dlt's resource state.
    """
    if stats is None:
        stats = {}
    stats.clear()
    stats.update(scanned=0, skipped=0, failed=0, deleted=0)
    page_token = state.get("page_token")
    known_ids = set(state.get("known_ids", []))
    versions = dict(state.get("versions", {}))

    def fetch_row(client, file_meta):
        local: dict[str, int] = {}
        return _file_to_row(client, file_meta, config, local), local

    if page_token is None:
        # Capture the changes cursor BEFORE the full listing, so a file edited
        # during a slow initial sync is caught on the next incremental run
        # rather than missed. Re-processing such a file is idempotent under
        # write_disposition="merge".
        start_token = _get_start_page_token(service, config.shared_drive_id)
        yielded = 0
        present_ids = set()
        fetched_versions = {}

        def files_to_fetch():
            for file_meta in _list_files_in_scope(service, config):
                # Extraction failure does not mean the file disappeared.
                present_ids.add(file_meta["id"])
                if _is_unchanged(versions, file_meta):
                    stats["scanned"] += 1
                    stats["unchanged"] = stats.get("unchanged", 0) + 1
                    fetched_versions[file_meta["id"]] = file_meta["modifiedTime"]
                    continue
                yield file_meta

        for file_meta, (row, local) in _ordered_map(
            fetch_row, files_to_fetch(), service, service_factory, config.max_workers
        ):
            _merge_stats(stats, local)
            if local.get("skipped_too_large"):
                # Left out by the caller's size limit: forget what an earlier run stored.
                # The version stays unrecorded, so a raised limit fetches the file again.
                stats["deleted"] += 1
                yield {"id": file_meta["id"], "_deleted": True}
                continue
            if not local.get("failed") and file_meta.get("modifiedTime"):
                fetched_versions[file_meta["id"]] = file_meta["modifiedTime"]
            if row is not None:
                yielded += 1
                yield row
        for file_id in sorted(known_ids - present_ids):
            stats["deleted"] += 1
            yield {"id": file_id, "_deleted": True}
        state["known_ids"] = sorted(present_ids)
        state["versions"] = fetched_versions
        if not stats["failed"]:
            state["page_token"] = start_token
        logger.info("Google Drive: initial sync yielded %d file(s).", yielded)
        return

    changed_ids, deleted_ids, new_page_token = _list_changed_file_ids(
        service, page_token, config.shared_drive_id
    )

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
        scope_folder_ids = _get_scope_folder_ids(service, config)

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
            if _is_unchanged(versions, file_meta):
                return file_meta, None, {"scanned": 1, "unchanged": 1}
            row, local = fetch_row(client, file_meta)
            return file_meta, row, local

        for file_id, (file_meta, row, local) in _ordered_map(
            fetch_changed, sorted(changed_ids), service, service_factory, config.max_workers
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
            if local.get("skipped_too_large"):
                versions.pop(file_id, None)
                stats["deleted"] += 1
                yield {"id": file_id, "_deleted": True}
                continue
            if not local.get("failed") and file_meta.get("modifiedTime"):
                versions[file_id] = file_meta["modifiedTime"]
            if row is not None:
                yielded += 1
                yield row

    # Retry transient extraction failures on the next run instead of losing
    # them behind a successfully advanced changes cursor.
    state["known_ids"] = sorted(known_ids)
    state["versions"] = versions
    if not stats["failed"]:
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
        raise RuntimeError(f"Google Drive: failed to get changes start page token: {e}") from e
    return response["startPageToken"]


def _list_changed_file_ids(
    service, page_token: str, shared_drive_id: str | None = None
) -> tuple[set[str], set[str], str]:
    changed_ids: set[str] = set()
    deleted_ids: set[str] = set()
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
                        "nextPageToken, newStartPageToken, changes(fileId, removed, file(trashed))"
                    ),
                    **params,
                )
                .execute(num_retries=API_RETRIES)
            )
        except Exception as e:
            raise RuntimeError(f"Google Drive: failed to list changes: {e}") from e

        for change in response.get("changes", []):
            file_id = change.get("fileId")
            if not file_id:
                # The account feed also includes shared-drive membership events.
                continue
            if change.get("removed") or (change.get("file") or {}).get("trashed"):
                deleted_ids.add(file_id)
            else:
                changed_ids.add(file_id)

        current_token = response.get("nextPageToken")
        if response.get("newStartPageToken"):
            new_start_token = response["newStartPageToken"]
        if not current_token:
            break

    return changed_ids - deleted_ids, deleted_ids, new_start_token


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
        logger.warning(
            "Skipping unsupported Drive file '%s' (%s): mime type '%s'.", name, file_id, mime_type
        )
        return None

    size = file_meta.get("size")
    if (
        config.max_file_size_mb is not None
        and size
        and int(size) > config.max_file_size_mb * 1024 * 1024
    ):
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
    except (ExportTooLargeError, LoaderUnavailableError) as exc:
        reason = (
            "export_too_large" if isinstance(exc, ExportTooLargeError) else "loader_unavailable"
        )
        stats["failed"] = stats.get("failed", 0) + 1
        stats[f"failed_{reason}"] = stats.get(f"failed_{reason}", 0) + 1
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


def _env_max_file_size_mb() -> int | None:
    value = os.getenv("GOOGLE_DRIVE_MAX_FILE_SIZE_MB")
    return int(value) if value else None


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
            max_file_size_mb if max_file_size_mb is not None else _env_max_file_size_mb()
        ),
        shared_drive_id=shared_drive_id,
        max_workers=(
            max_workers
            if max_workers is not None
            else int(os.getenv("GOOGLE_DRIVE_MAX_WORKERS", str(DEFAULT_MAX_WORKERS)))
        ),
    )
    if config.max_workers < 1:
        raise ValueError("max_workers must be at least 1.")
    if config.max_file_size_mb is not None and config.max_file_size_mb < 1:
        raise ValueError("max_file_size_mb must be at least 1.")

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
            credentials = load_drive_credentials(
                auth_mode=config.auth_mode,
                credentials_path=config.credentials_path,
                token_path=config.token_path,
            )

            def factory():
                return _build_client(credentials)

        client = service or factory()
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
