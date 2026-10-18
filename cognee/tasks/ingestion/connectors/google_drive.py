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
* **Content** — Google Docs/Sheets export to text/CSV, PDFs are parsed with the
  core ``pypdf`` dependency, and plain text/markdown/CSV is downloaded as-is.  A
  file that can't be parsed is skipped with a warning rather than failing the
  whole sync.
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

import io
import os
from collections.abc import Callable
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
PLAIN_TEXT_MIME_TYPES = {"text/plain", "text/markdown", "text/csv"}

# Native Google formats we know how to export as text.  Other native types
# (Slides, Drawings, Forms, ...) are skipped with a warning rather than failing
# the whole sync.
_EXPORT_MIME_MAP = {
    GOOGLE_DOC_MIME_TYPE: "text/plain",
    # Drive's export only returns the first sheet as CSV — a documented
    # limitation; multi-sheet extraction would require the Sheets API.
    GOOGLE_SHEET_MIME_TYPE: "text/csv",
}

FILE_FIELDS = "id, name, mimeType, modifiedTime, webViewLink, parents, driveId, trashed, size"


@dataclass(frozen=True)
class _DriveConfig:
    folder_id: str
    auth_mode: str
    credentials_path: str | None
    token_path: str | None
    include_subfolders: bool
    max_file_size_mb: int
    shared_drive_id: str | None = None


# ---------------------------------------------------------------------------
# Auth / service construction
# ---------------------------------------------------------------------------
def build_drive_service(
    *,
    auth_mode: str = "service_account",
    credentials_path: str | None = None,
    token_path: str | None = None,
) -> Any:
    """Build an authenticated Drive v3 API client.

    The Google client libraries are imported lazily so they remain an optional
    dependency (``pip install "cognee[google-drive]"``).
    """
    try:
        from google.auth.transport.requests import Request
        from google.oauth2 import service_account
        from google.oauth2.credentials import Credentials as UserCredentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
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

    return build("drive", "v3", credentials=credentials, cache_discovery=False)


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
def is_supported_mime_type(mime_type: str) -> bool:
    return (
        mime_type in _EXPORT_MIME_MAP
        or mime_type == PDF_MIME_TYPE
        or mime_type in PLAIN_TEXT_MIME_TYPES
    )


def extract_file_content(service: Any, file_id: str, mime_type: str, name: str) -> str | None:
    """Return extracted text for a Drive file, or None to skip it.

    A file that can't be parsed (corrupt PDF, export error, transient per-file
    hiccup) is logged and skipped rather than aborting the whole folder sync.
    Genuine auth / connectivity failures surface from the listing calls instead.
    """
    try:
        if mime_type in _EXPORT_MIME_MAP:
            data = (
                service.files()
                .export(fileId=file_id, mimeType=_EXPORT_MIME_MAP[mime_type])
                .execute()
            )
            return _decode(data)

        if mime_type == PDF_MIME_TYPE:
            data = service.files().get_media(fileId=file_id, supportsAllDrives=True).execute()
            return _extract_pdf_text(data)

        if mime_type in PLAIN_TEXT_MIME_TYPES:
            data = service.files().get_media(fileId=file_id, supportsAllDrives=True).execute()
            return _decode(data)
    except Exception as exc:
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


def _extract_pdf_text(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data), strict=False)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


# ---------------------------------------------------------------------------
# Sync state machine (pure given a service + state dict — unit-testable)
# ---------------------------------------------------------------------------
def _iter_rows(service, config: _DriveConfig, state: dict, stats: dict[str, int] | None = None):
    """Yield one row per in-scope file (full listing on the first run, deltas
    thereafter), plus hard-delete tombstones for removed files.

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

    if page_token is None:
        # Capture the changes cursor BEFORE the full listing, so a file edited
        # during a slow initial sync is caught on the next incremental run
        # rather than missed. Re-processing such a file is idempotent under
        # write_disposition="merge".
        start_token = _get_start_page_token(service, config.shared_drive_id)
        yielded = 0
        present_ids = set()
        for file_meta in _list_files_in_scope(service, config):
            # Extraction failure does not mean the file disappeared.
            present_ids.add(file_meta["id"])
            row = _file_to_row(service, file_meta, config, stats)
            if row is not None:
                yielded += 1
                yield row
        for file_id in sorted(known_ids - present_ids):
            stats["deleted"] += 1
            yield {"id": file_id, "_deleted": True}
        state["known_ids"] = sorted(present_ids)
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
        tombstoned += 1
        stats["deleted"] += 1
        yield {"id": file_id, "_deleted": True}

    # The scope set is only needed to check whether changed files are still in
    # the folder, so skip the subfolder walk entirely when nothing changed.
    if changed_ids:
        scope_folder_ids = _get_scope_folder_ids(service, config)
        for file_id in changed_ids:
            try:
                file_meta = (
                    service.files()
                    .get(fileId=file_id, fields=FILE_FIELDS, supportsAllDrives=True)
                    .execute()
                )
            except Exception as e:
                if _is_not_found(e):
                    known_ids.discard(file_id)
                    tombstoned += 1
                    stats["deleted"] += 1
                    yield {"id": file_id, "_deleted": True}
                    continue
                raise RuntimeError(
                    f"Google Drive: failed to fetch metadata for file '{file_id}': {e}"
                ) from e

            if file_meta.get("trashed") or not _is_in_scope(
                file_meta,
                scope_folder_ids,
                config.shared_drive_id if config.include_subfolders else None,
            ):
                known_ids.discard(file_id)
                tombstoned += 1
                stats["deleted"] += 1
                yield {"id": file_id, "_deleted": True}
                continue

            row = _file_to_row(service, file_meta, config, stats)
            known_ids.add(file_id)
            if row is not None:
                yielded += 1
                yield row

    # Retry transient extraction failures on the next run instead of losing
    # them behind a successfully advanced changes cursor.
    state["known_ids"] = sorted(known_ids)
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
                        pageSize=100,
                        pageToken=page_token,
                        supportsAllDrives=True,
                        includeItemsFromAllDrives=True,
                    )
                    .execute()
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
                "pageSize": 100,
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
            response = service.files().list(**params).execute()
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
        response = service.changes().getStartPageToken(**params).execute()
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
                .execute()
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
    if size and int(size) > config.max_file_size_mb * 1024 * 1024:
        _count_skip(stats, "too_large")
        logger.warning(
            "Skipping Drive file '%s' (%s): size exceeds max_file_size_mb=%d.",
            name,
            file_id,
            config.max_file_size_mb,
        )
        return None

    content = extract_file_content(service, file_id, mime_type, name)
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
    service: Any = None,
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
        max_file_size_mb: Skip files larger than this (default 25).
        shared_drive_id: Shared-drive id when ``folder_id`` is a shared-drive root.
        service: Pre-built Drive API client. Mainly an injection point for tests;
            when omitted a client is built from the auth settings above.
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
            else int(os.getenv("GOOGLE_DRIVE_MAX_FILE_SIZE_MB", "25"))
        ),
        shared_drive_id=shared_drive_id,
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
        client = service or build_drive_service(
            auth_mode=config.auth_mode,
            credentials_path=config.credentials_path,
            token_path=config.token_path,
        )
        yield from dlt_utils.guarded_rows(
            _iter_rows(client, config, dlt.current.resource_state(), stats), check_active
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
