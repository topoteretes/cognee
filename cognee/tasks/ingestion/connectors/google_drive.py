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
        primary_key="file_id",
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
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set, Tuple

from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion.dlt_utils import CONTENT_COLUMN_HINT_ATTR

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

FILE_FIELDS = "id, name, mimeType, modifiedTime, webViewLink, parents, trashed, size"


@dataclass(frozen=True)
class _DriveConfig:
    folder_id: str
    auth_mode: str
    credentials_path: Optional[str]
    token_path: Optional[str]
    include_subfolders: bool
    max_file_size_mb: int


# ---------------------------------------------------------------------------
# Auth / service construction
# ---------------------------------------------------------------------------
def build_drive_service(
    *,
    auth_mode: str = "service_account",
    credentials_path: Optional[str] = None,
    token_path: Optional[str] = None,
) -> Any:
    """Build an authenticated Drive v3 API client.

    The Google client libraries are imported lazily so they remain an optional
    dependency (``pip install "cognee[google-drive]"``).
    """
    try:
        from google.oauth2 import service_account
        from google.oauth2.credentials import Credentials as UserCredentials
        from google.auth.transport.requests import Request
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


def _load_oauth_credentials(
    UserCredentials, InstalledAppFlow, Request, client_secret_path, token_path
):
    creds = None
    if token_path and os.path.exists(token_path):
        creds = UserCredentials.from_authorized_user_file(token_path, DRIVE_READONLY_SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not client_secret_path or not os.path.exists(client_secret_path):
                raise FileNotFoundError(
                    f"Google Drive OAuth client secret not found at {client_secret_path!r}. "
                    "Set GOOGLE_DRIVE_CREDENTIALS_PATH to an OAuth client-secret JSON file "
                    "(Desktop app) downloaded from the Google Cloud Console."
                )
            # Interactive: opens a browser on first run. For headless / CI use,
            # pre-authorize a token file and point token_path at it.
            flow = InstalledAppFlow.from_client_secrets_file(
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


def extract_file_content(service: Any, file_id: str, mime_type: str, name: str) -> Optional[str]:
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
            data = service.files().get_media(fileId=file_id).execute()
            return _extract_pdf_text(data)

        if mime_type in PLAIN_TEXT_MIME_TYPES:
            data = service.files().get_media(fileId=file_id).execute()
            return _decode(data)
    except Exception as exc:
        logger.warning(
            "Skipping Drive file '%s' (%s): content extraction failed: %s", name, file_id, exc
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
def _iter_rows(service, config: _DriveConfig, state: dict):
    """Yield one row per in-scope file (full listing on the first run, deltas
    thereafter), plus hard-delete tombstones for removed files.

    Decoupled from dlt's resource-state machinery (which needs an active
    pipeline context) so it is directly unit-testable with a fake Drive service
    and a plain dict standing in for dlt's resource state.
    """
    page_token = state.get("page_token")

    if page_token is None:
        # Capture the changes cursor BEFORE the full listing, so a file edited
        # during a slow initial sync is caught on the next incremental run
        # rather than missed. Re-processing such a file is idempotent under
        # write_disposition="merge".
        start_token = _get_start_page_token(service)
        yielded = 0
        for file_meta in _list_files_in_scope(service, config):
            row = _file_to_row(service, file_meta, config)
            if row is not None:
                yielded += 1
                yield row
        state["page_token"] = start_token
        logger.info("Google Drive: initial sync yielded %d file(s).", yielded)
        return

    changed_ids, deleted_ids, new_page_token = _list_changed_file_ids(service, page_token)

    yielded = 0
    tombstoned = 0
    for file_id in deleted_ids:
        tombstoned += 1
        yield {"file_id": file_id, "_deleted": True}

    # The scope set is only needed to check whether changed files are still in
    # the folder, so skip the subfolder walk entirely when nothing changed.
    if changed_ids:
        scope_folder_ids = _get_scope_folder_ids(service, config)
        for file_id in changed_ids:
            try:
                file_meta = service.files().get(fileId=file_id, fields=FILE_FIELDS).execute()
            except Exception as e:
                if _is_not_found(e):
                    tombstoned += 1
                    yield {"file_id": file_id, "_deleted": True}
                    continue
                raise RuntimeError(
                    f"Google Drive: failed to fetch metadata for file '{file_id}': {e}"
                ) from e

            if file_meta.get("trashed") or not _is_in_scope(file_meta, scope_folder_ids):
                tombstoned += 1
                yield {"file_id": file_id, "_deleted": True}
                continue

            row = _file_to_row(service, file_meta, config)
            if row is not None:
                yielded += 1
                yield row

    state["page_token"] = new_page_token
    logger.info(
        "Google Drive: incremental sync yielded %d changed file(s), %d deletion(s).",
        yielded,
        tombstoned,
    )


def _get_scope_folder_ids(service, config: _DriveConfig) -> Set[str]:
    root = config.folder_id
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
    for folder_id in _get_scope_folder_ids(service, config):
        yield from _list_files_in_folder(service, folder_id)


def _list_files_in_folder(service, folder_id: str):
    page_token = None
    while True:
        try:
            response = (
                service.files()
                .list(
                    q=f"'{folder_id}' in parents and trashed=false",
                    fields=f"nextPageToken, files({FILE_FIELDS})",
                    pageSize=100,
                    pageToken=page_token,
                )
                .execute()
            )
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


def _get_start_page_token(service) -> str:
    try:
        response = service.changes().getStartPageToken().execute()
    except Exception as e:
        raise RuntimeError(f"Google Drive: failed to get changes start page token: {e}") from e
    return response["startPageToken"]


def _list_changed_file_ids(service, page_token: str) -> Tuple[Set[str], Set[str], str]:
    changed_ids: Set[str] = set()
    deleted_ids: Set[str] = set()
    current_token = page_token
    new_start_token = page_token

    while True:
        try:
            response = (
                service.changes()
                .list(
                    pageToken=current_token,
                    fields=(
                        "nextPageToken, newStartPageToken, changes(fileId, removed, file(trashed))"
                    ),
                )
                .execute()
            )
        except Exception as e:
            raise RuntimeError(f"Google Drive: failed to list changes: {e}") from e

        for change in response.get("changes", []):
            file_id = change["fileId"]
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


def _is_in_scope(file_meta: Dict[str, Any], scope_folder_ids: Set[str]) -> bool:
    return bool(set(file_meta.get("parents", [])) & scope_folder_ids)


def _is_not_found(e: Exception) -> bool:
    return getattr(getattr(e, "resp", None), "status", None) == 404


def _file_to_row(service, file_meta: Dict[str, Any], config: _DriveConfig) -> Optional[dict]:
    mime_type = file_meta.get("mimeType", "")
    name = file_meta.get("name", "")
    file_id = file_meta["id"]

    if not is_supported_mime_type(mime_type):
        logger.warning(
            "Skipping unsupported Drive file '%s' (%s): mime type '%s'.", name, file_id, mime_type
        )
        return None

    size = file_meta.get("size")
    if size and int(size) > config.max_file_size_mb * 1024 * 1024:
        logger.warning(
            "Skipping Drive file '%s' (%s): size exceeds max_file_size_mb=%d.",
            name,
            file_id,
            config.max_file_size_mb,
        )
        return None

    content = extract_file_content(service, file_id, mime_type, name)
    if content is None or not content.strip():
        # Unparseable/skipped, or an empty document — nothing to add to memory.
        return None

    return {
        "file_id": file_id,
        "name": name,
        "mime_type": mime_type,
        "web_view_link": file_meta.get("webViewLink"),
        "modified_time": file_meta.get("modifiedTime"),
        "parent_folder_id": (file_meta.get("parents") or [None])[0],
        "content": content,
        "_deleted": False,
    }


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------
def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def google_drive_source(
    folder_id: Optional[str] = None,
    *,
    auth_mode: Optional[str] = None,
    credentials_path: Optional[str] = None,
    token_path: Optional[str] = None,
    include_subfolders: Optional[bool] = None,
    max_file_size_mb: Optional[int] = None,
    service: Any = None,
):
    """Return a ``dlt`` resource yielding one row per in-scope Google Drive file.

    Any argument left as ``None`` falls back to the matching ``GOOGLE_DRIVE_*``
    environment variable.  Hand the result to ``cognee.remember(...)`` with
    ``write_disposition="merge"`` and ``primary_key="file_id"``.

    Args:
        folder_id: Drive folder ID to sync (found in the folder's URL).
        auth_mode: ``"service_account"`` (default) or ``"oauth"``.
        credentials_path: Path to the service-account key or OAuth client-secret JSON.
        token_path: Where the cached OAuth user token is read/written (oauth mode).
        include_subfolders: Recurse into subfolders (default True).
        max_file_size_mb: Skip files larger than this (default 25).
        service: Pre-built Drive API client. Mainly an injection point for tests;
            when omitted a client is built from the auth settings above.
    """
    try:
        import dlt
    except ImportError as exc:
        raise ImportError(
            'The Google Drive connector requires the dlt extra: pip install "cognee[dlt]".'
        ) from exc

    resolved_folder_id = folder_id or os.getenv("GOOGLE_DRIVE_FOLDER_ID")
    if not resolved_folder_id:
        raise ValueError(
            "folder_id is required (pass it explicitly or set GOOGLE_DRIVE_FOLDER_ID)."
        )

    config = _DriveConfig(
        folder_id=resolved_folder_id,
        auth_mode=auth_mode or os.getenv("GOOGLE_DRIVE_AUTH_MODE", "service_account"),
        credentials_path=credentials_path or os.getenv("GOOGLE_DRIVE_CREDENTIALS_PATH"),
        token_path=token_path or os.getenv("GOOGLE_DRIVE_TOKEN_PATH"),
        include_subfolders=(
            _env_bool("GOOGLE_DRIVE_INCLUDE_SUBFOLDERS", True)
            if include_subfolders is None
            else include_subfolders
        ),
        max_file_size_mb=(
            max_file_size_mb
            if max_file_size_mb is not None
            else int(os.getenv("GOOGLE_DRIVE_MAX_FILE_SIZE_MB", "25"))
        ),
    )

    @dlt.resource(
        name="google_drive_files",
        write_disposition="merge",
        primary_key="file_id",
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
        yield from _iter_rows(client, config, dlt.current.resource_state())

    resource = google_drive_files()
    # Self-describing: declare the column carrying file content so
    # resolve_dlt_sources routes rows through normal chunking + LLM graph
    # extraction (document mode) without the caller passing dlt_content_column.
    setattr(resource, CONTENT_COLUMN_HINT_ATTR, "content")
    return resource
