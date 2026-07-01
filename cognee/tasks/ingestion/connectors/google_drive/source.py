"""Google Drive dlt source: one row per in-scope file, incremental via the
Drive Changes API, deletions propagated as dlt merge hard-delete rows.

IMPORTANT — callers must pass ``write_disposition="merge"`` and
``primary_key="file_id"`` to ``cognee.remember()``/``cognee.add()``
explicitly. Those values also control ``resolve_dlt_sources``' own
orphan-cleanup gating and are not inherited from the resource's own hints.
Using the default ``write_disposition="replace"`` would drop and recreate
the destination table on every run, discarding unchanged files that
weren't re-yielded on an incremental pass.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from cognee.shared.logging_utils import get_logger
from .config import GoogleDriveConfig, get_google_drive_config
from .auth import build_drive_service
from .content import extract_file_content, is_supported_mime_type, GOOGLE_FOLDER_MIME_TYPE
from .exceptions import GoogleDriveAPIError, GoogleDriveConfigError

logger = get_logger("google_drive_source")

FILE_FIELDS = "id, name, mimeType, modifiedTime, webViewLink, parents, trashed, size"


def create_google_drive_source(
    folder_id: Optional[str] = None,
    *,
    auth_mode: Optional[str] = None,
    credentials_path: Optional[str] = None,
    token_path: Optional[str] = None,
    include_subfolders: Optional[bool] = None,
    max_file_size_mb: Optional[int] = None,
):
    """Build a dlt resource yielding one row per in-scope Google Drive file.

    Any argument left as ``None`` falls back to ``GoogleDriveConfig`` env
    vars (``GOOGLE_DRIVE_*``). Pass the result to::

        await cognee.remember(
            create_google_drive_source(folder_id="..."),
            dataset_name="my_drive_folder",
            primary_key="file_id",
            write_disposition="merge",
            dlt_content_column="content",
            max_rows_per_table=0,  # GOOGLE_DRIVE folders often exceed the default 50-row cap
        )
    """
    try:
        import dlt
    except ImportError as e:
        raise GoogleDriveConfigError(
            message="Google Drive ingestion requires the 'dlt' extra. Install with: pip install cognee[dlt]"
        ) from e

    env_config = get_google_drive_config()
    resolved_folder_id = folder_id or env_config.google_drive_folder_id
    if not resolved_folder_id:
        raise GoogleDriveConfigError(
            message="folder_id is required (pass explicitly or set GOOGLE_DRIVE_FOLDER_ID)."
        )

    config = GoogleDriveConfig(
        google_drive_folder_id=resolved_folder_id,
        google_drive_auth_mode=auth_mode or env_config.google_drive_auth_mode,
        google_drive_credentials_path=credentials_path or env_config.google_drive_credentials_path,
        google_drive_token_path=token_path or env_config.google_drive_token_path,
        google_drive_include_subfolders=(
            env_config.google_drive_include_subfolders
            if include_subfolders is None
            else include_subfolders
        ),
        google_drive_max_file_size_mb=max_file_size_mb or env_config.google_drive_max_file_size_mb,
    )

    @dlt.resource(
        name="google_drive_files",
        write_disposition="merge",
        primary_key="file_id",
        columns={"deleted": {"hard_delete": True}},
    )
    def google_drive_files():
        service = build_drive_service(config)
        state = dlt.current.resource_state()
        yield from _iter_rows(service, config, state)

    return google_drive_files()


def _iter_rows(service, config: GoogleDriveConfig, state: dict):
    """Core sync generator, decoupled from dlt's resource-state machinery
    (which requires an active pipeline context) so it's directly unit
    testable with a fake Drive service and a plain dict standing in for
    dlt's resource state.
    """
    page_token = state.get("page_token")
    active_file_ids: Set[str] = set(state.get("active_file_ids", []))

    if page_token is None:
        new_active_ids: Set[str] = set()
        for file_meta in _list_files_in_scope(service, config):
            row = _file_to_row(service, file_meta, config)
            if row is not None:
                new_active_ids.add(file_meta["id"])
                yield row
        state["active_file_ids"] = sorted(new_active_ids)
        state["page_token"] = _get_start_page_token(service)
        return

    changed_ids, deleted_ids, new_page_token = _list_changed_file_ids(service, page_token)
    scope_folder_ids = _get_scope_folder_ids(service, config)

    for file_id in deleted_ids:
        active_file_ids.discard(file_id)
        yield {"file_id": file_id, "deleted": True}

    for file_id in changed_ids:
        try:
            file_meta = service.files().get(fileId=file_id, fields=FILE_FIELDS).execute()
        except Exception as e:
            if _is_not_found(e):
                active_file_ids.discard(file_id)
                yield {"file_id": file_id, "deleted": True}
                continue
            raise GoogleDriveAPIError(
                message=f"Failed to fetch metadata for Drive file '{file_id}': {e}"
            ) from e

        if file_meta.get("trashed") or not _is_in_scope(file_meta, scope_folder_ids):
            active_file_ids.discard(file_id)
            yield {"file_id": file_id, "deleted": True}
            continue

        row = _file_to_row(service, file_meta, config)
        if row is not None:
            active_file_ids.add(file_id)
            yield row

    state["active_file_ids"] = sorted(active_file_ids)
    state["page_token"] = new_page_token


def _get_scope_folder_ids(service, config: GoogleDriveConfig) -> Set[str]:
    root = config.google_drive_folder_id
    if not config.google_drive_include_subfolders:
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
                raise GoogleDriveAPIError(
                    message=f"Failed to list subfolders of Drive folder '{current}': {e}"
                ) from e

            for f in response.get("files", []):
                if f["id"] not in scope_ids:
                    scope_ids.add(f["id"])
                    queue.append(f["id"])

            page_token = response.get("nextPageToken")
            if not page_token:
                break

    return scope_ids


def _list_files_in_scope(service, config: GoogleDriveConfig):
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
            raise GoogleDriveAPIError(
                message=f"Failed to list files in Drive folder '{folder_id}': {e}"
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
        raise GoogleDriveAPIError(
            message=f"Failed to get Drive changes start page token: {e}"
        ) from e
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
            raise GoogleDriveAPIError(message=f"Failed to list Drive changes: {e}") from e

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
    status_code = getattr(getattr(e, "resp", None), "status", None)
    return status_code == 404


def _file_to_row(service, file_meta: Dict[str, Any], config: GoogleDriveConfig) -> Optional[dict]:
    mime_type = file_meta.get("mimeType", "")
    name = file_meta.get("name", "")
    file_id = file_meta["id"]

    if not is_supported_mime_type(mime_type):
        logger.warning(
            "Skipping unsupported Drive file '%s' (%s): mime type '%s' is not supported.",
            name,
            file_id,
            mime_type,
        )
        return None

    size = file_meta.get("size")
    if size and int(size) > config.google_drive_max_file_size_mb * 1024 * 1024:
        logger.warning(
            "Skipping Drive file '%s' (%s): size exceeds max_file_size_mb=%d.",
            name,
            file_id,
            config.google_drive_max_file_size_mb,
        )
        return None

    content = extract_file_content(service, file_id, mime_type, name)
    if content is None:
        return None

    return {
        "file_id": file_id,
        "name": name,
        "mime_type": mime_type,
        "web_view_link": file_meta.get("webViewLink"),
        "modified_time": file_meta.get("modifiedTime"),
        "parent_folder_id": (file_meta.get("parents") or [None])[0],
        "content": content,
        "deleted": False,
    }
