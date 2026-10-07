"""Google Tasks connector for cognee — a ``dlt`` source that turns your tasks into memory.

Pull Google Tasks into cognee, incrementally and with forget-on-deletion.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from typing import Any

from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR

logger = get_logger("google_tasks_connector")

TASKS_READONLY_SCOPE = "https://www.googleapis.com/auth/tasks.readonly"
_NUM_RETRIES = 6

def build_tasks_service(
    credentials_path: str = "credentials.json",
    token_path: str = "token.json",
) -> Any:
    """Build an authenticated Google Tasks API client via OAuth2."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise ImportError(
            "The Google Tasks connector requires the 'google-tasks' extra. Install it with:\n"
            '    pip install "cognee[google-tasks]"\n'
        ) from exc

    scopes = [TASKS_READONLY_SCOPE]
    creds = None

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, scopes)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(credentials_path):
                raise FileNotFoundError(
                    f"Google Tasks OAuth client secrets not found at '{credentials_path}'."
                )
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, scopes)
            creds = flow.run_local_server(port=0)
        
        fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as token_file:
            token_file.write(creds.to_json())
        os.chmod(token_path, 0o600)

    return build("tasks", "v1", credentials=creds, cache_discovery=False)


def build_tasks_service_from_access_token(access_token: str) -> Any:
    """Build a Google Tasks client from a short-lived token."""
    try:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise ImportError(
            "The Google Tasks connector requires the 'google-tasks' extra. Install it with:\n"
            '    pip install "cognee[google-tasks]"\n'
        ) from exc
    credentials = Credentials(token=access_token, scopes=[TASKS_READONLY_SCOPE])
    return build("tasks", "v1", credentials=credentials, cache_discovery=False)


def _document_content(task: dict, tasklist_title: str) -> str:
    lines = [
        f"Task List: {tasklist_title}",
        f"Status: {task.get('status', 'needsAction')}",
    ]
    if task.get("due"):
        lines.append(f"Due: {task.get('due')}")
    if task.get("parent"):
        lines.append(f"Parent: {task.get('parent')}")
        
    for link in task.get("links", []):
        link_url = link.get("link")
        if link_url:
            lines.append(f"Link: {link_url}")
            
    notes = (task.get("notes") or "").strip()
    if notes:
        lines.extend(["", notes])
        
    return "\n".join(lines).strip()


def parse_task(task: dict, tasklist_title: str) -> dict[str, Any]:
    return {
        "id": task.get("id"),
        "title": task.get("title", ""),
        "content": _document_content(task, tasklist_title),
        "_deleted": False,
    }

def _deleted_row(task_id: str) -> dict[str, Any]:
    return {"id": task_id, "_deleted": True}


def sync_tasks(
    service: Any,
    state: dict,
    stats: dict[str, int] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield tasks from all task lists incrementally."""
    if stats is None:
        stats = {}
        
    last_updated_min = state.get("last_updated_min")
    sync_start_time = datetime.now(timezone.utc).isoformat()
    
    prev_tasklist_tasks = state.get("tasklist_tasks", {})
    new_tasklist_tasks = {}
    
    tasklists = []
    page_token = None
    while True:
        res = service.tasklists().list(maxResults=100, pageToken=page_token).execute(num_retries=_NUM_RETRIES)
        tasklists.extend(res.get("items", []))
        page_token = res.get("nextPageToken")
        if not page_token:
            break
            
    current_tlist_ids = {t["id"] for t in tasklists}
    for old_tlist_id, old_task_ids in prev_tasklist_tasks.items():
        if old_tlist_id not in current_tlist_ids:
            for task_id in old_task_ids:
                stats["deleted"] = stats.get("deleted", 0) + 1
                yield _deleted_row(task_id)

    for tlist in tasklists:
        tlist_id = tlist["id"]
        tlist_title = tlist.get("title", "")
        
        valid_task_ids = set(prev_tasklist_tasks.get(tlist_id, []))
        
        page_token = None
        while True:
            kwargs = {
                "tasklist": tlist_id,
                "showDeleted": True,
                "showHidden": True,
                "maxResults": 100,
            }
            if page_token:
                kwargs["pageToken"] = page_token
            if last_updated_min:
                kwargs["updatedMin"] = last_updated_min
                
            try:
                res = service.tasks().list(**kwargs).execute(num_retries=_NUM_RETRIES)
            except Exception as e:
                if getattr(getattr(e, "resp", None), "status", None) in (404, 410):
                    break
                stats["failed"] = stats.get("failed", 0) + 1
                raise
                
            for task in res.get("items", []):
                stats["scanned"] = stats.get("scanned", 0) + 1
                task_id = task["id"]
                
                if task.get("deleted") or task.get("hidden"):
                    stats["deleted"] = stats.get("deleted", 0) + 1
                    valid_task_ids.discard(task_id)
                    yield _deleted_row(task_id)
                else:
                    valid_task_ids.add(task_id)
                    yield parse_task(task, tlist_title)
                    
            page_token = res.get("nextPageToken")
            if not page_token:
                break
                
        new_tasklist_tasks[tlist_id] = list(valid_task_ids)
        
    state["tasklist_tasks"] = new_tasklist_tasks
    state["last_updated_min"] = sync_start_time


def google_tasks_source(
    *,
    resource_name: str = "google_tasks",
    check_active: Callable[[], None] | None = None,
    credentials_path: str = "credentials.json",
    token_path: str = "token.json",
    service: Any = None,
):
    try:
        import dlt
    except ImportError as exc:
        raise ImportError(
            "The Google Tasks connector requires the 'google-tasks' extra. Install it with:\n"
            '    pip install "cognee[google-tasks]"\n'
        ) from exc

    if getattr(dlt_utils, "DOCUMENT_SYNC_VERSION", 0) < 1:
        raise RuntimeError(
            "Google Tasks sync requires a Cognee build with table-scoped DLT document cleanup."
        )

    stats: dict[str, int] = {}

    @dlt.resource(
        name=resource_name,
        primary_key="id",
        write_disposition="merge",
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def tasks_resource():
        stats.clear()
        stats.update(scanned=0, skipped=0, failed=0, deleted=0)
        client = service or build_tasks_service(credentials_path, token_path)
        resource_state = dlt.current.resource_state()
        
        rows = sync_tasks(client, resource_state, stats=stats)
        yield from dlt_utils.guarded_rows(rows, check_active)

    resource = tasks_resource()
    setattr(resource, DOCUMENT_SOURCE_ATTR, "google_tasks")
    setattr(resource, dlt_utils.PIPELINE_SCOPE_ATTR, resource_name)
    resource.cognee_sync_stats = stats
    return resource
