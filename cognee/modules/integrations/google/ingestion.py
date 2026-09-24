"""SDK-owned Google sources used by the OAuth integrations."""

import asyncio
import json
from collections.abc import Callable
from hashlib import sha256
from importlib import import_module
from typing import Any

from cognee.modules.integrations.credentials import require_active_credential


def resource_name(provider: str, credential: Any, scope: str = "") -> str:
    """dlt cursors belong to resources, not destination datasets.

    Include the owner as well as the Google account so a later owner cannot
    inherit the former owner's cursor, including Drive's shared 'root' scope.
    """
    identity = [str(credential.user_id), str(credential.provider_account_id), scope]
    digest = sha256(json.dumps(identity).encode()).hexdigest()[:24]
    prefix = "gmail_messages" if provider == "gmail" else "google_drive_files"
    return f"{prefix}_{digest}"


def extraction_checkpoint(credential: Any) -> Callable[[], None]:
    """Bridge dlt's worker thread to the API loop's async credential store."""
    loop = asyncio.get_running_loop()

    def check_active() -> None:
        asyncio.run_coroutine_threadsafe(require_active_credential(credential), loop).result()

    return check_active


async def retire_resources(
    provider: str, credential: Any, dataset_name: str, retained: set[str]
) -> None:
    """Reconcile deselected tables, including legacy names, on explicit sync.

    Inventory comes from server-owned document metadata in this owner's
    dataset. Each removed table goes through normal DLT replacement and its
    deferred graph/vector cleanup; no dataset-wide forget is involved.
    """
    from sqlalchemy import select

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.models import Data, Dataset

    async with get_relational_engine().get_async_session() as db:
        result = await db.execute(
            select(Data.system_metadata)
            .join(Dataset, Data.dataset_id == Dataset.id)
            .where(Dataset.name == dataset_name, Dataset.owner_id == credential.user_id)
        )
        retired = {
            metadata["table_name"]
            for metadata in result.scalars()
            if isinstance(metadata, dict)
            and metadata.get("source") == provider
            and isinstance(metadata.get("table_name"), str)
        } - retained
    if not retired:
        return

    from cognee.api.v1.remember.remember import remember
    from cognee.modules.users.methods import get_user

    owner = await get_user(credential.user_id)
    for table in sorted(retired):
        await require_active_credential(credential)
        result = await remember(
            empty_resource(provider, table, extraction_checkpoint(credential)),
            dataset_name=dataset_name,
            user=owner,
            write_disposition="replace",
            primary_key="id",
            max_rows_per_table=0,
            self_improvement=False,
        )
        if getattr(result, "status", None) == "errored":
            raise RuntimeError("Failed to remove deselected Google resource data")


def empty_resource(provider: str, name: str, check_active=None):
    """Explicitly empty a staging table and reset its cursor for re-selection."""
    import dlt

    from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR, PIPELINE_SCOPE_ATTR

    @dlt.resource(name=name, columns={"id": {"data_type": "text", "primary_key": True}})
    def empty():
        if check_active is not None:
            check_active()
        dlt.current.resource_state().clear()
        yield []  # Explicit empty replace, not a zero-change merge.

    resource = empty()
    setattr(resource, DOCUMENT_SOURCE_ATTR, provider)
    setattr(resource, PIPELINE_SCOPE_ATTR, name)
    return resource


def source_factory(provider: str) -> Callable[..., Any]:
    module_name, factory_name = {
        "google_drive": (
            "cognee.tasks.ingestion.connectors.google_drive",
            "google_drive_source",
        ),
        "gmail": ("cognee.tasks.ingestion.connectors.gmail", "gmail_source"),
    }[provider]
    return getattr(import_module(module_name), factory_name)


def build_service(provider: str, access_token: str) -> Any:
    """Inject the core-owned OAuth token without a connector-side login flow."""
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    api, version = {"google_drive": ("drive", "v3"), "gmail": ("gmail", "v1")}[provider]
    return build(api, version, credentials=Credentials(token=access_token), cache_discovery=False)


def source_counts(source: Any) -> dict[str, int]:
    """Read count-only diagnostics published by the SDK source."""
    raw = getattr(source, "cognee_sync_stats", None)
    if not isinstance(raw, dict):
        return {}
    return {
        key: value
        for key, value in raw.items()
        if isinstance(key, str) and type(value) is int and value >= 0
    }


def add_source_counts(totals: dict[str, int], source: Any) -> None:
    for key, value in source_counts(source).items():
        totals[key] = totals.get(key, 0) + value


# This local API runs one worker. A multi-worker deployment must replace this
# process-local guard with a durable job queue / distributed lease.
_running_syncs: set[tuple[str, str]] = set()


def sync_is_running(provider: str, account_id: str) -> bool:
    return (provider, str(account_id)) in _running_syncs


async def run_sync(provider: str, credential: Any, sync_source: Any) -> None:
    """Serialize each account's DLT cursor and expose actual in-flight state."""
    from cognee.modules.integrations.credentials import record_sync_result

    key = (provider, str(credential.provider_account_id))
    if key in _running_syncs:
        return
    _running_syncs.add(key)
    counts = {"scanned": 0, "skipped": 0, "failed": 0}
    try:
        try:
            credential = await require_active_credential(credential)
            status, counts = await sync_source(credential, counts)
        except Exception as exc:
            counts["failed"] = max(1, counts["failed"])
            # Store a safe category, never provider exception text (which may
            # include message IDs, URLs, or user content).
            message = str(exc).lower()
            if any(term in message for term in ("ratelimitexceeded", "quota exceeded", "http 429")):
                counts["failed_rate_limit"] = 1
            await record_sync_result(credential, status="degraded", counts=counts)
            raise
        await record_sync_result(credential, status=status, counts=counts)
    finally:
        _running_syncs.discard(key)


async def dataset_summary(credential: Any, dataset_name: str) -> tuple[str | None, int]:
    """Report persisted items for this connection's owner, not source scan counts."""
    from sqlalchemy import func, select

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.models.Data import Data
    from cognee.modules.data.models.Dataset import Dataset

    async with get_relational_engine().get_async_session() as db:
        result = await db.execute(
            select(Dataset.id, func.count(Data.id))
            .outerjoin(Data, Data.dataset_id == Dataset.id)
            .where(Dataset.name == dataset_name, Dataset.owner_id == credential.user_id)
            .group_by(Dataset.id)
        )
        row = result.first()
        return (str(row[0]), row[1]) if row else (None, 0)
