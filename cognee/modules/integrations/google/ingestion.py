"""SDK-owned Google sources used by the OAuth integrations."""

from collections.abc import Callable
from importlib import import_module
from typing import Any


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
