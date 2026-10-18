"""Provider-neutral entry points to the SDK-owned DLT document sources.

Hosts (the Cloud pod, the SDK's own syncs) reach every document source through
these functions, keyed by provider. A provider is one entry in ``_PROVIDERS``:
its resource-name prefix, the source factory and a service builder. The table
is static and every heavy import is lazy, so importing this module needs none
of the providers' client libraries.
"""

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module
from typing import Any

from cognee.shared.logging_utils import get_logger

logger = get_logger("integrations_ingestion")


class RateLimitedError(RuntimeError):
    """A provider refused a call because its rate limit is spent."""


@dataclass(frozen=True)
class _Provider:
    prefix: str
    source_module: str
    source_name: str
    service_builder: Callable[[str], Any] | None = None


def _google_service_builder(api: str, version: str) -> Callable[[str], Any]:
    def build_service(access_token: str) -> Any:
        """Inject the core-owned OAuth token without a connector-side login flow."""
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        return build(
            api, version, credentials=Credentials(token=access_token), cache_discovery=False
        )

    return build_service


def _linear_service(access_token: str) -> Any:
    from cognee.tasks.ingestion.connectors.linear import build_linear_service

    return build_linear_service(access_token)


_PROVIDERS: dict[str, _Provider] = {
    "google_drive": _Provider(
        prefix="google_drive_files",
        source_module="cognee.tasks.ingestion.connectors.google_drive",
        source_name="google_drive_source",
        service_builder=_google_service_builder("drive", "v3"),
    ),
    "gmail": _Provider(
        prefix="gmail_messages",
        source_module="cognee.tasks.ingestion.connectors.gmail",
        source_name="gmail_source",
        service_builder=_google_service_builder("gmail", "v1"),
    ),
    "google_calendar": _Provider(
        prefix="google_calendar_events",
        source_module="cognee.tasks.ingestion.connectors.google_calendar",
        source_name="google_calendar_source",
        service_builder=_google_service_builder("calendar", "v3"),
    ),
    "linear": _Provider(
        prefix="linear",
        source_module="cognee.tasks.ingestion.connectors.linear",
        source_name="linear_source",
        service_builder=_linear_service,
    ),
}


def _provider(provider: str) -> _Provider:
    try:
        return _PROVIDERS[provider]
    except KeyError:
        raise KeyError(f"Unsupported document provider: {provider!r}") from None


def resource_name(provider: str, credential: Any, scope: str = "") -> str:
    """dlt cursors belong to resources, not destination datasets.

    Include the owner as well as the provider account so a later owner cannot
    inherit the former owner's cursor, including Drive's shared 'root' scope.

    The result names the DLT table and cursor of a connection, so it must stay
    byte-identical for a provider once tenants have synced with it.
    """
    prefix = _provider(provider).prefix
    identity = [str(credential.user_id), str(credential.provider_account_id), scope]
    digest = sha256(json.dumps(identity).encode()).hexdigest()[:24]
    return f"{prefix}_{digest}"


def dataset_name_for_account(prefix: str, email: str, account_id: str = "") -> str:
    """``{prefix}_{email slug}``, plus a 10 hex suffix of ``account_id`` when it is given.

    The name is a stored identifier, so the slug rule and the suffix length must not change
    once a provider has synced: every dataset already created under the old name would be
    orphaned. The suffix keeps accounts apart whose slugs collide (``a.b`` and ``a_b``).
    """
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", email or "").strip("_").lower()
    name = f"{prefix}_{slug or 'account'}"
    return f"{name}_{sha256(account_id.encode()).hexdigest()[:10]}" if account_id else name


def dataset_name_for_credential(prefix: str, credential: Any) -> str:
    """The dataset a connected account syncs into, named from its stored email and id."""
    email = (credential.provider_metadata or {}).get("email") or ""
    return dataset_name_for_account(prefix, str(email), str(credential.provider_account_id))


def source_factory(provider: str) -> Callable[..., Any]:
    entry = _provider(provider)
    return getattr(import_module(entry.source_module), entry.source_name)


def build_service(provider: str, access_token: str) -> Any:
    builder = _provider(provider).service_builder
    if builder is None:
        raise KeyError(f"Document provider {provider!r} has no service to build")
    return builder(access_token)


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


async def require_active_credential(credential: Any) -> Any:
    """Raise unless the connection is still active. Imported late to keep this module light."""
    from cognee.modules.integrations.credentials import require_active_credential as require

    return await require(credential)


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
            raise RuntimeError("Failed to remove deselected resource data")


async def sync_scopes(
    provider: str,
    credential: Any,
    counts: dict[str, int],
    *,
    scopes: list[str],
    dataset_name: str,
    make_source: Callable[[str, str, Callable[[], None]], Any],
    classify_error: Callable[[BaseException], str] | None = None,
) -> set[str]:
    """Ingest one source per scope into the connection's dataset and return the tables kept.

    ``make_source(scope, resource_name, check_active)`` builds the DLT source.
    A scope that fails is counted and logged, and the loop goes on with the
    others; an inactive connection stops it. ``classify_error`` maps a failure
    to the count key it is reported under (default ``failed_ingestion``).
    The caller retires deselected tables only when no scope failed.
    """
    from cognee.api.v1.remember.remember import remember
    from cognee.modules.integrations.credentials import CredentialInactiveError
    from cognee.modules.users.methods import get_user

    owner = await get_user(credential.user_id)
    retained: set[str] = set()
    for scope in scopes:
        await require_active_credential(credential)
        source = None
        try:
            name = resource_name(provider, credential, scope)
            source = make_source(scope, name, extraction_checkpoint(credential))
            retained.add(name)
            result = await remember(
                source,
                dataset_name=dataset_name,
                user=owner,
                write_disposition="merge",
                primary_key="id",
                max_rows_per_table=0,
                self_improvement=False,
            )
            if getattr(result, "status", None) == "errored":
                counts["failed"] += 1
                counts["failed_ingestion"] = counts.get("failed_ingestion", 0) + 1
        except CredentialInactiveError:
            raise
        except Exception as exc:
            key = (classify_error(exc) if classify_error else None) or "failed_ingestion"
            counts["failed"] += 1
            counts[key] = counts.get(key, 0) + 1
            logger.exception(
                "%s sync failed for account %s scope %s",
                provider,
                credential.provider_account_id,
                scope,
            )
        finally:
            add_source_counts(counts, source)
    return retained


async def sync_selected_resources(
    provider: str,
    credential: Any,
    counts: dict[str, int],
    *,
    selection: list[str] | None,
    selection_kwarg: str,
    dataset_name: str,
    access_token_for: Callable[[Any], Awaitable[str]],
) -> tuple[str, dict[str, int]]:
    """Ingest the connection's whole selection through one source and return the sync outcome.

    The source takes the selection as the ``selection_kwarg`` keyword (``label_ids``,
    ``calendar_ids``) and keeps its own per-resource state, so a changed selection
    needs no extra resource here. An empty selection ingests nothing and retires
    everything the account synced. A failed ingestion keeps the old tables and marks
    the sync degraded; the deselected ones are retired only after a clean run. The
    caller checks the shape of ``selection``.
    """
    from cognee.api.v1.remember.remember import remember
    from cognee.modules.users.methods import get_user

    if selection == []:
        await retire_resources(provider, credential, dataset_name, set())
        return "ok", {"scanned": 0, "skipped": 0, "failed": 0}

    make_source = source_factory(provider)
    service = build_service(provider, await access_token_for(credential))
    owner = await get_user(credential.user_id)
    name = resource_name(provider, credential)
    source = make_source(
        **{selection_kwarg: selection},
        service=service,
        resource_name=name,
        check_active=extraction_checkpoint(credential),
    )
    await require_active_credential(credential)
    try:
        result = await remember(
            source,
            dataset_name=dataset_name,
            user=owner,
            write_disposition="merge",
            primary_key="id",
            max_rows_per_table=0,
            self_improvement=False,
        )
    finally:
        add_source_counts(counts, source)
    if getattr(result, "status", None) == "errored":
        counts["failed"] += 1
        counts["failed_ingestion"] = 1
    else:
        await retire_resources(provider, credential, dataset_name, {name})
    return ("degraded" if counts["failed"] else "ok"), counts


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
            if isinstance(exc, RateLimitedError) or any(
                term in message for term in ("ratelimitexceeded", "quota exceeded", "http 429")
            ):
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
