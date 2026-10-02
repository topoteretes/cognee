"""Provider-neutral entry points to the SDK-owned DLT document sources.

Hosts (the Cloud pod, the SDK's own syncs) reach every document source through
these functions, keyed by provider. A provider is one entry in ``_PROVIDERS``:
its resource-name prefix, the source factory and a service builder. The table
is static and every heavy import is lazy, so importing this module needs none
of the providers' client libraries.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module
from typing import Any


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
