"""Lazy loading for Google community sources used by the SDK integrations."""

from collections.abc import Callable
from importlib import import_module
from typing import Any


def source_factory(provider: str) -> Callable[..., Any]:
    module_name, factory_name, package = {
        "google_drive": (
            "cognee_community_connector_google_drive",
            "google_drive_source",
            "google-drive",
        ),
        "gmail": ("cognee_community_connector_gmail", "gmail_source", "gmail"),
    }[provider]
    try:
        return getattr(import_module(module_name), factory_name)
    except ImportError as error:
        raise RuntimeError(
            f"{provider} sync requires cognee-community-connector-{package}. "
            "Install the community connector with its dependencies before syncing."
        ) from error


def build_service(provider: str, access_token: str) -> Any:
    """Inject the core-owned OAuth token without a connector-side login flow."""
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    api, version = {"google_drive": ("drive", "v3"), "gmail": ("gmail", "v1")}[provider]
    return build(api, version, credentials=Credentials(token=access_token), cache_discovery=False)


def source_counts(source: Any) -> dict[str, int]:
    """Read count-only diagnostics published by the community source."""
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
