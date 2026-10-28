"""Shared Google OAuth primitives used by the Drive, Gmail and Calendar connectors.

Moved from cognee.modules.integrations.google (cognee dev, 2026-10-26): the log-safe
OAuth client (``exchange_code``, ``refresh_access_token``, ``revoke_token``,
``fetch_userinfo``, ``GoogleAuthError``) and the settings base (``GoogleSettings``,
``require``) that package exported, plus the contract pieces the three service packages
build on (``google_descriptor``, ``GoogleConnector`` and the identity-only ``CONNECTOR``).
This module imports no host: cognee is needed only by
:mod:`cognee_connector_google.cognee_host`, which is installed with the ``cognee`` extra
and which nothing here imports.
"""

from cognee_connector_google.client import (
    GoogleAuthError,
    exchange_code,
    fetch_userinfo,
    refresh_access_token,
    revoke_token,
)
from cognee_connector_google.connector import (
    CONNECTOR,
    DESCRIPTOR,
    GoogleConnector,
    google_descriptor,
)
from cognee_connector_google.settings import GoogleSettings, require

__all__ = [
    "CONNECTOR",
    "DESCRIPTOR",
    "GoogleAuthError",
    "GoogleConnector",
    "GoogleSettings",
    "exchange_code",
    "fetch_userinfo",
    "google_descriptor",
    "refresh_access_token",
    "require",
    "revoke_token",
]
