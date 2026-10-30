"""Plugs the Google Drive connector into a cognee server (cognee >= 1.6.5).

Moved from cognee.modules.integrations.google_drive.__init__ (cognee dev, 2026-10-26),
where importing the package registered the Drive adapter with the integrations registry
as a side effect; here a host calls ``register()`` and nothing runs at import. Drive
mounts no routers of its own (its whole surface is the generic
``/api/v1/integrations/google_drive/*`` routes, and it receives no webhooks), so
``routers()`` is empty.

Everything under ``cognee_host/`` may import cognee; it is installed with the ``cognee``
extra (``pip install "cognee-connector-google-drive[cognee]"``). The package ``__init__``
never imports this subpackage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import APIRouter


def register() -> None:
    """Register this provider with cognee's integration registry (cognee >= 1.6.5)."""
    from cognee.modules.integrations.registry import use_integration

    from cognee_connector_google_drive.cognee_host.adapter import GoogleDriveIntegration

    use_integration(GoogleDriveIntegration())


def routers() -> list[tuple[APIRouter, str, list[str]]]:
    """Routers this provider mounts on a cognee server, as (router, prefix, tags)."""
    return []
