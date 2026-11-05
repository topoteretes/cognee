"""Plugs the OneNote connector into a cognee server (cognee >= 1.6.5): nothing to plug.

Everything under ``cognee_host/`` may import cognee; it is installed with the ``cognee``
extra (``pip install "cognee-connector-onenote[cognee]"``). The package ``__init__`` never
imports this subpackage, and nothing here runs at import: a host calls ``register()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import APIRouter


def register() -> None:
    """Register this provider with cognee's integration registry: a no-op, on purpose.

    cognee's registry (``cognee.modules.integrations.registry.use_integration``) takes an
    ``OAuthIntegration``: an adapter with an authorize URL, a code exchange and stored
    credentials, served on the ``/api/v1/integrations/{provider}/...`` routes. OneNote has
    none of that in cognee. It reads Microsoft Graph with a delegated access token the caller
    passes to ``onenote_source`` (the source never logs in and never refreshes it); cognee
    ships no adapter, no credential flow, no routes and no ``ONENOTE_*`` settings for it, and
    the SDK path (``remember(onenote_source(...))``) needs no registration. A host that stores
    Microsoft tokens builds the source itself, the way ``cognee.modules.integrations.ingestion``
    does (its provider table names ``onenote_source`` with no service builder). So there is
    nothing to register, and the package declares no ``cognee.integrations`` entry point; this
    function exists so every connector package offers the same call.
    """


def routers() -> list[tuple[APIRouter, str, list[str]]]:
    """Routers this provider mounts on a cognee server, as (router, prefix, tags): none."""
    return []
