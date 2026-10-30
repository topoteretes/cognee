"""Plugs the Obsidian connector into a cognee server (cognee >= 1.6.5): nothing to plug.

Everything under ``cognee_host/`` may import cognee; it is installed with the ``cognee``
extra (``pip install "cognee-connector-obsidian[cognee]"``). The package ``__init__`` never
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
    credentials, served on the ``/api/v1/integrations/{provider}/...`` routes. Obsidian has
    none of that in cognee, and cannot have it: a vault is a folder on the machine that runs
    the sync, which nothing hosted can reach, so cognee's integration provider table never
    listed it and the connector is SDK only. There is no account and no token, and the SDK
    path (``remember(obsidian_source(vault_path=...))``) needs no registration. So there is
    nothing to register, and the package declares no ``cognee.integrations`` entry point; this
    function exists so every connector package offers the same call.
    """


def routers() -> list[tuple[APIRouter, str, list[str]]]:
    """Routers this provider mounts on a cognee server, as (router, prefix, tags): none."""
    return []
