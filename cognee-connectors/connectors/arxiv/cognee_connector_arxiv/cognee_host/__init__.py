"""Plugs the arXiv connector into a cognee server (cognee >= 1.6.5): nothing to plug.

Everything under ``cognee_host/`` may import cognee; it is installed with the
``cognee`` extra (``pip install "cognee-connector-arxiv[cognee]"``). The package
``__init__`` never imports this subpackage, and nothing here runs at import: a host
calls ``register()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import APIRouter


def register() -> None:
    """Register this provider with cognee's integration registry: a no-op, on purpose.

    cognee's registry (``cognee.modules.integrations.registry.use_integration``) takes
    an ``OAuthIntegration``: an adapter with an authorize URL, a code exchange and
    stored credentials, served on the ``/api/v1/integrations/{provider}/...`` routes.
    arXiv has none of that: the API is public, so there is no account, no token and
    nothing to authorize. The SDK path (``remember(arxiv_source(categories=...))``)
    needs no registration, so there is nothing to register and the package declares no
    ``cognee.integrations`` entry point; this function exists so every connector
    package offers the same call.
    """


def routers() -> list[tuple[APIRouter, str, list[str]]]:
    """Routers this provider mounts on a cognee server, as (router, prefix, tags): none."""
    return []
