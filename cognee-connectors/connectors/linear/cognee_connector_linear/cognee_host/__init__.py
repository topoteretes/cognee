"""Plugs the Linear connector into a cognee server (cognee >= 1.6.5).

Everything under ``cognee_host/`` may import cognee; it is installed with the
``cognee`` extra (``pip install "cognee-connector-linear[cognee]"``). The package
``__init__`` never imports this subpackage, and nothing here runs at import: a host calls
``register()``, which replaces the import-time ``use_integration(LinearIntegration())``
cognee's own ``cognee.modules.integrations.linear`` package had.

Linear mounts no routers of its own: its webhooks, agent session events included, arrive
on the generic ``POST /api/v1/integrations/linear/events`` route. The worker that
continues syncs cut short by Linear's quota is an ASGI lifespan for the host to mount,
:func:`cognee_connector_linear.cognee_host.resume_sync.linear_resume_lifespan`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import APIRouter


def register() -> None:
    """Register this provider with cognee's integration registry (cognee >= 1.6.5)."""
    from cognee.modules.integrations.registry import use_integration

    from cognee_connector_linear.cognee_host.adapter import LinearIntegration

    use_integration(LinearIntegration())


def routers() -> list[tuple[APIRouter, str, list[str]]]:
    """Routers this provider mounts on a cognee server, as (router, prefix, tags)."""
    return []
