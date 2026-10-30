"""Plugs the GitHub connector into a cognee server (cognee >= 1.6.5).

Moved from cognee.modules.integrations.github.__init__ (cognee dev,
2026-10-26), where importing the package registered the adapter as a side
effect; here a host calls ``register()`` explicitly and nothing runs at
import. GitHub mounts no routers of its own — its webhooks arrive on the
generic ``POST /api/v1/integrations/github/events`` route — so ``routers()``
is empty.

Everything under ``cognee_host/`` may import cognee; it is installed with the
``cognee`` extra (``pip install "cognee-connector-github[cognee]"``). The
package ``__init__`` never imports this subpackage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import APIRouter


def register() -> None:
    """Register this provider with cognee's integration registry (cognee >= 1.6.5)."""
    from cognee.modules.integrations.registry import use_integration

    from cognee_connector_github.cognee_host.adapter import GithubIntegration

    use_integration(GithubIntegration())


def routers() -> list[tuple[APIRouter, str, list[str]]]:
    """Routers this provider mounts on a cognee server, as (router, prefix, tags)."""
    return []


__all__ = ["register", "routers"]
