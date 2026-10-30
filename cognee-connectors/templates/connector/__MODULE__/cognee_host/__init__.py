"""Plugs the __DISPLAY_NAME__ connector into a cognee server (cognee >= 1.6.5).

Everything under ``cognee_host/`` may import cognee; it is installed with the
``cognee`` extra (``pip install "__DIST__[cognee]"``). The package ``__init__`` never
imports this subpackage, and nothing here runs at import: a host calls ``register()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import APIRouter


def register() -> None:
    """Register this provider with cognee's integration registry (cognee >= 1.6.5)."""
    from cognee.modules.integrations.registry import use_integration

    from __MODULE__.cognee_host.adapter import __CLASS__Integration

    use_integration(__CLASS__Integration())


def routers() -> list[tuple[APIRouter, str, list[str]]]:
    """Routers this provider mounts on a cognee server, as (router, prefix, tags)."""
    return []
