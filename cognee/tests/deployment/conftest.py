"""Pytest fixtures and marker registration for deployment end-to-end tests.

Deployment tests build and run Docker images and are therefore opt-in via the
``deployment`` marker (select with ``pytest -m deployment``). They are skipped
automatically when no Docker daemon is reachable.
"""

from __future__ import annotations

import os
from typing import Iterator

import pytest

from mcp_harness import (
    MCPContainer,
    docker_available,
    ensure_mcp_image,
    run_mcp_http_container,
)

DEFAULT_MCP_IMAGE = os.getenv("COGNEE_MCP_IMAGE", "cognee-mcp:local")
HEALTH_TIMEOUT = float(os.getenv("COGNEE_DEPLOYMENT_HEALTH_TIMEOUT", "120"))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "deployment: end-to-end tests that build and run Docker images "
        "(opt-in; excluded from the default unit run).",
    )


@pytest.fixture(scope="session")
def mcp_image() -> str:
    """Ensure the cognee-mcp image exists locally and return its tag.

    Skips the whole suite (rather than failing) when Docker is unavailable or
    the image cannot be built, so the marker stays friendly on dev machines.
    """
    if not docker_available():
        pytest.skip("Docker daemon is not available; skipping deployment tests.")
    try:
        return ensure_mcp_image(DEFAULT_MCP_IMAGE)
    except Exception as exc:  # noqa: BLE001 - surface as skip with context
        pytest.skip(f"Could not build/find cognee-mcp image {DEFAULT_MCP_IMAGE!r}: {exc}")


@pytest.fixture(scope="module")
def mcp_http_container(mcp_image: str) -> Iterator[MCPContainer]:
    """A module-scoped cognee-mcp container in HTTP transport (direct) mode."""
    with run_mcp_http_container(mcp_image, health_timeout=HEALTH_TIMEOUT) as container:
        yield container
