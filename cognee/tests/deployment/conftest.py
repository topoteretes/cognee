"""Pytest fixtures and marker registration for deployment end-to-end tests.

Deployment tests run a prebuilt ``cognee-mcp`` Docker image and are therefore
opt-in via the ``deployment`` marker (select with ``pytest -m deployment``).
They are skipped automatically when Docker is unavailable or the image has not
been built.
"""

from __future__ import annotations

import os
from typing import Iterator

import pytest

from mcp_harness import (
    MCPContainer,
    docker_available,
    image_exists,
    run_mcp_http_container,
)

DEFAULT_MCP_IMAGE = os.getenv("COGNEE_MCP_IMAGE", "cognee-mcp:local")
HEALTH_TIMEOUT = float(os.getenv("COGNEE_DEPLOYMENT_HEALTH_TIMEOUT", "120"))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "deployment: end-to-end tests that run a prebuilt Docker image "
        "(opt-in; excluded from the default unit run).",
    )


@pytest.fixture(scope="session")
def mcp_image() -> str:
    """Return the tag of a locally available cognee-mcp image.

    Skips the whole suite (rather than failing) when Docker is unavailable or
    the image has not been built, so the marker stays friendly on dev machines.
    The image is built explicitly beforehand (CI does this) rather than from
    inside a fixture, so a bare ``pytest -m deployment`` never triggers a
    surprise multi-minute build.
    """
    if not docker_available():
        pytest.skip("Docker daemon is not available; skipping deployment tests.")
    if not image_exists(DEFAULT_MCP_IMAGE):
        pytest.skip(
            f"cognee-mcp image {DEFAULT_MCP_IMAGE!r} not found; build it first:\n"
            f"  docker build -f cognee-mcp/Dockerfile -t {DEFAULT_MCP_IMAGE} ."
        )
    return DEFAULT_MCP_IMAGE


@pytest.fixture(scope="module")
def mcp_http_container(mcp_image: str) -> Iterator[MCPContainer]:
    """A module-scoped cognee-mcp container in HTTP transport (direct) mode."""
    with run_mcp_http_container(mcp_image, health_timeout=HEALTH_TIMEOUT) as container:
        yield container
