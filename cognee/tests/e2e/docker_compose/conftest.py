"""Fixtures for the docker-compose full-stack e2e suite.

The suite assumes the stack is already up (the CI workflow brings it up before
invoking pytest; locally you run ``docker compose --profile postgres --profile
mcp up -d``). The session-scoped fixtures below block until each service is
healthy — this is the real replacement for the old ``sleep 30``.
"""

from __future__ import annotations

import os
import sys

# This suite is intentionally NOT part of the importable `cognee` package: it
# only talks to the running stack over HTTP, so it must collect on a minimal
# runner that has not installed cognee. Put this directory on sys.path so the
# sibling helper modules import as top-level (`from config import ...`) without
# dragging in `cognee/__init__.py` and its heavy dependency tree.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest  # noqa: E402

from compose_utils import wait_for_http_ok  # noqa: E402
from config import CONFIG  # noqa: E402


@pytest.fixture(scope="session")
def config():
    return CONFIG


@pytest.fixture(scope="session")
def api_ready(config):
    """Block until the main cognee API reports healthy on :8000."""
    wait_for_http_ok(config.health_url, name="cognee API (:8000)")
    return config.api_url


@pytest.fixture(scope="session")
def mcp_ready(config):
    """Block until the MCP service reports healthy on :8001."""
    wait_for_http_ok(config.mcp_health_url, name="cognee-mcp (:8001)")
    return config.mcp_url


@pytest.fixture(scope="session")
def requires_compose(config):
    """Skip compose-driving tests when this process can't run docker compose."""
    if not config.manage_compose:
        pytest.skip(
            "compose-driving test skipped (set COGNEE_E2E_MANAGE_COMPOSE=1 to enable)"
        )
    return config
