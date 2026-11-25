"""Shared fixtures: the package's module-level state is reset around every test.

The registry and the slug-collision history are process-wide dicts; without a
reset, tests registering throwaway connectors or probing dataset-name
collisions would be order-dependent on a same-process rerun.
"""

import pytest

from cognee_connectors_core import registry
from cognee_connectors_core.base import reset_slug_origins


@pytest.fixture(autouse=True)
def _isolated_module_state():
    registry.reset_registry()
    reset_slug_origins()
    yield
    registry.reset_registry()
    reset_slug_origins()
