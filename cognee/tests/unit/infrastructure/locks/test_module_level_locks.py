"""Guard: every module-level lock we converted stays loop-safe.

These locks used to be plain ``asyncio.Lock()`` objects created at import time,
which bind to the first event loop that contends them. They are now
``LoopBoundLock`` so they work from any number of event loops. This test fails if
a raw module-level ``asyncio.Lock`` is ever reintroduced at one of these sites.
"""

import importlib

import pytest

from cognee.infrastructure.locks.loop_bound_lock import LoopBoundLock


@pytest.mark.parametrize(
    ("module_path", "lock_name"),
    [
        ("cognee.infrastructure.locks.session_lock", "_registry_lock"),
        ("cognee.infrastructure.locks.session_lock", "_improve_registry_lock"),
        ("cognee.infrastructure.databases.relational.create_db_and_tables", "_create_db_lock"),
    ],
)
def test_module_level_lock_is_loop_bound(module_path, lock_name):
    module = importlib.import_module(module_path)
    assert isinstance(getattr(module, lock_name), LoopBoundLock)
