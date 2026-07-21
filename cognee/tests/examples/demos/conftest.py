"""Demos-local fixtures.

Some session / feedback demos require cognee's file-system session cache to be
enabled (they raise ``CogneeConfigurationError`` otherwise). ``cached_example_env``
layers that on top of the shared ``isolated_example_env`` so those demos run
keyless without changing the base harness.
"""

from __future__ import annotations

import pytest_asyncio


@pytest_asyncio.fixture
async def cached_example_env(monkeypatch, isolated_example_env):
    from cognee.infrastructure.databases.cache.config import get_cache_config

    monkeypatch.setenv("CACHING", "true")
    monkeypatch.setenv("CACHE_BACKEND", "fs")
    # get_cache_config is lru_cached and several demos read it at import time,
    # so clear it after setting the env or the stale default backend sticks.
    get_cache_config.cache_clear()
    yield isolated_example_env
    get_cache_config.cache_clear()
