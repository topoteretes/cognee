"""Regression tests: session-cache QA entries use a timezone-aware UTC timestamp.

The cache adapters previously stamped ``SessionQAEntry.time`` with the deprecated,
timezone-*naive* ``datetime.utcnow().isoformat()``. They now use
``datetime.now(timezone.utc).isoformat()``, which is timezone-aware and carries an
explicit UTC offset. These tests pin that behavior independently of the Python
version (``datetime.utcnow`` only emits a DeprecationWarning on 3.12+).
"""

from datetime import datetime

import pytest

from cognee.infrastructure.databases.cache.sql.SqlCacheAdapter import SqlCacheAdapter
from cognee.infrastructure.databases.cache.fscache.FsCacheAdapter import FSCacheAdapter


@pytest.mark.parametrize("adapter_cls", [SqlCacheAdapter, FSCacheAdapter])
def test_qa_entry_time_is_timezone_aware_utc(adapter_cls):
    dump = adapter_cls._build_qa_entry_dump(question="q", context="c", answer="a")

    parsed = datetime.fromisoformat(dump["time"])
    assert parsed.tzinfo is not None, (
        f"{adapter_cls.__name__} QA entry 'time' must be timezone-aware, got {dump['time']!r}"
    )
    assert parsed.utcoffset().total_seconds() == 0, (
        f"{adapter_cls.__name__} QA entry 'time' must be UTC, got {dump['time']!r}"
    )
