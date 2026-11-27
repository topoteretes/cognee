"""Helpers for tests of the streamed code graph load."""

import importlib
from unittest.mock import AsyncMock

from cognee.tasks.code_graph.extract_code_graph import (
    _SnapshotEnd,
    extract_code_graph,
    load_code_graph,
)
from cognee.tasks.code_graph.models import CodeRepository


async def extracted_points(*args, **kwargs) -> list:
    """Every DataPoint ``extract_code_graph`` streams, repository nodes first.

    The order ``map_facts_to_data_points`` returns; an unchanged snapshot,
    which streams nothing, gives [].
    """
    points = []
    async for item in extract_code_graph(*args, **kwargs):
        if not isinstance(item, _SnapshotEnd):
            points.extend(item)
    return [point for point in points if isinstance(point, CodeRepository)] + [
        point for point in points if not isinstance(point, CodeRepository)
    ]


async def load_with_node_writes_mocked(monkeypatch, **kwargs) -> AsyncMock:
    """Run ``load_code_graph`` with ``add_data_points`` replaced by a mock, which is returned.

    What is left is the graph read, the edges, the sweep, the detag and the
    stamp, against whatever graph engine the test installed.
    """
    write = AsyncMock()
    storage_module = importlib.import_module("cognee.tasks.storage.add_data_points")
    monkeypatch.setattr(storage_module, "add_data_points", write)
    await load_code_graph(**kwargs)
    return write
