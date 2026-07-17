from types import SimpleNamespace

import pytest

from cognee.modules.provenance import capture


@pytest.mark.asyncio
async def test_disabled_capture_is_zero_cost(monkeypatch):
    monkeypatch.setattr(
        capture,
        "get_provenance_config",
        lambda: SimpleNamespace(
            provenance_enabled=False,
            provenance_flush_threshold=10_000,
        ),
    )
    ctx = SimpleNamespace(provenance_buffer=None)

    assert await capture.capture_graph_provenance([object()], [], ctx) == 0
    assert ctx.provenance_buffer is None
