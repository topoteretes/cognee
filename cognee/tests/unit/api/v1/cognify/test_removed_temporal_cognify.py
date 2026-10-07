"""temporal_cognify is no longer a cognify()/remember() option.

The default pipeline extracts dates as Timestamp nodes and SearchType.TEMPORAL reads
them, so the event pipeline the flag switched to is no longer offered. cognify()
forwards unknown keyword arguments into the extraction LLM call, so a stale flag has
to raise up front instead of reaching it.
"""

import importlib

import pytest

cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")
remember_module = importlib.import_module("cognee.api.v1.remember.remember")


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [True, False])
async def test_cognify_rejects_temporal_cognify(value):
    with pytest.raises(TypeError, match="no longer takes temporal_cognify"):
        await cognify_module.cognify(temporal_cognify=value)


def test_remember_does_not_route_temporal_cognify():
    assert "temporal_cognify" not in remember_module._COGNIFY_ONLY
    assert "temporal_cognify" not in remember_module._ADD_ONLY
    assert "temporal_cognify" not in remember_module._SHARED


@pytest.mark.asyncio
async def test_remember_rejects_temporal_cognify():
    with pytest.raises(TypeError, match="Unexpected keyword arguments: temporal_cognify"):
        await remember_module.remember("some text", temporal_cognify=True)
