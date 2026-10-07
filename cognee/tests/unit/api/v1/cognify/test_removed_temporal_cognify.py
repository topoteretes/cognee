"""temporal_cognify is no longer a cognify()/remember() option; it is accepted and ignored.

The default pipeline extracts dates as Timestamp nodes and SearchType.TEMPORAL reads
them, so the event pipeline the flag switched to is no longer offered. Existing calls
keep working: cognify() drops the flag before its unknown-kwargs forwarding, so it
never reaches the extraction LLM call, and remember() still routes it to cognify().
"""

import importlib
from unittest.mock import AsyncMock, patch

import pytest

cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")
remember_module = importlib.import_module("cognee.api.v1.remember.remember")
serve_state_module = importlib.import_module("cognee.api.v1.serve.state")


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [True, False])
@patch.object(serve_state_module, "get_remote_client", return_value=None)
@patch.object(cognify_module, "get_pipeline_executor")
@patch.object(cognify_module, "get_temporal_tasks", new_callable=AsyncMock)
@patch.object(cognify_module, "get_default_tasks", new_callable=AsyncMock)
@patch("cognee.modules.migrations.startup.run_migrations_and_block", new_callable=AsyncMock)
@patch.object(cognify_module, "get_configured_ontology_resolver", return_value=None)
async def test_cognify_ignores_temporal_cognify(
    _resolver,
    _migrations,
    mock_get_default_tasks,
    mock_get_temporal_tasks,
    mock_get_pipeline_executor,
    _remote_client,
    value,
):
    mock_get_default_tasks.return_value = []
    mock_get_pipeline_executor.return_value = AsyncMock(return_value={})

    await cognify_module.cognify(extractor="llm", temporal_cognify=value)

    # The default pipeline runs, the event pipeline does not, and the flag is not
    # forwarded into the extraction tasks (and from there the LLM call).
    mock_get_default_tasks.assert_awaited_once()
    assert "temporal_cognify" not in mock_get_default_tasks.await_args.kwargs
    mock_get_temporal_tasks.assert_not_awaited()


def test_remember_still_routes_temporal_cognify_to_cognify():
    assert "temporal_cognify" in remember_module._COGNIFY_ONLY
