"""The deprecated temporal_cognify flag at the cognify() and remember() boundaries (SDK-830).

The event pipeline the flag switched to is gone, so any value is dropped with a
deprecation warning and the default pipeline runs. cognify() forwards unknown keyword
arguments into the extraction LLM call, so the flag must never travel on.
"""

import importlib
from unittest.mock import AsyncMock, patch

import pytest

import cognee
from cognee.api.v1.cognify.cognify import (
    TEMPORAL_COGNIFY_DEPRECATED,
    drop_deprecated_temporal_cognify,
)

cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")
remember_module = importlib.import_module("cognee.api.v1.remember.remember")
serve_state_module = importlib.import_module("cognee.api.v1.serve.state")


def test_kwargs_without_the_flag_are_left_alone():
    kwargs = {"graph_model": None}

    with patch.object(cognify_module, "logger") as logger:
        drop_deprecated_temporal_cognify(kwargs)

    assert kwargs == {"graph_model": None}
    logger.warning.assert_not_called()


@pytest.mark.parametrize("value", [True, False, None])
def test_the_flag_is_dropped_with_a_deprecation_warning(value):
    kwargs = {"temporal_cognify": value, "graph_model": None}

    with (
        patch.object(cognify_module, "logger") as logger,
        pytest.warns(DeprecationWarning, match="temporal_cognify is deprecated") as warned,
    ):
        drop_deprecated_temporal_cognify(kwargs)

    assert kwargs == {"graph_model": None}
    message = str(warned[0].message)
    assert "removed in the next release" in message
    assert "memory_only=True" in message
    # Logged too: asyncio.run(cognify(...)) attributes the warning to asyncio, where
    # the default filters hide it.
    logger.warning.assert_called_once_with(TEMPORAL_COGNIFY_DEPRECATED)


@pytest.mark.asyncio
@patch.object(serve_state_module, "get_remote_client", return_value=None)
@patch.object(cognify_module, "get_pipeline_executor")
@patch.object(cognify_module, "get_default_tasks", new_callable=AsyncMock)
@patch("cognee.modules.migrations.startup.run_migrations_and_block", new_callable=AsyncMock)
@patch.object(cognify_module, "get_configured_ontology_resolver", return_value=None)
async def test_cognify_runs_the_default_pipeline_without_forwarding_the_flag(
    _resolver,
    _migrations,
    mock_get_default_tasks,
    mock_get_pipeline_executor,
    _remote_client,
):
    mock_get_default_tasks.return_value = []
    mock_get_pipeline_executor.return_value = AsyncMock(return_value={})

    with pytest.warns(DeprecationWarning, match="temporal_cognify is deprecated"):
        await cognify_module.cognify(extractor="llm", temporal_cognify=True)

    mock_get_default_tasks.assert_awaited_once()
    assert "temporal_cognify" not in mock_get_default_tasks.await_args.kwargs


@pytest.mark.asyncio
async def test_remember_drops_the_flag_instead_of_rejecting_it():
    # setup() runs right after remember()'s kwarg check, so reaching it means the flag
    # did not trip "Unexpected keyword arguments".
    with (
        patch(
            "cognee.modules.engine.operations.setup.setup",
            new_callable=AsyncMock,
            side_effect=RuntimeError("reached setup"),
        ),
        pytest.warns(DeprecationWarning, match="temporal_cognify is deprecated"),
        pytest.raises(RuntimeError, match="reached setup"),
    ):
        await cognee.remember("Apollo 11 landed on 20 July 1969.", temporal_cognify=True)
