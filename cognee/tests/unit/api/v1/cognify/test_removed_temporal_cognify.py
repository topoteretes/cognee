"""The removed temporal_cognify flag at the cognify() and remember() boundaries (SDK-981).

cognify() forwards unknown keyword arguments into the extraction LLM call, so the flag
must never travel on: True asked for the deleted event pipeline and raises, a falsy
value was always a no-op and is dropped with a deprecation warning.
"""

import importlib
from unittest.mock import AsyncMock, patch

import pytest

import cognee
from cognee.api.v1.cognify.cognify import (
    TEMPORAL_COGNIFY_REMOVED,
    reject_removed_temporal_cognify,
)

cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")
serve_state_module = importlib.import_module("cognee.api.v1.serve.state")


def test_kwargs_without_the_flag_are_left_alone():
    kwargs = {"graph_model": None}

    reject_removed_temporal_cognify(kwargs)

    assert kwargs == {"graph_model": None}


def test_a_true_flag_raises_and_names_the_rebuild_path():
    with pytest.raises(TypeError, match="temporal_cognify was removed") as raised:
        reject_removed_temporal_cognify({"temporal_cognify": True})

    assert "memory_only=True" in str(raised.value)
    assert "triplet enrichment" in str(raised.value)


@pytest.mark.parametrize("value", [False, None])
def test_a_falsy_flag_is_dropped_with_a_deprecation_warning(value):
    kwargs = {"temporal_cognify": value, "graph_model": None}

    with (
        patch.object(cognify_module, "logger") as logger,
        pytest.warns(DeprecationWarning, match="temporal_cognify was removed"),
    ):
        reject_removed_temporal_cognify(kwargs)

    assert kwargs == {"graph_model": None}
    # Logged too: asyncio.run(cognify(...)) attributes the warning to asyncio, where
    # the default filters hide it.
    logger.warning.assert_called_once_with(TEMPORAL_COGNIFY_REMOVED)


@pytest.mark.asyncio
async def test_cognify_raises_before_any_pipeline_work():
    with (
        patch.object(serve_state_module, "get_remote_client", return_value=None),
        patch(
            "cognee.modules.migrations.startup.run_migrations_and_block",
            new_callable=AsyncMock,
            side_effect=AssertionError("migrations ran before validation"),
        ),
        patch.object(
            cognify_module,
            "get_default_tasks",
            new_callable=AsyncMock,
            side_effect=AssertionError("tasks were built before validation"),
        ),
        pytest.raises(TypeError, match="temporal_cognify was removed"),
    ):
        await cognify_module.cognify(temporal_cognify=True)


@pytest.mark.asyncio
async def test_remember_raises_the_same_message_instead_of_the_generic_one():
    with (
        patch(
            "cognee.modules.engine.operations.setup.setup",
            new_callable=AsyncMock,
            side_effect=AssertionError("setup ran before validation"),
        ),
        pytest.raises(TypeError) as raised,
    ):
        await cognee.remember("Apollo 11 landed on 20 July 1969.", temporal_cognify=True)

    assert str(raised.value) == TEMPORAL_COGNIFY_REMOVED
