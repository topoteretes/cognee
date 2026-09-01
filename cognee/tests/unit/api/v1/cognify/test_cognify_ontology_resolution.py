import importlib
from unittest.mock import AsyncMock, patch

import pytest

cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")
serve_state_module = importlib.import_module("cognee.api.v1.serve.state")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "input_config", [None, {}, {"ontology_config": {}}, {"ontology_config": None}]
)
@patch.object(serve_state_module, "get_remote_client", return_value=None)
@patch.object(cognify_module, "get_pipeline_executor")
@patch.object(cognify_module, "get_default_tasks", new_callable=AsyncMock)
@patch("cognee.modules.migrations.startup.run_migrations_and_block", new_callable=AsyncMock)
@patch.object(cognify_module, "get_configured_ontology_mode", return_value="annotate")
@patch.object(cognify_module, "get_configured_ontology_resolver")
async def test_cognify_normalizes_skip_config(
    mock_get_resolver,
    mock_get_mode,
    mock_migrations,
    mock_get_default_tasks,
    mock_get_pipeline_executor,
    mock_remote_client,
    input_config,
):
    mock_get_resolver.return_value = None
    mock_get_default_tasks.return_value = []
    mock_get_pipeline_executor.return_value = AsyncMock(return_value={})

    await cognify_module.cognify(config=input_config)

    mock_get_resolver.assert_called_once_with(input_config)
    mock_get_mode.assert_called_once_with(input_config)
    mock_get_default_tasks.assert_awaited_once()
    passed_config = mock_get_default_tasks.await_args.kwargs["config"]
    assert passed_config == {
        "ontology_config": {"ontology_resolver": None, "ontology_mode": "annotate"}
    }


@pytest.mark.asyncio
@patch.object(serve_state_module, "get_remote_client", return_value=None)
@patch.object(cognify_module, "get_configured_ontology_resolver")
@patch("cognee.modules.migrations.startup.run_migrations_and_block", new_callable=AsyncMock)
async def test_cognify_raises_on_invalid_env_before_pipeline(
    mock_migrations, mock_get_resolver, mock_remote_client
):
    mock_get_resolver.side_effect = EnvironmentError("Unsupported ontology resolver: bad")

    with pytest.raises(EnvironmentError):
        await cognify_module.cognify(config=None)
