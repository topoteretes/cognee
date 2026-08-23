import importlib
from types import SimpleNamespace
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
@patch.object(cognify_module, "get_configured_ontology_resolver")
async def test_cognify_normalizes_skip_config(
    mock_get_resolver,
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
    mock_get_default_tasks.assert_awaited_once()
    passed_config = mock_get_default_tasks.await_args.kwargs["config"]
    assert passed_config == {"ontology_config": {"ontology_resolver": None}}


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


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_model", [None, object()])
@patch.object(serve_state_module, "get_remote_client", return_value=None)
@patch.object(cognify_module, "get_pipeline_executor")
@patch.object(cognify_module, "get_default_tasks", new_callable=AsyncMock)
@patch("cognee.modules.migrations.startup.run_migrations_and_block", new_callable=AsyncMock)
@patch.object(cognify_module, "get_configured_ontology_resolver", return_value=None)
@patch.object(
    cognify_module,
    "get_graph_config",
    return_value=SimpleNamespace(graph_model="configured-model"),
)
async def test_cognify_uses_configured_graph_model_only_as_default(
    mock_graph_config,
    mock_get_resolver,
    mock_migrations,
    mock_get_default_tasks,
    mock_get_pipeline_executor,
    mock_remote_client,
    explicit_model,
):
    mock_get_default_tasks.return_value = []
    mock_get_pipeline_executor.return_value = AsyncMock(return_value={})

    await cognify_module.cognify(graph_model=explicit_model)

    passed_model = mock_get_default_tasks.await_args.kwargs["graph_model"]
    assert passed_model == ("configured-model" if explicit_model is None else explicit_model)
    if explicit_model is None:
        mock_graph_config.assert_called_once()
    else:
        mock_graph_config.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_model", [None, object()])
@patch.object(cognify_module, "Task")
@patch.object(cognify_module, "get_max_chunk_tokens", new_callable=AsyncMock)
@patch.object(cognify_module, "get_cognify_config")
@patch.object(
    cognify_module,
    "get_graph_config",
    return_value=SimpleNamespace(graph_model="configured-model"),
)
async def test_get_default_tasks_uses_graph_model_for_extraction_task(
    mock_graph_config,
    mock_get_cognify_config,
    mock_get_max_chunk_tokens,
    mock_task,
    explicit_model,
):
    mock_get_cognify_config.return_value = SimpleNamespace(
        triplet_embedding=False,
        contradiction_detection=False,
        provenance_tracking=False,
        chunks_per_batch=None,
    )
    mock_get_max_chunk_tokens.return_value = 128

    await cognify_module.get_default_tasks(graph_model=explicit_model)

    extraction_call = next(
        call
        for call in mock_task.call_args_list
        if call.args[0] is cognify_module.extract_graph_and_summarize
    )
    assert extraction_call.kwargs["graph_model"] == (
        "configured-model" if explicit_model is None else explicit_model
    )
    if explicit_model is None:
        mock_graph_config.assert_called_once()
    else:
        mock_graph_config.assert_not_called()
