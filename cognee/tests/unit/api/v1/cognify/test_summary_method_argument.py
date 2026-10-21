"""summary_method at the cognify() and remember() boundary (SDK-882)."""

import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.modules.cognify.config import get_cognify_config, resolve_summary_method

cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")
remember_module = importlib.import_module("cognee.api.v1.remember.remember")
serve_state_module = importlib.import_module("cognee.api.v1.serve.state")


def _config(summary_method):
    return get_cognify_config().model_copy(update={"summary_method": summary_method})


def test_argument_wins_over_the_setting():
    assert resolve_summary_method(None, _config("from_extraction")) == "from_extraction"
    assert resolve_summary_method("llm", _config("from_extraction")) == "llm"
    with pytest.raises(ValueError, match="Unknown summary_method"):
        resolve_summary_method("extraction", _config("llm"))


@pytest.mark.asyncio
async def test_cognify_rejects_a_misspelled_or_remote_summary_method(monkeypatch):
    with pytest.raises(ValueError, match="Unknown summary_method"):
        await cognify_module.cognify(summary_method="extraction")

    monkeypatch.setattr(serve_state_module, "get_remote_client", lambda: MagicMock())
    with pytest.raises(ValueError, match="remote Cognee instance"):
        await cognify_module.cognify(summary_method="from_extraction")


@pytest.mark.asyncio
@patch.object(serve_state_module, "get_remote_client", return_value=None)
@patch.object(cognify_module, "get_pipeline_executor")
@patch.object(cognify_module, "get_default_tasks", new_callable=AsyncMock, return_value=[])
@patch("cognee.modules.migrations.startup.run_migrations_and_block", new_callable=AsyncMock)
@patch.object(cognify_module, "get_configured_ontology_resolver", return_value=None)
async def test_cognify_passes_the_argument_to_get_default_tasks(
    mock_get_resolver, mock_migrations, mock_get_default_tasks, mock_executor, mock_remote
):
    mock_executor.return_value = AsyncMock(return_value={})

    await cognify_module.cognify(summary_method="from_extraction")

    assert mock_get_default_tasks.await_args.kwargs["summary_method"] == "from_extraction"


@pytest.mark.asyncio
@pytest.mark.parametrize("summary_method", [None, "llm", "from_extraction"])
async def test_get_default_tasks_binds_the_summary_method(summary_method, monkeypatch):
    config = _config("from_extraction")
    monkeypatch.setattr(cognify_module, "get_cognify_config", lambda: config)

    tasks = await cognify_module.get_default_tasks(chunk_size=512, summary_method=summary_method)

    extraction = next(
        task for task in tasks if task.executable.__name__ == "extract_graph_and_summarize"
    )
    assert extraction.default_params["kwargs"]["summary_method"] == (
        summary_method or "from_extraction"
    )


@pytest.mark.asyncio
@patch.object(serve_state_module, "get_remote_client", return_value=None)
@patch("cognee.modules.migrations.startup.run_migrations_and_block", new_callable=AsyncMock)
@patch.object(cognify_module, "get_configured_ontology_resolver", return_value=None)
@patch("cognee.modules.cognify.estimator.estimate_cognify_dry_run", new_callable=AsyncMock)
async def test_cognify_dry_run_estimates_the_chosen_method(mock_estimate, *_):
    await cognify_module.cognify(dry_run=True, summary_method="from_extraction")

    assert mock_estimate.await_args.kwargs["summary_method"] == "from_extraction"


def test_remember_routes_the_summary_method_to_cognify():
    assert "summary_method" in remember_module._COGNIFY_ONLY
    assert "summary_method" in remember_module.RememberKwargs.__annotations__


@pytest.mark.asyncio
@patch("cognee.modules.cognify.estimator.estimate_remember_dry_run", new_callable=AsyncMock)
async def test_remember_dry_run_estimates_the_chosen_method(mock_estimate):
    await remember_module.remember("text", dry_run=True, summary_method="from_extraction")

    assert mock_estimate.await_args.kwargs["summary_method"] == "from_extraction"


@pytest.mark.asyncio
@patch.object(serve_state_module, "get_remote_client", return_value=None)
@patch("cognee.shared.utils.send_telemetry")
@patch("cognee.api.v1.add.add", new_callable=AsyncMock)
async def test_remember_rejects_a_misspelled_summary_method_before_add(mock_add, *_):
    with pytest.raises(ValueError, match="Unknown summary_method"):
        await remember_module.remember("text", summary_method="extraction")

    mock_add.assert_not_awaited()


@pytest.mark.asyncio
async def test_session_remember_rejects_an_explicit_summary_method():
    with pytest.raises(ValueError, match="session_id"):
        await remember_module.remember(
            "text", session_id="session", summary_method="from_extraction"
        )
