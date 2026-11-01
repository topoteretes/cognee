"""The indexing pipeline cognify() runs before extraction, and its counterpart in the
cognify task lists (SDK-397)."""

import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.cognify.routing import CognifyRoute

cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")
remember_module = importlib.import_module("cognee.api.v1.remember.remember")
serve_state_module = importlib.import_module("cognee.api.v1.serve.state")
gliner_tasks_module = importlib.import_module("cognee.tasks.graph.gliner_demo.tasks")


def _config(early_chunk_indexing: bool):
    return get_cognify_config().model_copy(update={"early_chunk_indexing": early_chunk_indexing})


@pytest.mark.asyncio
async def test_indexing_tasks_chunk_and_store_without_the_llm():
    tasks = await cognify_module.get_indexing_tasks(chunk_size=512, chunks_per_batch=7)

    assert [task.executable.__name__ for task in tasks] == [
        "classify_documents",
        "extract_chunks_from_documents",
        "index_chunks_for_search",
    ]
    assert all(task.needs_llm is False for task in tasks)
    assert tasks[2].task_config["batch_size"] == 7


@pytest.mark.asyncio
async def test_both_runs_cut_the_same_chunks(monkeypatch):
    """The cognify run skips re-embedding only the chunk ids it finds in the
    store, so its chunking step must match the indexing run's exactly."""
    monkeypatch.setattr(cognify_module, "get_cognify_config", lambda: _config(True))
    chunker = MagicMock(name="chunker")

    indexing = await cognify_module.get_indexing_tasks(chunker=chunker, chunk_size=300)
    cognify = await cognify_module.get_default_tasks(chunker=chunker, chunk_size=300)

    assert indexing[1].default_params == cognify[1].default_params
    assert indexing[1].default_params["kwargs"]["chunker"] is chunker


@pytest.mark.asyncio
async def test_default_tasks_find_the_indexed_chunks_before_extraction(monkeypatch):
    monkeypatch.setattr(cognify_module, "get_cognify_config", lambda: _config(True))

    tasks = await cognify_module.get_default_tasks(chunk_size=512, chunks_per_batch=7)

    names = [task.executable.__name__ for task in tasks]
    assert names == [
        "classify_documents",
        "extract_chunks_from_documents",
        "find_indexed_chunks",
        "extract_graph_and_summarize",
        "add_data_points",
    ]
    lookup, final = tasks[2], tasks[4]
    assert lookup.needs_llm is False, "the lookup never calls the LLM"
    assert lookup.task_config["batch_size"] == 7, "batched like the extraction it precedes"
    # The skip in the final write rides on ctx.extras: both tasks must receive ctx,
    # or every chunk is silently embedded twice.
    assert lookup.accepts_ctx is True
    assert final.accepts_ctx is True


@pytest.mark.asyncio
async def test_the_setting_turns_the_step_off(monkeypatch):
    monkeypatch.setattr(cognify_module, "get_cognify_config", lambda: _config(False))

    tasks = await cognify_module.get_default_tasks(chunk_size=512)

    names = [task.executable.__name__ for task in tasks]
    assert "find_indexed_chunks" not in names
    assert names[:3] == [
        "classify_documents",
        "extract_chunks_from_documents",
        "extract_graph_and_summarize",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("argument", [True, False])
async def test_the_argument_wins_over_the_setting(argument, monkeypatch):
    monkeypatch.setattr(cognify_module, "get_cognify_config", lambda: _config(not argument))

    tasks = await cognify_module.get_default_tasks(chunk_size=512, early_chunk_indexing=argument)

    names = [task.executable.__name__ for task in tasks]
    assert ("find_indexed_chunks" in names) is argument


@pytest.mark.asyncio
@pytest.mark.parametrize("argument", [None, True, False])
async def test_gliner_route_gets_the_same_step(argument, monkeypatch):
    monkeypatch.setattr(gliner_tasks_module, "get_cognify_config", lambda: _config(True))

    with patch.object(gliner_tasks_module, "require_gliner2"):
        tasks = await gliner_tasks_module.get_gliner_demo_tasks(
            ["person"], chunk_size=512, chunks_per_batch=3, early_chunk_indexing=argument
        )

    names = [task.executable.__name__ for task in tasks]
    expected_on = argument is None or argument
    assert ("find_indexed_chunks" in names) is expected_on
    if expected_on:
        position = names.index("find_indexed_chunks")
        assert names[position - 1] == "extract_chunks_from_documents"
        assert names[position + 1] == "extract_graph_and_summarize_with_gliner"
        assert tasks[position].task_config["batch_size"] == 3
        assert tasks[position].needs_llm is False


def test_the_setting_is_on_by_default_and_reported():
    config = get_cognify_config()

    assert config.early_chunk_indexing is True
    assert config.to_dict()["early_chunk_indexing"] is True


def test_the_env_var_turns_the_setting_off(monkeypatch):
    monkeypatch.setenv("EARLY_CHUNK_INDEXING", "false")
    get_cognify_config.cache_clear()
    try:
        assert get_cognify_config().early_chunk_indexing is False
    finally:
        get_cognify_config.cache_clear()


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

    await cognify_module.cognify(early_chunk_indexing=False)

    assert mock_get_default_tasks.await_args.kwargs["early_chunk_indexing"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("argument", [True, False])
@patch.object(serve_state_module, "get_remote_client", return_value=None)
@patch.object(cognify_module, "get_pipeline_executor")
@patch.object(cognify_module, "get_default_tasks", new_callable=AsyncMock, return_value=[])
@patch("cognee.modules.migrations.startup.run_migrations_and_block", new_callable=AsyncMock)
@patch.object(cognify_module, "get_configured_ontology_resolver", return_value=None)
async def test_cognify_runs_the_indexing_pipeline_first_when_on(
    mock_get_resolver, mock_migrations, mock_get_default_tasks, mock_executor, mock_remote, argument
):
    executor = AsyncMock(return_value={})
    mock_executor.return_value = executor

    await cognify_module.cognify(early_chunk_indexing=argument)

    params = executor.await_args.kwargs
    assert params["pipeline"] is cognify_module.run_memory_pipelines
    assert params["pipeline_name"] == "cognify_pipeline"
    if argument:
        resolve = params["indexing_tasks"]
        standard = MagicMock(pipeline_status={})
        with patch.object(cognify_module, "cognify_route_for", return_value=CognifyRoute.STANDARD):
            names = [task.executable.__name__ for task in resolve(standard)]
        assert names == [
            "classify_documents",
            "extract_chunks_from_documents",
            "index_chunks_for_search",
        ]
        with patch.object(cognify_module, "cognify_route_for", return_value=CognifyRoute.CODE):
            assert resolve(standard) == []
    else:
        assert params["indexing_tasks"] is None


@pytest.mark.asyncio
async def test_cognify_rejects_the_argument_in_remote_mode(monkeypatch):
    monkeypatch.setattr(serve_state_module, "get_remote_client", lambda: MagicMock())

    with pytest.raises(ValueError, match="early_chunk_indexing is not supported"):
        await cognify_module.cognify(early_chunk_indexing=False)


def test_remember_routes_the_argument_to_cognify():
    assert "early_chunk_indexing" in remember_module._COGNIFY_ONLY


@pytest.mark.asyncio
async def test_remember_rejects_the_argument_in_remote_mode(monkeypatch):
    monkeypatch.setattr(serve_state_module, "get_remote_client", lambda: MagicMock())

    with pytest.raises(ValueError, match="early_chunk_indexing is not supported"):
        await remember_module.remember("some text", early_chunk_indexing=False)
