"""The staged memory build cognify() drives: caching, indexing, cognify (SDK-397)."""

import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.cognify.routing import CognifyRoute

cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")
remember_module = importlib.import_module("cognee.api.v1.remember.remember")
serve_state_module = importlib.import_module("cognee.api.v1.serve.state")
gliner_tasks_module = importlib.import_module("cognee.tasks.graph.gliner_demo.tasks")


def _config(early_chunk_indexing: bool = True, document_caching: bool = True):
    return get_cognify_config().model_copy(
        update={"early_chunk_indexing": early_chunk_indexing, "document_caching": document_caching}
    )


def _names(tasks):
    return [task.executable.__name__ for task in tasks]


@pytest.mark.asyncio
async def test_default_tasks_look_up_the_indexed_chunks_before_extraction(monkeypatch):
    monkeypatch.setattr(cognify_module, "get_cognify_config", lambda: _config(True))

    tasks = await cognify_module.get_default_tasks(chunk_size=512, chunks_per_batch=7)

    assert _names(tasks) == [
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
async def test_the_indexing_setting_turns_the_lookup_off(monkeypatch):
    monkeypatch.setattr(cognify_module, "get_cognify_config", lambda: _config(False))

    tasks = await cognify_module.get_default_tasks(chunk_size=512)

    assert "find_indexed_chunks" not in _names(tasks)
    assert _names(tasks)[:3] == [
        "classify_documents",
        "extract_chunks_from_documents",
        "extract_graph_and_summarize",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("argument", [True, False])
async def test_the_argument_wins_over_the_setting(argument, monkeypatch):
    monkeypatch.setattr(cognify_module, "get_cognify_config", lambda: _config(not argument))

    tasks = await cognify_module.get_default_tasks(chunk_size=512, early_chunk_indexing=argument)

    assert ("find_indexed_chunks" in _names(tasks)) is argument


@pytest.mark.asyncio
async def test_the_indexing_pipeline_cuts_the_same_chunks_and_stores_them(monkeypatch):
    monkeypatch.setattr(cognify_module, "get_cognify_config", lambda: _config(True))

    tasks = await cognify_module.get_indexing_tasks(chunk_size=512, chunks_per_batch=3)

    assert _names(tasks) == [
        "classify_documents",
        "extract_chunks_from_documents",
        "index_chunks_for_search",
    ]
    assert all(task.needs_llm is False for task in tasks)
    assert (
        tasks[1].default_params["kwargs"]["max_chunk_size"]
        == (
            (await cognify_module.get_default_tasks(chunk_size=512))[1].default_params["kwargs"][
                "max_chunk_size"
            ]
        )
    ), "the indexing run and the cognify run must cut identical chunks"
    assert tasks[2].task_config["batch_size"] == 3


def test_the_caching_pipeline_classifies_and_caches():
    tasks = cognify_module.get_caching_tasks()

    assert _names(tasks) == ["classify_documents", "cache_documents"]
    assert all(task.needs_llm is False for task in tasks)


@pytest.mark.asyncio
@pytest.mark.parametrize("argument", [None, True, False])
async def test_gliner_route_gets_the_same_lookup(argument, monkeypatch):
    monkeypatch.setattr(gliner_tasks_module, "get_cognify_config", lambda: _config(True))

    with patch.object(gliner_tasks_module, "require_gliner2"):
        tasks = await gliner_tasks_module.get_gliner_demo_tasks(
            ["person"], chunk_size=512, chunks_per_batch=3, early_chunk_indexing=argument
        )

    names = _names(tasks)
    expected_on = argument is None or argument
    assert ("find_indexed_chunks" in names) is expected_on
    if expected_on:
        position = names.index("find_indexed_chunks")
        assert names[position - 1] == "extract_chunks_from_documents"
        assert names[position + 1] == "extract_graph_and_summarize_with_gliner"
        assert tasks[position].task_config["batch_size"] == 3
        assert tasks[position].needs_llm is False


def test_both_settings_are_on_by_default_and_reported():
    config = get_cognify_config()

    assert config.early_chunk_indexing is True
    assert config.document_caching is True
    assert config.to_dict()["early_chunk_indexing"] is True
    assert config.to_dict()["document_caching"] is True


@pytest.mark.parametrize(
    "env_name,field",
    [
        ("EARLY_CHUNK_INDEXING", "early_chunk_indexing"),
        ("DOCUMENT_CACHING", "document_caching"),
    ],
)
def test_the_env_vars_turn_the_settings_off(monkeypatch, env_name, field):
    monkeypatch.setenv(env_name, "false")
    get_cognify_config.cache_clear()
    try:
        assert getattr(get_cognify_config(), field) is False
    finally:
        get_cognify_config.cache_clear()


async def _run_cognify(monkeypatch, **cognify_kwargs):
    """Run cognify() against a recording executor; returns the executor's kwargs."""
    calls = []

    async def fake_executor(**kwargs):
        calls.append(kwargs)
        return {}

    with (
        patch.object(serve_state_module, "get_remote_client", return_value=None),
        patch.object(cognify_module, "get_pipeline_executor", return_value=fake_executor),
        patch.object(cognify_module, "get_default_tasks", new=AsyncMock(return_value=["STD"])),
        patch.object(cognify_module, "get_indexing_tasks", new=AsyncMock(return_value=["IDX"])),
        patch.object(cognify_module, "get_caching_tasks", return_value=["CACHE"]),
        patch("cognee.modules.migrations.startup.run_migrations_and_block", new_callable=AsyncMock),
        patch.object(cognify_module, "get_configured_ontology_resolver", return_value=None),
    ):
        await cognify_module.cognify(datasets=["ds"], **cognify_kwargs)
    (call,) = calls
    return call


@pytest.mark.asyncio
async def test_cognify_chains_the_three_runs_through_one_executor_call(monkeypatch):
    monkeypatch.setattr(cognify_module, "get_cognify_config", lambda: _config(True, True))

    call = await _run_cognify(monkeypatch)

    assert call["pipeline"] is cognify_module.run_memory_pipelines
    assert call["pipeline_name"] == "cognify_pipeline"
    standard = type("Item", (), {"system_metadata": None, "extension": "txt"})()
    manifest = type("Item", (), {"system_metadata": {"source": "dlt_source"}, "extension": None})()
    code = type("Item", (), {"system_metadata": {"source": "code"}, "extension": "py"})()
    # Standard documents go through every stage; DLT manifests and code items
    # pass through the first two with an empty task list.
    assert call["caching_tasks"](standard) == ["CACHE"]
    assert call["indexing_tasks"](standard) == ["IDX"]
    assert call["tasks"](standard) == ["STD"]
    assert call["caching_tasks"](manifest) == [] and call["indexing_tasks"](manifest) == []
    assert call["caching_tasks"](code) == [] and call["indexing_tasks"](code) == []
    assert CognifyRoute.CODE is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs,caching,indexing",
    [
        ({"document_caching": False}, False, True),
        ({"early_chunk_indexing": False}, True, False),
        ({"document_caching": False, "early_chunk_indexing": False}, False, False),
    ],
)
async def test_each_stage_has_its_own_switch(monkeypatch, kwargs, caching, indexing):
    monkeypatch.setattr(cognify_module, "get_cognify_config", lambda: _config(True, True))

    call = await _run_cognify(monkeypatch, **kwargs)

    assert (call["caching_tasks"] is not None) is caching
    assert (call["indexing_tasks"] is not None) is indexing


@pytest.mark.asyncio
async def test_the_settings_switch_the_stages_too(monkeypatch):
    monkeypatch.setattr(cognify_module, "get_cognify_config", lambda: _config(False, False))

    call = await _run_cognify(monkeypatch)

    assert call["caching_tasks"] is None and call["indexing_tasks"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("argument", ["early_chunk_indexing", "document_caching"])
async def test_cognify_rejects_the_arguments_in_remote_mode(monkeypatch, argument):
    monkeypatch.setattr(serve_state_module, "get_remote_client", lambda: MagicMock())

    with pytest.raises(ValueError, match=f"{argument} is not supported"):
        await cognify_module.cognify(**{argument: False})


def test_remember_routes_both_arguments_to_cognify():
    assert "early_chunk_indexing" in remember_module._COGNIFY_ONLY
    assert "document_caching" in remember_module._COGNIFY_ONLY


@pytest.mark.asyncio
@pytest.mark.parametrize("argument", ["early_chunk_indexing", "document_caching"])
async def test_remember_rejects_the_arguments_in_remote_mode(monkeypatch, argument):
    """client.remember() would drop them, so the remote's own settings would win silently."""
    client = MagicMock()
    client.remember = AsyncMock()
    monkeypatch.setattr(serve_state_module, "get_remote_client", lambda: client)

    with pytest.raises(ValueError, match=f"{argument} is not supported"):
        await remember_module.remember("some text", **{argument: False})
    client.remember.assert_not_awaited()
