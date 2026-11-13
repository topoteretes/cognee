"""``codegraph_config`` validation and how ``batch_size`` reaches the CODE / CODE_REPO
task lists."""

import importlib

import pytest

from cognee.infrastructure.databases.graph.graph_db_interface import (
    DEFAULT_GRAPH_STREAM_BATCH_SIZE,
)
from cognee.tasks.code_graph.code_files import get_code_file_tasks
from cognee.tasks.code_graph.code_repo import get_code_repo_tasks
from cognee.tasks.code_graph.config import validate_codegraph_config


def test_none_is_an_empty_config():
    assert validate_codegraph_config(None) == {}


def test_a_key_set_to_none_counts_as_absent():
    assert validate_codegraph_config({"batch_size": None}) == {}


def test_unknown_key_is_rejected():
    with pytest.raises(ValueError, match="Unknown codegraph_config keys: batch\\."):
        validate_codegraph_config({"batch": 10})


def test_non_dict_is_rejected():
    with pytest.raises(TypeError, match="codegraph_config must be a dict"):
        validate_codegraph_config("batch_size")


def test_a_positive_batch_size_is_kept():
    assert validate_codegraph_config({"batch_size": 200000}) == {"batch_size": 200000}


@pytest.mark.parametrize("value", [True, False, "2000", 2000.0])
def test_a_batch_size_that_is_not_an_int_is_rejected(value):
    with pytest.raises(TypeError, match=r"codegraph_config\['batch_size'\] must be an int"):
        validate_codegraph_config({"batch_size": value})


@pytest.mark.parametrize("value", [0, -1])
def test_a_batch_size_below_one_is_rejected(value):
    with pytest.raises(ValueError, match=r"codegraph_config\['batch_size'\] must be >= 1"):
        validate_codegraph_config({"batch_size": value})


@pytest.mark.parametrize("get_tasks", [get_code_file_tasks, get_code_repo_tasks])
def test_routes_default_to_the_default_batch_size(get_tasks):
    (task,) = get_tasks()
    assert task.default_params["kwargs"]["code_graph_batch_size"] == (
        DEFAULT_GRAPH_STREAM_BATCH_SIZE
    )
    assert task.needs_llm is False


@pytest.mark.parametrize("get_tasks", [get_code_file_tasks, get_code_repo_tasks])
def test_batch_size_reaches_the_route_adapter(get_tasks):
    (task,) = get_tasks(batch_size=200000)
    assert task.default_params["kwargs"]["code_graph_batch_size"] == 200000
    # The runner's own batching of the adapter's input is untouched.
    assert task.task_config["batch_size"] == 1


@pytest.mark.asyncio
async def test_cognify_rejects_an_unknown_codegraph_config_key():
    # Validated first thing, before any config or database is touched.
    from cognee.api.v1.cognify.cognify import cognify

    with pytest.raises(ValueError, match="Unknown codegraph_config keys"):
        await cognify(codegraph_config={"batch": 10})


@pytest.mark.asyncio
async def test_cognify_rejects_a_bad_batch_size_before_touching_anything(monkeypatch):
    cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")

    def _touched(*args, **kwargs):
        raise AssertionError("cognify() read its config before validating codegraph_config")

    monkeypatch.setattr(cognify_module, "get_cognify_config", _touched)

    with pytest.raises(ValueError, match="must be >= 1"):
        await cognify_module.cognify(codegraph_config={"batch_size": 0})


def test_only_a_non_default_batch_size_requests_options():
    from cognee.tasks.code_graph.config import requests_codegraph_options

    assert requests_codegraph_options(None) is False
    assert requests_codegraph_options({}) is False
    assert requests_codegraph_options({"batch_size": DEFAULT_GRAPH_STREAM_BATCH_SIZE}) is False
    assert requests_codegraph_options({"batch_size": 1}) is True


@pytest.mark.asyncio
async def test_a_remote_instance_rejects_only_a_non_default_batch_size(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    serve_state = importlib.import_module("cognee.api.v1.serve.state")
    cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")
    client = MagicMock()
    client.cognify = AsyncMock(return_value={"remote": True})
    monkeypatch.setattr(serve_state, "get_remote_client", lambda: client)

    # Restating the default asks for nothing the remote would not do anyway.
    result = await cognify_module.cognify(codegraph_config={"batch_size": 2000})
    assert result == {"remote": True}

    with pytest.raises(ValueError, match="codegraph_config is not supported"):
        await cognify_module.cognify(codegraph_config={"batch_size": 1})
    client.cognify.assert_awaited_once()
