"""``codegraph_config`` validation and how ``index_vectors`` reaches the CODE / CODE_REPO
task lists."""

import pytest

from cognee.tasks.code_graph.code_files import get_code_file_tasks
from cognee.tasks.code_graph.code_repo import get_code_repo_tasks
from cognee.tasks.code_graph.config import validate_codegraph_config


def test_none_is_an_empty_config():
    assert validate_codegraph_config(None) == {}


def test_unknown_key_is_rejected():
    with pytest.raises(ValueError, match="Unknown codegraph_config keys: index_vector\\."):
        validate_codegraph_config({"index_vector": True})


def test_non_dict_is_rejected():
    with pytest.raises(TypeError, match="codegraph_config must be a dict"):
        validate_codegraph_config("index_vectors")


@pytest.mark.parametrize("get_tasks", [get_code_file_tasks, get_code_repo_tasks])
def test_routes_default_to_graph_only(get_tasks):
    (task,) = get_tasks()
    assert task.default_params["kwargs"]["index_vectors"] is False
    assert task.needs_llm is False


@pytest.mark.parametrize("get_tasks", [get_code_file_tasks, get_code_repo_tasks])
def test_index_vectors_reaches_the_route_adapter(get_tasks):
    (task,) = get_tasks(index_vectors=True)
    assert task.default_params["kwargs"]["index_vectors"] is True


@pytest.mark.asyncio
async def test_cognify_rejects_an_unknown_codegraph_config_key():
    # Validated first thing, before any config or database is touched.
    from cognee.api.v1.cognify.cognify import cognify

    with pytest.raises(ValueError, match="Unknown codegraph_config keys"):
        await cognify(codegraph_config={"index_vector": True})
