"""Code graph options that a remote instance cannot carry raise instead of vanishing.

``client.add()`` / ``client.cognify()`` send only the data, the dataset and a
few fields. A ``codegraph_config`` option with no field there would be dropped
on the wire and the remote would build something else, so each refuses it up
front. ``POST /remember`` carries the dict as one JSON field, so
``remember()`` forwards it instead.
"""

import importlib
from unittest.mock import AsyncMock

import pytest

serve_state = importlib.import_module("cognee.api.v1.serve.state")
add_module = importlib.import_module("cognee.api.v1.add.add")
cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")
remember_module = importlib.import_module("cognee.api.v1.remember.remember")


class _UnusedClient:
    """A connected remote whose methods must never be reached."""

    def __getattr__(self, name):
        raise AssertionError(f"the remote was called ({name}) for an unsupported option")


@pytest.fixture
def remote(monkeypatch):
    monkeypatch.setattr(serve_state, "get_remote_client", lambda: _UnusedClient())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config",
    [
        {"repo_credentials": "tok"},
        {"treat_as_repository": True},
        {"include_documents": False},
        {"index_vectors": True},
    ],
)
async def test_add_refuses_options_the_remote_cannot_carry(remote, config):
    with pytest.raises(ValueError, match="codegraph_config is not supported by add"):
        await add_module.add("https://github.com/acme/api", codegraph_config=config)


@pytest.mark.asyncio
async def test_add_forwards_a_config_that_only_restates_the_defaults(monkeypatch):
    client = AsyncMock()
    client.add = AsyncMock(return_value={"status": "ok"})
    monkeypatch.setattr(serve_state, "get_remote_client", lambda: client)

    await add_module.add(
        "a note", codegraph_config={"index_vectors": False, "include_documents": True}
    )

    client.add.assert_awaited_once_with("a note", "main_dataset")


@pytest.mark.asyncio
async def test_cognify_refuses_index_vectors_remotely(remote):
    with pytest.raises(ValueError, match="codegraph_config is not supported by cognify"):
        await cognify_module.cognify(codegraph_config={"index_vectors": True})


@pytest.mark.asyncio
async def test_remember_forwards_the_config_to_the_remote(monkeypatch):
    client = AsyncMock()
    client.remember = AsyncMock(return_value={"status": "completed", "dataset_name": "ds"})
    monkeypatch.setattr(serve_state, "get_remote_client", lambda: client)
    config = {"treat_as_repository": True, "include_documents": False, "repo_credentials": "tok"}

    result = await remember_module.remember(
        "https://bitbucket.org/acme/api", "ds", codegraph_config=config
    )

    # A remote remember() answers with the server's payload.
    assert result == {"status": "completed", "dataset_name": "ds"}
    assert client.remember.await_args.kwargs["codegraph_config"] == config
