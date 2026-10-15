"""Code graph options that a remote instance cannot carry raise instead of vanishing.

``client.add()`` / ``client.cognify()`` / ``POST /remember`` whitelist their
form fields. An option with no field there would be dropped on the wire and the
remote would build something else, so each surface refuses it up front.
"""

import importlib

import pytest

serve_state = importlib.import_module("cognee.api.v1.serve.state")
add_module = importlib.import_module("cognee.api.v1.add.add")
cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")
remember_module = importlib.import_module("cognee.api.v1.remember.remember")


class _UnusedClient:
    """A connected remote whose methods must never be reached."""

    def __getattr__(self, name):
        raise AssertionError(f"the remote was called ({name}) for an unsupported option")


@pytest.fixture(autouse=True)
def remote(monkeypatch):
    monkeypatch.setattr(serve_state, "get_remote_client", lambda: _UnusedClient())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config, message",
    [
        ({"repo_credentials": "tok"}, "repo_credentials is not supported"),
        ({"treat_as_repository": True}, "treat_as_repository is not supported"),
        ({"include_documents": False}, "include_documents=False is not supported"),
    ],
)
async def test_add_refuses_options_the_remote_cannot_carry(config, message):
    with pytest.raises(ValueError, match=message):
        await add_module.add("https://github.com/acme/api", codegraph_config=config)


@pytest.mark.asyncio
async def test_add_refuses_a_branch_remotely():
    with pytest.raises(ValueError, match="branch is not supported"):
        await add_module.add("https://github.com/acme/api", branch="dev")


@pytest.mark.asyncio
async def test_cognify_refuses_index_vectors_remotely():
    with pytest.raises(ValueError, match="index_vectors is not supported"):
        await cognify_module.cognify(codegraph_config={"index_vectors": True})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [
        {"repo_credentials": "tok"},
        {"codegraph_config": {"repo_credentials": "tok"}},
    ],
)
async def test_remember_refuses_repo_credentials_remotely(kwargs):
    with pytest.raises(ValueError, match="repo_credentials is not supported"):
        await remember_module.remember("https://github.com/acme/api", **kwargs)
