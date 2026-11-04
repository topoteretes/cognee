"""How a remote ``remember()`` carries repositories and ``codegraph_config`` (SDK-793).

``POST /remember`` takes the repository options as one JSON ``codegraph_config``
field, and repository specs as ``raw_data`` strings: a GitHub/GitLab URL, every
string of a ``treat_as_repository`` call, and every string of the removed
``content_type="code"`` call travel that way, so the server clones them instead
of storing the spec as a one-line text document. An undeclared ssh remote is
refused here with the same guidance the local path gives.
"""

import importlib
import json

import pytest

from cognee.api.v1.serve.cloud_client import CloudClient

# import_module, not `import ... as`: this test package is itself named
# `serve`, which shadows the dotted path inside a function body.
cloud_client_module = importlib.import_module("cognee.api.v1.serve.cloud_client")


class _FakeForm:
    def __init__(self):
        self.fields = []

    def add_field(self, name, value, **kwargs):
        self.fields.append((name, value))

    def values_for(self, name):
        return [value for field, value in self.fields if field == name]


@pytest.fixture
def form(monkeypatch):
    """Capture the multipart form instead of posting it."""
    built = _FakeForm()
    monkeypatch.setattr(cloud_client_module.aiohttp, "FormData", lambda *a, **k: built)

    class _Response:
        status = 200

        async def json(self):
            return {"status": "completed"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class _Session:
        def post(self, *args, **kwargs):
            return _Response()

    async def fake_get_session(self):
        return _Session()

    monkeypatch.setattr(CloudClient, "_get_session", fake_get_session)
    return built


def _client():
    """A client with just the attributes remember() touches; nothing connects."""
    client = CloudClient.__new__(CloudClient)
    client.service_url = "https://cognee.test"
    return client


@pytest.mark.asyncio
async def test_codegraph_config_travels_as_one_json_field(form):
    config = {"treat_as_repository": True, "include_documents": False, "repo_credentials": "tok"}

    await _client().remember("https://bitbucket.org/acme/api", "ds", codegraph_config=config)

    assert [json.loads(value) for value in form.values_for("codegraph_config")] == [config]
    assert form.values_for("content_type") == []
    assert form.values_for("index_vectors") == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "specs",
    [
        ["https://bitbucket.org/acme/api"],
        ["git@github.com:acme/api.git", "/srv/checkout/repo"],
    ],
)
async def test_declared_specs_are_sent_as_raw_data(form, specs):
    # The server resolves them; uploaded as text files they would be stored as
    # the spec's own text.
    await _client().remember(specs, "ds", codegraph_config={"treat_as_repository": True})

    assert form.values_for("raw_data") == specs
    assert form.values_for("data") == []


@pytest.mark.asyncio
async def test_a_detected_repository_url_is_sent_as_raw_data_without_a_config(form):
    await _client().remember(["https://github.com/acme/api", "a note"], "ds")

    assert form.values_for("raw_data") == ["https://github.com/acme/api"]
    assert len(form.values_for("data")) == 1
    assert form.values_for("codegraph_config") == []


@pytest.mark.asyncio
async def test_an_undeclared_ssh_remote_is_refused_not_uploaded(form):
    with pytest.raises(ValueError, match="treat_as_repository"):
        await _client().remember("git@github.com:acme/api.git", "ds")

    assert form.values_for("raw_data") == [] and form.values_for("data") == []


@pytest.mark.asyncio
async def test_a_code_content_type_call_sends_its_strings_as_raw_data(form, monkeypatch):
    # remember(content_type="code") passes repository specs, often a path on the
    # server: uploaded as text, the server would store the path string itself.
    remember_module = importlib.import_module("cognee.api.v1.remember.remember")
    serve_state = importlib.import_module("cognee.api.v1.serve.state")
    monkeypatch.setattr(serve_state, "get_remote_client", _client)

    with pytest.warns(FutureWarning, match="treat_as_repository"):
        await remember_module.remember("/srv/checkout/repo", "ds", content_type="code")

    assert form.values_for("raw_data") == ["/srv/checkout/repo"]
    assert form.values_for("data") == [] and form.values_for("content_type") == []
    assert form.values_for("codegraph_config") == []
