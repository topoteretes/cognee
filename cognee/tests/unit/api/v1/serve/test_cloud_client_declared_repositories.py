"""Repository options a remote ``POST /remember`` cannot carry raise instead of vanishing.

``POST /remember`` has no ``codegraph_config`` field; only ``index_vectors``
has a form field of its own. ``treat_as_repository`` and
``include_documents=False`` would be dropped on the wire -- the server would
sniff the specs (uploading a Bitbucket URL as a one-line text file) or ingest
the repository's documents -- so the client refuses them. The removed
``content_type='code'``, which once carried them, is ignored by the server.
"""

import importlib

import pytest

from cognee.api.v1.serve.cloud_client import CloudClient

# import_module, not `import ... as`: this test package is itself named
# `serve`, which shadows the dotted path inside a function body.
cloud_client_module = importlib.import_module("cognee.api.v1.serve.cloud_client")

UNDETECTED_REMOTES = [
    "https://bitbucket.org/acme/api",
    "https://gitlab.acme.com/team/api",
    "git@github.com:acme/api.git",
]


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
@pytest.mark.parametrize(
    "config",
    [
        {"treat_as_repository": True},
        {"treat_as_repository": True, "include_documents": False},
        {"treat_as_repository": True, "include_documents": True},
        {"treat_as_repository": True, "index_vectors": True},
    ],
)
@pytest.mark.parametrize("spec", [*UNDETECTED_REMOTES, ["https://bitbucket.org/acme/api"]])
async def test_treat_as_repository_raises_instead_of_being_dropped(form, spec, config):
    with pytest.raises(ValueError, match="treat_as_repository is not supported while connected"):
        await _client().remember(spec, "ds", codegraph_config=config)

    assert form.values_for("raw_data") == [] and form.values_for("data") == []
    assert form.values_for("content_type") == []


@pytest.mark.asyncio
async def test_include_documents_false_raises_instead_of_vanishing(form):
    # The server would ingest the repository's documents, with LLM cost,
    # against what was asked.
    with pytest.raises(ValueError, match="include_documents=False is not supported"):
        await _client().remember(
            "https://github.com/acme/api", "ds", codegraph_config={"include_documents": False}
        )
    assert form.values_for("raw_data") == [] and form.values_for("data") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("spec", UNDETECTED_REMOTES)
async def test_without_the_declaration_the_same_spec_does_not(form, spec):
    # The contrast. An ssh spec is refused (it can only be a repository, and
    # the server cannot clone an undeclared one); the http(s) ones upload as
    # text, which is right for a URL nobody declared a repository.
    if spec.startswith(("git@", "ssh://")):
        with pytest.raises(ValueError, match="ssh git remotes"):
            await _client().remember(spec, "ds")
        return
    await _client().remember(spec, "ds")
    assert form.values_for("raw_data") == []
    assert len(form.values_for("data")) == 1


@pytest.mark.asyncio
async def test_a_detected_repository_url_still_travels_as_raw_data(form):
    await _client().remember("https://github.com/acme/api", "ds")

    assert form.values_for("raw_data") == ["https://github.com/acme/api"]
    assert form.values_for("content_type") == [], "undeclared adds keep their documents"


@pytest.mark.asyncio
async def test_index_vectors_rides_its_own_field(form):
    await _client().remember(
        "https://github.com/acme/api", "ds", codegraph_config={"index_vectors": True}
    )

    assert form.values_for("index_vectors") == ["true"]
    assert form.values_for("raw_data") == ["https://github.com/acme/api"]
    assert form.values_for("content_type") == []
