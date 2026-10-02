"""A declared repository reaches the remote as a repository, not as a text upload.

``POST /remember`` has no ``codegraph_config`` field. It has one legacy field
that means the same pair of things — ``content_type='code'``, which the server
reads back as ``include_documents=False`` + ``treat_as_repository=True`` — so a
config that means exactly that travels as that field, with every spec in
``raw_data``.

Without this, ``remember("https://bitbucket.org/o/r", content_type="code")``
against a remote instance uploaded the URL as a one-line text file: the
declaration was dropped on the wire and the server stored a string where the
caller asked for a code graph.
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
@pytest.mark.parametrize("spec", UNDETECTED_REMOTES)
async def test_a_declared_spec_travels_as_raw_data(form, spec):
    await _client().remember(
        spec,
        "ds",
        codegraph_config={"treat_as_repository": True, "include_documents": False},
    )

    assert form.values_for("raw_data") == [spec]
    assert form.values_for("data") == [], "the spec was uploaded as a file"
    assert form.values_for("content_type") == ["code"]


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
async def test_a_combination_the_wire_cannot_express_raises(form):
    # Declared repositories WITH their documents has no form field. Sending
    # content_type='code' anyway would suppress the documents the caller asked
    # for -- the silent drop this whole mapping exists to avoid.
    with pytest.raises(ValueError, match="include_documents=True is not supported"):
        await _client().remember(
            "https://bitbucket.org/acme/api",
            "ds",
            codegraph_config={"treat_as_repository": True, "include_documents": True},
        )


@pytest.mark.asyncio
async def test_a_list_of_declared_specs_all_travel_as_raw_data(form):
    specs = ["https://bitbucket.org/acme/api", "https://gitlab.acme.com/team/web"]

    await _client().remember(specs, "ds", codegraph_config={"treat_as_repository": True})

    assert form.values_for("raw_data") == specs
    assert form.values_for("data") == []


@pytest.mark.asyncio
async def test_index_vectors_still_rides_alongside(form):
    await _client().remember(
        "https://bitbucket.org/acme/api",
        "ds",
        codegraph_config={"treat_as_repository": True, "index_vectors": True},
    )

    assert form.values_for("index_vectors") == ["true"]
    assert form.values_for("raw_data") == ["https://bitbucket.org/acme/api"]
