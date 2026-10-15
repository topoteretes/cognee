"""CloudClient.remember sends repositories as raw_data specs (SDK-793).

The server resolves a raw_data entry as add() would, so a repository URL or ssh
remote must travel as one: uploaded as text, it would be stored as a document
holding the URL. codegraph_config travels as a JSON form field.
"""

import json
from unittest.mock import patch

import pytest

from cognee.api.v1.serve.cloud_client import CloudClient


class _FakeResponse:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return {"status": "completed"}


class _FakeSession:
    def __init__(self):
        self.form = None
        self.timeout = None

    def post(self, _url, data=None, timeout=None, **kwargs):
        self.form = data
        self.timeout = timeout
        return _FakeResponse()


def _fields(form):
    """(name, value) per form field; uploads as their decoded text."""
    fields = []
    for type_options, _headers, value in form._fields:
        if hasattr(value, "read"):
            value = value.read().decode("utf-8")
        fields.append((type_options["name"], value))
    return fields


async def _remember(data, **kwargs):
    client = CloudClient(service_url="https://example.test", api_key="k")
    session = _FakeSession()
    with patch.object(client, "_get_session", return_value=session):
        await client.remember(data, dataset_name="ds", **kwargs)
    return client, session


@pytest.mark.asyncio
async def test_repositories_travel_as_raw_data_and_text_as_uploads():
    client, session = await _remember(
        ["https://github.com/org/repo", "git@github.com:org/other.git", "a note"]
    )

    fields = _fields(session.form)
    assert ("raw_data", "https://github.com/org/repo") in fields
    assert ("raw_data", "git@github.com:org/other.git") in fields
    assert ("data", "a note") in fields
    assert not any(name == "codegraph_config" for name, _ in fields)
    # A clone and a whole-repo parse do not fit the default request timeout.
    assert session.timeout is client.UPLOAD_TIMEOUT


@pytest.mark.asyncio
async def test_a_web_page_url_keeps_its_text_upload():
    client, session = await _remember("https://example.com/article")

    assert _fields(session.form)[-1] == ("data", "https://example.com/article")
    assert session.timeout is client.DEFAULT_TIMEOUT


@pytest.mark.asyncio
async def test_treat_as_repository_sends_every_item_as_a_spec(tmp_path):
    config = {"treat_as_repository": True, "include_documents": False}

    _client, session = await _remember(
        [tmp_path, "https://git.example.com/team/repo"], codegraph_config=config
    )

    fields = _fields(session.form)
    assert ("raw_data", str(tmp_path)) in fields
    assert ("raw_data", "https://git.example.com/team/repo") in fields
    (sent_config,) = [value for name, value in fields if name == "codegraph_config"]
    assert json.loads(sent_config) == config
