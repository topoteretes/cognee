"""CloudClient.cognify must forward a custom graph_model schema on the wire."""

from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from cognee.api.v1.serve.cloud_client import CloudClient
from cognee.shared.data_models import KnowledgeGraph
from cognee.shared.graph_model_utils import graph_model_to_graph_schema


class _FakeResponse:
    status = 200

    async def json(self):
        return {"ok": True}

    async def text(self):
        return ""


class _FakePost:
    def __init__(self, recorder, url, json):
        recorder.append((url, json))

    async def __aenter__(self):
        return _FakeResponse()

    async def __aexit__(self, *exc):
        return False


def _client_with_recorder():
    client = CloudClient.__new__(CloudClient)
    client.service_url = "http://server"
    posts = []
    session = type(
        "FakeSession", (), {"post": lambda self, url, json: _FakePost(posts, url, json)}
    )()
    client._get_session = AsyncMock(return_value=session)
    return client, posts


class LanguageType(BaseModel):
    name: str = "Programming Language"


class Language(BaseModel):
    name: str
    is_type: LanguageType


@pytest.mark.asyncio
async def test_cognify_forwards_graph_model_schema():
    client, posts = _client_with_recorder()
    schema = graph_model_to_graph_schema(Language)
    await client.cognify(["docs"], graph_model=schema, run_in_background=True)
    url, payload = posts[0]
    assert url == "http://server/api/v1/cognify"
    assert payload["graph_model"] == schema
    assert payload["run_in_background"] is True
    assert payload["datasets"] == ["docs"]


@pytest.mark.asyncio
async def test_cognify_omits_graph_model_by_default():
    client, posts = _client_with_recorder()
    await client.cognify(["docs"])
    _, payload = posts[0]
    assert "graph_model" not in payload
