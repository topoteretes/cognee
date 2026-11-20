"""recall() and search() send retriever_specific_config to the remote instance (#6692).

Before this, ``client.recall(query, retriever_specific_config={...})`` and
``client.search(query, retriever_specific_config={...})`` returned success while
the server ran with its own defaults: CloudClient.recall and CloudClient.search
built their JSON payload from a fixed field list that did not include
``retriever_specific_config``. The server endpoints (RecallPayloadDTO and
SearchPayloadDTO) already accept and validate the field; the gap was strictly
on the client side.
"""

import pytest

from cognee.api.v1.recall.routers.get_recall_router import RecallPayloadDTO
from cognee.api.v1.search.routers.get_search_router import SearchPayloadDTO
from cognee.api.v1.serve import state as state_mod
from cognee.api.v1.serve.cloud_client import CloudClient

CONFIG = {"chunks_top_k": 5, "include_external_metadata": True}


class _FakeResponse:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return []

    async def text(self):
        return ""


class _FakeSession:
    """Captures what the client POSTs: the JSON payload of a recall / search."""

    def __init__(self):
        self.calls = []

    def post(self, url, data=None, json=None, **kwargs):
        self.calls.append({"url": url, "form": data, "json": json})
        return _FakeResponse()

    async def close(self):
        pass


@pytest.fixture
def remote(monkeypatch):
    client = CloudClient("http://remote.invalid", "key")
    session = _FakeSession()

    async def get_session():
        return session

    monkeypatch.setattr(client, "_get_session", get_session)
    monkeypatch.setattr(state_mod, "_remote_client", client)
    return client, session


# ----- recall() -----


@pytest.mark.asyncio
async def test_recall_sends_retriever_specific_config(remote):
    client, session = remote
    await client.recall("What is in the document?", retriever_specific_config=dict(CONFIG))

    (call,) = session.calls
    assert call["url"] == "http://remote.invalid/api/v1/recall"
    # The server's own DTO reads the field back: what the client sends must be
    # exactly what the route accepts (either key form binds via populate_by_name).
    dto = RecallPayloadDTO.model_validate(call["json"])
    assert dto.retriever_specific_config == CONFIG


@pytest.mark.asyncio
async def test_recall_sends_no_retriever_specific_config_by_default(remote):
    client, session = remote
    await client.recall("What is in the document?")

    (call,) = session.calls
    assert "retriever_specific_config" not in call["json"]
    assert "retrieverSpecificConfig" not in call["json"]
    assert RecallPayloadDTO.model_validate(call["json"]).retriever_specific_config is None


# ----- search() -----


@pytest.mark.asyncio
async def test_search_sends_retriever_specific_config(remote):
    client, session = remote
    await client.search("What is in the document?", retriever_specific_config=dict(CONFIG))

    (call,) = session.calls
    assert call["url"] == "http://remote.invalid/api/v1/search"
    # The server's own DTO reads the field back (camelCase or snake_case both bind).
    dto = SearchPayloadDTO.model_validate(call["json"])
    assert dto.retriever_specific_config == CONFIG


@pytest.mark.asyncio
async def test_search_sends_no_retriever_specific_config_by_default(remote):
    client, session = remote
    await client.search("What is in the document?")

    (call,) = session.calls
    assert "retriever_specific_config" not in call["json"]
    assert "retrieverSpecificConfig" not in call["json"]
    assert SearchPayloadDTO.model_validate(call["json"]).retriever_specific_config is None
