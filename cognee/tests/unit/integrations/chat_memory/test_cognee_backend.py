"""Wire-contract tests for :class:`CogneeMemoryBackend`, with cognee faked.

Fast and keyless: they monkeypatch ``cognee.remember`` / ``recall`` / ``forget``
plus the dataset/user lookups, so they assert the exact calls the backend makes
without spinning up an LLM or a database. These guard the blockers the InMemory
fake backend cannot express:

* ingest goes through the durable path (never the session-cache-only branch that
  drops the text/metadata/data_id);
* per-user "forget me" resolves the user's rows by ``external_metadata["user"]``
  and *fully* deletes them (not ``memory_only``);
* citations resolve back to the source via the ``data_id`` recall reports.
"""

import importlib
import types
import uuid

import pytest

import cognee
from cognee.integrations.chat_memory import CogneeMemoryBackend


class _Row:
    def __init__(self, id, external_metadata):
        self.id = id
        self.external_metadata = external_metadata


class _Dataset:
    def __init__(self, id, name):
        self.id = id
        self.name = name


@pytest.fixture
def fake_cognee(monkeypatch):
    """Replace the cognee surface the backend touches with in-process fakes."""
    state = {"remember_calls": [], "forget_calls": [], "rows": [], "recall_responses": []}

    async def fake_remember(data, **kwargs):
        state["remember_calls"].append({"data": data, "kwargs": kwargs})

    async def fake_recall(query, **kwargs):
        return list(state["recall_responses"])

    async def fake_forget(**kwargs):
        state["forget_calls"].append(kwargs)
        state["rows"] = [row for row in state["rows"] if row.id != kwargs.get("data_id")]
        return {"status": "success"}

    async def fake_get_default_user():
        return _Dataset(uuid.uuid4(), "default_user")

    async def fake_get_authorized_dataset_by_name(name, user, permission):
        return _Dataset(uuid.uuid5(uuid.NAMESPACE_URL, name), name)

    async def fake_list_data(dataset_id, user=None):
        return list(state["rows"])

    monkeypatch.setattr(cognee, "remember", fake_remember)
    monkeypatch.setattr(cognee, "recall", fake_recall)
    monkeypatch.setattr(cognee, "forget", fake_forget)
    monkeypatch.setattr(cognee.datasets, "list_data", fake_list_data)
    monkeypatch.setattr("cognee.modules.users.methods.get_default_user", fake_get_default_user)
    # The backend imports this from the submodule, which the package re-export
    # shadows — patch the submodule object directly rather than by dotted path.
    gad_module = importlib.import_module(
        "cognee.modules.data.methods.get_authorized_dataset_by_name"
    )
    monkeypatch.setattr(
        gad_module, "get_authorized_dataset_by_name", fake_get_authorized_dataset_by_name
    )
    return state


@pytest.mark.asyncio
async def test_ingest_uses_durable_path_with_real_payload(fake_cognee):
    backend = CogneeMemoryBackend()
    await backend.remember(
        "we ship on friday",
        dataset="chat:slack:t1:c1",
        session="slack:t1:c1:th1",
        external_metadata={"user": "U1", "permalink": "https://src/1"},
        item_id="907ed94b-bfad-5d11-bbf2-90412c28dff1",
    )

    assert len(fake_cognee["remember_calls"]) == 1
    call = fake_cognee["remember_calls"][0]
    # Durable path: no session_id (which would drop everything), background on.
    assert call["kwargs"].get("session_id") is None
    assert call["kwargs"]["dataset_name"] == "chat:slack:t1:c1"
    assert call["kwargs"]["run_in_background"] is True
    # The real message + stamp + a stable data_id reach cognee (not "[DataItem]").
    item = call["data"]
    assert item.data == "we ship on friday"
    assert item.external_metadata == {"user": "U1", "permalink": "https://src/1"}
    assert str(item.data_id) == "907ed94b-bfad-5d11-bbf2-90412c28dff1"


@pytest.mark.asyncio
async def test_forget_user_full_deletes_only_that_user(fake_cognee):
    r1, r2 = _Row(uuid.uuid4(), {"user": "U1"}), _Row(uuid.uuid4(), {"user": "U1"})
    r3 = _Row(uuid.uuid4(), {"user": "U2"})
    fake_cognee["rows"] = [r1, r2, r3]

    backend = CogneeMemoryBackend()
    result = await backend.forget_user(dataset="chat:slack:t1:c1", user="U1")

    assert result["items_removed"] == 2
    assert {call["data_id"] for call in fake_cognee["forget_calls"]} == {r1.id, r2.id}
    # Full delete, not memory_only, and always scoped to the dataset.
    for call in fake_cognee["forget_calls"]:
        assert not call.get("memory_only")
        assert call["dataset_id"] is not None
    # U2 survives.
    assert [row.external_metadata["user"] for row in fake_cognee["rows"]] == ["U2"]


@pytest.mark.asyncio
async def test_recall_resolves_citation_via_data_id(fake_cognee):
    data_id = uuid.uuid4()
    fake_cognee["rows"] = [_Row(data_id, {"user": "U1", "permalink": "https://src/1"})]
    fake_cognee["recall_responses"] = [
        types.SimpleNamespace(
            source="graph", text="we ship on friday", score=0.9, metadata={"data_id": str(data_id)}
        )
    ]

    backend = CogneeMemoryBackend()
    citations = await backend.recall(
        "when do we ship", dataset="chat:slack:t1:c1", session="s", top_k=5
    )

    assert len(citations) == 1
    citation = citations[0]
    assert citation.text == "we ship on friday"
    assert citation.source == "graph"
    assert citation.permalink == "https://src/1"
    assert citation.user == "U1"


@pytest.mark.asyncio
async def test_recall_without_data_id_degrades_to_text_only(fake_cognee):
    # A synthesized graph answer carries no data_id; the citation is still valid,
    # just without a resolved permalink/author.
    fake_cognee["recall_responses"] = [
        types.SimpleNamespace(source="graph", text="a synthesized answer", score=None, metadata={})
    ]
    backend = CogneeMemoryBackend()
    citations = await backend.recall("q", dataset="chat:slack:t1:c1", session="s", top_k=5)

    assert len(citations) == 1
    assert citations[0].text == "a synthesized answer"
    assert citations[0].permalink is None
    assert citations[0].user is None
