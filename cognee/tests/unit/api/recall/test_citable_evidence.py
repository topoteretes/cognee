"""No network/LLM: real evidence projection, ACL boundary and version semantics."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from cognee.modules.recall.evidence import service
from cognee.modules.recall.evidence.models import EvidenceBatch, EvidenceLocator


@pytest.fixture
def store(monkeypatch):
    dataset, chunk, document, owner = uuid4(), uuid4(), uuid4(), uuid4()
    ref = EvidenceLocator(dataset_id=dataset, kind="segment", artifact_id=chunk)
    engine = SimpleNamespace(
        get_nodes=AsyncMock(
            return_value=[
                {
                    "id": str(chunk),
                    "type": "DocumentChunk",
                    "text": "Use bounded retries.",
                    "document_id": str(document),
                    "chunk_index": 3,
                }
            ]
        ),
        has_edge=AsyncMock(return_value=True),
    )
    auth = AsyncMock(return_value=[SimpleNamespace(id=dataset, owner_id=owner)])
    monkeypatch.setattr(service, "get_authorized_existing_datasets", auth)
    monkeypatch.setattr(service, "get_graph_engine", AsyncMock(return_value=engine))
    monkeypatch.setattr(service, "backend_access_control_enabled", lambda: True)
    calls = []

    @asynccontextmanager
    async def context(dataset_id, owner_id):
        calls.append((dataset_id, owner_id))
        yield

    monkeypatch.setattr(service, "set_database_global_context_variables", context)
    docs = AsyncMock(
        return_value={
            str(document): {
                "data_id": str(document),
                "document_name": "Retry policy",
                "source_version": "doc-v1",
            }
        }
    )
    monkeypatch.setattr(service, "_documents", docs)
    return SimpleNamespace(
        ref=ref,
        engine=engine,
        auth=auth,
        docs=docs,
        calls=calls,
        dataset=dataset,
        owner=owner,
        user=SimpleNamespace(id=uuid4()),
    )


@pytest.mark.asyncio
async def test_index_has_no_body_and_batch_expands_stable_version(store):
    row = {"metadata": {"evidence": [store.ref.model_dump(mode="json")]}}
    index = await service.evidence_index([row], store.user)
    item = index.items[0]
    assert item.text is None and item.status == "available"
    assert item.reference.version and item.citation.endswith(item.reference.version)
    expanded = await service.expand_evidence([item.reference] * 5, store.user)
    assert all(i.text == "Use bounded retries." for i in expanded)
    assert all(i.reference == item.reference for i in expanded)
    assert store.engine.get_nodes.await_count == 2  # one graph batch per dataset
    assert store.calls == [(store.dataset, store.owner)] * 2


@pytest.mark.asyncio
async def test_changed_payload_or_source_revision_never_silently_replaces_text(store):
    old = (await service.expand_evidence([store.ref], store.user))[0]
    store.engine.get_nodes.return_value[0]["text"] = "Use a different policy."
    new = (await service.expand_evidence([old.reference], store.user))[0]
    assert new.status == "changed" and new.text is None
    assert new.requested_version == old.current_version
    assert new.current_version != old.current_version
    accepted = (await service.expand_evidence([new.reference], store.user))[0]
    assert accepted.status == "available" and accepted.text == "Use a different policy."
    store.docs.return_value[next(iter(store.docs.return_value))]["source_version"] = "doc-v2"
    assert (await service.expand_evidence([accepted.reference], store.user))[0].status == "changed"


@pytest.mark.asyncio
async def test_revoked_access_deleted_chunk_and_deleted_document_are_unavailable(store):
    store.auth.return_value = []
    assert (await service.expand_evidence([store.ref], store.user))[0].status == "unavailable"
    store.engine.get_nodes.assert_not_awaited()
    store.auth.return_value = [SimpleNamespace(id=store.dataset, owner_id=store.owner)]
    store.docs.return_value = {}  # source deleted but graph chunk remains
    assert (await service.expand_evidence([store.ref], store.user))[0].text is None
    store.engine.get_nodes.return_value = []
    assert (await service.expand_evidence([store.ref], store.user))[0].status == "unavailable"


@pytest.mark.asyncio
async def test_foreign_ids_and_internal_nodes_cannot_leak(store):
    foreign = store.ref.model_copy(update={"dataset_id": uuid4()})
    rows = await service.expand_evidence([foreign, store.ref], store.user)
    assert rows[0].status == "unavailable" and rows[1].status == "available"
    store.engine.get_nodes.return_value[0]["is_internal"] = True
    assert (await service.expand_evidence([store.ref], store.user))[0].status == "unavailable"


@pytest.mark.asyncio
async def test_shared_graph_storage_explicitly_unsupported(store, monkeypatch):
    monkeypatch.setattr(service, "backend_access_control_enabled", lambda: False)
    assert (await service.expand_evidence([store.ref], store.user))[0].status == "unsupported"
    store.engine.get_nodes.assert_not_awaited()


@pytest.mark.asyncio
async def test_backend_failure_not_absence(store):
    store.engine.get_nodes.side_effect = RuntimeError("private storage details")
    row = (await service.expand_evidence([store.ref], store.user))[0]
    assert row.status == "error" and "private" not in row.model_dump_json()


def test_public_batch_bounds_and_locator_validation(store):
    with pytest.raises(ValidationError):
        EvidenceBatch(references=[store.ref] * 6)
    with pytest.raises(ValidationError):
        EvidenceBatch(references=[])
    with pytest.raises(ValidationError):
        EvidenceLocator(dataset_id=store.dataset, artifact_id=uuid4(), kind="graph_edge")


@pytest.mark.asyncio
async def test_index_bound_and_unaddressable_results(store):
    refs = [
        store.ref.model_copy(update={"artifact_id": uuid4()}).model_dump(mode="json")
        for _ in range(23)
    ]
    result = await service.evidence_index(
        [{"metadata": {"evidence": refs}}, {"text": "legacy"}], store.user
    )
    assert len(result.items) == 20 and result.omitted == 3
    assert result.unaddressable_results == 1
    assert all(i.text is None for i in result.items)


@pytest.mark.asyncio
async def test_exact_directed_edge_and_resolvable_citation(store):
    from urllib.parse import parse_qs, urlparse

    left, right = uuid4(), uuid4()
    name = "depends on / with spaces"
    edge_id = service.generate_edge_object_id(str(left), str(right), name)
    ref = EvidenceLocator(
        dataset_id=store.dataset,
        kind="graph_edge",
        artifact_id=edge_id,
        source_node_id=left,
        target_node_id=right,
        relationship_name=name,
    )
    store.engine.get_nodes.return_value = [
        {"id": str(left), "name": "Worker"},
        {"id": str(right), "name": "Queue"},
    ]
    item = (await service.expand_evidence([ref], store.user))[0]
    assert item.status == "available" and item.source["derived"]
    assert parse_qs(urlparse(item.citation).query)["relationship_name"] == [name]
    store.engine.has_edge.assert_awaited_once_with(str(left), str(right), name)
    forged = ref.model_copy(update={"artifact_id": uuid4()})
    assert (await service.expand_evidence([forged], store.user))[0].status == "unavailable"
    store.engine.has_edge.return_value = False
    assert (await service.expand_evidence([ref], store.user))[0].status == "unavailable"


@pytest.fixture
def client(store, monkeypatch):
    import importlib

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from cognee.modules.recall.types.RecallResponse import ResponseGraphEntry
    from cognee.modules.recall.types.SearchResultItem import SearchResultKind
    from cognee.modules.search.types import SearchType

    router = importlib.import_module("cognee.api.v1.recall.routers.get_recall_router")
    row = ResponseGraphEntry(
        source="graph",
        kind=SearchResultKind.CHUNK,
        search_type=SearchType.CHUNKS,
        text="Legacy context",
        dataset_id=str(store.dataset),
        metadata={"evidence": [store.ref.model_dump(mode="json")]},
    )
    recall = AsyncMock(return_value=[row])
    monkeypatch.setattr(importlib.import_module("cognee.api.v1.recall"), "recall", recall)
    app = FastAPI()
    app.include_router(router.get_recall_router(), prefix="/api/v1/recall")
    app.dependency_overrides[router.get_authenticated_user] = lambda: store.user
    return TestClient(app), recall


def test_http_index_context_and_detail_contract(client, store):
    http, recall = client
    response = http.post("/api/v1/recall", json={"query": "policy", "evidence_mode": "index"})
    assert response.status_code == 200, response.text
    item = response.json()["items"][0]
    assert item["text"] is None and item["reference"]["version"]
    assert recall.call_args.kwargs["only_context"] is True
    assert recall.call_args.kwargs["include_references"] is True
    details = http.post("/api/v1/recall/evidence", json={"references": [item["reference"]]})
    assert details.status_code == 200 and details.json()[0]["text"] == "Use bounded retries."
    assert (
        http.post(
            "/api/v1/recall/evidence", json={"references": [item["reference"]] * 6}
        ).status_code
        == 422
    )
    context = http.post("/api/v1/recall", json={"query": "policy", "evidence_mode": "context"})
    assert context.status_code == 200, context.text
    assert context.json()[0]["metadata"]["evidence_items"][0]["citation"] == item["citation"]
    assert (
        http.post(
            "/api/v1/recall", json={"query": "p", "evidence_mode": "index", "stream": True}
        ).status_code
        == 422
    )


@pytest.mark.asyncio
async def test_large_context_excerpt_keeps_full_version(store):
    from cognee.modules.recall.types.RecallResponse import ResponseGraphEntry
    from cognee.modules.recall.types.SearchResultItem import SearchResultKind
    from cognee.modules.search.types import SearchType

    store.engine.get_nodes.return_value[0]["text"] = "long source " * 1500
    full = (await service.expand_evidence([store.ref], store.user))[0]
    row = ResponseGraphEntry(
        source="graph",
        kind=SearchResultKind.CHUNK,
        search_type=SearchType.CHUNKS,
        text="legacy",
        metadata={"evidence": [store.ref.model_dump(mode="json")]},
    )
    attached = (await service.attach_evidence([row], store.user))[0].metadata["evidence_items"][0]
    assert len(attached["text"]) == 1400 and attached["truncated"]
    assert attached["current_version"] == full.current_version


@pytest.mark.asyncio
async def test_source_document_membership_uses_real_relational_rows():
    from sqlalchemy import delete

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.models import Data

    first, foreign = uuid4(), uuid4()
    first_id, foreign_id = uuid4(), uuid4()
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        session.add_all(
            [
                Data(id=first_id, dataset_id=first, name="allowed.txt", content_hash="v1"),
                Data(id=foreign_id, dataset_id=foreign, name="private.txt", content_hash="v2"),
            ]
        )
        await session.commit()
    try:
        docs = await service._documents(
            first,
            {
                "one": {"document_id": str(first_id)},
                "two": {"document_id": str(foreign_id)},
            },
        )
        assert list(docs) == [str(first_id)]
        assert docs[str(first_id)]["source_version"] == "v1"
        assert docs[str(first_id)]["source_ref_key"]
        assert "private.txt" not in str(docs)
    finally:
        async with engine.get_async_session() as session:
            await session.execute(delete(Data).where(Data.id.in_([first_id, foreign_id])))
            await session.commit()


def test_vector_and_lexical_retrieval_expose_actual_chunk_identities():
    from cognee.modules.retrieval.chunks_retriever import ChunksRetriever
    from cognee.modules.retrieval.lexical_retriever import LexicalRetriever

    dataset, chunk, document = uuid4(), uuid4(), uuid4()
    payload = {"id": str(chunk), "document_id": str(document), "text": "policy"}
    vector = ChunksRetriever()
    lexical = LexicalRetriever(str.split, lambda a, b: 1.0)
    for retriever, objects in (
        (vector, [SimpleNamespace(id=chunk, payload=payload, score=0.1)]),
        (lexical, [payload]),
        (lexical, [(payload, 1.0)]),
    ):
        refs = retriever.get_context_evidence(objects, dataset)
        assert len(refs) == 1
        assert refs[0].artifact_id == str(chunk) and refs[0].data_id == str(document)
        assert refs[0].dataset_id == str(dataset)


@pytest.mark.asyncio
async def test_access_timestamps_do_not_invalidate_a_citation(store):
    old = (await service.expand_evidence([store.ref], store.user))[0]
    source = store.docs.return_value[next(iter(store.docs.return_value))]
    source["updated_at"] = "2026-10-13T12:00:00Z"
    source["created_at"] = "2026-10-01T12:00:00Z"
    current = (await service.expand_evidence([old.reference], store.user))[0]
    assert current.status == "available" and current.current_version == old.current_version
