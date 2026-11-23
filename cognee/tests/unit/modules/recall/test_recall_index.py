"""Recall's index mode and expand-by-id (SDK-1099)."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.api.v1.health import get_capabilities
from cognee.api.v1.recall.query_router import (
    INDEX_SEARCH_TYPES,
    RoutingConstraints,
    route_query,
)
from cognee.api.v1.recall.recall import _validate_index_request
from cognee.exceptions import CogneeValidationError
from cognee.modules.data.content_hash import compute_content_hash
from cognee.modules.recall.methods import recall_index
from cognee.modules.recall.methods.recall_index import (
    INDEX_TEXT_CHARS,
    RECALL_INDEX_CAPABILITY,
    build_index_items,
    expand_records,
)
from cognee.modules.recall.types.RecallIndex import ExpandItem
from cognee.modules.search.models.SearchResultPayload import SearchResultPayload
from cognee.modules.search.types import SearchType

DATASET_ID = uuid4()
DATA_ID = str(uuid4())


def _frame(**overrides) -> dict:
    return {
        "history": "Q: earlier\nA: earlier answer",
        "template_prefix": "The question is: `q`\nContext:\n`",
        "template_suffix": "`",
        "guidance": "### Rules\n- be brief",
        "grounding": "",
        "global_context": "",
        **overrides,
    }


def _payload(retrieved: dict, **overrides) -> SearchResultPayload:
    fields = {
        "result_object": retrieved,
        "context": "rendered",
        "search_type": SearchType.HYBRID_COMPLETION,
        "only_context": True,
        "user_prompt": "full prompt",
        "system_prompt": "system",
        "context_frame": _frame(),
        "dataset_id": DATASET_ID,
        "dataset_name": "agent_sessions",
        **overrides,
    }
    return SearchResultPayload(**fields)


LONG_TEXT = " ".join(["Switching datasets moves saves to the new dataset."] * 10)


def _retrieved() -> dict:
    return {
        "chunks": [
            {
                "id": "chunk-1",
                "text": LONG_TEXT,
                "document_id": DATA_ID,
                "document_name": "notes.md",
                "conflicts": ["The tenant runs 1.6.9 and 1.6.8."],
            }
        ],
        "chunk_summaries": {},
        "chunk_scores": {"chunk-1": {"score": 0.06, "distance": 0.21}},
        "entities": [
            {
                "id": "entity-1",
                "name": "tenant server",
                "type": "software",
                "edges": [
                    {"text": "The tenant server runs cognee.", "edge_object_id": str(uuid4())}
                ],
                "score": 0.3,
                "conflicts": ["The tenant runs 1.6.9 and 1.6.8."],
            }
        ],
        "facts": [
            {"id": "fact-1", "text": "Pinning keeps recall on the active dataset.", "score": 0.4},
            # Shown in the conflicts block already, so not a fact row (format_facts' rule).
            {"id": "fact-2", "text": "The tenant runs 1.6.9 and 1.6.8.", "score": 0.5},
        ],
    }


@pytest.fixture(autouse=True)
def _no_relational_reads(monkeypatch):
    async def created_at(data_ids):
        return {data_id: "2026-11-20T10:00:00" for data_id in data_ids if data_id}

    async def evidence(edge_ids, dataset_id, **kwargs):
        return [
            SimpleNamespace(
                edge_id=edge_id, data_id=DATA_ID, chunk_id="chunk-1", document_name="notes.md"
            )
            for edge_id in edge_ids
        ]

    monkeypatch.setattr(recall_index, "_created_at_by_data_id", created_at)
    monkeypatch.setattr(recall_index, "get_edge_evidence_records", evidence)


@pytest.mark.asyncio
async def test_index_rows_follow_prompt_order_with_ids_versions_and_sources():
    rows = await build_index_items(_payload(_retrieved()))
    kinds = [row.metadata["record_kind"] for row in rows]
    assert kinds == [
        "session_history",
        "template_prefix",
        "passage",
        "entity",
        "fact",
        "template_suffix",
        "session_guidance",
    ]

    passage, entity, fact = rows[2], rows[3], rows[4]
    assert len(passage.text) <= INDEX_TEXT_CHARS
    assert passage.metadata["record_id"] == "chunk-1"
    assert passage.metadata["rank"] == 1
    assert passage.metadata["chars"] == len(LONG_TEXT)
    assert passage.metadata["version"] == compute_content_hash(LONG_TEXT)
    assert passage.metadata["score"] == 0.06
    assert passage.metadata["distance"] == 0.21
    assert passage.metadata["conflicts"] == ["The tenant runs 1.6.9 and 1.6.8."]
    assert passage.metadata["sources"] == [
        {
            "data_id": DATA_ID,
            "document_name": "notes.md",
            "chunk_id": "chunk-1",
            "created_at": "2026-11-20T10:00:00",
        }
    ]

    assert entity.text == "tenant server (software)"
    assert entity.metadata["conflicts"] == ["The tenant runs 1.6.9 and 1.6.8."]
    assert entity.metadata["sources"][0]["data_id"] == DATA_ID

    # A fact fits its preview: the text is the whole body, so it needs no expand.
    assert fact.metadata["chars"] == len(fact.text)
    assert fact.metadata["sources"] is None
    assert fact.metadata["conflicts"] == []
    assert all(row.dataset_id == str(DATASET_ID) for row in rows)


@pytest.mark.asyncio
async def test_frame_rows_carry_full_text_and_no_id():
    rows = await build_index_items(
        _payload(_retrieved(), context_frame=_frame(grounding="## Ontology grounding\n- x"))
    )
    grounding = next(row for row in rows if row.metadata["record_kind"] == "grounding")
    assert grounding.text == "## Ontology grounding\n- x"
    assert grounding.metadata["record_id"] is None
    assert grounding.metadata["version"] is None


@pytest.mark.asyncio
async def test_no_prompt_means_no_rows_and_unsplittable_template_is_refused():
    assert await build_index_items(_payload(_retrieved(), user_prompt=None)) == []
    with pytest.raises(ValueError):
        await build_index_items(
            _payload(_retrieved(), context_frame=_frame(template_prefix=None, template_suffix=None))
        )


@pytest.mark.asyncio
async def test_expand_returns_bodies_missing_and_forbidden_in_request_order(monkeypatch):
    readable = SimpleNamespace(id=DATASET_ID, name="agent_sessions", owner_id=uuid4())
    other_dataset = uuid4()

    async def readable_dataset(dataset_id, user):
        return readable if dataset_id == DATASET_ID else None

    @asynccontextmanager
    async def dataset_context(*args, **kwargs):
        yield

    calls = []

    class FakeRetriever:
        async def expand_records(self, query, ids_by_kind):
            calls.append((query, ids_by_kind))
            return {
                ("passage", "chunk-1"): {
                    "body": "full passage body",
                    "conflicts": [],
                    "item": {"id": "chunk-1", "document_id": DATA_ID, "document_name": "n.md"},
                }
            }

    async def retriever_factory(query_type, query_text, **kwargs):
        assert query_type is SearchType.HYBRID_COMPLETION
        assert kwargs["top_k"] == 5
        return FakeRetriever()

    monkeypatch.setattr(recall_index, "_readable_dataset", readable_dataset)
    monkeypatch.setattr(recall_index, "set_database_global_context_variables", dataset_context)
    monkeypatch.setattr(recall_index, "get_search_type_retriever_instance", retriever_factory)

    items = [
        ExpandItem(dataset_id=DATASET_ID, record_kind="passage", record_id="chunk-1"),
        ExpandItem(dataset_id=DATASET_ID, record_kind="fact", record_id="gone"),
        ExpandItem(dataset_id=other_dataset, record_kind="passage", record_id="chunk-9"),
        ExpandItem(dataset_id=DATASET_ID, record_kind="passage", record_id="chunk-1"),
    ]
    rows = await expand_records(items, query_text="q", user=object(), retriever_kwargs={"top_k": 5})

    assert [row.metadata["status"] for row in rows] == ["available", "missing", "forbidden"]
    assert rows[0].text == "full passage body"
    assert rows[0].metadata["version"] == compute_content_hash("full passage body")
    assert rows[0].metadata["chars"] == len("full passage body")
    assert rows[1].text == "" and rows[2].text == ""
    # One store switch for the readable dataset, with its ids grouped by kind.
    assert calls == [("q", {"passage": ["chunk-1"], "fact": ["gone"]})]


@pytest.mark.parametrize(
    "kwargs, error_name",
    [
        ({"result_mode": "rows"}, "InvalidResultModeError"),
        ({"result_mode": "index", "only_context": False}, "IndexRequiresOnlyContextError"),
        (
            {"result_mode": "index", "query_type": SearchType.CHUNKS},
            "IndexUnsupportedSearchTypeError",
        ),
        ({"result_mode": "index", "sources": ["session", "graph"]}, "IndexRequiresGraphScopeError"),
        ({"expand_items": []}, "InvalidExpandItemsError"),
        ({"expand_items": [object()] * 21}, "InvalidExpandItemsError"),
        ({"expand_items": [object()], "result_mode": "index"}, "InvalidExpandRequestError"),
    ],
)
def test_invalid_index_and_expand_requests_are_refused(kwargs, error_name):
    request = {
        "result_mode": "full",
        "expand_items": None,
        "only_context": True,
        "query_type": None,
        "sources": ["graph"],
        **kwargs,
    }
    with pytest.raises(CogneeValidationError) as raised:
        _validate_index_request(**request)
    assert raised.value.name == error_name


def test_valid_index_and_expand_requests_pass():
    _validate_index_request(
        result_mode="index",
        expand_items=None,
        only_context=True,
        query_type=SearchType.HYBRID_COMPLETION,
        sources=["graph"],
    )
    _validate_index_request(
        result_mode="full",
        expand_items=[object()] * 20,
        only_context=False,
        query_type=None,
        sources=["session", "graph"],
    )


def test_index_routes_unsupported_picks_to_the_default():
    quoted = '"exact phrase"'
    assert route_query(quoted, RoutingConstraints()).search_type is SearchType.CHUNKS_LEXICAL
    indexed = route_query(quoted, RoutingConstraints(index_types=INDEX_SEARCH_TYPES))
    assert indexed.search_type is SearchType.HYBRID_COMPLETION
    assert RoutingConstraints.from_config(None, result_mode="index").index_types == (
        INDEX_SEARCH_TYPES
    )
    assert RoutingConstraints.from_config(None).index_types is None


def test_health_lists_the_recall_index_capability():
    assert RECALL_INDEX_CAPABILITY == "recall_index_v1"
    assert RECALL_INDEX_CAPABILITY in get_capabilities()


def test_recall_request_body_accepts_index_and_expand_fields():
    from pydantic import ValidationError

    from cognee.api.v1.recall.routers.get_recall_router import RecallPayloadDTO

    body = RecallPayloadDTO(
        query="q",
        result_mode="index",
        expand_items=[
            {"dataset_id": str(DATASET_ID), "record_kind": "passage", "record_id": "chunk-1"}
        ],
    )
    assert body.result_mode == "index"
    assert body.expand_items[0].dataset_id == DATASET_ID
    with pytest.raises(ValidationError):
        RecallPayloadDTO(query="q", result_mode="rows")
