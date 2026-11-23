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
from cognee.api.v1.recall.recall import _expand_items_as_models, _validate_index_request
from cognee.exceptions import CogneeSystemError, CogneeValidationError
from cognee.modules.data.content_hash import compute_content_hash
from cognee.modules.engine.models import Entity
from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.recall.methods import recall_index
from cognee.modules.recall.methods.recall_index import (
    INDEX_TEXT_CHARS,
    RECALL_INDEX_CAPABILITY,
    build_index_items,
    expand_records,
    rebuild_user_prompt,
)
from cognee.modules.recall.types.RecallIndex import ExpandItem
from cognee.modules.retrieval.hybrid.context import format_hybrid_context, render_passage
from cognee.modules.retrieval.hybrid.entities import render_entity
from cognee.modules.search.models.SearchResultPayload import SearchResultPayload
from cognee.modules.search.types import SearchType

DATASET_ID = uuid4()
DATA_ID = str(uuid4())
CHUNK_ID = str(uuid4())
ENTITY_ID = str(uuid4())
FACT_ID = str(uuid4())
CONFLICT_FACT_ID = str(uuid4())
GONE_ID = str(uuid4())
OTHER_CHUNK_ID = str(uuid4())


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
                "id": CHUNK_ID,
                "text": LONG_TEXT,
                "document_id": DATA_ID,
                "document_name": "notes.md",
                "conflicts": ["The tenant runs 1.6.9 and 1.6.8."],
            }
        ],
        "chunk_summaries": {},
        "chunk_scores": {CHUNK_ID: {"score": 0.06, "distance": 0.21}},
        "entities": [
            {
                "id": ENTITY_ID,
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
            {"id": FACT_ID, "text": "Pinning keeps recall on the active dataset.", "score": 0.4},
            # Shown in the conflicts block already, so not a fact row (format_facts' rule).
            {"id": CONFLICT_FACT_ID, "text": "The tenant runs 1.6.9 and 1.6.8.", "score": 0.5},
        ],
    }


@pytest.fixture(autouse=True)
def _no_relational_reads(monkeypatch):
    async def created_at(data_ids):
        return {data_id: "2026-11-20T10:00:00" for data_id in data_ids if data_id}

    async def evidence(edge_ids, dataset_id, **kwargs):
        return [
            SimpleNamespace(
                edge_id=edge_id, data_id=DATA_ID, chunk_id=CHUNK_ID, document_name="notes.md"
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
    assert passage.metadata["record_id"] == CHUNK_ID
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
            "chunk_id": CHUNK_ID,
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
    with pytest.raises(CogneeValidationError) as raised:
        await build_index_items(
            _payload(_retrieved(), context_frame=_frame(template_prefix=None, template_suffix=None))
        )
    assert raised.value.name == "IndexTemplateNotSplittableError"


@pytest.mark.asyncio
async def test_a_dataset_hybrid_handed_to_graph_completion_is_refused_by_name():
    # hybrid_deferral_reason: a dataset without a chunk collection runs GRAPH_COMPLETION,
    # whose result is a list with no frame.
    payload_ = _payload(
        ["triplet"],
        search_type=SearchType.GRAPH_COMPLETION,
        context_frame=None,
        dataset_name="code_repo",
    )
    with pytest.raises(CogneeValidationError) as raised:
        await build_index_items(payload_)
    assert raised.value.name == "IndexRequiresHybridError"
    assert "'code_repo'" in raised.value.message


@pytest.mark.asyncio
async def test_a_hybrid_payload_without_its_frame_is_an_internal_error():
    with pytest.raises(CogneeSystemError) as raised:
        await build_index_items(_payload(_retrieved(), context_frame=None))
    assert raised.value.name == "IndexFrameMissingError"


@pytest.mark.asyncio
async def test_rows_rebuild_the_user_prompt_the_retriever_renders():
    retrieved = _retrieved()
    frame = _frame(grounding="## Ontology grounding\n- x", global_context="## Global context\n- y")
    # The prompt exactly as hybrid and build_completion_prompts assemble it.
    context = f"{frame['grounding']}\n\n" + format_hybrid_context(
        frame["global_context"], retrieved
    )
    rendered = frame["template_prefix"] + context + frame["template_suffix"]
    user_prompt = "\n\n".join((frame["history"], rendered, frame["guidance"]))

    rows = await build_index_items(_payload(retrieved, context_frame=frame))
    bodies = {("passage", CHUNK_ID): render_passage(retrieved["chunks"][0])}
    bodies[("entity", ENTITY_ID)] = render_entity(retrieved["entities"][0])

    assert rebuild_user_prompt(rows, bodies) == user_prompt
    assert "## Fact conflicts" in user_prompt


@pytest.mark.asyncio
async def test_an_edge_two_shown_entities_share_is_a_source_of_both():
    retrieved = _retrieved()
    shared_edge = retrieved["entities"][0]["edges"][0]
    retrieved["entities"].append(
        {"id": str(uuid4()), "name": "cognee", "type": "software", "edges": [shared_edge]}
    )
    rows = await build_index_items(_payload(retrieved))
    entities = [row for row in rows if row.metadata["record_kind"] == "entity"]
    assert len(entities) == 2
    assert all(row.metadata["sources"][0]["data_id"] == DATA_ID for row in entities)


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
                ("passage", CHUNK_ID): {
                    "body": "full passage body",
                    "conflicts": [],
                    "item": {"id": CHUNK_ID, "document_id": DATA_ID, "document_name": "n.md"},
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
        ExpandItem(dataset_id=DATASET_ID, record_kind="passage", record_id=CHUNK_ID),
        ExpandItem(dataset_id=DATASET_ID, record_kind="fact", record_id=GONE_ID),
        ExpandItem(dataset_id=other_dataset, record_kind="passage", record_id=OTHER_CHUNK_ID),
        ExpandItem(dataset_id=DATASET_ID, record_kind="passage", record_id=CHUNK_ID),
    ]
    rows = await expand_records(items, query_text="q", user=object(), retriever_kwargs={"top_k": 5})

    assert [row.metadata["status"] for row in rows] == ["available", "missing", "forbidden"]
    assert rows[0].text == "full passage body"
    assert rows[0].metadata["version"] == compute_content_hash("full passage body")
    assert rows[0].metadata["chars"] == len("full passage body")
    assert rows[1].text == "" and rows[2].text == ""
    # One store switch for the readable dataset, with its ids grouped by kind.
    assert calls == [("q", {"passage": [CHUNK_ID], "fact": [GONE_ID]})]


_VALID_REQUEST = {
    "result_mode": "full",
    "expand_items": None,
    "only_context": True,
    "query_type": None,
    "sources": ["graph"],
    "node_type": NodeSet,
    "neighborhood_depth": None,
    "feedback_influence": 0.0,
}


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
        # Requests hybrid would hand to GRAPH_COMPLETION (request_deferral_reason).
        ({"result_mode": "index", "node_type": Entity}, "IndexRequiresHybridError"),
        ({"result_mode": "index", "neighborhood_depth": 2}, "IndexRequiresHybridError"),
        # Also what a deployment's DEFAULT_FEEDBACK_INFLUENCE > 0 passes by default.
        ({"result_mode": "index", "feedback_influence": 0.3}, "IndexRequiresHybridError"),
    ],
)
def test_invalid_index_and_expand_requests_are_refused(kwargs, error_name):
    request = {**_VALID_REQUEST, **kwargs}
    with pytest.raises(CogneeValidationError) as raised:
        _validate_index_request(**request)
    assert raised.value.name == error_name


def test_valid_index_and_expand_requests_pass():
    _validate_index_request(
        **{**_VALID_REQUEST, "result_mode": "index", "query_type": SearchType.HYBRID_COMPLETION}
    )
    # Expand runs no hybrid search, so the graph-only options do not refuse it.
    _validate_index_request(
        **{
            **_VALID_REQUEST,
            "expand_items": [object()] * 20,
            "only_context": False,
            "sources": ["session", "graph"],
            "node_type": Entity,
            "feedback_influence": 0.3,
        }
    )


def test_expand_items_from_index_rows_are_accepted_as_dicts_in_either_spelling():
    item = ExpandItem(dataset_id=DATASET_ID, record_kind="passage", record_id=CHUNK_ID)
    snake = {"dataset_id": str(DATASET_ID), "record_kind": "passage", "record_id": CHUNK_ID}
    camel = {"datasetId": str(DATASET_ID), "recordKind": "passage", "recordId": CHUNK_ID.upper()}
    assert _expand_items_as_models([item, snake, camel]) == [item, item, item]
    assert _expand_items_as_models(None) is None


@pytest.mark.parametrize(
    "bad_item",
    [
        {"dataset_id": str(DATASET_ID), "record_kind": "template_prefix", "record_id": CHUNK_ID},
        {"dataset_id": str(DATASET_ID), "record_kind": "passage"},
        {"dataset_id": str(DATASET_ID), "record_kind": "passage", "record_id": "chunk-1"},
    ],
)
def test_an_invalid_expand_item_is_refused_with_its_position(bad_item):
    snake = {"dataset_id": str(DATASET_ID), "record_kind": "passage", "record_id": CHUNK_ID}
    with pytest.raises(CogneeValidationError) as raised:
        _expand_items_as_models([snake, bad_item])
    assert raised.value.name == "InvalidExpandItemsError"
    assert "expand_items[1]" in raised.value.message


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

    body = RecallPayloadDTO.model_validate(
        {
            "query": "q",
            "resultMode": "index",
            "expandItems": [
                {"dataset_id": str(DATASET_ID), "record_kind": "passage", "record_id": CHUNK_ID},
                {"datasetId": str(DATASET_ID), "recordKind": "passage", "recordId": CHUNK_ID},
            ],
        }
    )
    assert body.result_mode == "index"
    assert body.expand_items[0] == body.expand_items[1]
    assert body.expand_items[0].dataset_id == DATASET_ID
    with pytest.raises(ValidationError):
        RecallPayloadDTO(query="q", result_mode="rows")
