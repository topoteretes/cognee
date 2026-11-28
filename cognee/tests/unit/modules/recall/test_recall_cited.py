"""Recall's cited mode and expand-by-id (SDK-1099)."""

import re
from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.api.v1.health import get_capabilities
from cognee.api.v1.recall.query_router import (
    CITED_SEARCH_TYPES,
    RoutingConstraints,
    route_query,
)
from cognee.api.v1.recall.recall import _expand_items_as_models, _validate_cited_request
from cognee.exceptions import CogneeSystemError, CogneeValidationError
from cognee.modules.data.content_hash import compute_content_hash
from cognee.modules.engine.models import Entity
from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.recall.methods import recall_cited
from cognee.modules.recall.methods.recall_cited import (
    RECALL_CITED_CAPABILITY,
    build_cited_item,
    build_cited_items,
    expand_records,
    parse_record_tag,
    record_tag,
)
from cognee.modules.recall.types.RecallCited import ExpandItem
from cognee.modules.retrieval.hybrid.context import format_hybrid_context, render_passage
from cognee.modules.retrieval.hybrid.entities import render_entity, stamp_full_version
from cognee.modules.search.models.SearchResultPayload import SearchResultPayload
from cognee.modules.search.types import SearchType
from cognee.modules.users.exceptions import PermissionDeniedError

DATASET_ID = uuid4()
DATA_ID = str(uuid4())
CHUNK_ID = str(uuid4())
ENTITY_ID = str(uuid4())
FACT_ID = str(uuid4())
CONFLICT_FACT_ID = str(uuid4())
GONE_ID = str(uuid4())
OTHER_CHUNK_ID = str(uuid4())
TAG_LINE = re.compile(
    r"^\[mem:(passage|entity|fact)/([0-9a-f-]{36}/)?[0-9a-f-]{36}@[0-9a-f]{32}\]$"
)
SHOWN_EDGE = {"text": "The tenant server runs cognee.", "edge_object_id": str(uuid4())}
# An edge the query ranked out of the prompt: in the entity's full body only.
UNSHOWN_EDGE = {"text": "The tenant server hosts the API.", "edge_object_id": str(uuid4())}


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


def _payload(retrieved, **overrides) -> SearchResultPayload:
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
FACT_TEXT = "Pinning keeps recall on the active dataset."


def _entity(**overrides) -> dict:
    """An entity as the retriever returns it: the full edge list replaced by its hash."""
    entity = {
        "id": ENTITY_ID,
        "name": "tenant server",
        "type": "software",
        "edges": [SHOWN_EDGE],
        "all_edges": [UNSHOWN_EDGE, SHOWN_EDGE],
        "score": 0.3,
        "conflicts": ["The tenant runs 1.6.9 and 1.6.8."],
        **overrides,
    }
    stamp_full_version(entity, citable=True)
    return entity


FULL_ENTITY = render_entity(
    {"name": "tenant server", "type": "software", "edges": [UNSHOWN_EDGE, SHOWN_EDGE]}
)


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
        "entities": [_entity()],
        "facts": [
            {"id": FACT_ID, "text": FACT_TEXT, "score": 0.4},
            # Shown in the conflicts block already, so not a fact (format_facts' rule).
            {"id": CONFLICT_FACT_ID, "text": "The tenant runs 1.6.9 and 1.6.8.", "score": 0.5},
        ],
    }


def _regular_prompt(retrieved: dict, frame: dict) -> str:
    """The prompt exactly as hybrid and build_completion_prompts assemble it."""
    context = format_hybrid_context(frame["global_context"], retrieved)
    if frame["grounding"]:
        context = f"{frame['grounding']}\n\n{context}"
    rendered = frame["template_prefix"] + context + frame["template_suffix"]
    return "\n\n".join(part for part in (frame["history"], rendered, frame["guidance"]) if part)


def _without_tags(text: str) -> str:
    return "\n".join(line for line in text.split("\n") if not TAG_LINE.match(line))


@pytest.fixture(autouse=True)
def _no_relational_reads(monkeypatch):
    async def created_at(data_ids):
        return {data_id: "2026-11-20T10:00:00" for data_id in data_ids if data_id}

    async def no_evidence_read(dataset_id, edge_ids):
        raise AssertionError("a cited recall reads no edge evidence")

    monkeypatch.setattr(recall_cited, "_created_at_by_data_id", created_at)
    monkeypatch.setattr(recall_cited, "get_edge_sources", no_evidence_read)


@pytest.mark.asyncio
async def test_cited_prompt_is_the_regular_prompt_with_a_tag_before_every_record():
    retrieved = _retrieved()
    frame = _frame(grounding="## Ontology grounding\n- x", global_context="## Global context\n- y")

    (item,) = await build_cited_item(_payload(retrieved, context_frame=frame))

    assert _without_tags(item.text) == _regular_prompt(retrieved, frame)
    assert item.system_prompt == "system"
    assert item.raw == {"value": item.text}
    lines = item.text.split("\n")
    tags = [line for line in lines if TAG_LINE.match(line)]
    assert tags == [record["tag"] for record in item.metadata["records"]]
    # Uniform: each tag line is directly followed by its record's body.
    for record in item.metadata["records"]:
        assert f"{record['tag']}\n" in item.text
    # The frame and the section headers carry no tag.
    for untagged in ("## Ontology grounding", "## Global context", "## Relevant passages"):
        (position,) = [i for i, line in enumerate(lines) if line.endswith(untagged)]
        assert not TAG_LINE.match(lines[position - 1])
    assert "## Fact conflicts\n- The tenant runs 1.6.9 and 1.6.8." in item.text


@pytest.mark.asyncio
async def test_records_carry_ids_versions_scores_and_sources_in_prompt_order():
    retrieved = _retrieved()
    (item,) = await build_cited_item(_payload(retrieved))
    passage, entity, fact = item.metadata["records"]

    passage_body = render_passage(retrieved["chunks"][0])
    assert passage["record_kind"] == "passage" and passage["record_id"] == CHUNK_ID
    assert passage["dataset_id"] == str(DATASET_ID)
    assert passage["version"] == compute_content_hash(passage_body)
    assert passage["chars"] == len(passage_body)
    assert passage["tag"] == record_tag("passage", DATASET_ID, CHUNK_ID, passage["version"])
    assert (passage["rank"], passage["score"], passage["distance"]) == (1, 0.06, 0.21)
    assert passage["conflicts"] == ["The tenant runs 1.6.9 and 1.6.8."]
    assert passage["sources"] == [
        {
            "data_id": DATA_ID,
            "document_name": "notes.md",
            "chunk_id": CHUNK_ID,
            "created_at": "2026-11-20T10:00:00",
        }
    ]

    # An entity's prompt shows the ranked bullets; its version and full_chars describe
    # every edge, the body expand returns, whatever the query ranked. The retriever
    # hashed that body and dropped the edge list; its sources come from expand.
    shown_entity = render_entity(retrieved["entities"][0])
    assert "all_edges" not in retrieved["entities"][0]
    assert UNSHOWN_EDGE["text"] in FULL_ENTITY and UNSHOWN_EDGE["text"] not in shown_entity
    assert entity["record_id"] == ENTITY_ID
    assert entity["chars"] == len(shown_entity)
    assert entity["full_chars"] == len(FULL_ENTITY)
    assert entity["version"] == compute_content_hash(FULL_ENTITY)
    assert f"{entity['tag']}\n{shown_entity}" in item.text
    assert entity["sources"] is None
    assert passage["full_chars"] == passage["chars"]

    # A fact's body is its line in Related facts; its sources are not tracked.
    assert fact["record_id"] == FACT_ID
    assert fact["version"] == compute_content_hash(f"- {FACT_TEXT}")
    assert fact["sources"] is None
    assert f"{fact['tag']}\n- {FACT_TEXT}" in item.text


@pytest.mark.asyncio
async def test_no_prompt_means_no_item_and_unsplittable_template_is_refused():
    assert await build_cited_item(_payload(_retrieved(), user_prompt=None)) == []
    with pytest.raises(CogneeValidationError) as raised:
        await build_cited_item(
            _payload(_retrieved(), context_frame=_frame(template_prefix=None, template_suffix=None))
        )
    assert raised.value.name == "CitedTemplateNotSplittableError"


def _handed_off(name: str) -> SearchResultPayload:
    # hybrid_deferral_reason: a dataset without a chunk collection runs GRAPH_COMPLETION,
    # whose result is a list with no frame.
    return _payload(
        ["triplet"],
        search_type=SearchType.GRAPH_COMPLETION,
        context_frame=None,
        dataset_id=uuid4(),
        dataset_name=name,
    )


@pytest.mark.asyncio
async def test_a_dataset_hybrid_handed_to_graph_completion_is_left_out():
    items = await build_cited_items([_handed_off("code_repo"), _payload(_retrieved())])
    assert [item.dataset_name for item in items] == ["agent_sessions"]


@pytest.mark.asyncio
async def test_cited_recall_is_refused_when_no_dataset_can_be_cited():
    with pytest.raises(CogneeValidationError) as raised:
        await build_cited_items(
            [_handed_off("code_repo"), _payload(_retrieved(), user_prompt=None)]
        )
    assert raised.value.name == "CitedRequiresHybridError"
    assert "'code_repo'" in raised.value.message
    # Nothing found at all is not a refusal.
    assert await build_cited_items([_payload(_retrieved(), user_prompt=None)]) == []


@pytest.mark.asyncio
async def test_a_shared_store_result_is_tagged_without_a_dataset():
    # Access control off and several datasets: one search over the shared store, so the
    # payload names no dataset.
    retrieved = _retrieved()

    (item,) = await build_cited_items([_payload(retrieved, dataset_id=None)])

    assert _without_tags(item.text) == _regular_prompt(retrieved, _frame())
    for record in item.metadata["records"]:
        assert record["dataset_id"] is None
        assert record["tag"] == record_tag(
            record["record_kind"], None, record["record_id"], record["version"]
        )
        assert parse_record_tag(record["tag"]).dataset_id is None


@pytest.mark.parametrize("citable, readable", [(False, True), (True, False)])
@pytest.mark.asyncio
async def test_an_entity_expand_cannot_rebuild_is_shown_without_a_tag(citable, readable):
    # An ontology-pinned concept (not citable), and an entity whose neighborhood read
    # failed (no all_edges): expand would answer with another version, or with missing.
    unstamped = {
        "id": str(uuid4()),
        "name": "Customer",
        "type": "ontology class",
        "edges": [SHOWN_EDGE],
        **({"all_edges": [SHOWN_EDGE]} if readable else {}),
    }
    stamp_full_version(unstamped, citable=citable)
    assert not {"all_edges", "full_version", "full_chars"} & set(unstamped)
    retrieved = _retrieved()
    retrieved["entities"].insert(0, unstamped)

    (item,) = await build_cited_item(_payload(retrieved))

    assert _without_tags(item.text) == _regular_prompt(retrieved, _frame())
    lines = item.text.split("\n")
    assert not TAG_LINE.match(lines[lines.index("### Customer (ontology class)") - 1])
    entities = [r for r in item.metadata["records"] if r["record_kind"] == "entity"]
    assert [record["record_id"] for record in entities] == [ENTITY_ID]


@pytest.mark.asyncio
async def test_a_hybrid_payload_without_its_frame_is_an_internal_error():
    with pytest.raises(CogneeSystemError) as raised:
        await build_cited_item(_payload(_retrieved(), context_frame=None))
    assert raised.value.name == "CitedFrameMissingError"


def _evidence(edge_id, chunk_id=CHUNK_ID):
    return SimpleNamespace(
        edge_id=edge_id, data_id=DATA_ID, chunk_id=chunk_id, document_name="notes.md"
    )


@pytest.mark.asyncio
async def test_expanded_entity_sources_are_complete_and_shared_edges_count_for_both(monkeypatch):
    from uuid import UUID

    # More evidence rows than the citation lookup's 50-row cut, which expand must not use.
    edges = [{"text": f"edge {n}", "edge_object_id": str(uuid4())} for n in range(60)]
    first = {"id": ENTITY_ID, "all_edges": edges}
    second = {"id": str(uuid4()), "all_edges": edges[:1]}

    async def sources(dataset_id, edge_ids):
        assert dataset_id == DATASET_ID
        return [_evidence(edge_id, chunk_id=str(uuid4())) for edge_id in edge_ids]

    monkeypatch.setattr(recall_cited, "get_edge_sources", sources)

    by_entity = await recall_cited._entity_sources([first, second], [DATASET_ID])

    assert len(by_entity[ENTITY_ID]) == 60
    assert len(by_entity[second["id"]]) == 1
    assert UUID(by_entity[second["id"]][0].data_id) == UUID(DATA_ID)


@pytest.mark.asyncio
async def test_a_failing_sources_lookup_fails_the_expand(monkeypatch):
    async def broken(dataset_id, edge_ids):
        raise RuntimeError("evidence table unavailable")

    monkeypatch.setattr(recall_cited, "get_edge_sources", broken)
    with pytest.raises(RuntimeError, match="evidence table unavailable"):
        await recall_cited._entity_sources(
            [{"id": ENTITY_ID, "all_edges": [SHOWN_EDGE]}], [DATASET_ID]
        )


@pytest.mark.parametrize("error", [PermissionDeniedError, None])
@pytest.mark.asyncio
async def test_an_unknown_or_unreadable_dataset_is_not_readable(monkeypatch, error):
    async def authorized(datasets, permission_type, user):
        assert permission_type == "read"
        if error is not None:
            raise error("no read access")
        return []

    monkeypatch.setattr(recall_cited, "get_authorized_existing_datasets", authorized)
    assert await recall_cited._readable_dataset(uuid4(), object()) is None


def test_a_tag_names_the_record_it_was_made_for():
    version = compute_content_hash("body")
    tag = record_tag("entity", DATASET_ID, ENTITY_ID, version)
    assert parse_record_tag(f"  {tag} ") == ExpandItem(
        dataset_id=DATASET_ID, record_kind="entity", record_id=ENTITY_ID, version=version
    )
    # A shared-store tag (access control off) names no dataset.
    shared = record_tag("entity", None, ENTITY_ID, version)
    assert shared == f"[mem:entity/{ENTITY_ID}@{version}]"
    assert parse_record_tag(shared) == ExpandItem(
        record_kind="entity", record_id=ENTITY_ID, version=version
    )
    for malformed in (
        tag[:-1],
        tag.replace("entity", "grounding"),
        f"[mem:entity/{DATASET_ID}/chunk-1@{version}]",
        f"{tag} trailing",
    ):
        with pytest.raises(ValueError):
            parse_record_tag(malformed)


@pytest.mark.asyncio
async def test_expand_returns_bodies_missing_and_forbidden_in_request_order(monkeypatch):
    readable = SimpleNamespace(id=DATASET_ID, name="agent_sessions", owner_id=uuid4())
    other_dataset = uuid4()
    llm_config, embedding_config = object(), object()

    async def readable_dataset(dataset_id, user):
        return readable if dataset_id == DATASET_ID else None

    contexts = []

    @asynccontextmanager
    async def dataset_context(*args, **kwargs):
        contexts.append((args, kwargs))
        yield

    calls = []

    class FakeRetriever:
        async def expand_records(self, ids_by_kind):
            calls.append(ids_by_kind)
            return {
                ("passage", CHUNK_ID): {
                    "body": "full passage body",
                    "conflicts": [],
                    "item": {"id": CHUNK_ID, "document_id": DATA_ID, "document_name": "n.md"},
                },
                ("fact", FACT_ID): {"body": FACT_TEXT, "conflicts": [], "item": {}},
            }

    async def retriever_factory(query_type, query_text, **kwargs):
        assert query_type is SearchType.HYBRID_COMPLETION
        assert kwargs["include_references"] is True
        return FakeRetriever()

    monkeypatch.setattr(recall_cited, "_readable_dataset", readable_dataset)
    monkeypatch.setattr(recall_cited, "set_database_global_context_variables", dataset_context)
    monkeypatch.setattr(recall_cited, "get_search_type_retriever_instance", retriever_factory)
    monkeypatch.setattr(recall_cited, "backend_access_control_enabled", lambda: True)

    gone_version = compute_content_hash("gone")
    items = [
        ExpandItem(dataset_id=DATASET_ID, record_kind="passage", record_id=CHUNK_ID),
        ExpandItem(dataset_id=DATASET_ID, record_kind="fact", record_id=FACT_ID),
        parse_record_tag(record_tag("fact", DATASET_ID, GONE_ID, gone_version)),
        ExpandItem(dataset_id=other_dataset, record_kind="passage", record_id=OTHER_CHUNK_ID),
        ExpandItem(dataset_id=DATASET_ID, record_kind="passage", record_id=CHUNK_ID),
        # A shared-store tag needs access control off.
        parse_record_tag(record_tag("passage", None, OTHER_CHUNK_ID, gone_version)),
    ]
    rows = await expand_records(
        items,
        user=object(),
        retriever_kwargs={"include_references": True},
        llm_config=llm_config,
        embedding_config=embedding_config,
    )

    assert [row.metadata["status"] for row in rows] == [
        "available",
        "available",
        "missing",
        "forbidden",
        "available",
        "forbidden",
    ]
    assert rows[0].text == "full passage body"
    assert rows[0].metadata["version"] == compute_content_hash("full passage body")
    assert rows[0].metadata["tag"] == record_tag(
        "passage", DATASET_ID, CHUNK_ID, rows[0].metadata["version"]
    )
    # One result per requested item: the repeated item repeats its row.
    assert rows[4] == rows[0]
    # The same body, and so the same version, a cited prompt shows for this fact.
    assert rows[1].text == f"- {FACT_TEXT}"
    assert rows[1].metadata["version"] == compute_content_hash(f"- {FACT_TEXT}")
    # A status row echoes the requested tag, so it can be sent again; an item sent
    # without a version has no tag to echo.
    assert rows[2].text == "" and rows[3].text == ""
    assert rows[2].metadata["tag"] == record_tag("fact", DATASET_ID, GONE_ID, gone_version)
    assert parse_record_tag(rows[2].metadata["tag"]) == items[2]
    assert rows[3].metadata["tag"] is None
    assert rows[5].metadata["dataset_id"] is None
    # One store switch for the readable dataset, with its ids grouped by kind and the
    # caller's configs forwarded (a dataset built with another embedding width).
    assert calls == [{"passage": [CHUNK_ID], "fact": [FACT_ID, GONE_ID]}]
    assert contexts == [
        (
            (DATASET_ID, readable.owner_id),
            {"llm_config": llm_config, "embedding_config": embedding_config},
        )
    ]


@pytest.mark.asyncio
async def test_expand_reads_a_shared_store_tag_with_access_control_off(monkeypatch):
    user = SimpleNamespace(id=uuid4())
    contexts = []

    @asynccontextmanager
    async def dataset_context(*args, **kwargs):
        contexts.append(args)
        yield

    class FakeRetriever:
        async def expand_records(self, ids_by_kind):
            return {("fact", FACT_ID): {"body": FACT_TEXT, "conflicts": [], "item": {}}}

    async def retriever_factory(query_type, query_text, **kwargs):
        assert kwargs["dataset"] is None
        return FakeRetriever()

    async def authorized(datasets, permission_type, user):
        return [SimpleNamespace(id=uuid4())]

    monkeypatch.setattr(recall_cited, "backend_access_control_enabled", lambda: False)
    monkeypatch.setattr(recall_cited, "get_authorized_existing_datasets", authorized)
    monkeypatch.setattr(recall_cited, "set_database_global_context_variables", dataset_context)
    monkeypatch.setattr(recall_cited, "get_search_type_retriever_instance", retriever_factory)

    version = compute_content_hash(f"- {FACT_TEXT}")
    (row,) = await expand_records(
        [parse_record_tag(record_tag("fact", None, FACT_ID, version))],
        user=user,
        retriever_kwargs={},
    )
    assert row.metadata["status"] == "available"
    assert row.metadata["version"] == version
    assert row.metadata["tag"] == record_tag("fact", None, FACT_ID, version)
    assert contexts == [(None, user.id)]


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
        ({"result_mode": "index"}, "InvalidResultModeError"),
        ({"result_mode": "cited", "only_context": False}, "CitedRequiresOnlyContextError"),
        (
            {"result_mode": "cited", "query_type": SearchType.CHUNKS},
            "CitedUnsupportedSearchTypeError",
        ),
        ({"result_mode": "cited", "sources": ["session", "graph"]}, "CitedRequiresGraphScopeError"),
        ({"expand_items": []}, "InvalidExpandItemsError"),
        ({"expand_items": [object()] * 21}, "InvalidExpandItemsError"),
        ({"expand_items": [object()], "result_mode": "cited"}, "InvalidExpandRequestError"),
        # Requests hybrid would hand to GRAPH_COMPLETION (request_deferral_reason).
        ({"result_mode": "cited", "node_type": Entity}, "CitedRequiresHybridError"),
        ({"result_mode": "cited", "neighborhood_depth": 2}, "CitedRequiresHybridError"),
        # Also what a deployment's DEFAULT_FEEDBACK_INFLUENCE > 0 passes by default.
        ({"result_mode": "cited", "feedback_influence": 0.3}, "CitedRequiresHybridError"),
    ],
)
def test_invalid_cited_and_expand_requests_are_refused(kwargs, error_name):
    with pytest.raises(CogneeValidationError) as raised:
        _validate_cited_request(**{**_VALID_REQUEST, **kwargs})
    assert raised.value.name == error_name


def test_valid_cited_and_expand_requests_pass():
    _validate_cited_request(
        **{**_VALID_REQUEST, "result_mode": "cited", "query_type": SearchType.HYBRID_COMPLETION}
    )
    # Expand runs no hybrid search, so the graph-only options do not refuse it.
    _validate_cited_request(
        **{
            **_VALID_REQUEST,
            "expand_items": [object()] * 20,
            "only_context": False,
            "sources": ["session", "graph"],
            "node_type": Entity,
            "feedback_influence": 0.3,
        }
    )


def test_expand_items_are_accepted_as_tags_models_and_dicts_in_either_spelling():
    item = ExpandItem(dataset_id=DATASET_ID, record_kind="passage", record_id=CHUNK_ID)
    version = compute_content_hash("body")
    tag = record_tag("passage", DATASET_ID, CHUNK_ID, version)
    snake = {"dataset_id": str(DATASET_ID), "record_kind": "passage", "record_id": CHUNK_ID}
    camel = {"datasetId": str(DATASET_ID), "recordKind": "passage", "recordId": CHUNK_ID.upper()}
    assert _expand_items_as_models([tag, item, snake, camel]) == [
        item.model_copy(update={"version": version}),
        item,
        item,
        item,
    ]
    assert _expand_items_as_models(None) is None


@pytest.mark.parametrize(
    "bad_item",
    [
        {"dataset_id": str(DATASET_ID), "record_kind": "template_prefix", "record_id": CHUNK_ID},
        {"dataset_id": str(DATASET_ID), "record_kind": "passage"},
        {"dataset_id": str(DATASET_ID), "record_kind": "passage", "record_id": "chunk-1"},
        f"[mem:passage/{DATASET_ID}/{CHUNK_ID}]",
        # A version is a content hash; anything else would be echoed into a tag no
        # parser accepts.
        {
            "dataset_id": str(DATASET_ID),
            "record_kind": "passage",
            "record_id": CHUNK_ID,
            "version": "<script>x</script>",
        },
    ],
)
def test_an_invalid_expand_item_is_refused_with_its_position(bad_item):
    snake = {"dataset_id": str(DATASET_ID), "record_kind": "passage", "record_id": CHUNK_ID}
    with pytest.raises(CogneeValidationError) as raised:
        _expand_items_as_models([snake, bad_item])
    assert raised.value.name == "InvalidExpandItemsError"
    assert "expand_items[1]" in raised.value.message


def test_cited_routes_unsupported_picks_to_the_default():
    quoted = '"exact phrase"'
    assert route_query(quoted, RoutingConstraints()).search_type is SearchType.CHUNKS_LEXICAL
    cited = route_query(quoted, RoutingConstraints(cited_types=CITED_SEARCH_TYPES))
    assert cited.search_type is SearchType.HYBRID_COMPLETION
    assert RoutingConstraints.from_config(None, result_mode="cited").cited_types == (
        CITED_SEARCH_TYPES
    )
    assert RoutingConstraints.from_config(None).cited_types is None


def test_health_lists_the_recall_cited_capability():
    assert RECALL_CITED_CAPABILITY == "recall_cited_v1"
    assert RECALL_CITED_CAPABILITY in get_capabilities()


def test_recall_request_body_accepts_cited_and_expand_fields():
    from pydantic import ValidationError

    from cognee.api.v1.recall.routers.get_recall_router import RecallPayloadDTO

    tag = record_tag("passage", DATASET_ID, CHUNK_ID, compute_content_hash("body"))
    body = RecallPayloadDTO.model_validate(
        {
            "query": "q",
            "resultMode": "cited",
            "expandItems": [
                {"dataset_id": str(DATASET_ID), "record_kind": "passage", "record_id": CHUNK_ID},
                {"datasetId": str(DATASET_ID), "recordKind": "passage", "recordId": CHUNK_ID},
                tag,
            ],
        }
    )
    assert body.result_mode == "cited"
    assert body.expand_items[0] == body.expand_items[1]
    assert body.expand_items[0].dataset_id == DATASET_ID
    assert body.expand_items[2] == tag
    with pytest.raises(ValidationError):
        RecallPayloadDTO(query="q", result_mode="index")


@pytest.mark.asyncio
async def test_the_real_prompt_chain_strips_back_to_the_regular_prompt(monkeypatch):
    """The contract end to end: hybrid's context builder, the only_context prompt, the
    payload frame and the cited rebuild, with every frame part filled."""
    from cognee.modules.retrieval import hybrid_retriever, only_context_prompt
    from cognee.modules.retrieval.utils.completion import SessionPrompt
    from cognee.modules.search.methods.get_retriever_output import _context_frame

    async def grounding(query, context):
        return "## Ontology grounding\n- customer is a class"

    async def global_context(self, query, retrieved_objects):
        return "## Global context\nroot summary"

    monkeypatch.setattr(hybrid_retriever, "grounding_block_for", grounding)
    monkeypatch.setattr(
        hybrid_retriever.HybridRetriever, "_build_global_context_section", global_context
    )
    monkeypatch.setattr(
        only_context_prompt,
        "load_read_only_session_prompt",
        lambda *args, **kwargs: _async(SessionPrompt(history="Q: earlier", guidance="- be brief")),
    )
    retrieved = _retrieved()
    retrieved["chunks"].append({"id": str(uuid4()), "text": "   ", "document_id": None})
    retrieved["chunks"].append({"id": OTHER_CHUNK_ID, "text": "Second passage."})
    retrieved["entities"].append({"id": str(uuid4()), "name": "", "edges": []})

    retriever = hybrid_retriever.HybridRetriever()
    get_context = hybrid_retriever.HybridRetriever.get_context_from_objects
    get_context = getattr(get_context, "__wrapped__", get_context)
    context = await get_context(retriever, query="q?", retrieved_objects=retrieved)
    prompts = await only_context_prompt.build_only_context_prompt(
        retriever, query="q?", context=context
    )
    payload_ = _payload(
        retrieved,
        context=context,
        user_prompt=prompts[0],
        system_prompt=prompts[1],
        context_frame=_context_frame(retriever, prompts),
    )

    (item,) = await build_cited_item(payload_)

    assert _without_tags(item.text) == prompts[0]
    assert "## Ontology grounding" in item.text and "## Global context" in item.text
    # The empty passage and the nameless entity get no tag.
    kinds = [record["record_kind"] for record in item.metadata["records"]]
    assert kinds == ["passage", "passage", "entity", "fact"]


async def _async(value):
    return value


@pytest.mark.parametrize("rendered", ["no context here", "MARK twice MARK"])
def test_a_template_without_exactly_one_context_cannot_be_split(monkeypatch, rendered):
    from cognee.modules.retrieval import only_context_prompt

    monkeypatch.setattr(
        only_context_prompt,
        "render_prompt",
        lambda path, variables: rendered.replace("MARK", variables["context"]),
    )
    assert only_context_prompt._split_user_template("template.txt", "q") == (None, None)
