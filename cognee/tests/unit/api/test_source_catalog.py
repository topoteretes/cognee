"""Discovery is provider-independent and authorization happens before metadata/LLM reads."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.modules.data import source_catalog as catalog


def dataset(name="workspace"):
    return SimpleNamespace(id=uuid4(), name=name)


@pytest.fixture(autouse=True)
def _fresh_catalog_cache():
    catalog.clear_source_catalog_cache()
    yield
    catalog.clear_source_catalog_cache()


class _Session:
    """An async session stand-in that returns fixed rows and counts reads."""

    def __init__(self, rows, reads):
        self.rows, self.reads = rows, reads

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, statement):
        self.reads.append(statement)
        return SimpleNamespace(all=lambda: list(self.rows))


@pytest.mark.asyncio
async def test_catalog_descriptors_are_cached_briefly_per_user_and_datasets(monkeypatch):
    one, other = dataset("research"), dataset("ops")
    rows = [(one.id, ["research"], None, "Doc", "source", None)]
    reads, clock = [], [0.0]
    monkeypatch.setattr(catalog, "_monotonic", lambda: clock[0])
    monkeypatch.setattr(
        catalog,
        "get_relational_engine",
        lambda: SimpleNamespace(get_async_session=lambda: _Session(rows, reads)),
    )
    permission = AsyncMock(return_value=[one])
    monkeypatch.setattr(catalog, "get_all_user_permission_datasets", permission)
    alice, bob = SimpleNamespace(id=uuid4()), SimpleNamespace(id=uuid4())

    first = await catalog.source_catalog(alice)
    assert len(first["items"]) == 2 and len(reads) == 1
    # Within the TTL the rows are not re-read, but authorization still runs live.
    assert await catalog.source_catalog(alice) == first
    assert await catalog.source_catalog(alice, [one.id]) == first
    assert len(reads) == 1 and permission.await_count == 3
    # Another user is another entry.
    await catalog.source_catalog(bob)
    assert len(reads) == 2
    # An explicit refresh bypasses the entry and replaces it.
    await catalog.source_catalog(alice, refresh=True)
    assert len(reads) == 3
    # Past the TTL the next call rebuilds.
    clock[0] += catalog.CATALOG_CACHE_TTL_SECONDS + 1
    await catalog.source_catalog(alice)
    assert len(reads) == 4
    # A changed grant changes the key: the cached list for the old grant is not served.
    permission.return_value = [one, other]
    assert len((await catalog.source_catalog(alice))["items"]) == 3 and len(reads) == 5
    permission.return_value = []
    assert (await catalog.source_catalog(alice))["items"] == [] and len(reads) == 5
    # The cache is bounded: the oldest entry is evicted first.
    monkeypatch.setattr(catalog, "CATALOG_CACHE_MAX_ENTRIES", 2)
    permission.return_value = [one]
    carol = SimpleNamespace(id=uuid4())
    await catalog.source_catalog(bob)
    await catalog.source_catalog(carol)
    await catalog.source_catalog(alice)
    assert len(catalog._catalog_cache) == 2
    await catalog.source_catalog(bob)
    assert reads and len(reads) == 9  # bob was evicted and rebuilt


def test_new_sources_and_node_sets_need_no_provider_registration():
    data = dataset()
    rows = [
        (
            data.id,
            [f"custom-source:{i}"],
            None,
            f"Lab {i} research",
            f"provider-{i}",
            f"Knowledge about experiment {i}",
        )
        for i in range(300)
    ]
    result = catalog.build_catalog([data], rows)
    assert len(result) == 301
    assert any("provider-299" in r["aliases"] for r in result)
    assert len({r["id"] for r in result}) == len(result)
    assert catalog.build_catalog([data], rows) == result


def test_same_node_set_in_two_datasets_has_distinct_identity():
    one, two = dataset(), dataset()
    rows = [(d.id, ["same"], None, None, None, None) for d in (one, two)]
    targets = catalog.build_catalog([one, two], rows)
    nodes = [t for t in targets if t["kind"] == "node_set"]
    assert len(nodes) == 2 and nodes[0]["id"] != nodes[1]["id"]


@pytest.mark.asyncio
async def test_unauthorized_dataset_fails_before_database_or_llm(monkeypatch):
    one, hidden = dataset(), dataset()
    monkeypatch.setattr(catalog, "get_all_user_permission_datasets", AsyncMock(return_value=[one]))
    monkeypatch.setattr(
        catalog, "get_relational_engine", lambda: pytest.fail("unauthorized DB read")
    )
    with pytest.raises(PermissionError):
        await catalog.source_catalog(SimpleNamespace(id=uuid4()), [hidden.id])


@pytest.mark.asyncio
async def test_hundreds_of_descriptors_are_routed_in_bounded_batches(monkeypatch):
    data = dataset()
    items = catalog.build_catalog(
        [data], [(data.id, [f"unknown:{i}"], None, f"Project {i}", None, None) for i in range(200)]
    )
    target = next(item for item in items if item["name"] == "unknown:177")
    monkeypatch.setattr(
        catalog, "source_catalog", AsyncMock(return_value={"items": items, "complete": True})
    )
    sizes = []

    async def rank(query, hint, descriptors):
        sizes.append(len(descriptors))
        return catalog.SourceChoices(
            choices=[
                catalog.SourceChoice(
                    source_id=target["id"], relevance=0.9, reason="Matches the requested project"
                )
            ]
            if any(d["id"] == target["id"] for d in descriptors)
            else []
        )

    monkeypatch.setattr(catalog, "rank_descriptors", rank)
    result = await catalog.route_sources(None, "project 177")
    assert [t["id"] for t in result["targets"]] == [target["id"]]
    assert max(sizes) <= 64
    assert result["catalog_entries"] == 201
    assert result["searched_content"] is False


@pytest.mark.asyncio
async def test_llm_cannot_invent_an_unauthorized_target(monkeypatch):
    items = catalog.build_catalog([dataset("a"), dataset("b")], [])
    monkeypatch.setattr(
        catalog, "source_catalog", AsyncMock(return_value={"items": items, "complete": True})
    )
    monkeypatch.setattr(
        catalog,
        "rank_descriptors",
        AsyncMock(
            return_value=catalog.SourceChoices(
                choices=[
                    catalog.SourceChoice(source_id=str(uuid4()), relevance=1, reason="injected")
                ]
            )
        ),
    )
    with pytest.raises(ValueError, match="unknown catalog ID"):
        await catalog.route_sources(None, "question")


@pytest.mark.asyncio
async def test_metadata_budget_is_not_silent_catalog_truncation(monkeypatch):
    items = catalog.build_catalog([dataset(str(i)) for i in range(20)], [])
    monkeypatch.setattr(
        catalog, "source_catalog", AsyncMock(return_value={"items": items, "complete": True})
    )
    rank = AsyncMock()
    monkeypatch.setattr(catalog, "rank_descriptors", rank)
    result = await catalog.route_sources(None, "question", max_catalog_entries=10)
    assert result["status"] == "catalog_budget_exceeded"
    assert not result["targets"]
    rank.assert_not_awaited()


@pytest.mark.asyncio
async def test_inconclusive_routing_does_not_claim_no_content(monkeypatch):
    monkeypatch.setattr(
        catalog, "source_catalog", AsyncMock(return_value={"items": [], "complete": True})
    )
    result = await catalog.route_sources(None, "question")
    assert result["status"] == "inconclusive"
    assert result["searched_content"] is False


@pytest.mark.asyncio
async def test_router_drops_agent_memory_but_keeps_the_callers_context(monkeypatch):
    from cognee.context_global_variables import llm_config
    from cognee.infrastructure.llm.LLMGateway import LLMGateway
    from cognee.modules.agent_memory.runtime import (
        _agent_memory_context_var,
        get_current_agent_memory_context,
    )
    from cognee.modules.operations.usage_accumulator import (
        get_active_operation_usage,
        operation_usage_scope,
    )

    override, seen = object(), {}

    async def operation(**kwargs):
        seen["memory"] = get_current_agent_memory_context()
        seen["llm_config"] = llm_config.get()
        seen["usage"] = get_active_operation_usage()
        return catalog.RoutingChoices()

    monkeypatch.setattr(LLMGateway, "acreate_structured_output", operation)
    memory = _agent_memory_context_var.set(SimpleNamespace(memory_context="another session"))
    config = llm_config.set(override)
    try:
        with operation_usage_scope() as usage:
            await catalog.rank_descriptors("question", None, [])
    finally:
        llm_config.reset(config)
        _agent_memory_context_var.reset(memory)
    # Another agent's memory never reaches the router prompt, while the caller's
    # LLM override and operation usage tally stay in effect for the routing call.
    assert seen["memory"] is None
    assert seen["llm_config"] is override
    assert seen["usage"] is usage


@pytest.mark.asyncio
async def test_single_candidate_is_selected_without_a_routing_call(monkeypatch):
    items = catalog.build_catalog([dataset("only")], [])
    monkeypatch.setattr(
        catalog, "source_catalog", AsyncMock(return_value={"items": items, "complete": True})
    )
    rank = AsyncMock(side_effect=AssertionError("one candidate needs no LLM"))
    monkeypatch.setattr(catalog, "rank_descriptors", rank)
    result = await catalog.route_sources(None, "anything")
    assert [t["id"] for t in result["targets"]] == [items[0]["id"]]
    assert result["routing_llm_calls"] == 0
    assert result["source_resolution"] == "single_candidate"
    rank.assert_not_awaited()


def test_routing_descriptor_is_bounded():
    data = dataset()
    rows = [(data.id, ["set"], None, f"label {i}", f"alias {i}", "d" * 500) for i in range(12)]
    item = next(i for i in catalog.build_catalog([data], rows) if i["kind"] == "node_set")
    assert len(item["aliases"]) == 12  # the catalog keeps everything for callers
    descriptor = catalog.routing_descriptor(item)
    assert "id" not in descriptor and "dataset_id" not in descriptor
    assert len(descriptor["aliases"]) == 3
    assert len(descriptor["descriptions"]) == 1 and len(descriptor["descriptions"][0]) == 120
    assert descriptor["documents"] == 12 and descriptor["retrieval_method"] == "chunks"


@pytest.mark.asyncio
async def test_large_catalog_reduces_winners_and_excludes_searched_targets(monkeypatch):
    items = catalog.build_catalog([dataset(str(i)) for i in range(900)], [])
    monkeypatch.setattr(
        catalog, "source_catalog", AsyncMock(return_value={"items": items, "complete": True})
    )
    excluded = items[0]["id"]
    sizes = []

    async def rank(query, hint, descriptors):
        assert excluded not in [d["id"] for d in descriptors]
        sizes.append(len(descriptors))
        return catalog.SourceChoices(
            choices=[
                catalog.SourceChoice(source_id=d["id"], relevance=0.9, reason="relevant")
                for d in descriptors[:8]
            ]
        )

    monkeypatch.setattr(catalog, "rank_descriptors", rank)
    from uuid import UUID

    result = await catalog.route_sources(
        None, "broad query", max_catalog_entries=1000, exclude_source_ids=[UUID(excluded)]
    )
    assert result["status"] == "selected"
    assert len(result["targets"]) == 6
    assert max(sizes) <= 64
    assert result["routing_llm_calls"] == len(sizes)


@pytest.mark.asyncio
async def test_native_metadata_projection_pages_aliases_and_revocation(monkeypatch):
    """Use a real isolated relational table; no graph, source fetch, or live state."""
    from uuid import UUID

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from cognee.modules.data.methods import resolve_data_id

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    engine_wrapper = SimpleNamespace(get_async_session=sessions)
    monkeypatch.setattr(catalog, "get_relational_engine", lambda: engine_wrapper)
    # The native alias resolver uses the same isolated relational engine.
    monkeypatch.setitem(
        resolve_data_id.__globals__, "get_relational_engine", lambda: engine_wrapper
    )
    one, hidden = dataset("research"), dataset("private")
    permission = AsyncMock(return_value=[one])
    monkeypatch.setattr(catalog, "get_all_user_permission_datasets", permission)
    monkeypatch.setattr(catalog, "get_authorized_existing_datasets", permission)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(catalog.Data.__table__.create)
        async with sessions() as db:
            for index in range(1, 6):
                db.add(
                    catalog.Data(
                        id=UUID(int=(10 << 124) + index),
                        dataset_id=one.id if index < 5 else hidden.id,
                        legacy_id=UUID(int=(10 << 124) + 99) if index == 1 else None,
                        node_set=["research"] if index != 2 else ["other"],
                        label=f"Document {index}",
                        raw_data_location="must-not-be-exposed",
                        external_metadata={"source": "new-provider", "secret": "never-in-catalog"},
                    )
                )
            await db.commit()
        result = await catalog.source_catalog(None)
        assert len(result["items"]) == 3
        assert all(item["dataset_id"] == str(one.id) for item in result["items"])
        assert "never-in-catalog" not in str(result)
        assert "must-not-be-exposed" not in str(result)
        assert any("new-provider" in item["aliases"] for item in result["items"])
        source_id = catalog.target_id(one.id, "research")
        # Paging authorizes the one source directly; it never rebuilds the catalog.
        real_source_catalog = catalog.source_catalog
        monkeypatch.setattr(
            catalog, "source_catalog", AsyncMock(side_effect=AssertionError("catalog rebuilt"))
        )
        page = await catalog.source_documents(None, source_id, limit=2)
        assert page["source"]["id"] == source_id and page["source"]["node_sets"] == ["research"]
        assert [item["id"] for item in page["items"]] == [
            str(UUID(int=(10 << 124) + 1)),
            str(UUID(int=(10 << 124) + 3)),
        ]
        page2 = await catalog.source_documents(
            None, source_id, after=UUID(page["next_cursor"]), limit=2
        )
        assert [item["id"] for item in page2["items"]] == [str(UUID(int=(10 << 124) + 4))]
        assert page2["next_cursor"] is None
        # A sparse source never scans the dataset unboundedly: with the scan cap at
        # one row, pages come back short (even empty, for the non-matching row) but
        # the cursor keeps advancing, and following it covers every document.
        monkeypatch.setattr(catalog, "MAX_SCAN_ROWS", 1)
        collected, cursor, pages = [], None, 0
        while True:
            page = await catalog.source_documents(None, source_id, after=cursor, limit=2)
            assert len(page["items"]) <= 1
            collected.extend(item["id"] for item in page["items"])
            pages += 1
            cursor = page["next_cursor"]
            if cursor is None:
                break
        assert pages == 4
        assert collected == [str(UUID(int=(10 << 124) + i)) for i in (1, 3, 4)]
        monkeypatch.setattr(catalog, "MAX_SCAN_ROWS", 10_000)
        old = await catalog.source_document(None, one.id, UUID(int=(10 << 124) + 99))
        assert old["id"] == str(UUID(int=(10 << 124) + 1))
        # One batched load per source: exact and legacy ids resolve, keyed as
        # requested; another dataset's row and an unknown id are simply absent.
        batch = await catalog.source_documents_by_id(
            None,
            one.id,
            [
                str(UUID(int=(10 << 124) + 99)),
                UUID(int=(10 << 124) + 3),
                UUID(int=(10 << 124) + 5),
                uuid4(),
            ],
        )
        assert {k: v["id"] for k, v in batch.items()} == {
            str(UUID(int=(10 << 124) + 99)): str(UUID(int=(10 << 124) + 1)),
            str(UUID(int=(10 << 124) + 3)): str(UUID(int=(10 << 124) + 3)),
        }
        assert batch[str(UUID(int=(10 << 124) + 3))]["node_sets"] == ["research"]
        with pytest.raises(PermissionError):
            await catalog.source_document(None, one.id, UUID(int=(10 << 124) + 5))
        monkeypatch.setattr(catalog, "source_catalog", real_source_catalog)
        monkeypatch.setattr(catalog, "MAX_METADATA_ROWS", 2)
        assert (await catalog.source_catalog(None, refresh=True))["complete"] is False
        # A dataset target needs no metadata scan at all, and scoping a node-set
        # lookup to its dataset keeps it reachable when the user's other datasets
        # would exhaust the row cap first.
        permission.return_value = [one, hidden]
        monkeypatch.setattr(catalog, "MAX_METADATA_ROWS", 1)
        whole = await catalog.source_documents(None, catalog.target_id(hidden.id))
        assert [item["id"] for item in whole["items"]] == [str(UUID(int=(10 << 124) + 5))]
        scoped = await catalog.source_documents(
            None, catalog.target_id(hidden.id, "research"), dataset_id=hidden.id
        )
        assert [item["id"] for item in scoped["items"]] == [str(UUID(int=(10 << 124) + 5))]
        with pytest.raises(PermissionError):
            # A dataset the user cannot read is not a valid scope, whatever the id.
            await catalog.source_documents(None, source_id, dataset_id=uuid4())
        permission.return_value = []
        with pytest.raises(PermissionError):
            await catalog.source_documents(None, source_id, after=page["next_cursor"])
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_named_source_cannot_expand_to_mixed_topic_group(monkeypatch):
    data = dataset()
    items = catalog.build_catalog(
        [data],
        [
            (data.id, ["meetings", "origin:custom-a"], None, "meeting A", "Custom A", None),
            (data.id, ["meetings", "origin:custom-b"], None, "meeting B", "Custom B", None),
        ],
    )
    monkeypatch.setattr(
        catalog, "source_catalog", AsyncMock(return_value={"items": items, "complete": True})
    )

    # An exact name is a constraint the catalog resolves itself: no routing call.
    rank = AsyncMock(side_effect=AssertionError("exact match needs no LLM"))
    monkeypatch.setattr(catalog, "rank_descriptors", rank)
    result = await catalog.route_sources(None, "commitments", source_hint="Custom A")
    assert result["source_resolution"] == "exact_metadata"
    assert result["routing_llm_calls"] == 0
    assert [t["name"] for t in result["targets"]] == ["origin:custom-a"]
    assert result["targets"][0]["node_sets"] == ["origin:custom-a"]
    rank.assert_not_awaited()
    from uuid import UUID

    empty = await catalog.route_sources(
        None,
        "commitments",
        source_hint="Custom A",
        exclude_source_ids=[UUID(result["targets"][0]["id"])],
    )
    assert empty["status"] == "inconclusive" and empty["targets"] == []


@pytest.mark.asyncio
async def test_llm_indexes_resolve_only_to_supplied_targets(monkeypatch):
    from cognee.infrastructure.llm.LLMGateway import LLMGateway

    items = catalog.build_catalog([dataset()], [])
    method = AsyncMock(
        return_value=catalog.RoutingChoices(
            choices=[catalog.RoutingChoice(index=0, relevance=1, reason="Relevant")]
        )
    )
    monkeypatch.setattr(LLMGateway, "acreate_structured_output", method)
    result = await catalog.rank_descriptors("question", None, items)
    assert result.choices[0].source_id == items[0]["id"]
    assert items[0]["id"] not in method.await_args.kwargs["text_input"]
    method.return_value = catalog.RoutingChoices(
        choices=[catalog.RoutingChoice(index=999, relevance=1, reason="Invalid")]
    )
    with pytest.raises(ValueError, match="unknown catalog index"):
        await catalog.rank_descriptors("question", None, items)
